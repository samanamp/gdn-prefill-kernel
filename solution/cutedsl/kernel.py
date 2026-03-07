import math
import operator
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack, make_ptr
from cutlass import Float32, Int32, Int64, Boolean
import cutlass.torch as cutlass_torch

@cute.kernel
def gdn_prefill_kernel(
    mOldV: cute.Tensor,       # [total_seq_len, num_v_heads, head_size] fp32
    mNewV: cute.Tensor,       # [total_seq_len, num_v_heads, head_size] fp32
    mG: cute.Tensor,          # [total_seq_len, num_v_heads] fp32
    mQ: cute.Tensor,          # [total_seq_len, num_q_heads, head_size] bf16
    mK: cute.Tensor,          # [total_seq_len, num_k_heads, head_size] bf16
    mV: cute.Tensor,          # [total_seq_len, num_v_heads, head_size] bf16
    mState: cute.Tensor,      # [num_seqs, num_v_heads, head_size, head_size] fp32 [N,H,V,K]
    mA_log: cute.Tensor,      # [num_v_heads] fp32
    mA: cute.Tensor,          # [total_seq_len, num_v_heads] bf16
    mDtBias: cute.Tensor,     # [num_v_heads] fp32
    mBgate: cute.Tensor,      # [total_seq_len, num_v_heads] bf16
    mOutput: cute.Tensor,     # [total_seq_len, num_v_heads, head_size] bf16
    mNewState: cute.Tensor,   # [num_seqs, num_v_heads, head_size, head_size] fp32 [N,H,V,K]
    mCuSeqlens: cute.Tensor,  # [num_seqs + 1] int64
    scale: Float32,
    has_state: cutlass.Constexpr[bool],
    seq_end_limit: Int32,     # = total_seq_len, for bounds checking
):
    tidx, _, _ = cute.arch.thread_idx()
    seq_idx, v_head, v_tile_idx = cute.arch.block_idx()

    k_idx = tidx  # thread i owns K-dimension index i
    v_start = v_tile_idx * V_TILE_SIZE
    qk_head = v_head // GVA_RATIO

    # Load sequence bounds
    seq_start = mCuSeqlens[seq_idx]
    seq_end = mCuSeqlens[seq_idx + 1]

    # Load per-head constants (broadcast to all threads — redundant loads, but simple)
    A_log_val = mA_log[v_head]
    dt_bias_val = mDtBias[v_head]

    # ================================================================
    # Initialize state tile in registers: state[k_idx, v_start:v_start+V_TILE_SIZE]
    # Memory layout is [N, H, V, K] so state[seq, head, v, k]
    # ================================================================
    state_0 = Float32(0.0)
    state_1 = Float32(0.0)
    state_2 = Float32(0.0)
    state_3 = Float32(0.0)
    state_4 = Float32(0.0)
    state_5 = Float32(0.0)
    state_6 = Float32(0.0)
    state_7 = Float32(0.0)

    if cutlass.const_expr(has_state):
        state_0 = mState[(seq_idx, v_head, v_start + 0, k_idx)]
        state_1 = mState[(seq_idx, v_head, v_start + 1, k_idx)]
        state_2 = mState[(seq_idx, v_head, v_start + 2, k_idx)]
        state_3 = mState[(seq_idx, v_head, v_start + 3, k_idx)]
        state_4 = mState[(seq_idx, v_head, v_start + 4, k_idx)]
        state_5 = mState[(seq_idx, v_head, v_start + 5, k_idx)]
        state_6 = mState[(seq_idx, v_head, v_start + 6, k_idx)]
        state_7 = mState[(seq_idx, v_head, v_start + 7, k_idx)]

    # ================================================================
    # Shared memory for block-level reductions
    # Shape: (V_TILE_SIZE, NUM_WARPS) for cross-warp reduction
    # ================================================================
    smem = utils.SmemAllocator()
    reduce_buf = smem.allocate_tensor(
        Float32,
        cute.make_layout((V_TILE_SIZE, NUM_WARPS)),
        byte_alignment=16,
    )

    warp_idx = cute.arch.warp_idx()
    lane_idx = cute.arch.lane_idx()

    # ================================================================
    # Main loop: iterate over timesteps in this sequence
    # ================================================================
    for t in cutlass.range(seq_start, seq_end):

        # Load k[t, qk_head, k_idx] and q[t, qk_head, k_idx]
        k_val = mK[(t, qk_head, k_idx)].to(Float32)
        q_val = mQ[(t, qk_head, k_idx)].to(Float32)

        # Compute gate g and beta for (t, v_head)
        a_val = mA[(t, v_head)].to(Float32)
        b_val = mBgate[(t, v_head)].to(Float32)

        # g = exp(-exp(A_log) * softplus(a + dt_bias))
        x = a_val + dt_bias_val
        # softplus(x) = log(1 + exp(x))
        softplus_x = cute.math.log(Float32(1.0) + cute.math.exp(x, fastmath=True), fastmath=True)
        # softplus_x = x
        # if x <= Float32(20.0):
        #     softplus_x = cute.math.log(Float32(1.0) + cute.math.exp(x, fastmath=False), fastmath=False)
        g = cute.math.exp(-cute.math.exp(A_log_val, fastmath=True) * Float32(softplus_x), fastmath=True)
        mG[(t, v_head)] = softplus_x
        # cute.printf(g)
        # beta = sigmoid(b) = 1 / (1 + exp(-b))
        beta = Float32(1.0) / (Float32(1.0) + cute.math.exp(-b_val, fastmath=True))

        # ---- Step 1: Decay state ----
        state_0 = g * state_0
        state_1 = g * state_1
        state_2 = g * state_2
        state_3 = g * state_3
        state_4 = g * state_4
        state_5 = g * state_5
        state_6 = g * state_6
        state_7 = g * state_7

        # ---- Step 2: old_v[vi] = sum_k(k[k] * state[k, vi]) — reduce over K ----
        # Compute partial products (each thread has one K-element)
        partial_0 = k_val * state_0
        partial_1 = k_val * state_1
        partial_2 = k_val * state_2
        partial_3 = k_val * state_3
        partial_4 = k_val * state_4
        partial_5 = k_val * state_5
        partial_6 = k_val * state_6
        partial_7 = k_val * state_7

        # Warp reduction for each vi
        w0 = cute.arch.warp_reduction(partial_0, operator.add)
        w1 = cute.arch.warp_reduction(partial_1, operator.add)
        w2 = cute.arch.warp_reduction(partial_2, operator.add)
        w3 = cute.arch.warp_reduction(partial_3, operator.add)
        w4 = cute.arch.warp_reduction(partial_4, operator.add)
        w5 = cute.arch.warp_reduction(partial_5, operator.add)
        w6 = cute.arch.warp_reduction(partial_6, operator.add)
        w7 = cute.arch.warp_reduction(partial_7, operator.add)

        # Block reduction: lane 0 of each warp writes to smem
        if lane_idx == 0:
            reduce_buf[(0, warp_idx)] = w0
            reduce_buf[(1, warp_idx)] = w1
            reduce_buf[(2, warp_idx)] = w2
            reduce_buf[(3, warp_idx)] = w3
            reduce_buf[(4, warp_idx)] = w4
            reduce_buf[(5, warp_idx)] = w5
            reduce_buf[(6, warp_idx)] = w6
            reduce_buf[(7, warp_idx)] = w7
        cute.arch.barrier()

        # All threads read the 4 warp-reduced values and sum them
        old_v_0 = reduce_buf[(0, 0)] + reduce_buf[(0, 1)] + reduce_buf[(0, 2)] + reduce_buf[(0, 3)]
        old_v_1 = reduce_buf[(1, 0)] + reduce_buf[(1, 1)] + reduce_buf[(1, 2)] + reduce_buf[(1, 3)]
        old_v_2 = reduce_buf[(2, 0)] + reduce_buf[(2, 1)] + reduce_buf[(2, 2)] + reduce_buf[(2, 3)]
        old_v_3 = reduce_buf[(3, 0)] + reduce_buf[(3, 1)] + reduce_buf[(3, 2)] + reduce_buf[(3, 3)]
        old_v_4 = reduce_buf[(4, 0)] + reduce_buf[(4, 1)] + reduce_buf[(4, 2)] + reduce_buf[(4, 3)]
        old_v_5 = reduce_buf[(5, 0)] + reduce_buf[(5, 1)] + reduce_buf[(5, 2)] + reduce_buf[(5, 3)]
        old_v_6 = reduce_buf[(6, 0)] + reduce_buf[(6, 1)] + reduce_buf[(6, 2)] + reduce_buf[(6, 3)]
        old_v_7 = reduce_buf[(7, 0)] + reduce_buf[(7, 1)] + reduce_buf[(7, 2)] + reduce_buf[(7, 3)]
        mOldV[(t, v_head, v_start + 0)] = old_v_0
        mOldV[(t, v_head, v_start + 1)] = old_v_1
        mOldV[(t, v_head, v_start + 2)] = old_v_2
        mOldV[(t, v_head, v_start + 3)] = old_v_3
        mOldV[(t, v_head, v_start + 4)] = old_v_4
        mOldV[(t, v_head, v_start + 5)] = old_v_5
        mOldV[(t, v_head, v_start + 6)] = old_v_6
        mOldV[(t, v_head, v_start + 7)] = old_v_7
        cute.arch.barrier()

        # ---- Step 3: new_v = beta * v + (1 - beta) * old_v ----
        v_0 = mV[(t, v_head, v_start + 0)].to(Float32)
        v_1 = mV[(t, v_head, v_start + 1)].to(Float32)
        v_2 = mV[(t, v_head, v_start + 2)].to(Float32)
        v_3 = mV[(t, v_head, v_start + 3)].to(Float32)
        v_4 = mV[(t, v_head, v_start + 4)].to(Float32)
        v_5 = mV[(t, v_head, v_start + 5)].to(Float32)
        v_6 = mV[(t, v_head, v_start + 6)].to(Float32)
        v_7 = mV[(t, v_head, v_start + 7)].to(Float32)

        one_minus_beta = Float32(1.0) - beta
        new_v_0 = beta * v_0 + one_minus_beta * old_v_0
        new_v_1 = beta * v_1 + one_minus_beta * old_v_1
        new_v_2 = beta * v_2 + one_minus_beta * old_v_2
        new_v_3 = beta * v_3 + one_minus_beta * old_v_3
        new_v_4 = beta * v_4 + one_minus_beta * old_v_4
        new_v_5 = beta * v_5 + one_minus_beta * old_v_5
        new_v_6 = beta * v_6 + one_minus_beta * old_v_6
        new_v_7 = beta * v_7 + one_minus_beta * old_v_7

        mNewV[(t, v_head, v_start + 0)] = new_v_0
        mNewV[(t, v_head, v_start + 1)] = new_v_1
        mNewV[(t, v_head, v_start + 2)] = new_v_2
        mNewV[(t, v_head, v_start + 3)] = new_v_3
        mNewV[(t, v_head, v_start + 4)] = new_v_4
        mNewV[(t, v_head, v_start + 5)] = new_v_5
        mNewV[(t, v_head, v_start + 6)] = new_v_6
        mNewV[(t, v_head, v_start + 7)] = new_v_7

        # ---- Step 4: state[k, v] += k[k] * (new_v[v] - old_v[v]) ----
        # This is purely thread-local: each thread owns k_val and state[vi]
        state_0 = state_0 + k_val * (new_v_0 - old_v_0)
        state_1 = state_1 + k_val * (new_v_1 - old_v_1)
        state_2 = state_2 + k_val * (new_v_2 - old_v_2)
        state_3 = state_3 + k_val * (new_v_3 - old_v_3)
        state_4 = state_4 + k_val * (new_v_4 - old_v_4)
        state_5 = state_5 + k_val * (new_v_5 - old_v_5)
        state_6 = state_6 + k_val * (new_v_6 - old_v_6)
        state_7 = state_7 + k_val * (new_v_7 - old_v_7)

        # ---- Step 5: output[v] = scale * sum_k(q[k] * state[k, v]) — reduce over K ----
        p0 = q_val * state_0
        p1 = q_val * state_1
        p2 = q_val * state_2
        p3 = q_val * state_3
        p4 = q_val * state_4
        p5 = q_val * state_5
        p6 = q_val * state_6
        p7 = q_val * state_7

        o0 = cute.arch.warp_reduction(p0, operator.add)
        o1 = cute.arch.warp_reduction(p1, operator.add)
        o2 = cute.arch.warp_reduction(p2, operator.add)
        o3 = cute.arch.warp_reduction(p3, operator.add)
        o4 = cute.arch.warp_reduction(p4, operator.add)
        o5 = cute.arch.warp_reduction(p5, operator.add)
        o6 = cute.arch.warp_reduction(p6, operator.add)
        o7 = cute.arch.warp_reduction(p7, operator.add)

        if lane_idx == 0:
            reduce_buf[(0, warp_idx)] = o0
            reduce_buf[(1, warp_idx)] = o1
            reduce_buf[(2, warp_idx)] = o2
            reduce_buf[(3, warp_idx)] = o3
            reduce_buf[(4, warp_idx)] = o4
            reduce_buf[(5, warp_idx)] = o5
            reduce_buf[(6, warp_idx)] = o6
            reduce_buf[(7, warp_idx)] = o7
        cute.arch.barrier()

        # Thread 0 of warp 0 writes the output (only one thread needs to do the store)
        if warp_idx == 0:
            if lane_idx == 0:
                out_0 = scale * (reduce_buf[(0, 0)] + reduce_buf[(0, 1)] + reduce_buf[(0, 2)] + reduce_buf[(0, 3)])
                out_1 = scale * (reduce_buf[(1, 0)] + reduce_buf[(1, 1)] + reduce_buf[(1, 2)] + reduce_buf[(1, 3)])
                out_2 = scale * (reduce_buf[(2, 0)] + reduce_buf[(2, 1)] + reduce_buf[(2, 2)] + reduce_buf[(2, 3)])
                out_3 = scale * (reduce_buf[(3, 0)] + reduce_buf[(3, 1)] + reduce_buf[(3, 2)] + reduce_buf[(3, 3)])
                out_4 = scale * (reduce_buf[(4, 0)] + reduce_buf[(4, 1)] + reduce_buf[(4, 2)] + reduce_buf[(4, 3)])
                out_5 = scale * (reduce_buf[(5, 0)] + reduce_buf[(5, 1)] + reduce_buf[(5, 2)] + reduce_buf[(5, 3)])
                out_6 = scale * (reduce_buf[(6, 0)] + reduce_buf[(6, 1)] + reduce_buf[(6, 2)] + reduce_buf[(6, 3)])
                out_7 = scale * (reduce_buf[(7, 0)] + reduce_buf[(7, 1)] + reduce_buf[(7, 2)] + reduce_buf[(7, 3)])

                mOutput[(t, v_head, v_start + 0)] = out_0.to(cutlass.BFloat16)
                mOutput[(t, v_head, v_start + 1)] = out_1.to(cutlass.BFloat16)
                mOutput[(t, v_head, v_start + 2)] = out_2.to(cutlass.BFloat16)
                mOutput[(t, v_head, v_start + 3)] = out_3.to(cutlass.BFloat16)
                mOutput[(t, v_head, v_start + 4)] = out_4.to(cutlass.BFloat16)
                mOutput[(t, v_head, v_start + 5)] = out_5.to(cutlass.BFloat16)
                mOutput[(t, v_head, v_start + 6)] = out_6.to(cutlass.BFloat16)
                mOutput[(t, v_head, v_start + 7)] = out_7.to(cutlass.BFloat16)
        cute.arch.barrier()

    # ================================================================
    # Write back final state: [N, H, V, K] layout
    # ================================================================
    # cute.printf(f"scale_val exact: {scale:.15e}")
    mNewState[(seq_idx, v_head, v_start + 0, k_idx)] = state_0
    mNewState[(seq_idx, v_head, v_start + 1, k_idx)] = state_1
    mNewState[(seq_idx, v_head, v_start + 2, k_idx)] = state_2
    mNewState[(seq_idx, v_head, v_start + 3, k_idx)] = state_3
    mNewState[(seq_idx, v_head, v_start + 4, k_idx)] = state_4
    mNewState[(seq_idx, v_head, v_start + 5, k_idx)] = state_5
    mNewState[(seq_idx, v_head, v_start + 6, k_idx)] = state_6
    mNewState[(seq_idx, v_head, v_start + 7, k_idx)] = state_7


# ============================================================================
# Host function
# ============================================================================

@cute.jit
def gdn_prefill_host(
    # old_v_ptr: cute.Pointer,
    # new_v_ptr: cute.Pointer,
    # g_ptr: cute.Pointer,
    q_ptr: cute.Pointer,
    k_ptr: cute.Pointer,
    v_ptr: cute.Pointer,
    state_ptr: cute.Pointer,
    A_log_ptr: cute.Pointer,
    a_ptr: cute.Pointer,
    dt_bias_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    output_ptr: cute.Pointer,
    new_state_ptr: cute.Pointer,
    cu_seqlens_ptr: cute.Pointer,
    total_seq_len: Int32,
    num_seqs: Int32,
    scale: Float32,
    has_state: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
    NUM_V_HEADS: Int32,
    HEAD_SIZE: Int32,
    NUM_K_HEADS: Int32,
    V_TILES: Int32,
    V_TILE_SIZE: Int32,
    NUM_WARPS: Int32,
    THREADS_PER_CTA: Int32,
):
    # [total_seq_len, num_v_heads, head_size]
    # mOldV = cute.make_tensor(
    #     old_v_ptr,
    #     cute.make_layout(
    #         (total_seq_len, NUM_V_HEADS, HEAD_SIZE),
    #         stride=(NUM_V_HEADS * HEAD_SIZE, HEAD_SIZE, 1),
    #     ),
    # )

    # # [total_seq_len, num_v_heads, head_size]
    # mNewV = cute.make_tensor(
    #     new_v_ptr,
    #     cute.make_layout(
    #         (total_seq_len, NUM_V_HEADS, HEAD_SIZE),
    #         stride=(NUM_V_HEADS * HEAD_SIZE, HEAD_SIZE, 1),
    #     ),
    # )
    
    # mG = cute.make_tensor(
    #     g_ptr,
    #     cute.make_layout(
    #         (total_seq_len, NUM_V_HEADS),
    #         stride=(NUM_V_HEADS, 1),
    #     ),
    # )
    # Q: [total_seq_len, num_q_heads, head_size], row-major
    mQ = cute.make_tensor(
        q_ptr,
        cute.make_layout(
            (total_seq_len, NUM_K_HEADS, HEAD_SIZE),
            stride=(NUM_K_HEADS * HEAD_SIZE, HEAD_SIZE, 1),
        ),
    )

    # K: [total_seq_len, num_k_heads, head_size]
    mK = cute.make_tensor(
        k_ptr,
        cute.make_layout(
            (total_seq_len, NUM_K_HEADS, HEAD_SIZE),
            stride=(NUM_K_HEADS * HEAD_SIZE, HEAD_SIZE, 1),
        ),
    )

    # V: [total_seq_len, num_v_heads, head_size]
    mV = cute.make_tensor(
        v_ptr,
        cute.make_layout(
            (total_seq_len, NUM_V_HEADS, HEAD_SIZE),
            stride=(NUM_V_HEADS * HEAD_SIZE, HEAD_SIZE, 1),
        ),
    )

    # State: [num_seqs, num_v_heads, head_size, head_size] — [N, H, V, K]
    mState = cute.make_tensor(
        state_ptr,
        cute.make_layout(
            (num_seqs, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE),
            stride=(NUM_V_HEADS * HEAD_SIZE * HEAD_SIZE, HEAD_SIZE * HEAD_SIZE, HEAD_SIZE, 1),
        ),
    )

    # A_log: [num_v_heads]
    mA_log = cute.make_tensor(
        A_log_ptr,
        cute.make_layout((NUM_V_HEADS,), stride=(1,)),
    )

    # a: [total_seq_len, num_v_heads]
    mA = cute.make_tensor(
        a_ptr,
        cute.make_layout(
            (total_seq_len, NUM_V_HEADS),
            stride=(NUM_V_HEADS, 1),
        ),
    )

    # dt_bias: [num_v_heads]
    mDtBias = cute.make_tensor(
        dt_bias_ptr,
        cute.make_layout((NUM_V_HEADS,), stride=(1,)),
    )

    # b: [total_seq_len, num_v_heads]
    mBgate = cute.make_tensor(
        b_ptr,
        cute.make_layout(
            (total_seq_len, NUM_V_HEADS),
            stride=(NUM_V_HEADS, 1),
        ),
    )

    # Output: [total_seq_len, num_v_heads, head_size]
    mOutput = cute.make_tensor(
        output_ptr,
        cute.make_layout(
            (total_seq_len, NUM_V_HEADS, HEAD_SIZE),
            stride=(NUM_V_HEADS * HEAD_SIZE, HEAD_SIZE, 1),
        ),
    )

    # NewState: [num_seqs, num_v_heads, head_size, head_size]
    mNewState = cute.make_tensor(
        new_state_ptr,
        cute.make_layout(
            (num_seqs, NUM_V_HEADS, HEAD_SIZE, HEAD_SIZE),
            stride=(NUM_V_HEADS * HEAD_SIZE * HEAD_SIZE, HEAD_SIZE * HEAD_SIZE, HEAD_SIZE, 1),
        ),
    )

    # CuSeqlens: [num_seqs + 1]
    mCuSeqlens = cute.make_tensor(
        cu_seqlens_ptr,
        cute.make_layout((num_seqs + 1,), stride=(1,)),
    )

    # Grid: (num_seqs, num_v_heads, V_TILES)
    grid = (num_seqs, NUM_V_HEADS, V_TILES)

    # Shared memory: V_TILE_SIZE * NUM_WARPS * sizeof(float32)
    smem_bytes = V_TILE_SIZE * NUM_WARPS * 4

    gdn_prefill_kernel(
        # mOldV, mNewV, mG,
        mQ, mK, mV, mState,
        mA_log, mA, mDtBias, mBgate,
        mOutput, mNewState, mCuSeqlens,
        scale, has_state, total_seq_len,
    ).launch(
        grid=grid,
        block=(THREADS_PER_CTA, 1, 1),
        smem=smem_bytes,
        stream=stream,
    )

def cuteGDN(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, out, state_out):
    B, Hv, K, V = state.shape
    Hq = q.shape[1]
    total_seq_len = a.shape[0]
    V_TILE_SIZE = 8     # V elements per CTA
    V_TILES = K // V_TILE_SIZE  # 16
    THREADS_PER_CTA = 128  # one thread per K element
    NUM_WARPS = THREADS_PER_CTA // 32  # 4
    current_stream = cutlass_torch.default_stream()
    # old_v_cute_ptr = make_ptr(cutlass.Float32, old_v_cute.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    # new_v_cute_ptr = make_ptr(cutlass.Float32, new_v_cute.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    # g_cute = make_ptr(cutlass.Float32, g_zero.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    q_ptr = make_ptr(cutlass.BFloat16, q.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    k_ptr = make_ptr(cutlass.BFloat16, k.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    v_ptr = make_ptr(cutlass.BFloat16, v.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    state_ptr = make_ptr(cutlass.Float32, state.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    A_log_ptr = make_ptr(cutlass.Float32, A_log.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    dt_bias_ptr = make_ptr(cutlass.Float32, dt_bias.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    b_ptr = make_ptr(cutlass.BFloat16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    output_ptr = make_ptr(cutlass.BFloat16, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    new_state_ptr = make_ptr(cutlass.Float32, state_out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    # cu_seqlens_ptr = make_ptr(cutlass.Int64, cu_seqlens.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
    cu_seqlens_ptr = make_ptr(cutlass.Int32, cu_seqlens.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)

    gdn_prefill_host(
        # old_v_cute_ptr, new_v_cute_ptr, g_cute, 
        q_ptr, k_ptr, v_ptr, state_ptr,
        A_log_ptr, a_ptr, dt_bias_ptr, b_ptr,
        output_ptr, new_state_ptr, cu_seqlens_ptr,
        Int32(total_seq_len), Int32(B), Float32(scale),
        has_state=True,
        stream=current_stream,
        NUM_V_HEADS=Hv,
        HEAD_SIZE=K,
        NUM_K_HEADS=Hq,
        V_TILE_SIZE=V_TILE_SIZE,
        V_TILES=V_TILES,
        THREADS_PER_CTA=THREADS_PER_CTA,
        NUM_WARPS=NUM_WARPS
    )