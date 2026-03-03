"""
Triton Kernel Template for FlashInfer Competition.

Implement your kernel logic here. The entry point function name should match
the `entry_point` setting in config.toml.

See the track definition for required function signature and semantics.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def gdn_kernel(
    state_ptr, state_out_ptr, out_ptr,
    q_ptr, k_ptr, v_ptr,
    a_ptr, dt_bias_ptr, alog_ptr, b_ptr, cu_seqlens_ptr,
    debug_ptr,
    # state strides [num_seqs, Hv, V, K]
    stride_state_b, stride_state_h, stride_state_v, stride_state_k,
    # q strides [total_seq_len, Hq, K]
    stride_q_t, stride_q_h, stride_q_k,
    # k strides [total_seq_len, Hk, K]
    stride_k_t, stride_k_h, stride_k_k,
    # v strides [total_seq_len, Hv, K]
    stride_v_t, stride_v_h, stride_v_v,
    # a strides [total_seq_len, Hv]
    stride_a_t, stride_a_h,
    # b strides [total_seq_len, Hv]
    stride_b_t, stride_b_h,
    # out strides [total_seq_len, Hv, K]
    stride_out_t, stride_out_h, stride_out_v,
    # scalar strides
    stride_dt_bias,  # [Hv]
    stride_alog,     # [Hv]
    stride_debug_t, stride_debug_h, stride_debug_v,
    # dims
    K: tl.constexpr,
    V: tl.constexpr,  # =128, same as K but for clarity
    BLOCK_V: tl.constexpr,
    Hv: tl.constexpr,
    Hq: tl.constexpr,
    scale: tl.constexpr,
    MAX_LEN: tl.constexpr,
):

    seq_idx = tl.program_id(0)
    h = tl.program_id(1)
    v_block = tl.program_id(2)
    # scale = tl.load(scale_ptr)
    offs_v = v_block * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_v = offs_v < V
    offs_k = tl.arange(0, K)

    alog = tl.load(alog_ptr + h * stride_alog).to(tl.float32) # scalar
    g_alog = -tl.exp(alog)
    dt_bias = tl.load(dt_bias_ptr + h * stride_dt_bias).to(tl.float32) # scalar
    
    # [Block_v, K]
    state_tile = tl.load(state_ptr + seq_idx*stride_state_b + h*stride_state_h + offs_v[:,None]*stride_state_v + offs_k[None,:]*stride_state_k, mask=mask_v[:, None], other=0.0)
    seq_start = tl.load(cu_seqlens_ptr + seq_idx)
    seq_end = tl.load(cu_seqlens_ptr + seq_idx + 1)

    rep = Hv // Hq
    k_head = h // rep
    # L = seq_end - seq_start
    for i in range(MAX_LEN, step=1, num_stages=1):
        t = seq_start + i
        
        active = t < seq_end
        # if active:
        #     tl.device_print('t', t)
        # scalar
        a = tl.load(a_ptr + t*stride_a_t + h*stride_a_h, mask=active, other=0.0).to(tl.float32)
        g = tl.exp(g_alog * tl.log(1+tl.exp(a+dt_bias)))
        
        x = a + dt_bias
        # Numerically stable softplus: for large x, softplus(x) ~ x
        absx = tl.abs(x)
        softplus = tl.maximum(x, 0.0) + tl.log(1.0 + tl.exp(-absx))
        g = tl.exp(-tl.exp(alog) * softplus)
        # scalar
        b = tl.load(b_ptr + t*stride_b_t + h*stride_b_h, mask=active, other=0.0).to(tl.float32)
        beta = tl.sigmoid(b)
        
        # state_tile = g * state_tile
        state_tile = tl.where(active, g * state_tile, state_tile).to(tl.float32)
        # [K]
        k_K = tl.load(k_ptr + t*stride_k_t + k_head*stride_k_h + offs_k * stride_k_k, mask=active, other=0.0).to(tl.float32)
        
        # old_v: [BLOCK_V, K] @ [K, 1] -> [BLOCK_V]
        old_v = tl.sum(state_tile*k_K[None,:], axis=1)
        tl.store(debug_ptr+t*stride_debug_t+h*stride_debug_h+offs_v*stride_debug_v, old_v, mask=active)
        # # # # [block_v]
        v_BLOCKV = tl.load(v_ptr + t*stride_v_t+ h * stride_v_h + offs_v * stride_v_v, mask=active & mask_v, other=0.0).to(tl.float32)
        new_v_BLOCKV = beta * v_BLOCKV + (1-beta) * old_v
        
        # # [block_v]
        delta_v_BLOCKV = new_v_BLOCKV - old_v
        # # # state_tile += k_K[None, :] * delta_v_BLOCKV[:, None]
        
        state_tile = tl.where(active, state_tile + k_K[None, :] * delta_v_BLOCKV[:, None], state_tile)
        q_K = tl.load(q_ptr + t * stride_q_t + k_head * stride_q_h + offs_k * stride_q_k, mask=active, other=0.0).to(tl.float32)
        # [block_v]
        curr_output = tl.sum(q_K[None,:]*state_tile, axis=1)
        tl.store(out_ptr + t * stride_out_t + h * stride_out_h + offs_v * stride_out_v, (scale * curr_output).to(tl.bfloat16), mask=active & mask_v)

    tl.store(state_out_ptr + seq_idx*stride_state_b + h*stride_state_h + offs_v[:,None]*stride_state_v + offs_k[None,:]*stride_state_k, state_tile, mask=mask_v[:,None])

def tritonGDN(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale, out, state_out):
    B, Hv, K, V = state.shape
    Hq = q.shape[1]
    BLOCK_V = min(64, triton.next_power_of_2(V))
    grid = (B, Hv, triton.cdiv(V, BLOCK_V))

    debug_var = torch.zeros((a.shape[0], Hv, V), device=q.device, dtype=torch.float32)
    gdn_kernel[grid](
        state, state_out, out,
        q, k, v,
        a, dt_bias, A_log, b, cu_seqlens,
        debug_var,
        # state [num_seqs, Hv, V=128, K=128]
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        # q [total_seq_len, Hq, 128]
        q.stride(0), q.stride(1), q.stride(2),
        # k [total_seq_len, Hk, 128]
        k.stride(0), k.stride(1), k.stride(2),
        # v [total_seq_len, Hv, 128]
        v.stride(0), v.stride(1), v.stride(2),
        # a [total_seq_len, Hv]
        a.stride(0), a.stride(1),
        # b [total_seq_len, Hv]
        b.stride(0), b.stride(1),
        # out [total_seq_len, Hv, 128]
        out.stride(0), out.stride(1), out.stride(2),
        # scalars [Hv]
        dt_bias.stride(0),
        A_log.stride(0),
        debug_var.stride(0), debug_var.stride(1), debug_var.stride(2),
        # constexprs
        K=K, V=V, BLOCK_V=BLOCK_V,
        Hv=Hv, Hq=Hq, scale=scale,
        MAX_LEN=64,
    )
    print(debug_var.shape)
    return debug_var