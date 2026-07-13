# FlashAttention3 Forward Compatibility

This document is a quick reference for the current FlashAttention-3-compatible **forward** coverage provided by MATE on MUSA.

## At a Glance

| Area | Status | Notes |
| --- | --- | --- |
| Q Mode | ✅ Supported | `Normal`, `Ragged`, `Padded` |
| KV Mode | ✅ Supported | `Normal`, `Ragged`, `Padded`, `Paged` |
| Append New KV | ✅ Supported | `flash_attn_with_kvcache` appends via `k` / `v`; packed new KV is supported via `cu_seqlens_k_new` |
| RoPE Input | ✅ Supported | `flash_attn_with_kvcache` supports `rotary_cos` / `rotary_sin`, `Interleaved`, and `Non-interleaved` |
| Cache Index Options | ✅ Supported | `cache_batch_idx`, `cache_leftpad` |
| Mask Mode | ✅ Supported | `None`, `Causal`, `Local`, `Local + attention_chunk` |
| Score Mode | ✅ Supported | Standard softmax and `softcap` |
| Page Size | ✅ Supported | `1`, `16`, `64`, and arbitrary page sizes |
| Dtype | ✅ Supported | `bf16`, `fp16`, and `torch.float8_e4m3fn` forward inputs; FP8 uses `q_descale` / `k_descale` / `v_descale` scaling |
| QV Input | ✅ Supported | The forward path supports an optional `qv` input, including FP8 inputs with `qv` |
| HeadDim | ✅ Supported | Any `headdim <= 512` |
| Optimization | ✅ Supported | `SplitKV`, `PackGQA`, `SchedulerMetadata` |
| Output | ✅ Supported | `out`, `softmax_lse` |

## MATE Extensions

| Extension | Status | Notes |
| --- | --- | --- |
| Context Parallel | ✅ Supported | `cp_world_size`, `cp_rank`, `cp_tot_seqused_k` |
| Learnable Sink | ✅ Supported | Supported on the local-attention path |

## Notes

- This page summarizes the compatibility surface, not every internal kernel detail.
- The statement `Any headdim <= 512` refers to the supported forward-path head-dimension range.
- FP8 forward support currently refers to `torch.float8_e4m3fn`; pass optional
  `q_descale`, `k_descale`, and `v_descale` tensors with shape
  `(batch_size, num_heads_kv)` when scale factors are required.
- When both `q` and the optional `qv` input are FP8, `q_descale` applies to
  both query tensors; `k_descale` and `v_descale` still apply to the KV inputs.
- RoPE is supported only when appending new KV through `k` / `v`; `rotary_dim` must be `<= headdim` and divisible by 16.
- `Local + attention_chunk` requires MUSA SDK >= 5.1.0.
- FP8 attention works on the forward path today. For best performance, use
  MUSA SDK 5.2.0 or newer when available.
- For wrapper-level usage, see the {doc}`FlashAttention wrapper page </wrappers/flash_attention_wrapper>`.
