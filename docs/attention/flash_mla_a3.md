# Experimental FlashMLA on Ascend A3

This implementation provides the CANN FlashMLA Python calling convention using
native Triton kernels on Ascend 910C (A3). It is an experimental functional
backend. It is slower than CANN FIA in the measured decode cases and should not
replace the production attention backend by default. A2 and A5 are unverified.

The interface reference is the official
[CANN FlashMLA specification](https://gitcode.com/cann/ops-transformer/blob/f4de7a8e47e112224f2a08dba9d0592f0c972426/attention/flash_mla/docs/torchapi_flash_mla.md),
frozen at commit `f4de7a8e47e112224f2a08dba9d0592f0c972426`.
The A5 `arch35` source is not compiled or copied into this package. This backend
implements the attention equations independently using the repository's existing
Triton paged MLA approach.

## Interface and supported semantics

```python
from sgl_kernel_npu.attention.flash_mla import flash_mla, flash_mla_metadata

# q: [T, H, 576], cache: [blocks, page_size, 1, 576]
# lengths: device int32 [B], cu_q: device int32 [B+1]
metadata = flash_mla_metadata(lengths, q.shape[1], 1, cu_q, layout_q="TND")
out, lse = flash_mla(
    q, cache, block_table, lengths, cu_q,
    metadata=metadata,
    layout_q="TND", layout_out="TND",
    softmax_scale=576**-0.5,
    return_softmax_lse=True,
)
```

The default scale is `1.0`; callers supply their model's scale explicitly.
Both functions accept the keyword parameters in the reference Python API.

- BF16/FP16 queries and caches; QK dimension 576, V dimension 512, one KV head,
  and 1 through 128 query heads. Values are the first 512 cache channels.
- Query layouts `TND`, `BSND`, `BNSD`; output layouts `TND`/`NTD` for `TND`,
  and the matching layout for fixed batches. Outputs are contiguous.
- KV layouts `PA_BBND`, `PA_BNBD`, `PA_NZ`; 16-aligned page sizes up to 1024.
- Ragged physical query prefix sums and independent effective query lengths.
  Prefix sums count allocated query rows, including padded rows. Effective rows
  are the first `seqused_q[b]` rows in each request.
- Mask mode 0, right-aligned causal mode 3, and sliding-window mode 4, with
  optional per-head FP32 softmax sinks. Causal/window alignment uses the
  effective query length, including when physical query rows contain padding.
- Natural-log FP32 LSE, shaped `[H,T]` or `[B,H,S]`. Disabled LSE is an empty
  FP32 tensor. Fully masked or inactive rows return zero and positive-infinite
  LSE, including when sinks are present.

For mask modes 3/4, provide the canonical device INT8 `[2048,2048]` upper
triangular mask required by CANN. The kernel computes its canonical semantics
analytically; arbitrary mask contents are unsupported. Window mode uses
`p = query_position + kv_length - effective_query_length` and the inclusive
interval `[p-win_left, p+win_right]`, where `-1` means unbounded.

Metadata is an opaque, backend-specific device INT32 `[B,4]` tensor; **CANN A5
metadata cannot be reused**. This differs from CANN's one-dimensional scheduling
buffer. Generate metadata with this module, using the same query layout and
lengths. For fixed layouts, a supplied `max_seqlen_q` describes the physical
query length when effective lengths are omitted; omitting both uses the query's
physical length at execution. `max_seqlen_kv` is an accepted compatibility hint;
the kernel always reads actual device KV lengths.

The main function also accepts `metadata=None` and generates it internally.
When lengths change, regenerate metadata, preferably inside the captured graph.
Both metadata generation and attention avoid host reads of device lengths.
Valid prefix sums, nonnegative lengths within allocated capacities, effective
query lengths within physical query blocks, and valid block IDs are caller
preconditions; the wrapper does not synchronize to validate tensor contents.

## Implementation

Metadata generation snapshots query offsets, effective query lengths and KV
lengths on device. Attention uses separate 512-channel content and 64-channel
RoPE dot products, online FP32 softmax, and split-KV. A second kernel combines
partial outputs and adds each sink exactly once.

The A3 schedule uses 32-head by 64-key tiles for more than 16 heads when cache
capacity is at least 512 tokens. It retains 16-head tiles for short contexts,
small windows and NZ layouts. Fully bounded windows can use smaller 16/32-key
tiles. Split counts use a shape-based budget of 128 logical tasks, are rounded
to powers of two, capped at 32, and limited using cache/window capacity. This is
not a length-aware per-core scheduler; it never reads actual device lengths on
the host. Changing lengths therefore preserves the captured launch grid.

Each request partitions only its visible KV interval on device. Causal bounds
use effective query lengths; windows start at their exact lower bound and use
the paged block map across page boundaries. Inactive query rows do no KV work.
Window arithmetic uses int64 so INT32_MAX window attributes cannot overflow.

The schedule was informed by A3 KV reuse and FlashDecode separation in
[sgl-kernel-npu's arch22 SFA](https://github.com/sgl-project/sgl-kernel-npu/blob/6fba476a4c4387fdbb4aec09f0619415b9495f0d/csrc/sparse_flash_attention/op_kernel/arch22/sparse_flash_attention_service_cube_mla.h)
and CANN's A3 FIA
[Cube block](https://gitcode.com/cann/ops-transformer/blob/f4de7a8e47e112224f2a08dba9d0592f0c972426/attention/fused_infer_attention_score/op_kernel/arch22/fia_block_cube_nonquant_mla.h)
and [FlashDecode block](https://gitcode.com/cann/ops-transformer/blob/f4de7a8e47e112224f2a08dba9d0592f0c972426/attention/fused_infer_attention_score/op_kernel/arch22/fia_block_vec_flashdecode_arch22.h).
Their explicit L1 buffering and Cube/vector pipeline are not ported here, and
no source from those implementations is copied into the Triton kernel.

The A3 compiler requires conservative tiling: 128-key tiles exceeded the 192 KiB
UB limit with the full 512-channel accumulator. Enabling the dynamic CV pipeline also failed
the numerical comparison on the tested compiler; the kernel explicitly uses the
validated static CV pipeline. Large-head decode still repeats cache loads across
head groups and has substantial per-tile overhead. Further cache reuse and
Cube/vector pipelining is required before enabling this backend in production.

Temporary storage is approximately
`T * H * splits * (512 + 1) * 4` bytes, excluding output and layout conversions.
This backend targets decode and short verification queries; large prefill
workloads have not been benchmarked. The wider head group and split rounding
can increase scratch storage: the measured B=4, H=96, KV=4096 case uses about
12 MiB for partials/LSE, versus 4.5 MiB in the original prototype.

## Build and validation

Build/install the normal Ascend910 package using the repository's documented
build workflow. The new operator is Python/Triton only and requires no new C++
binary or external CANN custom-op extension.

From the repository root with `sgl_kernel_npu` installed:

```bash
python3 tests/python/sgl_kernel_npu/test_flash_mla.py -v
python3 tests/python/sgl_kernel_npu/bench_flash_mla.py --include-fia
```

To compare with the original prototype on the same inputs:

```bash
git show 1dba6fd:python/sgl_kernel_npu/sgl_kernel_npu/attention/flash_mla.py > /tmp/flash_mla_before_tiling.py
python3 tests/python/sgl_kernel_npu/bench_flash_mla.py --include-fia --baseline-file /tmp/flash_mla_before_tiling.py
```

The tests use a separate FP32 PyTorch reference with shuffled page tables. They
cover ragged multi-query causality, padding, completely masked rows, empty KV,
empty query requests, small and 96-head groups, layouts, sinks, windows, LSE,
and argument rejection. The graph test replays the same graph with device KV
lengths 129, 17, 0 and 101 for small heads, and 2049, 17, 0 and 1025 for
96 heads. Tests also cover 17/33/65/128 head boundaries, long-context cache
layouts and INT32_MAX windows. A separate window/sink graph replays lengths
2049, 129, 127, 17, 0 and 1025, including windows crossing page boundaries.
BF16 output tolerance is `atol=0.003, rtol=0.02`;
FP16 uses `atol=0.0004, rtol=0.02`; LSE uses `atol=0.004, rtol=0.002`.

The development wheel reused the environment's unchanged `libsgl_kernel_npu.so`
and packaged the updated Python modules with the normal Ascend910 provider
staging. This validates the Python-only migration and wheel import; it does not
claim a fresh build of all unrelated C++ operators.

## Paired A3 tiling optimization results

The installed Ascend910 wheel passed all 19 unittest methods. The following
measurements compare with prototype commit `1dba6fd` on the same inputs and
device, using the environment described below. Decode timings exclude metadata
and are medians of five groups of 30 warmed graph replays.

| Batch | Heads | KV length | Before | Optimized | Speedup | FIA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 4096 | 55.5 us | 56.5 us | 0.98x | 34.1 us |
| 1 | 96 | 16384 | 608.4 us | 320.6 us | 1.90x | 68.6 us |
| 4 | 96 | 4096 | 628.4 us | 327.0 us | 1.92x | 72.1 us |

The 16-head case has no measured gain and is about 2% slower in this run.
The wider-head gains do not establish superiority over FIA: all three points
remain slower than FIA. With metadata included, the optimized 96-head points
take 326.9 us and 333.3 us respectively.

Additional FP32-reference-checked cases include metadata and use medians of
three groups of 20 graph replays. All have 96 heads and BF16 inputs:

| Case | Before | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| KV=16384, 128-token window, sink | 607.1 us | 35.5 us | 17.08x |
| Ragged causal, KV=4097/2049, physical Q=3/2, used Q=2/1 | 614.3 us | 265.7 us | 2.31x |
| Fixed causal, KV=4096, physical Q=4, used Q=3 | 636.6 us | 273.2 us | 2.33x |

These gains come from KV reuse, limiting split count for small visible ranges,
and skipping invisible/padded work. A separate eager profile at B=1, H=96,
KV=16384 measured 293.0 us in partial attention, 1.6 us metadata and 8.1 us
merge: about 96.8% of summed kernel time remains in partial attention.
The exact measurements, source/wheel hashes and experiment summaries are in
[flash_mla_a3_tiling_results.json](flash_mla_a3_tiling_results.json).
Whole-model performance and large prefill have not been measured.

## Original prototype measurements

This section records the original `1dba6fd` prototype, before the optimization
above. It is retained as the earlier baseline.

Measured on server 88, one idle `Ascend910_9362` die (physical device 4),
2026-10-24, CANN 9.1.0 A3 B103 image, PyTorch `2.10.0+cpu`,
torch-npu `2.10.0.post6`, triton-ascend `3.2.0`. Inputs are BF16, page size 128,
one query per request. Device-event timings are the median of five groups of
30 graph replays after warm-up; compilation is excluded. These are operator
measurements, not end-to-end Kimi K3 measurements.

| Batch | Heads | KV length | FlashMLA | Including metadata | Existing decode_mla | CANN FIA |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 4096 | 56.4 us | 58.2 us | 1077.4 us | 34.1 us |
| 1 | 96 | 16384 | 598.6 us | 603.5 us | 4280.9 us | 67.8 us |
| 4 | 96 | 4096 | 633.7 us | 630.5 us | 2176.1 us | 71.4 us |

FIA uses separately packed content/RoPE caches and pads 96 query heads to 128,
as required by that decode path. Packing and padding are outside the timed
region. FIA receives static host length lists; these timings do not validate
changing its lengths during graph replay. The native FlashMLA graph test does
validate changing device lengths. Both baselines are checked numerically before
their timing results are reported. Despite beating the repository's existing
Triton decoder, this prototype is approximately 1.7–8.9 times slower than FIA.
Small differences between timings with/without metadata are measurement noise.

The installed wheel passed all 14 unittest methods, including page sizes 16
and 1024. In a separate eager profile at batch 1, 96 heads and KV length 16384,
the mixed Cube/vector partial-attention task took 579.8 us, metadata 1.7 us,
and merge 8.2 us. Partial attention accounts for about 98.3% of the summed kernel
time; optimizing metadata or merge alone will not close the gap to FIA.
The recorded inputs, software versions and exact timings are in
[flash_mla_a3_results.json](flash_mla_a3_results.json).
