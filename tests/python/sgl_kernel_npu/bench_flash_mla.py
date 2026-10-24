"""Compare A3 FlashMLA with the existing single-query paged MLA kernel."""

import argparse
import json
import statistics

import torch
import torch_npu  # noqa: F401
import triton
from sgl_kernel_npu.attention.decode_attention import decode_mla
from sgl_kernel_npu.attention.flash_mla import flash_mla, flash_mla_metadata


def latency(function, repetitions):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = function()
    samples = []
    for _ in range(5):
        start, end = torch.npu.Event(enable_timing=True), torch.npu.Event(
            enable_timing=True
        )
        start.record()
        for _ in range(repetitions):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repetitions)
    return statistics.median(samples), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--include-fia", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(19)
    results = []
    for batch, heads, length in ((1, 16, 4096), (1, 96, 16384), (4, 96, 4096)):
        page = 128
        pages = (length + page - 1) // page
        q = torch.randn(batch, heads, 576, dtype=torch.bfloat16, device="npu") * 0.3
        cache = (
            torch.randn(batch * pages, page, 1, 576, dtype=q.dtype, device="npu") * 0.3
        )
        table = torch.arange(batch * pages, dtype=torch.int32, device="npu").reshape(
            batch, pages
        )
        lengths = torch.full((batch,), length, dtype=torch.int32, device="npu")
        cu = torch.arange(batch + 1, dtype=torch.int32, device="npu")
        meta = flash_mla_metadata(lengths, heads, 1, cu, layout_q="TND")

        def new(with_metadata=False):
            return flash_mla(
                q,
                cache,
                table,
                lengths,
                cu,
                metadata=None if with_metadata else meta,
                softmax_scale=1 / 24,
                layout_q="TND",
                layout_out="TND",
            )

        old_output = torch.empty(batch, heads, 512, device="npu", dtype=q.dtype)

        def old():
            decode_mla(
                q,
                cache[..., :512],
                cache[..., 512:],
                old_output,
                lengths,
                1 / 24,
                page,
                table,
            )
            return old_output

        new_us, (output, _) = latency(new, args.repetitions)
        full_us, _ = latency(lambda: new(True), args.repetitions)
        old_us, baseline = latency(old, args.repetitions)
        torch.testing.assert_close(
            output.float(), baseline.float(), atol=0.003, rtol=0.03
        )
        results.append(
            dict(
                batch=batch,
                heads=heads,
                kv_length=length,
                page=page,
                flash_mla_us=new_us,
                flash_mla_with_metadata_us=full_us,
                decode_mla_us=old_us,
                speedup=old_us / new_us,
            )
        )
        if args.include_fia:
            padded_heads = triton.next_power_of_2(heads)
            padded = torch.zeros(
                batch, 1, padded_heads, 576, dtype=q.dtype, device=q.device
            )
            padded[:, 0, :heads] = q
            qn, qr = padded[..., :512].contiguous(), padded[..., 512:].contiguous()
            kn = cache[..., :512].permute(0, 2, 1, 3).contiguous()
            kr = cache[..., 512:].permute(0, 2, 1, 3).contiguous()

            def fia():
                return torch_npu.npu_fused_infer_attention_score(
                    qn,
                    kn,
                    kn,
                    query_rope=qr,
                    key_rope=kr,
                    num_heads=padded_heads,
                    num_key_value_heads=1,
                    input_layout="BSND",
                    scale=1 / 24,
                    sparse_mode=0,
                    block_table=table,
                    block_size=page,
                    actual_seq_lengths_kv=[length] * batch,
                )

            fia_us, (fia_out, _) = latency(fia, args.repetitions)
            torch.testing.assert_close(
                output.float(), fia_out[:, 0, :heads].float(), atol=0.003, rtol=0.03
            )
            results[-1].update(
                fia_us=fia_us, fia_heads=padded_heads, speedup_vs_fia=fia_us / new_us
            )
        print(json.dumps(results[-1]), flush=True)
    print(
        json.dumps(
            dict(
                device=torch.npu.get_device_name(0),
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                results=results,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
