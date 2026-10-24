import unittest

import torch
import torch_npu  # noqa: F401
from sgl_kernel_npu.attention.flash_mla import flash_mla, flash_mla_metadata


class TestFlashMLA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.npu.set_device(0)
        cls.mask = torch.triu(torch.ones((2048, 2048), dtype=torch.int8), 1).npu()

    def case(
        self,
        dtype=torch.bfloat16,
        heads=16,
        lengths=(129, 53),
        counts=(1, 1),
        used=None,
        mode=0,
        window=(-1, -1),
        layout="TND",
        cache_layout="PA_BBND",
        sink=False,
        metadata_max=True,
        output_layout=None,
        send_used=True,
        page_size=32,
    ):
        torch.manual_seed(17)
        batch, page = len(lengths), page_size
        used = counts if used is None else used
        pages = max(1, (max(lengths) + page - 1) // page)
        table = torch.randperm(batch * pages).reshape(batch, pages).int()
        cache = torch.randn(batch * pages, page, 1, 576, dtype=dtype) * 0.3
        q = torch.randn(sum(counts), heads, 576, dtype=dtype) * 0.3
        cu = torch.tensor(
            [0] + list(torch.tensor(counts).cumsum(0).tolist()), dtype=torch.int32
        )
        sinks = torch.randn(heads) if sink else None
        expected = torch.zeros(sum(counts), heads, 512)
        lse = torch.full((sum(counts), heads), float("inf"))
        for b in range(batch):
            kv = cache[table[b].long()].reshape(-1, 576)[: lengths[b]].float()
            for i in range(used[b]):
                t = int(cu[b]) + i
                score = q[t].float() @ kv.T / 24
                position = torch.arange(lengths[b])
                anchor = i + lengths[b] - used[b]
                visible = torch.ones(lengths[b], dtype=torch.bool)
                if mode == 3:
                    visible &= position <= anchor
                if mode == 4:
                    if window[0] >= 0:
                        visible &= position >= anchor - window[0]
                    if window[1] >= 0:
                        visible &= position <= anchor + window[1]
                if visible.any():
                    score[:, ~visible] = -float("inf")
                    denominator = (
                        score
                        if sinks is None
                        else torch.cat((score, sinks[:, None]), 1)
                    )
                    lse[t] = denominator.logsumexp(1)
                    expected[t] = (score - lse[t, :, None]).exp() @ kv[:, :512]
        qn = q.npu()
        kn = cache.npu()
        if cache_layout == "PA_BNBD":
            kn = kn.permute(0, 2, 1, 3).contiguous()
        if cache_layout == "PA_NZ":
            kn = kn.reshape(-1, page, 1, 36, 16).permute(0, 2, 3, 1, 4).contiguous()
        if layout != "TND":
            self.assertEqual(len(set(counts)), 1)
            qn = qn.reshape(batch, counts[0], heads, 576)
            if layout == "BNSD":
                qn = qn.permute(0, 2, 1, 3).contiguous()
        lens, cun, un = (
            torch.tensor(lengths, dtype=torch.int32).npu(),
            cu.npu(),
            torch.tensor(used, dtype=torch.int32).npu(),
        )
        args = dict(
            cache_seqlens=lens,
            cu_seqlens_q=cun if layout == "TND" else None,
            seqused_q=un if send_used else None,
            max_seqlen_q=max(counts),
            mask_mode=mode,
            win_left=window[0],
            win_right=window[1],
            layout_q=layout,
        )
        meta_args = dict(args)
        if not metadata_max:
            meta_args["max_seqlen_q"] = None
        meta = flash_mla_metadata(num_heads_q=heads, num_heads_kv=1, **meta_args)
        out, actual_lse = flash_mla(
            qn,
            kn,
            table.npu(),
            metadata=meta,
            sinks=sinks.npu() if sink else None,
            attn_mask=self.mask if mode else None,
            softmax_scale=1 / 24,
            layout_kv=cache_layout,
            layout_out=output_layout or layout,
            return_softmax_lse=True,
            **args
        )
        if layout == "BNSD":
            out = out.permute(0, 2, 1, 3)
        if output_layout == "NTD":
            out = out.permute(1, 0, 2)
        actual_lse = (
            actual_lse.T
            if layout == "TND"
            else actual_lse.permute(0, 2, 1).reshape(-1, heads)
        )
        actual = out.reshape(-1, heads, 512).float().cpu()
        torch.testing.assert_close(
            actual,
            expected,
            atol=0.003 if dtype == torch.bfloat16 else 0.0004,
            rtol=0.02,
        )
        torch.testing.assert_close(actual_lse.cpu(), lse, atol=0.004, rtol=0.002)

    def test_decode(self):
        self.case()

    def test_ragged_causal(self):
        self.case(heads=24, counts=(3, 2), used=(2, 0), mode=3)

    def test_small_heads_fp16(self):
        self.case(dtype=torch.float16, heads=1, counts=(2, 1), mode=3)

    def test_layouts(self):
        for layout in ("BSND", "BNSD"):
            for cache in ("PA_BBND", "PA_BNBD", "PA_NZ"):
                with self.subTest(layout=layout, cache=cache):
                    self.case(
                        heads=8,
                        counts=(2, 2),
                        mode=3,
                        layout=layout,
                        cache_layout=cache,
                    )

    def test_window_sink(self):
        for window in ((2, 1), (-1, -1), (2, -1), (-1, 0), (0, 0)):
            with self.subTest(window=window):
                self.case(heads=16, counts=(3, 2), mode=4, window=window, sink=True)

    def test_empty_kv(self):
        self.case(lengths=(0, 17), counts=(2, 1), mode=4, window=(0, 0), sink=True)

    def test_long_kv(self):
        self.case(heads=16, lengths=(4097,), counts=(1,))

    def test_long_kv_layouts_many_heads(self):
        for cache in ("PA_BBND", "PA_BNBD", "PA_NZ"):
            with self.subTest(cache=cache):
                self.case(heads=96, lengths=(2049,), counts=(1,), cache_layout=cache)

    def test_head_group_boundaries(self):
        for heads in (17, 33, 65, 128):
            with self.subTest(heads=heads):
                self.case(heads=heads, lengths=(1025,), counts=(1,))
        self.case(dtype=torch.float16, heads=33, lengths=(1025,), counts=(1,))

    def test_window_on_long_cache(self):
        self.case(
            heads=96, lengths=(4097,), counts=(1,), mode=4, window=(127, 0), sink=True
        )

    def test_int32_max_window(self):
        self.case(counts=(3, 2), mode=4, window=(2147483647, 2147483647))

    def test_page_size_boundaries(self):
        for page in (16, 1024):
            with self.subTest(page=page):
                self.case(page_size=page)

    def test_many_heads(self):
        self.case(heads=96, counts=(2, 1), mode=3)

    def test_empty_query_batch(self):
        self.case(counts=(0, 2), mode=3, output_layout="NTD")

    def test_fixed_metadata_without_maximum(self):
        self.case(counts=(3, 3), used=(2, 0), mode=3, layout="BSND", metadata_max=False)
        self.case(counts=(2, 2), layout="BNSD", metadata_max=False, send_used=False)

    def test_fully_masked_causal(self):
        self.case(lengths=(1, 2), counts=(3, 2), mode=3)

    def test_invalid_arguments(self):
        lengths = torch.ones(1, dtype=torch.int32, device="npu")
        with self.assertRaisesRegex(ValueError, "one KV head"):
            flash_mla_metadata(lengths, 16, 2, max_seqlen_q=1)
        with self.assertRaisesRegex(ValueError, "cu_seqlens_q"):
            flash_mla_metadata(lengths, 16, 1, layout_q="TND")
        with self.assertRaisesRegex(ValueError, "window"):
            flash_mla_metadata(lengths, 16, 1, max_seqlen_q=1, win_left=3)

    def _check_graph_device_lengths(self, heads, page, replay_lengths, window=None):
        torch.manual_seed(23)
        pages = (max(replay_lengths) + page - 1) // page
        q = torch.randn((1, heads, 576), dtype=torch.bfloat16, device="npu") * 0.3
        cache = torch.randn((pages, page, 1, 576), dtype=q.dtype, device="npu") * 0.3
        table = torch.arange(pages, dtype=torch.int32, device="npu").reshape(1, pages)
        lengths = torch.tensor([replay_lengths[0]], dtype=torch.int32, device="npu")
        cu = torch.tensor([0, 1], dtype=torch.int32, device="npu")
        sinks = torch.randn(heads, device="npu") if window is not None else None

        def run():
            return flash_mla(
                q,
                cache,
                table,
                lengths,
                cu,
                sinks=sinks,
                attn_mask=self.mask if window is not None else None,
                mask_mode=4 if window is not None else 0,
                win_left=window[0] if window is not None else -1,
                win_right=window[1] if window is not None else -1,
                softmax_scale=1 / 24,
                layout_q="TND",
                layout_out="TND",
                return_softmax_lse=True,
            )

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            out, lse = run()
        for length in replay_lengths:
            lengths.fill_(length)
            graph.replay()
            torch.npu.synchronize()
            kv = cache.cpu().reshape(-1, 576)[:length].float()
            if length:
                if window is not None:
                    kv = kv[max(0, length - 1 - window[0]) :]
                score = q.cpu()[0].float() @ kv.T / 24
                denominator = (
                    score
                    if sinks is None
                    else torch.cat((score, sinks.cpu()[:, None]), 1)
                )
                expected_lse = denominator.logsumexp(-1)
                expected = (score - expected_lse[:, None]).exp() @ kv[:, :512]
                torch.testing.assert_close(
                    out.cpu()[0].float(), expected, atol=0.003, rtol=0.02
                )
                torch.testing.assert_close(
                    lse.cpu()[:, 0], expected_lse, atol=0.004, rtol=0.002
                )
            else:
                self.assertEqual(out.count_nonzero().item(), 0)
                self.assertTrue(lse.isposinf().all().item())

    def test_graph_device_lengths(self):
        self._check_graph_device_lengths(8, 32, (129, 17, 0, 101))
        self._check_graph_device_lengths(96, 128, (2049, 17, 0, 1025))

    def test_graph_window_device_lengths(self):
        self._check_graph_device_lengths(
            96, 128, (2049, 129, 127, 17, 0, 1025), window=(127, 0)
        )


if __name__ == "__main__":
    unittest.main()
