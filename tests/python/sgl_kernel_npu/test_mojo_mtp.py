# SPDX-License-Identifier: Apache-2.0
"""Paged DSpark GQA against FP32 and FIA, including mutable NPU graph inputs."""

import math
from unittest.mock import patch

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
if not torch.npu.is_available():
    pytest.skip("Ascend device required", allow_module_level=True)

from sgl_kernel_npu.attention.mojo_mtp import mojo_paged_mtp_decode  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    torch.npu.set_device(0)
    torch.set_num_threads(4)


def make_case(lengths, gamma, group, dim, page, dtype, hkv=2):
    generator = torch.Generator().manual_seed(26)
    batch = len(lengths)
    pages = max(1, math.ceil(max(lengths) / page))
    table = (
        torch.randperm(batch * pages, generator=generator).reshape(batch, pages).int()
    )
    q = (torch.randn(batch * gamma, hkv * group, dim, generator=generator) * 0.3).to(
        dtype
    )
    k = torch.full((batch * pages, page, hkv * dim), float("nan"), dtype=dtype)
    v = k.clone()
    for b, length in enumerate(lengths):
        keys = (torch.randn(length, hkv * dim, generator=generator) * 0.3).to(dtype)
        values = (torch.randn(length, hkv * dim, generator=generator) * 0.3).to(dtype)
        for index in range(length):
            physical = table[b, index // page]
            k[physical, index % page] = keys[index]
            v[physical, index % page] = values[index]
    args = dict(
        query=q.npu(),
        k_cache=k.npu(),
        v_cache=v.npu(),
        seq_lens=torch.tensor(lengths, dtype=torch.int32, device="npu"),
        cu_q_lens=torch.arange(
            0, (batch + 1) * gamma, gamma, dtype=torch.int32, device="npu"
        ),
        block_tables=table.npu(),
        page_size=page,
        num_q_heads=hkv * group,
        num_kv_heads=hkv,
        head_dim=dim,
        gamma=gamma,
        softmax_scale=dim**-0.5,
        max_kv_len=pages * page,
    )
    return args, dict(q=q, k=k, v=v, table=table)


def reference(args, data, lengths, causal):
    gamma, hkv, dim = args["gamma"], args["num_kv_heads"], args["head_dim"]
    group = args["num_q_heads"] // hkv
    result = torch.zeros_like(data["q"], dtype=torch.float32)
    for b, length in enumerate(lengths):
        if length == 0:
            continue
        keys = data["k"][data["table"][b].long()].reshape(-1, hkv, dim)[:length]
        values = data["v"][data["table"][b].long()].reshape(-1, hkv, dim)[:length]
        keys = keys.float().repeat_interleave(group, 1)
        values = values.float().repeat_interleave(group, 1)
        query = data["q"][b * gamma : (b + 1) * gamma].float()
        scores = torch.einsum("qhd,khd->hqk", query, keys) * args["softmax_scale"]
        if causal:
            visible = torch.arange(length)[None, :] <= (
                length - gamma + torch.arange(gamma)[:, None]
            )
            scores.masked_fill_(~visible[None, :, :], -float("inf"))
        probabilities = scores.softmax(-1).nan_to_num(0)
        result[b * gamma : (b + 1) * gamma] = torch.einsum(
            "hqk,khd->qhd", probabilities, values
        )
    return result


def check_output(output, expected, lengths, gamma):
    actual = output.float().cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.03)
    for b, length in enumerate(lengths):
        if length == 0:
            assert torch.count_nonzero(actual[b * gamma : (b + 1) * gamma]) == 0


CASES = [
    (1, 1, 16, 16, [0, 1, 17]),
    (7, 3, 48, 16, [7, 15, 17]),
    (7, 4, 64, 128, [0, 127, 134]),
    (8, 2, 32, 32, [31, 32, 33]),
    (7, 8, 128, 128, [0, 128, 262]),
    (1, 1, 64, 128, [1, 129, 1025]),
    (7, 1, 128, 32, [0, 7, 65]),
    (8, 4, 256, 128, [8, 127, 256]),
    (7, 4, 64, 128, [2049]),
]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("gamma,group,dim,page,lengths", CASES)
def test_paged_gqa_matches_fp32(dtype, causal, gamma, group, dim, page, lengths):
    args, data = make_case(lengths, gamma, group, dim, page, dtype)
    table = data["table"].clone()
    for b, length in enumerate(lengths):
        if length == 0:
            table[b].fill_(-1)
    args["block_tables"].copy_(table)
    with patch.object(
        torch.Tensor, "cpu", side_effect=AssertionError("unexpected D2H")
    ):
        output = mojo_paged_mtp_decode(**args, causal=causal)
    check_output(output, reference(args, data, lengths, causal), lengths, gamma)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [False, True])
def test_paged_gqa_matches_fia(dtype, causal):
    args, data = make_case([129, 2049], 7, 4, 64, 128, dtype, hkv=4)
    mask = (
        torch.ones((2048, 2048), device="npu", dtype=torch.bool).triu_(1)
        if causal
        else None
    )
    fia, _ = torch_npu.npu_fused_infer_attention_score(
        args["query"],
        args["k_cache"],
        args["v_cache"],
        block_table=args["block_tables"],
        block_size=128,
        num_heads=16,
        num_key_value_heads=4,
        input_layout="TND",
        atten_mask=mask,
        scale=64**-0.5,
        actual_seq_lengths=[7, 14],
        actual_seq_lengths_kv=[129, 2049],
        sparse_mode=3 if causal else 0,
    )
    output = mojo_paged_mtp_decode(**args, causal=causal)
    torch.testing.assert_close(output, fia, atol=0.004, rtol=0.03)


@pytest.mark.parametrize("lengths", [[129, 0], [2049], [16007] * 16])
def test_graph_replay_reads_changed_lengths_and_query(lengths):
    capacity_lengths = [max(129, max(lengths))] * len(lengths)
    args, data = make_case(capacity_lengths, 7, 4, 64, 128, torch.bfloat16, hkv=4)
    args["seq_lens"].copy_(torch.tensor(lengths, dtype=torch.int32, device="npu"))
    for _ in range(3):
        mojo_paged_mtp_decode(**args, causal=True)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = mojo_paged_mtp_decode(**args, causal=True)
    batch = len(lengths)
    for values in [lengths, [17] * batch, [0] * batch, [129] * batch]:
        args["seq_lens"].copy_(torch.tensor(values, dtype=torch.int32, device="npu"))
        table = data["table"].clone()
        for b, length in enumerate(values):
            if length == 0:
                table[b].fill_(-1)
        args["block_tables"].copy_(table)
        graph.replay()
        check_output(output, reference(args, data, values, True), values, 7)
    data["q"].mul_(1.5)
    args["query"].copy_(data["q"])
    graph.replay()
    check_output(output, reference(args, data, [129] * batch, True), [129] * batch, 7)


def test_packed_query_accepts_noncontiguous_tnd_input():
    args, data = make_case([129], 7, 4, 64, 128, torch.bfloat16, hkv=4)
    backing = torch.empty(
        (*args["query"].shape[:-1], 128), dtype=torch.bfloat16, device="npu"
    )
    view = backing[..., ::2]
    view.copy_(args["query"])
    args["query"] = view
    assert not view.is_contiguous()
    output = mojo_paged_mtp_decode(**args, causal=True)
    check_output(output, reference(args, data, [129], True), [129], 7)
