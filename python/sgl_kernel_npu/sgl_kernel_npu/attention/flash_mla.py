"""Experimental paged MLA for Ascend 910C (A3), with device scheduling.

The Python interface follows CANN FlashMLA. Metadata is private to this backend;
it is not binary compatible with CANN's Ascend 950 scheduling tensor.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(
    L,
    CU,
    USED,
    M,
    B: tl.constexpr,
    S: tl.constexpr,
    RAGGED: tl.constexpr,
    HAS_USED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = b < B
    start = b * S
    end = start + S
    if RAGGED:
        start = tl.load(CU + b, valid, 0)
        end = tl.load(CU + b + 1, valid, 0)
    used = end - start
    if HAS_USED:
        used = tl.load(USED + b, valid, 0)
    length = tl.load(L + b, valid, 0)
    tl.store(M + b * 4, start, valid)
    tl.store(M + b * 4 + 1, end, valid)
    tl.store(M + b * 4 + 2, used, valid)
    tl.store(M + b * 4 + 3, length, valid)


@triton.jit
def _partial(
    Q,
    K,
    TABLE,
    META,
    PART,
    LSE,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    QS: tl.constexpr,
    QH: tl.constexpr,
    QD: tl.constexpr,
    KS: tl.constexpr,
    KP: tl.constexpr,
    KD: tl.constexpr,
    TB: tl.constexpr,
    TC: tl.constexpr,
    PAGE: tl.constexpr,
    NZ: tl.constexpr,
    RAGGED: tl.constexpr,
    S: tl.constexpr,
    SPLITS: tl.constexpr,
    SCALE: tl.constexpr,
    MODE: tl.constexpr,
    LEFT: tl.constexpr,
    RIGHT: tl.constexpr,
    BN: tl.constexpr = 64,
    BH: tl.constexpr = 16,
):
    token = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    batch = token // S
    if RAGGED:
        lo = 0
        hi = B
        while lo < hi:
            mid = (lo + hi) // 2
            end = tl.load(META + mid * 4 + 1)
            go_right = token >= end
            lo = tl.where(go_right, mid + 1, lo)
            hi = tl.where(go_right, hi, mid)
        batch = tl.minimum(lo, B - 1)
    start = tl.load(META + batch * 4)
    used = tl.load(META + batch * 4 + 2)
    if not RAGGED:
        start = batch * S
        used = tl.where(used < 0, S, used)
    length = tl.load(META + batch * 4 + 3)
    active = (token - start) < used
    anchor = token - start + length - used
    h = group * BH + tl.arange(0, BH)
    d = tl.arange(0, 512)
    r = tl.arange(0, 64)
    q = tl.load(Q + token * QS + h[:, None] * QH + d[None, :] * QD, h[:, None] < H, 0)
    qr = tl.load(
        Q + token * QS + h[:, None] * QH + (512 + r[None, :]) * QD, h[:, None] < H, 0
    )
    # Partition only visible tiles. In particular a short window must not
    # iterate over the entire cache, and padded queries need no KV work.
    visible_begin = 0
    visible_end = length
    if MODE == 3:
        visible_end = tl.minimum(length, tl.maximum(0, anchor + 1))
    if MODE == 4:
        # Windows may be INT32_MAX; their arithmetic must not wrap.
        anchor64 = anchor.to(tl.int64)
        if LEFT >= 0:
            visible_begin = tl.maximum(0, anchor64 - LEFT)
        if RIGHT >= 0:
            visible_end = tl.minimum(length, tl.maximum(0, anchor64 + RIGHT + 1))
    visible_end = tl.where(active, visible_end, 0)
    if MODE == 4 and LEFT >= 0:
        visible_begin = tl.minimum(visible_begin, visible_end)
    tiles = tl.cdiv(visible_end - visible_begin, BN)
    per_split = tl.cdiv(tiles, SPLITS)
    begin = per_split * split
    end = tl.minimum(per_split * (split + 1), tiles)
    m = tl.full((BH,), -float("inf"), tl.float32)
    z = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, 512), tl.float32)
    for tile in range(begin, end):
        # Start at the exact lower bound, without an extra masked tile for
        # unaligned windows. Page crossings use the per-token block map.
        n = visible_begin + tile * BN + tl.arange(0, BN)
        valid = n < visible_end
        page = tl.load(TABLE + batch * TB + (n // PAGE) * TC, n < length, 0)
        base = page * KS + (n % PAGE) * KP
        kd = d * KD
        kr = (512 + r) * KD
        if NZ:
            kd = (d // 16) * KD + d % 16
            kr = ((512 + r) // 16) * KD + r % 16
        k = tl.load(K + base[:, None] + kd[None, :], valid[:, None], 0)
        rope = tl.load(K + base[:, None] + kr[None, :], valid[:, None], 0)
        score = (tl.dot(q, tl.trans(k)) + tl.dot(qr, tl.trans(rope))) * SCALE
        score = tl.where(valid[None, :], score, -float("inf"))
        new_m = tl.maximum(m, tl.max(score, 1))
        safe_m = tl.where(new_m == -float("inf"), 0.0, new_m)
        alpha = tl.exp(m - safe_m)
        p = tl.exp(score - safe_m[:, None])
        z = z * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(k.dtype), k)
        m = new_m
    denom = tl.where(z > 0, z, 1.0)
    offset = (token * H + h) * SPLITS + split
    tl.store(
        PART + offset[:, None] * 512 + d[None, :], acc / denom[:, None], h[:, None] < H
    )
    tl.store(LSE + offset, tl.where(z > 0, m + tl.log(denom), -float("inf")), h < H)


@triton.jit
def _merge(
    PART,
    LSE,
    SINK,
    OUT,
    FINAL_LSE,
    H: tl.constexpr,
    SPLITS: tl.constexpr,
    SS: tl.constexpr,
    HAS_SINK: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    token = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, SS)
    d = tl.arange(0, 512)
    base = (token * H + h) * SPLITS
    l = tl.load(LSE + base + s, s < SPLITS, -float("inf"))
    m = tl.max(l, 0)
    valid = m != -float("inf")
    if HAS_SINK:
        sink = tl.load(SINK + h)
        m = tl.maximum(m, sink)
    safe = tl.where(m == -float("inf"), 0.0, m)
    w = tl.exp(l - safe)
    z = tl.sum(w, 0)
    if HAS_SINK:
        z += tl.exp(sink - safe)
    z = tl.where(z > 0, z, 1.0)
    p = tl.load(PART + (base + s[:, None]) * 512 + d[None, :], s[:, None] < SPLITS, 0)
    out = tl.sum(p * w[:, None], 0) / z
    tl.store(OUT + (token * H + h) * 512 + d, out)
    if RETURN_LSE:
        tl.store(
            FINAL_LSE + token * H + h, tl.where(valid, safe + tl.log(z), float("inf"))
        )


def _lengths(tensor, batch, device, name):
    if (
        tensor is None
        or tensor.dtype != torch.int32
        or tensor.shape != (batch,)
        or tensor.device != device
        or not tensor.is_contiguous()
    ):
        raise ValueError(f"{name} must be contiguous device int32 [{batch}]")


def _options(head_dim_qk, head_dim_v, mask_mode, win_left, win_right, layout_q):
    if (head_dim_qk, head_dim_v) != (576, 512):
        raise ValueError("FlashMLA requires QK dimension 576 and V dimension 512")
    if layout_q not in ("TND", "BSND", "BNSD"):
        raise ValueError("unsupported query layout")
    if mask_mode not in (0, 3, 4):
        raise ValueError("mask_mode must be 0, 3 or 4")
    if min(win_left, win_right) < -1 or (
        mask_mode != 4 and (win_left, win_right) != (-1, -1)
    ):
        raise ValueError("invalid attention window")


def flash_mla_metadata(
    cache_seqlens,
    num_heads_q,
    num_heads_kv,
    cu_seqlens_q=None,
    seqused_q=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    head_dim_qk=576,
    head_dim_v=512,
    mask_mode=None,
    win_left=-1,
    win_right=-1,
    layout_q="BSND",
):
    """Build A3 metadata on device; regenerate when device lengths change."""
    mode = 0 if mask_mode is None else mask_mode
    _options(head_dim_qk, head_dim_v, mode, win_left, win_right, layout_q)
    if num_heads_kv != 1 or not 1 <= num_heads_q <= 128:
        raise ValueError("requires one KV head and 1..128 query heads")
    batch = cache_seqlens.numel()
    if not batch or cache_seqlens.device.type != "npu":
        raise ValueError("requires nonempty NPU cache_seqlens")
    _lengths(cache_seqlens, batch, cache_seqlens.device, "cache_seqlens")
    if seqused_q is not None:
        _lengths(seqused_q, batch, cache_seqlens.device, "seqused_q")
    if layout_q == "TND":
        _lengths(cu_seqlens_q, batch + 1, cache_seqlens.device, "cu_seqlens_q")
    elif cu_seqlens_q is not None:
        raise ValueError("cu_seqlens_q requires TND layout")
    if max_seqlen_q is not None and max_seqlen_q < -1:
        raise ValueError("max_seqlen_q must be -1 or nonnegative")
    result = torch.empty((batch, 4), device=cache_seqlens.device, dtype=torch.int32)
    _metadata[(triton.cdiv(batch, 128),)](
        cache_seqlens,
        cu_seqlens_q,
        seqused_q,
        result,
        batch,
        -1 if max_seqlen_q is None else max_seqlen_q,
        layout_q == "TND",
        seqused_q is not None,
        128,
    )
    return result


def flash_mla(
    q,
    k_cache,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
):
    """Compute paged latent attention without transferring lengths to the CPU.

    Valid device lengths, prefix sums and block IDs are caller preconditions.
    Mask modes 3/4 use the canonical CANN triangular template analytically.
    """
    _options(q.shape[-1], head_dim_v, mask_mode, win_left, win_right, layout_q)
    if q.device.type != "npu" or q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be NPU float16 or bfloat16")
    if k_cache.device != q.device or k_cache.dtype != q.dtype:
        raise ValueError("cache must match query device and dtype")
    batch = cache_seqlens.numel() if cache_seqlens is not None else 0
    _lengths(cache_seqlens, batch, q.device, "cache_seqlens")
    if layout_q == "TND":
        if q.ndim != 3 or layout_out not in ("TND", "NTD"):
            raise ValueError("TND query requires TND/NTD output")
        flat = q
        physical_q = 1
    else:
        if q.ndim != 4 or q.shape[0] != batch or layout_out != layout_q:
            raise ValueError("fixed query and output layouts must match")
        shaped = q if layout_q == "BSND" else q.permute(0, 2, 1, 3)
        physical_q = shaped.shape[1]
        flat = shaped.reshape(-1, shaped.shape[2], 576)
    tokens, heads, _ = flat.shape
    if not batch or not tokens or not 1 <= heads <= 128:
        raise ValueError("requires nonempty queries and 1..128 heads")
    if layout_kv == "PA_BBND" and k_cache.ndim == 4 and k_cache.shape[2:] == (1, 576):
        page = k_cache.shape[1]
        ks, kp, kd = k_cache.stride(0), k_cache.stride(1), k_cache.stride(3)
    elif (
        layout_kv == "PA_BNBD"
        and k_cache.ndim == 4
        and k_cache.shape[1] == 1
        and k_cache.shape[3] == 576
    ):
        page = k_cache.shape[2]
        ks, kp, kd = k_cache.stride(0), k_cache.stride(2), k_cache.stride(3)
    elif (
        layout_kv == "PA_NZ"
        and k_cache.ndim == 5
        and k_cache.shape[1:3] == (1, 36)
        and k_cache.shape[4] == 16
        and k_cache.stride(4) == 1
    ):
        page = k_cache.shape[3]
        ks, kp, kd = k_cache.stride(0), k_cache.stride(3), k_cache.stride(2)
    else:
        raise ValueError("unsupported cache shape/layout")
    if page % 16 or not 16 <= page <= 1024:
        raise ValueError("page size must be 16-aligned in [16,1024]")
    if (
        block_table is None
        or block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.dtype != torch.int32
        or block_table.device != q.device
    ):
        raise ValueError("block_table must be device int32 [batch, pages]")
    if mask_mode == 0:
        if attn_mask is not None:
            raise ValueError("unmasked attention does not accept attn_mask")
    elif (
        attn_mask is None
        or attn_mask.shape != (2048, 2048)
        or attn_mask.dtype != torch.int8
        or attn_mask.device != q.device
    ):
        raise ValueError(
            "masked attention requires canonical device int8 [2048,2048] mask"
        )
    has_sink = sinks is not None and sinks.numel() != 0
    if has_sink and (
        sinks.shape != (heads,)
        or sinks.dtype != torch.float32
        or sinks.device != q.device
        or not sinks.is_contiguous()
    ):
        raise ValueError("sinks must be device float32 [heads]")
    if metadata is None:
        metadata = flash_mla_metadata(
            cache_seqlens,
            heads,
            1,
            cu_seqlens_q,
            seqused_q,
            physical_q if layout_q != "TND" else max_seqlen_q,
            max_seqlen_kv,
            mask_mode=mask_mode,
            win_left=win_left,
            win_right=win_right,
            layout_q=layout_q,
        )
    if (
        metadata.shape != (batch, 4)
        or metadata.dtype != torch.int32
        or metadata.device != q.device
        or not metadata.is_contiguous()
    ):
        raise ValueError("metadata must come from the A3 flash_mla_metadata backend")
    # A3's 192 KiB UB permits 32-head/64-key tiles with this static pipeline.
    # Reuse KV across more heads for long contexts; retain the smaller group
    # for short contexts/windows and NZ, where extra layout scratch is needed.
    block_n = 64
    capacity = block_table.shape[1] * page
    visible_capacity = capacity
    if mask_mode == 4 and win_left >= 0 and win_right >= 0:
        visible_capacity = min(capacity, win_left + win_right + 1)
        block_n = max(
            16, min(block_n, triton.next_power_of_2(max(1, visible_capacity)))
        )
    block_h = (
        32 if heads > 16 and visible_capacity >= 512 and layout_kv != "PA_NZ" else 16
    )
    programs = tokens * triton.cdiv(heads, block_h)
    max_tiles = max(1, triton.cdiv(capacity, block_n))
    if mask_mode == 4 and win_left >= 0 and win_right >= 0:
        max_tiles = min(max_tiles, max(1, triton.cdiv(visible_capacity, block_n)))
    splits = min(
        32,
        triton.next_power_of_2(triton.cdiv(128, programs)),
        triton.next_power_of_2(max_tiles),
    )
    partial = torch.empty(
        (tokens, heads, splits, 512), device=q.device, dtype=torch.float32
    )
    lse = torch.empty((tokens, heads, splits), device=q.device, dtype=torch.float32)
    out = torch.empty((tokens, heads, 512), device=q.device, dtype=q.dtype)
    final = torch.empty(
        (tokens, heads) if return_softmax_lse else (0,),
        device=q.device,
        dtype=torch.float32,
    )
    _partial[(tokens, triton.cdiv(heads, block_h), splits)](
        flat,
        k_cache,
        block_table,
        metadata,
        partial,
        lse,
        batch,
        tokens,
        heads,
        *flat.stride(),
        ks,
        kp,
        kd,
        *block_table.stride(),
        page,
        layout_kv == "PA_NZ",
        layout_q == "TND",
        physical_q,
        splits,
        softmax_scale,
        mask_mode,
        win_left,
        win_right,
        BH=block_h,
        BN=block_n,
        enable_mixed_cv=True,
        # The dynamic CV pipeline miscompiles this loop on the tested A3
        # compiler (triton-ascend 3.2.0). Keep the validated static pipeline.
        enable_dynamic_cv_pipeline=False,
    )
    _merge[(tokens, heads)](
        partial,
        lse,
        sinks,
        out,
        final,
        heads,
        splits,
        triton.next_power_of_2(splits),
        has_sink,
        return_softmax_lse,
    )
    if layout_q == "TND":
        return (
            out if layout_out == "TND" else out.permute(1, 0, 2).contiguous(),
            final.T.contiguous() if return_softmax_lse else final,
        )
    out = out.reshape(batch, physical_q, heads, 512)
    if layout_out == "BNSD":
        out = out.permute(0, 2, 1, 3).contiguous()
    if return_softmax_lse:
        final = final.reshape(batch, physical_q, heads).permute(0, 2, 1).contiguous()
    return out, final
