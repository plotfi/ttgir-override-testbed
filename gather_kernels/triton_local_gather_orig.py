"""
Based on https://github.com/pytorch/pytorch/issues/121661
"""

import torch

import triton
import triton.language as tl
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride


@triton.autotune(
    configs=[
        triton.Config(
            {
                "XBLOCK": 1,
                "RBLOCK": 2048,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 8,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 4,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 8,
                "RBLOCK": 512,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 8,
                "RBLOCK": 256,
            },
            num_stages=1,
            num_warps=8,
        ),
        triton.Config(
            {
                "XBLOCK": 64,
                "RBLOCK": 64,
            },
            num_stages=1,
            num_warps=8,
        ),
    ],
    key=["xnumel", "rnumel"],
)
# pass in 2 1D index pointers, we gather from base_ptr to get a 2D tensor of [XBLOCK, RBLOCK]
# base_ptr will be 1D with size xnumel, ts_0_ptrs will have size xnumel, ts_1_ptrs will have size rnumel
# will perform a matrix vector operation on the 2D tensor with vec_ptr
@triton.jit
def triton_local_gather(base_ptr, vec_ptr, ts_0_ptrs, ts_1_ptrs, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    rbase = tl.arange(0, RBLOCK)[None, :].to(tl.int64)
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    # base_ptr has size xnumel, we can prefetch its content, then use local_gather inside the loop
    # ts_0 is a slice of ts_0_ptrs with size XBLOCK, ts_1 is a slice of ts_1_ptrs with size RBLOCK
    # the index matrix is formed from ts_0 and ts_1 [XBLOCK, RBLOCK]
    ts_0 = tl.load(ts_0_ptrs + x0) # [XBLOCK, 1]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        r1 = rindex # size (1, RBLOCK)
        tmp7 = tl.load(vec_ptr + (r1), None, eviction_policy='evict_last').to(tl.float32)
        ts_1 = tl.load(ts_1_ptrs + rindex)
        ts = ts_0 - ts_1
        ts = tl.where(ts > 0, ts, 0)
        ts = tl.where(ts < xnumel, ts, xnumel-1)
        # ts is calculated from ts_0 [XBLOCK, 1] and ts_1 [1, RBLOCK]
        # its value is between [0, xnumel)
        tmp4 = tl.load(base_ptr + ts, None, eviction_policy='evict_first')
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp6 * tmp8 # (XBLOCK, RBLOCK) * (1, RBLOCK)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tmp12
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tmp11.to(tl.float32)
    tl.store(out_ptr1 + (x0), tmp13, None)


def triton_gemv_0(base, vec, ts_0, ts_1):
    S, = vec.shape
    assert_size_stride(base, (2*S, ), (1, ))
    assert_size_stride(vec, (S, ), (1, ))
    assert_size_stride(ts_0, (2*S, ), (1, ))
    assert_size_stride(ts_1, (S, ), (1, ))
    xnumel = 2*S
    rnumel = S
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        # size will be double
        buf1 = empty_strided_cuda((2*S, ), (1, ), torch.bfloat16)
        grid = lambda META: (
            triton.cdiv(2*S, META["XBLOCK"]),
        )
        triton_local_gather[grid](base, vec, ts_0, ts_1, buf1, xnumel, rnumel)
    return (reinterpret_tensor(buf1, (2, S), (S, 1), 0), )

