"""
Based on https://github.com/pytorch/pytorch/issues/121661
"""

def nothing(a, b, c):
    print("do nothing")

import torch

import triton
import triton.language as tl
empty_strided_cuda = torch.empty_strided
reinterpret_tensor = torch.as_strided
assert_size_stride = nothing

import time
from torch._dynamo.testing import rand_strided

import csv
import os
import statistics
from typing import Any, Callable, Generator, List, Optional

import numpy

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
def triton_global_gather(base_ptr, vec_ptr, ts_0_ptrs, ts_1_ptrs, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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

    ### TRITON_SMEM, local_gather preload to smem
    ### this is expected to produce ttg.local_alloc(tt.load)
    if False:
        base_ptr_smem = tl.load(base_ptr, None, eviction_policy='evict_first', shared=True)
    else:
        base_ptr_smem = base_ptr

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

        ### TRITON_SMEM, local_gather read from smem
        ### this is expected to produce ttg.local_gather(base_ptr_smem, ts)
        tmp4 = tl.load(base_ptr_smem + ts, None, eviction_policy='evict_first')

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

benchmark_iters = 10000

def triton_gemv_0(base, vec, ts_0, ts_1, benchmark: bool = False):
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
        buf1 = empty_strided_cuda((2*S, ), (1, ), dtype=torch.bfloat16, device='cuda:0')
        grid = lambda META: (
            triton.cdiv(2*S, META["XBLOCK"]),
        )

        is_local_gather = os.environ.get("TRITON_LOCAL_GATHER") == "1"
        gather_kernel = triton_local_gather if is_local_gather else triton_global_gather

        if benchmark:
            print(f"Benchmarking {'local' if is_local_gather else 'global'} gather for shape [{S}]")
            # Warm up GPU
            for _ in range(10):
                gather_kernel[grid](base, vec, ts_0, ts_1, buf1, xnumel, rnumel)
            torch.cuda.synchronize()

            # Timing the matmul
            start_time = time.time()
            for _ in range(benchmark_iters):
                gather_kernel[grid](base, vec, ts_0, ts_1, buf1, xnumel, rnumel)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            print(f"Elapsed gather kernel time for {benchmark_iters} iterations: {elapsed_time:.6f} seconds")
        else:
            print(f"Running gather for shape [{S}]")
            triton_local_gather[grid](base, vec, ts_0, ts_1, buf1, xnumel, rnumel)


    return (reinterpret_tensor(buf1, (2, S), (S, 1), 0), )


triton_test_0 = triton_gemv_0

BUILDIN_SHAPES = [
    (2048),
    # (4096),
    # (8192),
    # (16384),
]


shapes = BUILDIN_SHAPES

def get_x_val(example_inputs) -> float:
    base, vec, ts_0, ts_1 = example_inputs
    s = vec.size()
    return s

for shape in shapes:
    S = shape
    base = rand_strided((2*S, ), (1, ), device='cuda:0', dtype=torch.int8)
    vec = rand_strided((S, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    ts_0 = rand_strided((2*S, ), (1, ), device='cuda:0', dtype=torch.int32)
    ts_1 = rand_strided((S, ), (1, ), device='cuda:0', dtype=torch.int32)
    triton_test_0(base, vec, ts_0, ts_1, benchmark=(os.environ.get("TRITON_GATHER_BENCHMARK") == "1" ))

