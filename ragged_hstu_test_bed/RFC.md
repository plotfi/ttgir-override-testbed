Triton SMEM Write Up + RFC

https://github.com/plotfi/triton/pull/3/files

https://github.com/plotfi/triton/tree/plotfi-local-gather-minimal

TODO:
* Provide pros and cons of ways to do this, tl.max value range versus others
    * Hint can be tunable, or take variable of the program
    * 
* tl.preload/prefetch
* Automate as much as possible instead of hint


This is a request for comment on a new ttgir instruction we have been experimenting with for one of our kernels at Meta. The goal of this work is to inch closer to the kind of control over shared memory accesses that is present in Cuda through a new ttg.local_gather instruction. Much like a tt.load+addr_add global gather this new local_gather instruction takes a base offset and an index tensor for computing a load from base per entry of the index tensor, except with the local_gather the base comes from a ptr<3> shared memory memdesc.

The TTGIR tablegen looks something like:

def TTG_LocalGatherOp : TTG_Op<"local_gather", [ ... ]> {
    let summary = "Gather from a local memory buffer into a distributed tensor";
    let arguments = (
        ins TT_MemDescType:$src,
        TT_IntTensor:$indices,
        Optional<TT_BoolLike>:$mask,
        Optional<TT_Type>:$other
    );
}

Because the base source argument is a memdesc that means that local_gather works in conjunction with local_alloc or a async_copy_global_to_local. The most obvious way to use a local_gather instead of a global gather via tt::LoadOp is in the body of a loop. An example of a gather kernel that could stand to benefit from local gathering from shared memory is the following:

```
@triton.jit
def kernel(base_ptr, index_ptrs, out_ptr, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]

    acc = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        indices = tl.load(index_ptrs + (roffset + rbase))
        gathered = tl.load(base_ptr + indices)
        acc = acc + gathered

    acc = tl.sum(acc, 1)[:, None]
    tl.store(out_ptr + xindex, acc, None)
```

Here we have a tight loop that gathers from global memory. Despite the access patern not being contiguous or swizzleable, there are potential upsites to pre-loading the value tensor’s content from global memory into shared memory. Below is a mixture of Triton source and TTGIR to illustrate this:


```
@triton.jit
def kernel(base_ptr, index_ptrs, out_ptr, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]

    bucket_size = tl.arange(0, 2048)
    tmp = tl.load(base_ptr + bucket_size)
    base_ptr = ttg.local_alloc(tmp) # TTGIR

    acc = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        indices = tl.load(index_ptrs + (roffset + rbase))
        gathered = ttg.local_gather(base_ptr, indices) # TTGIR
        acc = acc + gathered

    acc = tl.sum(acc, 1)[:, None]
    tl.store(out_ptr + xindex, acc, None)
```

In the recently published  ragged HSTU kernel from “Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations” (https://arxiv.org/abs/2402.17152), we have experimented with ttg.local_gather and found that for smaller block sizes (ie 64x64) there is a 6% reduction in kernel execution time.

The source for the ragger HSTU kernel is available at https://github.com/facebookresearch/generative-recommenders. And out experimentation with the local gather instruction has been done using the TRITON_KERNEL_OVERRIDE env var toggle to run the kernel from modified TTGIR in the cache. The test bed for these experiments lives at https://github.com/plotfi/ttgir-override-testbed/tree/main/ragged_hstu_test_bed.


