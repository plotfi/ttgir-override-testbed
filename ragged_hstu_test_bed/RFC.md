# RFC: ttg.local_gather

### Introduction
This is a request for comment on a new TTGIR instruction currently being tested on attention kernels at Meta. We call our proposed instruction `ttg.local_gather`, which allows loading data from multiple arbitrary shared memory addresses. Similar to a global gather (using `tt.load` with `tt.addptr`), the local_gather instruction takes a base offset and an index tensor to determine memory load locations. However, local_gather operates on shared memory that has had its data pre-copied from global memory.
local_gather description

The TableGen MLIR operation description for local_gather is structured as follows. The base is determined using a MemDesc to address space 3 (NVPTX and AMDGPU shared memory), and the offsets are derived from an index tensor:

```
def TTG_LocalGatherOp : TTG_Op<"local_gather", [ ... ]> {
    let summary = "Gather from local memory buffer into tensor";
    let arguments = ( ins 
        TT_MemDescType:$src,
        TT_IntTensor:$indices,
        Optional<TT_BoolLike>:$mask,
        Optional<TT_Type>:$other
    );
}
```

For a local_gather instruction with base address `B` and index tensor `I` of size `N`, the `matchToRewrite` lowering of TTGIR to LLIR generates `N/(threadsPerWarp * warpsPerCTA)` LLVM-IR address space 3 loads from `B + I[n]` (where `0 ≤ n < N`).

Because the base source argument is a MemDesc, local_gather is naturally used in conjunction with local_alloc or async_copy_global_to_local. We envision the most common use for local_gather in loops, where repeated local gathering from shared memory is supplied by allocation to the MemDesc location before the loop body would offer an advantage over repeated global gathering.

### Example Usage

The following is an example of a gather kernel that could stand to benefit from local gathering:

```
@triton.jit
def kernel(base_ptr, index_ptrs, out_ptr,
           rnumel,
           XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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

Here we have a tight loop that gathers from global memory, and because the access pattern is not contiguous or swizzle-able a `ttg.local_load` would not work for potentially squeezing more performance through use of shared memory. There are still upsides to preloading to shared memory, depending on reuse of offsets in the index tensor across loop iterations. 

Below is a mixture of Triton source and TTGIR to illustrate how this boilerplate kernel could preload the gathered contents:

```
@triton.jit
def kernel(base_ptr, index_ptrs, out_ptr,
           rnumel,
           BUCKET_SIZE : tl.constexpr,
           XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]

    # This is a coalesced read from dram into sram through the use of
    # tl.load with ttg.local_alloc, but cp.async could also work by using
    # ttg.async_copy_global_to_local instead
    bucket_range = tl.arange(0, BUCKET_SIZE)
    tmp = tl.load(base_ptr + bucket_range)
    base_ptr = ttg.local_alloc(tmp) # TTGIR

    acc = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        indices = tl.load(index_ptrs + (roffset + rbase))
        # Reads from shared memory
        gathered = ttg.local_gather(base_ptr, indices) # TTGIR
        acc = acc + gathered
    acc = tl.sum(acc, 1)[:, None]
    tl.store(out_ptr + xindex, acc, None)
```

### Experimentation Results in Production Kernels

We have conducted experiments using ttg.local_gather on the recently published Ragged HSTU Attention kernel from Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (https://arxiv.org/abs/2402.17152). The source for this work is available at https://github.com/facebookresearch/generative-recommenders, and the file for the Triton kernel experimented with is at https://github.com/facebookresearch/generative-recommenders/blob/main/ops/triton/triton_ragged_hstu_attention.py.

Our experiments were conducted using the TRITON_KERNEL_OVERRIDE environment variable to run the Ragged HSTU Attention kernel from cached TTGIR that was modified to use ttg.local_gather. They show that using shared memory to preload and gather non-contiguous accesses has promising results for smaller block sizes:

* There is a 6% reduction in kernel execution time for size 64x64 when using ttg.local_gather with the HSTU kernel’s Position Bias and Time Bias tensors on H100. Config is as follows: `triton.Config( {"BLOCK_M": 64, "BLOCK_N": 64}, num_stages=2, num_warps=4 )`
* There is also a roughly 5% reduction in kernel execution time for size 32x32 tile kernels on H100. Config is as follows: `triton.Config( {"BLOCK_M": 32, "BLOCK_N": 32}, num_stages=2, num_warps=2 )`
* <<< ADD RESULTS FOR AMD >>>


The test bed for these experiments lives at https://github.com/plotfi/ttgir-override-testbed, and the latest compiler changes to support this are at https://github.com/plotfi/triton/tree/plotfi-local-gather-rfc.
Upstreaming Proposal

We are interested in upstreaming the ttg.local_gather instruction as we further experiment to determine its potential for enhancing the performance of loop-gathered kernels like Ragged HSTU Attention. We are open to renaming local_gather to something more indicative of its experimental nature, similar to experimental_descriptor_load/store (e.g., experimental_local_gather).

For the time being this instruction will only be accessible through modified TTGIR using the TRITON_KERNEL_OVERRIDE environment variable. We do also have some ideas around how to expose local memory to Triton at the source level as well, which we cover in the next section.
Ideas for Triton Language Level Exposure

### 1) Compiler Hint

A compiler hint could work much like tl.max_contiguious etc. Let's call this hint tl.preload and imagine it could be used like so:

```
@triton.jit
def kernel(base_ptr, index_ptrs, out_ptr,
           rnumel,
           BUCKET_SIZE : tl.constexpr,
           XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]
    acc = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        indices = tl.load(index_ptrs + (roffset + rbase))
        # HINT:
        gathered = tl.preload(tl.load(base_ptr + indices), buckets=BUCKET_SIZE)
        acc = acc + gathered
    acc = tl.sum(acc, 1)[:, None]
    tl.store(out_ptr + xindex, acc, None)
```

Here the hint contains information like the size of the bucket as well as the notion that the contents of the load could be preloaded into shared memory (and hoisted) before the loop.

It could also be possible to encode such a hint into the tt:LoadOp/tl.load itself, in much the same way the tt::LoadOp’s eviction_policy or cache_modifier are set:


`gathered = tl.load(base_ptr + indices, preload_with_buckets=BUCKET_SIZE)`

To clarify, the purpose of the hint goes beyond merely suggesting to the compiler that a global gather can be optimized into a local gather. It also serves as a mechanism to convey information about tuning parameters and variables for shared memory allocation within the local context. Additionally, the hint must specify the `constexpr` bucket size used to allocate the chunk of shared memory for the local context.

Additionally, we need to consider how these hints can be integrated into the auto-tuner. We believe that using an autotuner APIs like `prune_configs_by` to integrate hinting with the auto-tuner can allow us to prune configurations where there is insufficient shared memory for preloading or where the `constexpr` bucket size does not match the non-constexpr number of elements. This way, shared memory is only used to locally gather when it makes sense to and when it stands a chance at boosting performance.

Pros: The advantage of this approach is that it strikes a balance between programmer control and platform independence. By using hints, kernel writers can explicitly highlight optimization opportunities with suggested bucket sizes (or other tuning variables) without necessitating that the compiler automatically manage local memory allocation and hoisting for every Triton backend. It also requires less explicit intervention by the kernel writer and allows for the optimizer to improve things with ongoing releases of Triton.

Cons: The drawbacks of this approach include the lack of transparency in the developer’s Triton code regarding code hoisting and shared memory allocations. While the hint explicitly signals the compiler to perform certain optimizations, the specifics of these actions remain opaque. For instance in the presence of loop nests even experienced kernel writers might find it unclear at which loop level the transfer from global to local memory takes place.

### 2) Explicit TL operators for preloading

### Part 1: Explicit Preload, Explicit Local Gather
Explicit operators for preload and gather offer another promising method for expressing shared memory operations in Triton:

```
@triton.jit
def kernel(base_ptr, index_ptrs, out_ptr,
           rnumel,
           BUCKET_SIZE : tl.constexpr,
           XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]
    acc = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    # Preload into shared memory (or warm caches) here explicitly, before loop:
    base_ptr = tl.preload(base_ptr, buckets=BUCKET_SIZE)
    for roffset in range(0, rnumel, RBLOCK):
        indices = tl.load(index_ptrs + (roffset + rbase))

        # Explicit ttg.local_gather, materialize into ld.shared or equivalent:
        gathered = tl.local_gather(base_ptr, indices)
        acc = acc + gathered
    acc = tl.sum(acc, 1)[:, None]
    tl.store(out_ptr + xindex, acc, None)
```

For most GPU targets it should be feasible to map such explicit operations to the target hardware’s local memory. For alternative targets like the CPU, we propose that the same explicit preload operation serves as the mechanism for warming caches before the entry into a tight loop.

### Part 2: Explicit Preload, Implicit Local Gather
A slight variation on the previous approach is to implicitly express the local gather using tl.load, all the while basing the LLVM operation materialization for the gather on the resulting MemDesc type from tl.preload. This will eventually materialize as ttg.local_gather later in the compiler pipeline:

```
@triton.jit
def kernel(base_ptr, index_ptrs, out_ptr,
           rnumel,
           BUCKET_SIZE : tl.constexpr,
           XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    rbase = tl.arange(0, RBLOCK)[None, :]
    acc = tl.full([XBLOCK, RBLOCK], 0, tl.float32)

    # Preload into shared memory (or warm caches) here explicitly, before loop:
    base_ptr = tl.preload(base_ptr, buckets=BUCKET_SIZE)
    for roffset in range(0, rnumel, RBLOCK):
        indices = tl.load(index_ptrs + (roffset + rbase))

        # Implicitly local_gather, materializes as a ttg.local_gather
        # (and later materialize as ld.shared or equivalent)
        # instead of a tt.LoadOp:
        gathered = tl.load(base_ptr + indices)
        acc = acc + gathered
    acc = tl.sum(acc, 1)[:, None]
    tl.store(out_ptr + xindex, acc, None)
```

The advantage to this explicit tl.preload with an implicit-context-aware tl.load approach is that the kernel writer can still maintain control over where the preload is hoisted. The behavior of the `tt.load` as a local gather in this approach is implicit however, and the gather location in code is unmodified from the original. Admittedly this does make the behavior of tl.load somewhat magical but it also affords very usable control over hand optimizing memory accesses.

For targets where local memory (or a cache warming equivalent) is either not feasible or not implemented the tl.preload could simply materialize as a memory pointer passthrough no-op. It should also be noted that the same auto-tuning and pruning integration mentioned above in the hinting approach could be integrated with tl.preload as well. 

Pros: The explicit approach gives the kernel writer control over the location of the loop hoisted preload rather than allowing for the compiler to handle this implicitly (and potentially in a sub-optimum fashion). It also makes for more readily apparent intent when reading PRs that modify kernels to take  advantage of shared memory using local gathers.

Cons: The disadvantage to this approach is mostly that we have to be very careful in how we implement it so that written Triton code is not coupled to any specific hardware vendor or class of compute devices (i.e. just GPUs, just NV etc).

### 3) Fully Automated Compiler Optimization
Another avenue for incorporating local gathering into Triton is to implement a compiler pass that analyzes and detects when a loop-nested global gather can be transformed into a hoisted-preloaded local gather without any need for hints or manual code intervention. However, handling such a transformation without any indicator could be risky due to the potential of running out of limited shared memory.

Pros: No manual intervention needed. This also means that there would be no potential cross platform worries. 

Cons: Relying too heavily on compiler optimization. Such approaches in compilers have shown to have the propensity of resulting in unpredictable performance. Introduces complexity by turning the shared memory problem into the equivalent of another register allocation problem.

### Summary
We have presented ttg.local_gather for upstreaming consideration. Based on our internal testing we see a 5-6% reduction in kernel overhead for the HSTU kernel on H100, and we wish to continue experimenting. We look forward to your comments and ideas around this proposal as well as how to expose such an operation at the Triton language level.
