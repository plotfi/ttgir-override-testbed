# TTGIR Test Bed for Triton Kernels

The following is a test bed for experimenting with modified TTGIR Triton kernels.

For testing out ragged HSTU try out the run.sh script in the ragged_hstu_test_bed directory:

```
cd ragged_hstu_test_bed
HSTU_BENCH_EXPERIMENT=4 bash run.sh
```

By default the run.sh script runs the kernel 3 times to get an average. The output will look like this:

```
local_gather for TW AND PW: 6% reduction

Running 3 times and printing the average overhead reduction (higher % is better)

Overriding kernel with file _ragged_hstu_attn_fwd.ttgir

P50 latency is 1.36168 ms
P20 latency is 1.35772 ms
P80 latency is 1.36673 ms

P50 latency is 1.36346 ms
P20 latency is 1.35869 ms
P80 latency is 1.36879 ms

P50 latency is 1.36536 ms
P20 latency is 1.35964 ms
P80 latency is 1.37253 ms

TOTAL RUNNING TIME FOR ALL 3x3 RUNS: 12.27460
Overhead Reduction for HSTU: 5.77649225346220305200%
```

For AMD targets use `USE_ROCM=1` and to only run the benchmark once use `RUN_ONCE=1`

If you're curious about the `HSTU_BENCH_EXPERIMENT` number, read the run.sh script code, but heres a run down:

```
These are all 64x64 num_stages=2 num_warps=4:

HSTU_BENCH_EXPERIMENT=1: DROP masks for TW and PW tl.loads
HSTU_BENCH_EXPERIMENT=2: TW with local_gather
HSTU_BENCH_EXPERIMENT=3: TW with local_gather, drop PW mask
HSTU_BENCH_EXPERIMENT=4: local_gather for TW AND PW
HSTU_BENCH_EXPERIMENT=5: Original HSTU TTGIR Unmodified
HSTU_BENCH_EXPERIMENT=6: local_gather for PW
HSTU_BENCH_EXPERIMENT=7: local_gather for TW AND PW, PW no mask
HSTU_BENCH_EXPERIMENT=8: local_gather for TW AND PW, no mask for either
HSTU_BENCH_EXPERIMENT=9: local_gather for TW AND PW, TW no mask


Tile Size 128x128:

HSTU_BENCH_EXPERIMENT=4128: local_gather for TW AND PW
HSTU_BENCH_EXPERIMENT=5128: Original HSTU Kernel

Tile Size 32x32 num_stages=2 num_warps=2:

HSTU_BENCH_EXPERIMENT=432: local_gather for TW AND PW
HSTU_BENCH_EXPERIMENT=532: Original HSTU Kernel

```

