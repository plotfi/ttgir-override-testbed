#!/bin/bash

# export TRITON_CACHE_DIR=`pwd`/triton_cache_dir
export TRITON_CACHE_DIR=~/.triton
export TRITON_KERNEL_DUMP=0
export TRITON_KERNEL_OVERRIDE=1

export MLIR_ENABLE_DIAGNOSTICS=1
export MLIR_ENABLE_DUMP=0
export LLVM_IR_ENABLE_DUMP=0
export CUDA_VISIBLE_DEVICES=$(($RANDOM % 8))

HSTU_OVERRIDE_FILE=./_triton/override/cdaeab8c4765d8de364c7f0b39651b55ff5d6d5ed6e2341b1bae05029098ceb6/_ragged_hstu_attn_fwd.ttgir
HSTU_OVERRIDE_FILE_128x128x1x8=./_triton/override/2d6e2950bb6d59995d363b3a3935309d13f9be4ec25b8d6cc4008cda2d5a05e6/_ragged_hstu_attn_fwd.ttgir
BENCH=$HSTU_BENCH_EXPERIMENT
TOTAL_RUN_BASE=4.34237

cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_64x64x2x4.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py

if [[ $BENCH -eq 1 ]]; then
  echo "DROP masks for TW and PW tl.loads: 2-3% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_gmem_nomask.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 2 ]]; then
  echo "KEEP PW mask, replace TW with local_gather: 1-2% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_local_gather_PW_mask.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 3 ]]; then
  echo "DROP PW mask, replace TW with local_gather: 3.6-8% reduction, but sometimes negative or 0... weird. Should be 4.5% or so."
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_local_gather_PW_nomask.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 4 ]]; then
  echo "local_gather for TW AND PW: 10-12% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 5 ]]; then
  echo "Original HSTU ttgir dump: should be 0% change"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_orig.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 6 ]]; then
  echo "local_gather for PW, GMEM load for TW: ????% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_gmem_TW_smem_PW_local_gather.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 7 ]]; then
  echo "local_gather for TW AND PW, PW no mask: ???% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather_PW_nomask.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 8 ]]; then
  echo "local_gather for TW AND PW, TW & PW no mask: ???% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather_TW_PW_nomask.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 9 ]]; then
  echo "local_gather for TW AND PW, TW no mask: ???% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather_TW_nomask.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 10 ]]; then
  echo "local_gather with no mask for TW, leave PW with gmem+mask alone ???% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_local_gather_TW_nomask.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 4128 ]]; then
  HSTU_OVERRIDE_FILE=$HSTU_OVERRIDE_FILE_128x128x1x8
  TOTAL_RUN_BASE=7.51967
  echo "local_gather for TW AND PW, block 128x128x1x8: ???% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_128x128x1x8.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather_128_128_1_8.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 5128 ]]; then
  HSTU_OVERRIDE_FILE=$HSTU_OVERRIDE_FILE_128x128x1x8
  TOTAL_RUN_BASE=7.51967
  echo "Original HSTU ttgir dump, block 128x128x1x8: should be 0% change"
  git checkout $HSTU_OVERRIDE_FILE
  cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_128x128x1x8.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py
  cp ttgir/_ragged_hstu_attn_fwd_orig_128_128_1_8.ttgir $HSTU_OVERRIDE_FILE
else
  echo "Running what's at $HSTU_OVERRIDE_FILE"
fi

rm -rf $TRITON_CACHE_DIR
rm -f hstu*.out
cp -r ./_triton $TRITON_CACHE_DIR

if [[ $RUN_NCU -eq 1 ]]; then
  rm -rf ./ncu_tmp; mkdir ncu_tmp
  export TMPDIR=`pwd`/ncu_tmp
  CUDA_INJECTION64_PATH=none ncu --set full --target-processes all -o $BENCH.prof -f --import-source yes python ragged_hstu_attention_bench.py
  exit 0
fi

if [[ $RUN_ONCE -eq 1 ]]; then
  echo "Running once and copying PTX to PTX/$BENCH.ptx..."
  python ragged_hstu_attention_bench.py
  find $TRITON_CACHE_DIR -type f -iname "*.ptx" -print0 -exec cp {} ./PTX/$BENCH.ptx \;
  exit 0
fi

if [[ $RUN_DEBUG -eq 1 ]]; then
  # ~/opt/tools/cgdb/build/cgdb/cgdb --args
  # gdb --args
  lldb -- python ragged_hstu_attention_bench.py
  exit 0
fi

echo "Running 3 times and printing the average overhead reduction (higher % is better)"

export CUDA_VISIBLE_DEVICES=$(($RANDOM % 8))
python ragged_hstu_attention_bench.py 2>&1 | tee hstu1.out
find $TRITON_CACHE_DIR -type f -iname "*.ptx" -print0 -exec cp {} ./PTX/$BENCH.ptx \;
export CUDA_VISIBLE_DEVICES=$(($RANDOM % 8))
python ragged_hstu_attention_bench.py 2>&1 | tee hstu2.out
export CUDA_VISIBLE_DEVICES=$(($RANDOM % 8))
python ragged_hstu_attention_bench.py 2>&1 | tee hstu3.out

cat hstu*.out | grep latency | cut -f4 -d' ' | tr '\n' '+' | xargs -I% echo "% 0" | bc -l

cat hstu*.out | grep latency | cut -f4 -d' ' | tr '\n' '+' | \
xargs -I% echo "100 - (100 * ((% 0) / ($TOTAL_RUN_BASE * 3)))" | bc -l | \
  xargs -I$ echo "Overhead Reduction for HSTU: $%"
# rm -rf $TRITON_CACHE_DIR ./hammer/generative_recommenders/ops/triton/__pycache__
rm -f hstu*.out
