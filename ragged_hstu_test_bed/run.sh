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
HSTU_OVERRIDE_FILE_32x32x2x2=./_triton/override/ed17ebb3a69f0d0b7d116894712de989509539327a9a5fe60511cf55cc90a654/_ragged_hstu_attn_fwd.ttgir
BENCH=$HSTU_BENCH_EXPERIMENT
TOTAL_RUN_BASE=4.34237

ROCM_FILE_MOD=""
if [[ USE_ROCM -eq 1 ]]; then
  git checkout hammer/generative_recommenders/ops/triton/triton_ragged_hstu*
  patch -p1 < rocm.patch
  ROCM_FILE_MOD="_ROCM"
fi

cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_64x64x2x4.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py
# cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_32x32x2x2.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py

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
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather$ROCM_FILE_MOD.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 5 ]]; then
  echo "Original HSTU ttgir dump: should be 0% change"
  git checkout $HSTU_OVERRIDE_FILE
  cp ttgir/_ragged_hstu_attn_fwd_orig$ROCM_FILE_MOD.ttgir $HSTU_OVERRIDE_FILE
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
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather_128_128_1_8$ROCM_FILE_MOD.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 5128 ]]; then
  HSTU_OVERRIDE_FILE=$HSTU_OVERRIDE_FILE_128x128x1x8
  TOTAL_RUN_BASE=7.51967
  echo "Original HSTU ttgir dump, block 128x128x1x8: should be 0% change"
  git checkout $HSTU_OVERRIDE_FILE
  cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_128x128x1x8.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py
  cp ttgir/_ragged_hstu_attn_fwd_orig_128_128_1_8$ROCM_FILE_MOD.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 432 ]]; then
  HSTU_OVERRIDE_FILE=$HSTU_OVERRIDE_FILE_32x32x2x2
  TOTAL_RUN_BASE=(14.70786/3)
  echo "local_gather for TW AND PW, block 32x32x2x2: ???% reduction"
  git checkout $HSTU_OVERRIDE_FILE
  cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_32x32x2x2.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py
  cp ttgir/_ragged_hstu_attn_fwd_smem_TW_PW_local_gather_32_32_2_2$ROCM_FILE_MOD.ttgir $HSTU_OVERRIDE_FILE
elif [[ $BENCH -eq 532 ]]; then
  HSTU_OVERRIDE_FILE=$HSTU_OVERRIDE_FILE_32x32x2x2
  TOTAL_RUN_BASE=(14.70786/3)
  echo "Original HSTU ttgir dump, block 32x32x2x2: should be 0% change"
  git checkout $HSTU_OVERRIDE_FILE
  cp hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention_32x32x2x2.py hammer/generative_recommenders/ops/triton/triton_ragged_hstu_attention.py
  cp ttgir/_ragged_hstu_attn_fwd_orig_32_32_2_2$ROCM_FILE_MOD.ttgir $HSTU_OVERRIDE_FILE
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
  git checkout hammer/generative_recommenders/ops/triton/triton_ragged_hstu*
  exit 0
fi

if [[ $RUN_ONCE -eq 1 ]]; then
  which python
  echo "Running once and copying PTX/AMDGCN to PTX/$BENCH.ptx or AMDGCN/$BENCH.amdgcn ..."
  python ragged_hstu_attention_bench.py
  find $TRITON_CACHE_DIR -type f -iname "*.ptx" -print0 -exec cp {} ./PTX/$BENCH.ptx \;
  find $TRITON_CACHE_DIR -type f -iname "*.amdgcn" -print0 -exec cp {} ./AMDGCN/$BENCH.amdgcn \;
  git checkout hammer/generative_recommenders/ops/triton/triton_ragged_hstu*
  exit 0
fi

if [[ $RUN_DEBUG -eq 1 ]]; then
  # ~/opt/tools/cgdb/build/cgdb/cgdb --args
  # gdb --args
  lldb -- python ragged_hstu_attention_bench.py
  git checkout hammer/generative_recommenders/ops/triton/triton_ragged_hstu*
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

echo -n "TOTAL RUNNING TIME FOR ALL 3x3 RUNS: "
cat hstu*.out | grep latency | cut -f4 -d' ' | tr '\n' '+' | xargs -I% echo "% 0" | bc -l

cat hstu*.out | grep latency | cut -f4 -d' ' | tr '\n' '+' | \
xargs -I% echo "100 - (100 * ((% 0) / ($TOTAL_RUN_BASE * 3)))" | bc -l | \
  xargs -I$ echo "Overhead Reduction for HSTU: $%"
# rm -rf $TRITON_CACHE_DIR ./hammer/generative_recommenders/ops/triton/__pycache__
rm -f hstu*.out
git checkout hammer/generative_recommenders/ops/triton/triton_ragged_hstu*
