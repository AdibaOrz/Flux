#!/bin/bash
# FLUX-only ADNI runs with fixed lr/bs and hidden_dim=128 (ablation vs tuned hd=256).
# Total: 4 partitions x 5 seeds = 20 runs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

PARTITIONS="modality_unique med_intra_0p0 low_intra_0p0 stratified"
SEEDS="42 43 44 45 46"
ALLOWED_GPUS=(0 1 2 3 4 5 6 7)
NUM_ALLOWED_GPUS=${#ALLOWED_GPUS[@]}
gpu_index=0

LOCAL_LR=1e-4
LOCAL_BS=8
FLUX_HIDDEN_DIM=128

NOTE="${NOTE:-$(date +%y%m%d)_flux_adni}"

count=0
for partition in $PARTITIONS; do
  for seed in $SEEDS; do

    gpu=${ALLOWED_GPUS[$gpu_index]}
    gpu_index=$(( (gpu_index + 1) % NUM_ALLOWED_GPUS ))

    echo "Starting flux (partition=$partition, lr=$LOCAL_LR, bs=$LOCAL_BS, hd=$FLUX_HIDDEN_DIM, seed=$seed) on gpu=$gpu"
    python -m adni_pipeline.run_adni \
                   --method flux \
                   --partition_name $partition \
                   --local_lr $LOCAL_LR \
                   --local_bs $LOCAL_BS \
                   --flux_hidden_dim $FLUX_HIDDEN_DIM \
                   --note "${NOTE}" \
                   --seed $seed \
                   --gpu $gpu &

    count=$((count + 1))
    if [ $((count % NUM_ALLOWED_GPUS)) -eq 0 ]; then
      wait
    fi
  done
done

wait
echo "FLUX ADNI (hd=128) runs completed. Total: $count"
