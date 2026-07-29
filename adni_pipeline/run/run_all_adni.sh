#!/bin/bash
# Run all 8 ADNI methods across 4 partition types with 5 seeds.
# Best configs are loaded automatically from adni_config.py — no CLI overrides needed.
# Total: 8 methods x 4 partitions x 5 seeds = 160 runs

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}" || exit 1

METHODS="fedavg fedprox fedper fedrod moon pmcmfl pepsy flux" # flux (ours)
PARTITIONS="modality_unique med_intra_0p0 low_intra_0p0 stratified"
SEEDS="42 43 44 45 46"
ALLOWED_GPUS=(0 1 2 3 4 5 6 7)
NUM_ALLOWED_GPUS=${#ALLOWED_GPUS[@]}
gpu_index=0

NOTE="${NOTE:-$(date +%y%m%d)_adni_main}"

count=0
for partition in $PARTITIONS; do
  for method in $METHODS; do
    for seed in $SEEDS; do

      gpu=${ALLOWED_GPUS[$gpu_index]}
      gpu_index=$(( (gpu_index + 1) % NUM_ALLOWED_GPUS ))

      echo "Starting $method (partition=$partition, seed=$seed) on gpu=$gpu"
      python -m adni_pipeline.run_adni \
                     --method $method \
                     --partition_name $partition \
                     --note "${NOTE}_${method}" \
                     --seed $seed \
                     --gpu $gpu &

      count=$((count + 1))
      if [ $((count % NUM_ALLOWED_GPUS)) -eq 0 ]; then
        wait
      fi
    done
  done
done

wait
echo "All ADNI main runs completed. Total: $count"
