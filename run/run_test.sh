#!/usr/bin/env bash
# FLUX smoke test: pamap2 only; availability {complete, medium, low} × stability {medium, low} × seeds {42..46}.

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1

gpus=(0 1 2 3 4 5 6 7)
gpu_idx=-1
count=0

METHOD=flux
DATASET=pamap2

extra_args=()
if [ "${NO_WANDB:-0}" = "1" ]; then
  extra_args=(--no-wandb)
fi

run_note="${NOTE:-$(date +%y%m%d)_flux_smoke_test}"

for availability in complete medium low
do
  for stability in medium low
  do
    for seed in 42 43 44 45 46
    do
      gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
      gpu=${gpus[gpu_idx]}
      echo "Run dataset=$DATASET method=$METHOD note=$run_note " \
            "seed=$seed gpu=$gpu avail=$availability stab=$stability"
      python main.py --dataset "$DATASET" \
                     --method "$METHOD" \
                     --gpu "$gpu" \
                     --seed "$seed" \
                     --availability "$availability" \
                     --stability "$stability" \
                     --note "$run_note" \
                     "${extra_args[@]}" &
      count=$((count + 1))
      if [ $((count % ${#gpus[@]})) -eq 0 ]; then
        wait
      fi
    done
  done
done
wait

echo "All FLUX test runs finished OK (30 jobs: 3×2×5)."
