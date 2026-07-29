#!/usr/bin/env bash

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.." || exit 1

gpus=(0 1 2 3 4 5 6 7)
gpu_idx=-1
count=0

METHOD=fedper
RUN_DATE=$(date +%y%m%d)

for dataset in pamap2 realworldhar sleepedf20
do
  run_note="${NOTE:-${RUN_DATE}_${METHOD}_${dataset}}"
  for availability in complete medium low
  do
    for stability in medium low
    do
      for seed in 42 43 44 45 46
      do
        gpu_idx=$(( (gpu_idx + 1) % ${#gpus[@]} ))
        gpu=${gpus[gpu_idx]}
        echo "Run dataset=$dataset method=$METHOD note=$run_note " \
              "seed=$seed gpu=$gpu avail=$availability stab=$stability"
        python main.py --dataset $dataset  \
                       --method $METHOD  \
                       --note "$run_note"  \
                       --seed $seed  \
                       --gpu $gpu \
                       --availability $availability \
                       --stability $stability &
        count=$((count + 1))
        if [ $((count % ${#gpus[@]})) -eq 0 ]; then
          wait
        fi
      done
    done
  done
done
wait
