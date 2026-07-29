#!/usr/bin/env bash
# Run every method launcher in this directory once each, in order, one script after another

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$RUN_DIR/.." && pwd)"
cd "$ROOT" || exit 1


METHOD_SCRIPTS=(
  flux.sh
  fedavg.sh
  fedprox.sh
  fedper.sh
  fedrod.sh
  moon.sh
  pmcmfl.sh
  pepsy.sh
)

for script in "${METHOD_SCRIPTS[@]}"; do
  path="$RUN_DIR/$script"
  if [ ! -f "$path" ]; then
    echo "Missing: $path" >&2
    exit 1
  fi
  echo "=============================="
  echo "Starting $script"
  echo "=============================="
  bash "$path" || exit 1
  echo "Finished $script"
  echo
done

echo "run_all.sh: all method scripts completed successfully."
