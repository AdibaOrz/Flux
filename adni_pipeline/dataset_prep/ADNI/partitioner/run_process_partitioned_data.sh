#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADNI_PREP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATH_CONFIG="${ADNI_PATH_CONFIG:-${ADNI_PREP_DIR}/adni_paths.json}"

if [ ! -f "${PATH_CONFIG}" ]; then
    echo "Missing ADNI path config: ${PATH_CONFIG}" >&2
    echo "Copy ${ADNI_PREP_DIR}/adni_paths_example.json to adni_paths.json and update it." >&2
    exit 1
fi

BASE_DIR="$(python -c 'import json,sys; from pathlib import Path; print(Path(json.load(open(sys.argv[1]))["adni_root"]).expanduser())' "${PATH_CONFIG}")/partitioned"

# Partition dirs used in main ADNI experiments (see ADNI_PARTITION_MAP in adni_config.py
# and run_all_adni.sh / run_flux_adni.sh). Run simulate_missingness.py for med/low only.
PARTITION_DIRS=(
    "$BASE_DIR/stratified_5silos"
    "$BASE_DIR/modality_unique_5silos"
    "$BASE_DIR/stratified_5silos_sim_avail_med_intra0p0"
    "$BASE_DIR/stratified_5silos_sim_avail_low_intra0p0"
)

# Modalities to process
MODALITIES=("clinical" "biospecimen" "genomic")

# Processing modes
# MODES=("centralized" "federated")
MODES=("federated")
# Run for all combinations
for split_dir in "${PARTITION_DIRS[@]}"; do
    for modality in "${MODALITIES[@]}"; do
        for mode in "${MODES[@]}"; do
            echo "Processing: $split_dir | $modality | $mode"
            python "${SCRIPT_DIR}/process_partitioned_data.py" \
                --path_config "${PATH_CONFIG}" \
                --split_dir "$split_dir" \
                --modality "$modality" \
                --mode "$mode"
        done
    done
done

echo "All processing complete!"