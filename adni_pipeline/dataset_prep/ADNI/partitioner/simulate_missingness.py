"""Simulate modality missingness on top of an existing partition.

Two layers of simulated missingness are applied on top of natural missingness:
  1. Inter-client (structural): Each silo draws p_a ~ Beta(alpha, beta), then each
     modality is kept with Bernoulli(p_a). Dropped modalities are absent for ALL
     samples in the silo — mimicking hospitals without certain equipment.
  2. Intra-client (MCAR): For each sample, each remaining observed modality is
     independently dropped with probability p_intra — mimicking per-patient
     data collection failures.

Output (saved in a NEW partition-specific directory):
  - split_metadata.json  (same patient splits, augmented with simulation params)
  - labels.csv           (modified observed masks — original labels.csv is untouched)

Main ADNI experiments use stratified_5silos as the base partition with
p_intra=0.0 (inter-client missingness only). Partition names med_intra_0p0 and
low_intra_0p0 map to:
  - stratified_5silos_sim_avail_med_intra0p0
  - stratified_5silos_sim_avail_low_intra0p0

Usage (main experiments — run both after create_partitions.py):
  cd partitioner
  python simulate_missingness.py --availability avail_med --p_intra 0.0 --seed 42
  python simulate_missingness.py --availability avail_low --p_intra 0.0 --seed 42

Then run run_process_partitioned_data.sh (or process_partitioned_data.py per dir).

Optional intra-client ablations: pass a non-zero --p_intra (e.g. 0.3 or 0.5).
This writes a new directory such as stratified_5silos_sim_avail_med_intra0p3.
Those variants are not used by the default run_all_adni.sh sweep; process them
manually if you need them.

Full path overrides:
  python simulate_missingness.py \
      --base_partition /path/to/partitioned/stratified_5silos \
      --labels_path    /path/to/processed/labels.csv \
      --availability   avail_med \
      --p_intra        0.0 \
      --seed           42
"""

import os
import sys
import json
import copy
import argparse
import datetime
from pathlib import Path
import numpy as np
import pandas as pd

ADNI_PREP_DIR = Path(__file__).resolve().parents[1]
if str(ADNI_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(ADNI_PREP_DIR))

from path_config import load_adni_paths

MODAL_COLS = ['Image', 'Genomic', 'Clinical', 'Biospecimen']

# Beta(alpha, beta) parameters for inter-client availability.
# E[available modalities per silo] ≈ 4 × alpha / (alpha + beta)
# High (alpha+beta) keeps sampled p_a tightly around the mean — important with only M=4 modalities.
AVAILABILITY_PRESETS = {
    'avail_nat':  None,       # No inter-client drops (natural missingness only)
    'avail_med': (17, 3),     # E[p] = 0.85 → E[avail] ≈ 3.4 / 4  (most silos keep 3-4 modalities)
    'avail_low': (6, 14),     # E[p] = 0.30 → E[avail] ≈ 1.2 / 4  (most silos keep 1 modality)
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simulate inter- and intra-client modality missingness '
                    'on top of an existing partition.')
    parser.add_argument('--path_config', type=str, default=None,
                        help='Path to adni_paths.json (default: ADNI_PATH_CONFIG or ../adni_paths.json)')
    parser.add_argument('--base_partition', type=str, default=None,
                        help='Override base partition dir (default: partitioned/stratified_5silos)')
    parser.add_argument('--labels_path', type=str, default=None,
                        help='Override the original labels.csv path')
    parser.add_argument('--availability', type=str, default='avail_nat',
                        choices=list(AVAILABILITY_PRESETS.keys()),
                        help='Inter-client availability level')
    parser.add_argument('--p_intra', type=float, default=0.0,
                        help='Intra-client MCAR drop probability [0.0 = no drop]')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_base_partition(base_dir):
    """Load split_metadata.json from the base partition directory."""
    meta_path = os.path.join(base_dir, 'split_metadata.json')
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'split_metadata.json not found in {base_dir}')
    with open(meta_path) as f:
        return json.load(f)


def get_silo_ptids(split_meta):
    """Return {silo_name: set(all PTIDs across train/val/test)} from federated splits."""
    silo_dict = split_meta['data']['federated']
    return {
        silo: set().union(*(set(splits[s]) for s in ('train', 'val', 'test')))
        for silo, splits in silo_dict.items()
    }


def build_output_dir(base_partition, availability, p_intra):
    """Construct the output directory path from simulation parameters.

    E.g. .../stratified_5silos  →  .../stratified_5silos_sim_avail_med_intra0p3
    """
    p_str = str(p_intra).replace('.', 'p')
    tag = f'sim_{availability}_intra{p_str}'
    parent = os.path.dirname(base_partition.rstrip('/'))
    base_name = os.path.basename(base_partition.rstrip('/'))
    return os.path.join(parent, f'{base_name}_{tag}')


# ---------------------------------------------------------------------------
# Step 2 — Inter-client structural missingness
# ---------------------------------------------------------------------------

def apply_inter_client_missingness(labels_df, silo_ptids, availability, rng):
    """For each silo, sample p_a ~ Beta(α, β), then draw per-modality availability
    via Bernoulli(p_a). If a modality is unavailable (=0), zero it out for ALL
    patients in the silo.

    Returns avail_matrix: {silo_name: {modality_col: 0 or 1}}.
    """
    beta_params = AVAILABILITY_PRESETS[availability]
    if beta_params is None:
        # 'complete' — no inter-client drops
        return {silo: {m: 1 for m in MODAL_COLS} for silo in silo_ptids}

    alpha_a, beta_a = beta_params
    avail_matrix = {}

    for silo, ptids in silo_ptids.items():
        p_a = rng.beta(alpha_a, beta_a)
        avail = rng.binomial(1, p_a, size=len(MODAL_COLS))

        # Safety: at least 1 modality must remain per silo
        if avail.sum() == 0:
            avail[rng.integers(len(MODAL_COLS))] = 1

        avail_dict = {m: int(a) for m, a in zip(MODAL_COLS, avail)}
        avail_matrix[silo] = avail_dict

        # Zero-out unavailable modalities for every patient in this silo
        ptid_mask = labels_df['PTID'].isin(ptids)
        for m, a in avail_dict.items():
            if a == 0:
                labels_df.loc[ptid_mask, m] = 0

        print(f'  {silo}: p_a={p_a:.3f}, avail={avail_dict}')

    return avail_matrix


# ---------------------------------------------------------------------------
# Step 3 — Intra-client MCAR drops
# ---------------------------------------------------------------------------

def apply_intra_client_missingness(labels_df, silo_ptids, p_intra, rng):
    """For each patient, independently drop each currently-observed modality
    with probability p_intra. At least 1 modality is always preserved."""
    if p_intra <= 0:
        return

    total_dropped = 0
    total_eligible = 0

    for silo, ptids in silo_ptids.items():
        silo_indices = labels_df.index[labels_df['PTID'].isin(ptids)]

        for idx in silo_indices:
            observed = [m for m in MODAL_COLS if labels_df.at[idx, m] == 1]
            if len(observed) <= 1:
                continue  # Cannot drop the last modality

            total_eligible += len(observed)
            to_drop = [m for m in observed if rng.random() < p_intra]

            # Safety: never drop all — keep one random survivor
            if len(to_drop) == len(observed):
                keep = observed[rng.integers(len(observed))]
                to_drop.remove(keep)

            for m in to_drop:
                labels_df.at[idx, m] = 0
            total_dropped += len(to_drop)

    elig = max(total_eligible, 1)
    print(f'  Intra-client MCAR: dropped {total_dropped}/{total_eligible} '
          f'sample-modality pairs ({total_dropped / elig * 100:.1f}%)')


# ---------------------------------------------------------------------------
# Step 4 — Remove all-zero samples
# ---------------------------------------------------------------------------

def cleanup_empty_samples(labels_df, split_meta):
    """Remove patients whose observed mask is all-zero after simulation.

    This can happen when inter-client drops eliminate the only modalities a
    patient naturally had. Such samples carry no information and would cause
    issues during training (zero-input forward pass).

    Modifies labels_df and split_meta in-place.
    Returns the total number of removed patients.
    """
    all_zero_mask = labels_df[MODAL_COLS].sum(axis=1) == 0
    empty_ptids = set(labels_df.loc[all_zero_mask, 'PTID'])

    if not empty_ptids:
        print('  No all-zero samples found — nothing to remove.')
        return 0

    # Remove from labels_df
    labels_df.drop(labels_df.index[all_zero_mask], inplace=True)
    labels_df.reset_index(drop=True, inplace=True)

    # Remove from split_metadata PTID lists (federated + centralized)
    removed_per_silo = {}
    for silo, splits in split_meta['data']['federated'].items():
        silo_removed = 0
        for s in ('train', 'val', 'test'):
            before = len(splits[s])
            splits[s] = [p for p in splits[s] if p not in empty_ptids]
            silo_removed += before - len(splits[s])
        removed_per_silo[silo] = silo_removed

    for s in ('train', 'val', 'test'):
        split_meta['data']['centralized'][s] = [
            p for p in split_meta['data']['centralized'][s] if p not in empty_ptids
        ]

    total_removed = len(empty_ptids)
    print(f'  Removed {total_removed} all-zero samples:')
    for silo, count in removed_per_silo.items():
        if count > 0:
            print(f'    {silo}: {count} removed')

    return total_removed


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

def print_summary(labels_df, silo_ptids, avail_matrix, original_labels_df):
    """Print per-silo modality counts (before vs. after simulation)."""
    print('\n' + '=' * 65)
    print(f'{"Silo":<10} {"N":>5}  ', end='')
    for m in MODAL_COLS:
        print(f'{m:>12}', end='')
    print()
    print('-' * 65)

    for silo in sorted(silo_ptids):
        ptids = silo_ptids[silo]
        mask = labels_df['PTID'].isin(ptids)
        after = labels_df.loc[mask, MODAL_COLS].sum()
        before = original_labels_df.loc[original_labels_df['PTID'].isin(ptids), MODAL_COLS].sum()
        n = mask.sum()

        print(f'{silo:<10} {n:>5}  ', end='')
        for m in MODAL_COLS:
            print(f'{int(after[m]):>5}/{int(before[m]):<5}', end=' ')
        print()

    # Global effective modality availability
    all_fed_ptids = set().union(*silo_ptids.values())
    fed_mask = labels_df['PTID'].isin(all_fed_ptids)
    total_possible = fed_mask.sum() * len(MODAL_COLS)
    total_observed = labels_df.loc[fed_mask, MODAL_COLS].sum().sum()
    orig_observed = original_labels_df.loc[
        original_labels_df['PTID'].isin(all_fed_ptids), MODAL_COLS
    ].sum().sum()
    print('-' * 65)
    print(f'Effective modality availability:  '
          f'{int(total_observed)}/{int(total_possible)} '
          f'({total_observed / total_possible * 100:.1f}%)  '
          f'[was {int(orig_observed)}/{int(total_possible)} '
          f'({orig_observed / total_possible * 100:.1f}%) before simulation]')
    print('=' * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    paths = load_adni_paths(args.path_config)
    base_partition = args.base_partition or str(paths["partitioned_dir"] / "stratified_5silos")
    labels_path = args.labels_path or str(paths["processed_dir"] / "labels.csv")
    rng = np.random.default_rng(args.seed)

    print('=== Simulate Missingness ===')
    print(f'  Base partition : {base_partition}')
    print(f'  Availability   : {args.availability}')
    print(f'  p_intra        : {args.p_intra}')
    print(f'  Seed           : {args.seed}')

    # Load base partition and original labels
    split_meta = load_base_partition(base_partition)
    labels_df = pd.read_csv(labels_path)
    original_labels_df = labels_df.copy()  # Keep pristine copy for reporting
    silo_ptids = get_silo_ptids(split_meta)

    num_silos = len(silo_ptids)
    total_patients = sum(len(p) for p in silo_ptids.values())
    print(f'  Silos          : {num_silos}  ({total_patients} patients total)')


    # Step 2: Inter-client structural missingness
    print(f'\n[Step 2] Inter-client structural missingness (availability={args.availability})')
    avail_matrix = apply_inter_client_missingness(labels_df, silo_ptids, args.availability, rng)

    # Step 3: Intra-client MCAR drops
    print(f'\n[Step 3] Intra-client MCAR (p_intra={args.p_intra})')
    apply_intra_client_missingness(labels_df, silo_ptids, args.p_intra, rng)


    # Step 4: Remove samples that ended up with all modalities missing
    print(f'\n[Step 4] Cleanup all-zero samples')
    n_removed = cleanup_empty_samples(labels_df, split_meta)

    # Refresh silo_ptids after cleanup (some PTIDs may have been removed)
    if n_removed > 0:
        silo_ptids = get_silo_ptids(split_meta)

    # Summary
    print_summary(labels_df, silo_ptids, avail_matrix, original_labels_df)

    # --- Save outputs ---
    out_dir = build_output_dir(base_partition, args.availability, args.p_intra)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Modified labels (partition-specific, original labels.csv is untouched)
    labels_out = os.path.join(out_dir, 'labels.csv')
    labels_df.to_csv(labels_out, index=False)
    print(f'\n[SAVE] Modified labels  → {labels_out}')

    # 2) Split metadata (same splits, augmented with simulation info)
    sim_meta = copy.deepcopy(split_meta)
    sim_meta['metadata']['simulation'] = {
        'base_partition': os.path.abspath(base_partition),
        'availability': args.availability,
        'availability_beta_params': AVAILABILITY_PRESETS.get(args.availability),
        'avail_matrix': avail_matrix,
        'p_intra': args.p_intra,
        'seed': args.seed,
        'samples_removed_all_zero': n_removed,
        'generated_on': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    meta_out = os.path.join(out_dir, 'split_metadata.json')
    with open(meta_out, 'w') as f:
        json.dump(sim_meta, f, indent=4)
    print(f'[SAVE] Split metadata  → {meta_out}')


if __name__ == '__main__':
    main()
