# FLUX: Multimodal Federated Learning for Dual-Axis Missingness

This repository contains the official implementation of **FLUX** and the federated learning benchmark for multimodal sensor data under modality incompleteness.

## Overview

Real-world multimodal federated learning faces **modality incompleteness** — sensors may be unavailable (static) or unstable (dynamic) across clients. FLUX uses **per-modality confidence heads** and **confidence tempering** (temperature scaling from predicted log-variance) so fusion stays robust when modalities are partially missing.

This codebase supports:
- **FLUX** (our method)
- **7 baselines**: FedAvg, FedProx, FedPer, FedRoD, MOON, PmcmFL, PEPSY
- **Three public sensor datasets** for the main FL entry (`main.py`): PAMAP2, RealWorld HAR, Sleep-EDF-20
- **ADNI pipeline**: A separate clinical multimodal path under `adni_pipeline/`
- Configurable modality incompleteness simulation (availability and stability)

Sleep-EDF-20 preprocessing in this repo expects a merged CSV produced from the Sleep-EDF-Expanded pipeline; see [`data_preprocessor/preprocess_sleepedf20/readme_sleepedf20.md`](data_preprocessor/preprocess_sleepedf20/readme_sleepedf20.md). Configure a `sleepedfx` entry in `dataset_paths.json` for those intermediate paths (see [`data_preprocessor/dataset_paths_example.json`](data_preprocessor/dataset_paths_example.json)).

## Repository Structure

```
.
├── main.py                     # Main entry point for FL experiments
├── arg_parser.py               # CLI argument definitions
├── configs.py                  # Dataset configs and tuned hyperparameters
├── flux_env_minimal.yaml       # Minimal conda environment
├── flux_env_exact.yaml         # Conda environment for exact reproduction
│
├── models/
│   ├── common/                 # Shared components (encoders, fusion, heads)
│   ├── method_specific/        # Method-specific model architectures (incl. flux.py)
│   └── model_utils.py          # Model factory (get_model)
│
├── ml_federated/
│   ├── client/                 # Per-method FL client implementations
│   └── server/                 # Per-method FL server implementations
│
├── data_preprocessor/          # Raw data -> processed CSV pipelines (per dataset)
├── data_simulation/            # Simulate modality incompleteness (availability/stability)
├── data_loader/                # Load preprocessed tensor data for FL training
│
├── utils/                      # Utilities (seeding, wandb helpers, pretty printing)
├── run/                        # Shell scripts
│
└── adni_pipeline/              # Self-contained ADNI experiment pipeline
    ├── configs/                # ADNI-specific configuration
    ├── data/                   # ADNI data loader
    ├── fl/                     # ADNI FL servers and clients
    ├── models/                 # ADNI model architectures
    └── run/                    # ADNI batch run scripts
```

## Setup

### 1. Environment

Both environment files target **Linux with NVIDIA GPUs (CUDA 11.8)** via `pytorch-cuda`.

**Minimal:** [`flux_env_minimal.yaml`](flux_env_minimal.yaml) — packages required to run the code in this repo.

```bash
conda env create -f flux_env_minimal.yaml
conda activate flux
```

**Exact reproduction:** [`flux_env_exact.yaml`](flux_env_exact.yaml) — fully pinned versions from the development environment. Use this if you need to match exact dependency versions.

```bash
conda env create -f flux_env_exact.yaml
conda activate flux
```

### 2. Data Preprocessing

Use the preprocessing scripts and READMEs for the datasets you run:

- [`data_preprocessor/preprocess_pamap2/readme_pamap2.md`](data_preprocessor/preprocess_pamap2/readme_pamap2.md)
- [`data_preprocessor/preprocess_realworldhar/readme_realworldhar.md`](data_preprocessor/preprocess_realworldhar/readme_realworldhar.md)
- [`data_preprocessor/preprocess_sleepedf20/readme_sleepedf20.md`](data_preprocessor/preprocess_sleepedf20/readme_sleepedf20.md)

### 3. Configure Data Paths

Copy [`data_preprocessor/dataset_paths_example.json`](data_preprocessor/dataset_paths_example.json) to `data_preprocessor/dataset_paths.json` and update the paths to your local dataset directories. Include `sleepedfx` paths if you follow the Sleep-EDF-20 README that builds from expanded Sleep-EDF data.

For Weights & Biases logging, copy [`wandb_config_example.json`](wandb_config_example.json) to `wandb_config.json` and set your entity and project names (`wearable_project` for `main.py`, `adni_project` for the ADNI pipeline). CLI flags `--project`, `--wandb_entity`, and `--wandb_project` override these defaults.

### 4. Simulate Modality Incompleteness

After preprocessing, simulate sensor availability and stability:

```bash
cd data_simulation

python simulate_incompleteness.py --dataset_name pamap2 --availability medium --stability low

bash run_simulation.sh
```

Then create the windowed tensor data (missing sensor values are **zero-imputed** here; default `--local_imp zero`):

```bash
python create_tensor_data.py --dataset pamap2 --scaling_method standard
```

`main.py` loads the resulting `*_tensors.pt` files; presence masks are stored alongside inputs for training.

## Running Experiments

Local training for `main.py` uses **SGD** (momentum 0.9). The ADNI pipeline (`adni_pipeline/run_adni.py`) uses **Adam** (fixed in ADNI client code).

### Single Run

```bash
python main.py --dataset pamap2 \
               --method flux \
               --availability medium \
               --stability low \
               --seed 42 \
               --gpu 0
```

### Available Methods

| Method   | `--method` flag |
|----------|----------------|
| FLUX     | `flux`         |
| FedAvg   | `fedavg`       |
| FedProx  | `fedprox`      |
| FedPer   | `fedper`       |
| FedRoD   | `fedrod`       |
| MOON     | `moon`         |
| PmcmFL   | `pmcmfl`       |
| PEPSY    | `pepsy`        |


### Available Datasets (`main.py`)

| Dataset         | `--dataset` flag |
|-----------------|-----------------|
| PAMAP2          | `pamap2`        |
| RealWorld HAR   | `realworldhar`  |
| Sleep-EDF-20    | `sleepedf20`    |

### Batch Experiments

One launcher per method under `run/` (e.g. `flux.sh`, `fedavg.sh`). Each script `cd`s to the repo root; `--note` defaults to `<method>_<dataset>` for each dataset block (override with `NOTE=mytag ./run/fedavg.sh`). Edit GPU visibility and loops inside the script to match your sweep.

```bash
bash run/flux.sh
bash run/fedavg.sh
bash run/run_all.sh   # all method launchers in sequence (excludes run_test.sh)
```

Learning rates and method-specific knobs for each dataset are in [`configs.py`](configs.py).

### Logging with Weights & Biases

Defaults come from `wandb_config.json` (see setup above). Override per run if needed:

```bash
python main.py --dataset pamap2 --method flux --note "my_experiment"
# Weights & Biases is enabled by default; use --no-wandb to turn it off.
```

## ADNI Pipeline

ADNI requires credentialed data access. Follow the preprocessing and local path
configuration instructions in
[`adni_pipeline/dataset_prep/ADNI/README_ADNI.md`](adni_pipeline/dataset_prep/ADNI/README_ADNI.md)
(copy `adni_paths_example.json` to `adni_paths.json`, same idea as
`dataset_paths.json` for wearable datasets). `--adni_root` can override the
configured dataset root in `run_adni.py`.

```bash
python -m adni_pipeline.run_adni --method flux --gpu 0 --seed 42

bash adni_pipeline/run/run_all_adni.sh
```

## Citation

Coming soon.
