## Incompleteness simulation

This folder contains scripts to simulate modality incompleteness (static **availability** and dynamic **stability**).

Supported datasets: `pamap2`, `realworldhar`, `sleepedf20` (see `simulate_incompleteness.py`).

Availability and stability levels are **`complete`**, **`medium`**, and **`low`** only (the former `high` setting is omitted to keep the experiment grid smaller).

Example:

```bash
python simulate_incompleteness.py --dataset_name pamap2 \
                                  --availability medium \
                                  --stability low
```

Or loop over grids with `run_simulation.sh`.

After simulation, build windowed tensors with `create_tensor_data.py`. Use the same `--scaling_method` as in `simulate_incompleteness.py` (default: `standard`). Missing values are filled with **zero imputation** at this step (default `--local_imp zero`); `main.py` loads the saved tensors and does not re-impute.

```bash
python create_tensor_data.py --dataset pamap2 --scaling_method standard
```
