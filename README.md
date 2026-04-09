# DeCUR Training Engine

This repo now includes a top-level multimodal DeCUR training engine at
`src/train.py`. It covers the four training-orchestration responsibilities:

1. paired SAR and optical data ingestion,
2. forward/backward execution with AMP and gradient accumulation,
3. validation-time representation tracking,
4. checkpointing, scheduler updates, and logging.

**Repo Structure**

```text
.
|-- src/
|   |-- data_loader/
|   |-- models/decur/
|   |-- Multi_modal_src/DeCURLoss.py
|   |-- eval.py
|   `-- train.py
|-- scripts/
`-- tests/
```

**Quickstart (PowerShell)**

1. Create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

3. Launch DeCUR pretraining:

```powershell
python src/train.py `
  --combined-root "datasets/FilteredSouthAsia/Combined" `
  --epochs 20 `
  --batch-size 4 `
  --learning-rate 1e-4 `
  --validation-split 0.1 `
  --patch-size 256 `
  --patch-overlap 0.2 `
  --amp `
  --output-dir "runs/decur_pretrain"
```

**What The Engine Does**

- Loads strict SAR/optical pairs from the combined manifest via `FusedDataset`.
- Optionally expands training samples with `PatchDataset`.
- Builds two augmented views per modality for the DeCUR objective.
- Tracks train loss, validation loss, cross-modal cosine alignment, view consistency, learning rate, and embedding norms.
- Writes checkpoints to `runs/.../checkpoints/{last,best}.pt`.
- Writes TensorBoard logs to `runs/.../logs` when TensorBoard is installed.

**Data Loader**

See `src/data_loader/README.md` for dataset details and patching behavior.
