# Training Experiment Ledger

Create one file per training run. Keep model checkpoints in the configured artifact storage rather than committing large binary files into normal Git history.

Each run record should capture:

- Run ID and timestamp
- Source commit
- Dataset identifiers, hashes, provenance, and license
- Model architecture and hyperparameters
- Starting checkpoint and checkpoint hash
- Compute device and runtime constraints
- Steps completed and training/validation metrics
- Evaluation tests and observed results
- Output checkpoint location and hash
- Whether the run is reproducible and safe to resume

Do not claim that a run happened unless a real training process produced its logs and checkpoint. Resuming requires compatible model configuration and a verified checkpoint.
