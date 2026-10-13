# Clodex

**Clodex** is an experimental coding-agent and language-model project by Llab. The intended implementation remains Rust and C++: Rust owns orchestration, metadata, datasets, and guarded project tools; C++ is the planned home for model kernels and training operations.

> **Current status:** the checked-in engine is only a CPU byte-level RNN reference implementation. It cannot load or fine-tune a pretrained transformer checkpoint. The target workflow is to start from a licensed pretrained coding model near 3B parameters, train its actual weights (not LoRA), and research carefully initialized expansion toward 7B. That transformer loader, full-weight trainer, and expansion path are not implemented yet. See [the pretrained training plan](docs/training.md).

## Quick start

Requirements: stable Rust, a C++17 compiler, and CMake.

```sh
cargo test
cmake -S engine -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

To test the current reference RNN only, put a plain UTF-8 corpus at `data/train.txt` (not committed), then run:

```sh
cargo run --release -- inspect-corpus
cargo run --release -- train --steps 100 --publish-release
cargo run --release -- status
cargo run --release -- generate --prompt "fn main" --max-new-tokens 128
```

Training resumes from `artifacts/clodex.latest.ckpt` when a compatible local checkpoint exists. If it is missing, Clodex attempts to fetch the `clodex-latest` GitHub Release asset through an authenticated `gh` CLI session. Add `--publish-release` to upload the completed checkpoint and run manifest after a session; use short sessions if you need to sync frequently. Use `--fresh` only when intentionally resetting weights. Clodex checks the saved corpus SHA-256 and refuses mismatched data unless you explicitly pass `--allow-corpus-change`. Periodic checkpoints retain the optimizer state and update count. See [docs/training.md](docs/training.md).

## Coding-agent tools

The guarded local project tools are available as:

```sh
cargo run -- agent list --workspace /path/to/project
cargo run -- agent read --workspace /path/to/project --path src/main.rs
cargo run -- agent write --workspace /path/to/project --path src/main.rs --content-file /tmp/proposed.rs
cargo run -- agent test --workspace /path/to/project --command "cargo test"
cargo run --release -- agent run --workspace /path/to/project --task "Find and fix the parser bug" --iterations 6
```

Writes preview the size and first lines of the proposed change and require explicit confirmation. Paths cannot traverse outside the workspace. Only a small test-command allowlist is enabled; arbitrary shell execution is not offered. The `agent run` command connects generation to a structured tool loop (list/read/search/write/test/finish). Invalid model outputs are rejected and retried; writes require explicit confirmation and tests use the allowlist. This is an end-to-end experimental agent architecture, but the current small model is not yet capable enough to plan and reliably complete coding tasks autonomously.

## Pretrained-model direction and compute

Do not use the current RNN checkpoint as if it were compatible with a pretrained transformer. Keep the original base checkpoint immutable and outside Git. Pin its model revision, tokenizer, license, dataset fingerprint, architecture, and hashes in run metadata. The planned trainer must prove it loaded the intended pretrained weights, report real training progress, save optimizer state for resuming, and evaluate before/after behavior.

A 15 GB T4 is severely constrained for full-weight training at 3B and is not sufficient for ordinary 7B full training with standard optimizer state. Quantization, activation checkpointing, gradient accumulation, and CPU offload may help but need measured validation; quantized weights alone do not make full training fit. Free compute can be interrupted, so checkpoint/resume is mandatory. Model expansion must use a documented initialization method and is not expected to improve intelligence automatically.

See [docs/architecture.md](docs/architecture.md) for component boundaries, safety constraints, and the next architecture milestones.
