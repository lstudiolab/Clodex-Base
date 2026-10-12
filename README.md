# Clodex

**Clodex** is an experimental coding-agent and language-model project by Llab. The implementation is Rust and C++ only: Rust owns the CLI, orchestration, metadata, and guarded project tools; C++ implements the trainable CPU reference model.

> **Current status:** a runnable training/continuation foundation—not a pretrained coding model. The initial neural engine is a compact byte-level recurrent language model. It is not yet a transformer, does not include a GPU backend, and will not behave like a modern coding model without substantial architecture, data, and training work.

## Quick start

Requirements: stable Rust, a C++17 compiler, and CMake.

```sh
cargo test
cmake -S engine -B build
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Put a plain UTF-8 training corpus at `data/train.txt` (not committed), then run:

```sh
cargo run --release -- inspect-corpus
cargo run --release -- train --steps 100 --publish-release
cargo run --release -- status
cargo run --release -- generate --prompt "fn main" --max-new-tokens 128
```

Training resumes from `artifacts/clodex.latest.ckpt` when a compatible local checkpoint exists. If it is missing, Clodex attempts to fetch the `clodex-latest` GitHub Release asset through an authenticated `gh` CLI session. Add `--publish-release` to upload the completed checkpoint and run manifest after a session; use short sessions if you need to sync frequently. Use `--fresh` only when intentionally resetting weights. Periodic checkpoints retain the optimizer state and update count. See [docs/training.md](docs/training.md).

## Coding-agent tools

The guarded local project tools are available as:

```sh
cargo run -- agent list --workspace /path/to/project
cargo run -- agent read --workspace /path/to/project --path src/main.rs
cargo run -- agent write --workspace /path/to/project --path src/main.rs --content-file /tmp/proposed.rs
cargo run -- agent test --workspace /path/to/project --command "cargo test"
```

Writes preview the size and first lines of the proposed change and require explicit confirmation. Paths cannot traverse outside the workspace. Only a small test-command allowlist is enabled; arbitrary shell execution is not offered. This tool layer is a safe foundation for an agent, but the current small model is not yet capable enough to plan and reliably complete coding tasks autonomously.

## Weights and compute

Source belongs in Git; trained checkpoints and private data do not. Keep artifacts out of Git history and publish useful checkpoints as release assets. Free compute services can terminate sessions; Clodex can resume on the next run from the last saved checkpoint but cannot prevent a provider from reclaiming its machine. The current C++ training engine uses CPU and is intended for tests and small experiments, not large-scale pretraining.

See [docs/architecture.md](docs/architecture.md) for component boundaries, safety constraints, and the next architecture milestones.
