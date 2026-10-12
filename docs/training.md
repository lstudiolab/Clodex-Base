# Training and resuming

1. Install a stable Rust toolchain, C++17 compiler, and CMake.
2. Put a UTF-8 corpus in `data/train.txt` (not committed; that path is ignored).
3. Build/test with `cargo test`.
4. Train with `cargo run --release -- train --corpus data/train.txt --steps 1000`.
5. Stop a run only after the periodic checkpoint has been written when possible. The default save cadence is 100 updates.
6. Resume the same corpus and checkpoint using the same command. Resume is the default; pass `--fresh` only to intentionally start new weights. Keep the same hidden size when resuming.
7. Sample with `cargo run --release -- generate --prompt "fn main" --max-new-tokens 128`.

The initial engine trains on raw UTF-8 bytes. The loss/task quality will be limited by corpus quality, model capacity, training time, and the CPU reference engine. Do not interpret a successful build or a low training loss as proof of coding competence; test generated code separately.

## Artifact storage

Do not commit checkpoints to the repository. A maintainer may upload a checkpoint to a GitHub Release after creating a versioned tag/release, for example using the GitHub CLI from a trusted machine:

```sh
gh release create clodex-checkpoint-v1 artifacts/clodex.latest.ckpt artifacts/clodex.latest.run.json --title "Clodex checkpoint v1" --notes "Training checkpoint and run manifest"
```

The upload command is a manual publish step, not part of ordinary training. Do not publish private corpora, secrets, or weights derived from data you are not authorized to redistribute.
