# Clodex architecture

## Initial architecture

- **Rust control plane**: command-line interface, input validation, corpus hashing, run metadata, agent workspace boundaries, and orchestration.
- **C++ neural engine**: a compact byte-level recurrent language model, softmax language-model loss, backpropagation through a truncated sequence, Adam updates, sampling, and binary checkpoint persistence.
- **Tokenizer contract**: UTF-8 byte IDs 0–255, BOS 256, EOS 257. This avoids an untrained vocabulary-merging stage and keeps Unicode representable, at the cost of long token sequences and weak semantic efficiency.
- **Resumable training**: checkpoints save model parameters, optimizer moments, architecture dimensions, and update count. Re-running `clodex train` uses a compatible checkpoint by default. The run manifest records the corpus hash so changed data is visible.
- **Coding-agent loop**: model output is parsed as a typed JSON tool action, then routed to list/read/search/write/test/finish tools. Malformed output is rejected and retried. File writes show a preview and require user confirmation; test commands use a strict allowlist. Arbitrary shell access, network access, credential access, and unrestricted destructive commands are intentionally not enabled.

## Training and storage

The initial engine is a real but deliberately small **recurrent language-model baseline**, not a transformer and not an already-trained intelligent model. A training engine and source code do not constitute learned knowledge: useful behavior depends on quality/quantity of data and actual optimization runs.

Use a clean plain-text corpus with appropriate rights to train on. Keep corpora, logs, and checkpoints out of normal Git history. The default checkpoint path is `artifacts/clodex.latest.ckpt`, which is intended to be ignored by Git. Publish checkpoint artifacts through a GitHub Release (or a compatible large-file/artifact store) once they are worth preserving. GitHub Actions can compile/test source, but should not run expensive training by default.

For continuation after a free GPU/CPU session ends, the next invocation resumes a saved checkpoint. A process cannot keep a provider's free compute allocation alive after that provider terminates it. This first C++ backend is CPU-only; it is suitable for correctness experiments and small datasets, not large-scale pretraining. No GPU support or sustained free GPU availability is implied.

## Next architectural milestones

1. Stabilize tests and checkpoint-format/version compatibility.
2. Add a SIMD/CUDA-capable tensor backend while retaining a deterministic CPU reference path.
3. Replace/augment the RNN baseline with a decoder-only transformer (multi-head causal attention, normalization, feed-forward blocks, AdamW, gradient clipping, and loss/evaluation reporting).
4. Add subword tokenizer training, packed datasets, validation splits, and reproducible data manifests.
5. Add explicit tool approval policies, transactional edits/diffs, patch-based writes, per-command timeouts, and full audit logs.
6. Add release artifact validation and automated checkpoint download/resume. Credentials should be provided via environment or platform secret management and never checked into the repository.

## Security boundary

Agent-generated text and tool output are untrusted. File tools must never escape the selected workspace. New tools should be least-privileged, tested independently, and require confirmation for external side effects. Do not store secrets or user-private datasets in the repository.
