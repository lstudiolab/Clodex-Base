# Clodex architecture

## Initial architecture

- **Rust control plane**: command-line interface, input validation, corpus hashing, run metadata, agent workspace boundaries, and orchestration.
- **C++ neural engine**: a compact byte-level recurrent language model, softmax language-model loss, backpropagation through a truncated sequence, Adam updates, sampling, and binary checkpoint persistence.
- **Tokenizer contract**: UTF-8 byte IDs 0–255, BOS 256, EOS 257. This avoids an untrained vocabulary-merging stage and keeps Unicode representable, at the cost of long token sequences and weak semantic efficiency.
- **Resumable training**: checkpoints save model parameters, optimizer moments, architecture dimensions, and update count. Re-running `clodex train` uses a compatible checkpoint by default. The run manifest records the corpus hash so changed data is visible.
- **Coding-agent loop**: model output is parsed as a typed JSON tool action, then routed to list/read/search/write/test/finish tools. Malformed output is rejected and retried. File writes show a preview and require user confirmation; test commands use a strict allowlist. Arbitrary shell access, network access, credential access, and unrestricted destructive commands are intentionally not enabled.

## Training and storage

The checked-in engine is a real but deliberately small **recurrent language-model baseline**, not a transformer and not an already-trained intelligent model. Its byte-tokenizer and checkpoint format are incompatible with pretrained transformer weights. Do not use its checkpoints for the new pretrained-model path.

Use a clean plain-text corpus with appropriate rights to train on. Keep corpora, logs, and checkpoints out of normal Git history. The default checkpoint path is `artifacts/clodex.latest.ckpt`, which is intended to be ignored by Git. Publish checkpoint artifacts through a GitHub Release (or a compatible large-file/artifact store) once they are worth preserving. GitHub Actions can compile/test source, but should not run expensive training by default.

For continuation after a free GPU/CPU session ends, the next invocation resumes a saved checkpoint. A process cannot keep a provider's free compute allocation alive after that provider terminates it. The current C++ backend is CPU-only and suitable only for correctness experiments and small datasets. It is not the intended trainer for a pretrained 3B/7B transformer. No GPU support or sustained free GPU availability is implied.

## Pretrained model and expansion milestones

1. Select a coding-focused pretrained model near 3B parameters; review its license, model card, tokenizer, architecture, and exact revision before downloading weights.
2. Implement a transformer checkpoint loader and inference smoke test. Keep the original checkpoint immutable and verify tensor shapes and parameter count so random initialization cannot be mistaken for pretrained weights.
3. Implement **full-weight continued training** (not LoRA): gradients for the intended base parameters, optimizer/scheduler state, gradient accumulation, memory-aware activation checkpointing, real progress reporting, and resumable checkpoints.
4. Add dataset provenance, deduplication, validation splits, reproducible fingerprints, and held-out code/compile/test evaluations.
5. Only after 3B loading, training, resume, and evaluation work, prototype a documented architecture-expansion strategy toward 7B. Initialize added weights deliberately and test for capability regressions; adding parameters alone does not guarantee better results.
6. Benchmark memory and throughput on available hardware before attempting long runs. A 15 GB T4 does not fit ordinary full-weight 7B training with standard optimizer states; even 3B full-weight training requires careful memory planning.
7. Keep the existing RNN as a small CPU reference/test engine until the pretrained transformer path has its own tests. Credentials should be provided via environment or platform secret management and never checked into the repository.

## Current limitations

No pretrained transformer loader, full-weight transformer trainer, CUDA tensor backend, or architecture-expansion implementation exists in this repository yet. The docs describe the target method, not completed capabilities.

## Security boundary

Agent-generated text and tool output are untrusted. File tools must never escape the selected workspace. New tools should be least-privileged, tested independently, and require confirmation for external side effects. Do not store secrets or user-private datasets in the repository.
