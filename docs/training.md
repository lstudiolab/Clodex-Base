# Pretrained model training and expansion

## Goal

Clodex's intended path is to start from a capable pretrained coding model near 3B parameters, improve the actual model weights with continued training, and investigate architecture expansion toward 7B and later sizes. LoRA is not the intended core training method.

**This goal is not implemented by the current engine.** The current C++ engine is a small byte-level RNN with a CPU-only backend. Its checkpoints are not compatible with Qwen or other transformer checkpoints. Do not use `clodex train` to claim that a pretrained transformer has been fine-tuned or expanded.

## Staged plan

1. **Select and pin a base model.** Review the model's license, permitted uses, model-card requirements, tokenizer, architecture configuration, and checkpoint format. Record the exact model revision and file hashes. Do not commit model weights or credentials.
2. **Validate inference first.** Load the original pretrained model with a compatible runtime and verify a fixed set of coding prompts before any training. Keep the original checkpoint unchanged.
3. **Prepare data.** Build a deduplicated, quality-filtered dataset with source/license records. Separate training and held-out evaluation examples. Never place credentials, private code, or unauthorized data in the corpus.
4. **Implement full-weight continued training.** Every intended base-model parameter remains trainable; this is not LoRA. Implement and test gradient accumulation, optimizer-state handling, gradient checkpointing, mixed precision where supported, resumable checkpoints, and deterministic run metadata. Verify that gradients and parameters actually change.
5. **Evaluate.** Compare the base and trained checkpoints on the same held-out coding, reasoning, and compile/test tasks. Track loss, failures, throughput, peak memory, and data provenance.
6. **Explore 3B-to-7B expansion separately.** Increasing layer count or width changes the architecture and tensor shapes. Use a documented model-expansion/initialization method, preserve a copy of the 3B model, then continue training the expanded model. Added parameters do not automatically add intelligence; evaluate for capability gains and regressions.
7. **Scale only when measured resources allow.** A free Colab T4 with 15 GB VRAM is not enough for ordinary full-parameter training of a 7B model with standard optimizer states. Even 3B full-weight training is memory-constrained. Quantized weights alone do not solve all training-memory needs. Benchmark memory and throughput before committing to a long run.

## Current repository commands (reference engine only)

The existing byte-RNN smoke/reference engine can still be built and tested:

```sh
cargo test --release
cmake -S engine -B build
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Its small-corpus training command is useful only for validating the existing RNN pipeline:

```sh
cargo run --release -- train --corpus data/train.txt --steps 100
```

It does **not** load a pretrained transformer, train 3B parameters, or expand 3B to 7B.

## Artifact and resume policy

Keep base-model files, datasets, and checkpoints out of Git. Store model ID/revision, tokenizer ID/revision, dataset fingerprint, optimizer/configuration, precision, architecture, training steps, and checkpoint hashes in run metadata. A resume must reject incompatible architecture/tokenizer changes and unexpected dataset changes unless the operator explicitly starts a new run or authorizes the change. Never publish private corpora, secrets, or weights unless redistribution is permitted by the model and dataset licenses.

## Required acceptance checks for the new trainer

- Loads the pinned pretrained checkpoint and tokenizer without silently substituting random weights.
- Reports model architecture and parameter count, and verifies expected checkpoint tensor shapes.
- Reports real update progress, loss, elapsed time, learning rate, and checkpoint-save success.
- Saves enough optimizer/scheduler/RNG state to resume the same run.
- Can resume after interruption and validates the checkpoint before continuing.
- Includes a short end-to-end training test that proves trainable weights changed.
- Includes a separate inference smoke test and held-out evaluation.
- Refuses unsupported model architectures or memory settings with a clear error.
