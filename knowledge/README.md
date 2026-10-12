# Clodex Knowledge Vault

This directory is Clodex's repository-native, Markdown-first knowledge base. It is designed to grow through content discovery and versioned modules rather than hardcoded lists.

## Architecture

Read [the architecture specification](ARCHITECTURE.md) and [the extension workflow](notes/extending-the-vault.md). The example [TOML configuration](clodex-knowledge.example.toml) shows configurable vault roots, extensions, exclusions, indexing, retrieval budgets, and optional modules.

## Layout

- `notes/`: concepts, languages, libraries, APIs, and engineering references.
- `tasks/`: task summaries, verified fixes, and lessons learned.
- `experiments/`: training runs, evaluations, and dataset provenance.
- `index.md`: human-readable entry point and curated links.
- `ARCHITECTURE.md`: runtime, retrieval, module, storage, lifecycle, and safety design.
- `clodex-knowledge.example.toml`: example runtime configuration.

## Growth model

- Add a note or a directory under a configured root; discovery should make it searchable without a code change.
- Add a new file format or algorithm through a versioned Rust module interface, not by rewriting the core.
- Keep the local index rebuildable from source files.
- Add new metadata fields compatibly and version any breaking schema changes.
- Keep model checkpoints and large datasets in artifact storage, not ordinary Git history.

## Training boundary

Adding or indexing a note does **not** change neural-network weights. Retrieval can use notes at inference time; explicit training changes weights and writes checkpoints. Training examples must be reviewed for correctness, licensing, secrets, duplicates, and evaluation impact before approval.
