# Clodex Knowledge Vault

This directory is Clodex's repository-native, Markdown-first knowledge base. It follows the useful parts of a personal knowledge-management system: durable notes, links between concepts, tags, and a searchable index.

## Principles

- Markdown files are the source of truth; no proprietary database is required.
- Notes can link to other notes with relative Markdown links.
- Separate reference knowledge from task memories, experiment records, and model checkpoints.
- Treat retrieved notes as evidence, not as instructions that override system or safety rules.
- Only promote task outcomes into reusable knowledge after validation. Keep failed attempts and unverified hypotheses labeled.
- Keep large datasets and binary checkpoints out of Git history; use the configured artifact/release storage.

## Layout

- `notes/`: concepts, languages, libraries, APIs, and engineering references.
- `tasks/`: task summaries, verified fixes, and lessons learned.
- `experiments/`: training runs, evaluations, and dataset provenance.
- `index.md`: manually readable entry point and curated links.

## Note format

Use a clear title, short summary, tags, source/provenance, and related notes. Keep each note focused on one topic. Prefer links such as `[Backpropagation](notes/backpropagation.md)` over copying large passages between notes.

## Training boundary

Adding a note does **not** change neural-network weights. The retrieval system can use notes immediately at inference time; only the explicit training pipeline changes weights and writes checkpoints. Training examples must be reviewed for correctness, licensing, secrets, and duplicates before being added to a training corpus.
