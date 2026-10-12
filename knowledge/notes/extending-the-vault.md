# Extending the Vault Without Source Changes

tags: knowledge, extensibility, workflow

## Add a topic

Create a Markdown file in the appropriate vault folder. Include a title, summary, tags, sources, and links to related notes. The runtime should discover it during incremental refresh or the next reconciliation; no source-code change should be needed.

## Add a collection

Add a directory under a configured vault root. The runtime should discover its files recursively, subject to configured include/exclude patterns and safety limits. Do not hardcode every collection name into the application.

## Add a new parser or ranking strategy

Implement the relevant Rust interface as a module, add tests, register its stable module ID, and enable it in configuration. Keep lexical retrieval as a fallback. Do not let note content install or enable code modules automatically.

## Add training examples

Keep candidate examples separate from reference notes. Record provenance and licensing, remove secrets and personal data, deduplicate, validate examples, and run evaluation before accepting them. An indexed note is not automatically training data.

## Add model capacity

A larger knowledge vault does not make the neural network itself larger. To increase model capacity, the architecture and checkpoint format must support the new model dimensions, and training must run on adequate compute. Treat that as a versioned model migration with compatibility checks rather than silently loading mismatched weights.
