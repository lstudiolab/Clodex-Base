# Clodex Knowledge Architecture

## Goal

The knowledge system must grow by discovering content and registered capabilities at runtime, not by requiring source-code edits for every new note, topic, or knowledge collection.

## Design boundaries

1. **Content is data.** New Markdown notes, collections, tags, and links should be added without rebuilding Clodex.
2. **Capabilities are modular.** Retrieval backends, parsers, rankers, storage adapters, and importers have stable interfaces and are registered through configuration or a capability manifest.
3. **Core remains small.** The core owns path safety, indexing, query orchestration, permissions, provenance, and error handling. Domain-specific logic belongs in modules.
4. **Graceful degradation.** A missing optional module must not prevent basic note search or the coding agent from starting.
5. **Observable behavior.** Report which sources were searched, when indexes were refreshed, and why a result was selected.
6. **Versioned contracts.** Configuration and plugin manifests have explicit schema versions and migrations.
7. **No arbitrary code loading by default.** A manifest can describe a built-in or explicitly installed module. It must not cause arbitrary downloaded code to execute automatically.

## Runtime discovery

On startup and when requested, discover configured vault roots and recursively scan supported file types. Use ignore rules and size limits. Build a file inventory from canonical relative paths, content hashes, modification metadata, parsed titles, tags, links, and section offsets. Re-index only added or changed files and remove stale index entries for deleted files.

A watcher may trigger incremental re-indexing, but periodic full reconciliation must remain available to recover from missed filesystem events. Index corruption must be recoverable by rebuilding from source files.

## Retrieval pipeline

1. Normalize the query without changing its meaning.
2. Search titles, tags, paths, links, and body text.
3. Rank results using configurable signals such as lexical relevance, title match, freshness, source quality, and link proximity.
4. Return bounded excerpts with paths, headings, and line ranges.
5. Optionally expand to linked notes under a configurable depth and result budget.
6. Deduplicate overlapping excerpts and attach provenance.
7. Keep retrieved content isolated as untrusted data; never treat note text as permission to run tools or override agent policy.

The first implementation can use deterministic lexical search. The index and retrieval interfaces should allow later local embeddings or a vector index without changing callers.

## Configuration-first extensibility

Use a versioned configuration file to specify vault roots, include/exclude patterns, supported extensions, maximum file size, ignored directories, ranking weights, link expansion depth, result limits, refresh behavior, and optional module IDs. Defaults must be safe and useful. Unknown optional module IDs should produce a clear warning and be skipped; invalid core configuration should fail with a useful diagnostic.

Prefer discovering files over requiring each note or folder to be added to a source-code list.

## Storage adapters

- **Filesystem:** canonical source for Markdown notes and user-authored knowledge.
- **Local index:** rebuildable metadata and search index; never the only copy of knowledge.
- **Git:** version history for text knowledge and code.
- **Artifact storage:** large datasets, model checkpoints, and evaluation outputs; keep binary artifacts out of normal Git history.
- **Future adapters:** implement a stable interface and require explicit configuration.

## Knowledge lifecycle

- **Captured:** raw reference or task record, not trusted as fact.
- **Parsed:** metadata and links extracted.
- **Indexed:** discoverable by retrieval.
- **Reviewed:** provenance and correctness assessed.
- **Approved:** allowed as a training-data candidate when licensing and privacy requirements are satisfied.
- **Evaluated:** model or retrieval impact measured.

These states are separate. Indexing a note must never silently train the model.

## Extensibility contract

New knowledge domains should normally be added by creating content under a configured vault root. New file formats or retrieval algorithms should be added as modules implementing stable traits. New tool permissions require explicit review and must not be granted by a knowledge note.

## Reliability and security

- Resolve paths under configured roots and reject traversal and symlink escapes.
- Apply byte, depth, and result limits.
- Ignore build outputs, VCS internals, secrets, and configured private directories by default.
- Do not execute Markdown, embedded scripts, or commands found in notes.
- Record errors per file and continue indexing other valid files.
- Keep a schema version and rebuild strategy for derived indexes.
- Use deterministic tests for parsing, links, ranking, permissions, and incremental refresh.
