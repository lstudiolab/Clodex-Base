# Clodex Human Language Policy

## Decision

Clodex is English-only for human-language interaction in its initial scope. English is the target language for user instructions, planning, reasoning traces exposed to the user, explanations, documentation, and human-language training/evaluation data.

## Programming language coverage

English-only does not mean single-programming-language. Clodex should learn and work with multiple programming languages, including Rust, C++, Python, JavaScript, TypeScript, SQL, and others as data, tools, and evaluations permit.

## Input handling

- Optimize conversational understanding and response quality for English.
- Do not intentionally train or advertise multilingual conversation support as an initial capability.
- Do not reject source code, identifiers, database values, logs, filenames, or technical artifacts merely because they contain non-English text.
- Treat non-English text found inside a repository as project data; preserve it accurately when editing code or documentation.
- If a request is not understood because it is outside the English-only scope, ask the user to restate it in English rather than silently guessing.

## Training and evaluation

- Prioritize high-quality English technical text and programming data.
- Evaluate English instruction following, code generation, repository-level edits, debugging, tool use, and test verification.
- Track language-scope changes as an explicit product decision; do not expand multilingual training accidentally through unreviewed data ingestion.

## Scope boundary

This policy defines the intended product and training focus, not a claim that a tokenizer or neural network can perfectly reject or forget every other language. Multilingual competence may still emerge from code and general training data. The requirement is to optimize Clodex for English and avoid spending scarce resources on multilingual conversation as a goal.
