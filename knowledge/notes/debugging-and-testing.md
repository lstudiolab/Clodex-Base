# Debugging and Testing

tags: debugging, testing, engineering

## Summary

A reliable fix starts with a reproducible failure, a test that exposes it, and evidence that the fix resolves the failure without breaking existing behavior.

## Workflow

1. Reproduce the problem and capture the exact error and environment.
2. Reduce it to the smallest failing case.
3. Identify the likely cause using logs, tests, and code inspection.
4. Add or update a regression test before changing behavior when practical.
5. Make the smallest coherent fix.
6. Run the targeted test, then the relevant broader suite.
7. Record the root cause, fix, and verification result.

## Confidence labels

- **Verified:** the fix passed the stated tests in a recorded environment.
- **Partially verified:** some checks passed, but important checks remain.
- **Hypothesis:** plausible explanation that has not yet been tested.

Never describe an unrun test as passing. Link to the source file, test, commit, or experiment where possible.

## Related notes

- [Programming fundamentals](programming-fundamentals.md)
