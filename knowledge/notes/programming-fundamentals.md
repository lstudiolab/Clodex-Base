# Programming Fundamentals

tags: programming, fundamentals

## Summary

Programs transform inputs into outputs through explicit operations and state. Good implementations make their assumptions, interfaces, and failure modes clear.

## Practical rules

- Define expected inputs, outputs, and error behavior before implementation.
- Prefer small functions with one responsibility and explicit interfaces.
- Validate external input at boundaries.
- Handle errors deliberately; do not silently discard failures.
- Add tests for normal behavior, edge cases, and invalid inputs.
- Measure performance before optimizing, then add a regression test for important fixes.

## Related notes

- [Debugging and testing](debugging-and-testing.md)
