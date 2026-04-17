# AGENTS.md

These instructions apply to all future code changes in this repository.

## Priority Order (Highest to Lowest)

1. Keep generated code as simple as possible.
2. Keep all code, prompts, comments, and messages in English.
3. When necessary, use the conda environment `custom_rag_agent2026`.
4. For significant code changes, always generate tests using `pytest`.
5. Store tests in the `tests` folder and ensure they are executable from the project root.
6. After significant code changes, run the test suite with `pytest` from the project root.
7. Every Python file must start with a header containing:
   - `Author: L. Saetta`
   - `Date last modified`
   - `License: MIT`
   - `Description` with a brief summary
8. After every code change, run `black` for formatting.
9. After formatting, run `pylint`.
10. Check and fix all `pylint` warnings/errors for touched files.

## Standard Workflow

1. Write or update the code, keeping it simple.
2. For significant changes, add or update `pytest` tests in `tests`.
3. Run `pytest` from the root folder.
4. Run `black`.
5. Run `pylint`.
6. Fix all `pylint` issues in touched files.
