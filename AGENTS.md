# AGENTS.md

These instructions apply to all future code changes in this repository.

## Project Goal

Guide users from zero to building AI solutions and AI agents on OCI Enterprise
AI through a structured, progressive, and practical learning path.

## Priority Order (Highest to Lowest)

1. Keep generated code as simple as possible.
2. Keep all code, prompts, comments, and messages in English.
3. When necessary, use the conda environment `oci_agents_workshop`.
4. For significant code changes, always generate tests using `pytest`.
5. Store tests in the `tests` folder and ensure they are executable from the project root.
6. After significant code changes, run the test suite with `pytest` from the project root.
7. Every Python file must start with a header containing:
   - `Author: L. Saetta`
   - `Date last modified`
   - `License: MIT`
   - `Description` with a brief summary
8. Write clear docstrings in English for public modules, classes, functions, and methods; include `Args`, `Returns`, and `Raises` sections when applicable.
9. When reusable functions are identified during development, move them into appropriate modules under `common` instead of duplicating logic across scripts.
10. After every code change, run `black` for formatting.
11. After formatting, run `pylint`.
12. Check and fix all `pylint` warnings/errors for touched files.

## Standard Workflow

1. Write or update the code, keeping it simple.
2. Use editable installation for local development: `pip install -e ".[dev]"`.
3. Extract reusable functions into appropriate modules under `common`.
4. Add or update docstrings to keep code intent, inputs, and outputs clear.
5. For significant changes, add or update `pytest` tests in `tests`.
6. Run `pytest` from the root folder.
7. Run `black`.
8. Run `pylint`.
9. Fix all `pylint` issues in touched files.
