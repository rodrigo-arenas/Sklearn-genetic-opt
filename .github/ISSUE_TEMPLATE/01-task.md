---
name: Contributor task / good first issue
about: A small, well-scoped task with implementation guidance (great for newcomers)
title: ''
labels: 'help wanted'

---

## Summary

A short description of the task and the problem it solves.

## Why this helps

Who benefits and in which workflow. Keep it concrete — for example "new users
often copy a `param_grid` and then add `PopulationConfig(warm_start_configs=[...])`".

## Suggested implementation

Start by reading these files:

- `path/to/file.py`
  - Mention the key function(s) and what they already do.
- `path/to/other_file.py`
  - Point to the place where the change should live.

Describe the approach at a high level so a contributor knows where to make the
change and why.

## Tests to add

Add focused tests in `path/to/tests/test_*.py` for:

1. the main happy path
2. an edge / failure case
3. (add more as needed)

## Validation

Run:

```bash
pytest path/to/tests/test_file.py
```

## Notes for contributors

Why this is a good task to pick up (small, well-contained, clear value).
