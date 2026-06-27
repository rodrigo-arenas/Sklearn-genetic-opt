---
name: Refactor / maintenance
about: Internal cleanup, tech debt, or dependency/test maintenance
title: '[MAINT]'
labels: 'maintenance'

---

## Summary

What should be refactored or maintained, and what's the current pain?

## Why this helps

The benefit — readability, fewer bugs, faster CI, easier contributions, etc.
Note that this should not change behavior for users (or call out the exception).

## Suggested implementation

Start by reading these files:

- `path/to/file.py`
  - What is there today and what should change.

Outline the approach and any ordering/migration concerns.

## Risk & validation

- What could break:
- How to verify nothing regresses:

```bash
pytest
```

## Notes for contributors

Anything that makes this easier to pick up.
