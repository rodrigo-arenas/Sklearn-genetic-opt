# sklearn-genetic-opt — Claude Code Instructions

## Documentation Migration: VitePress + GitHub Pages

The project is migrating from Sphinx/Read the Docs to **VitePress** (static site) deployed to **GitHub Pages**.

Source docs live in `docs-vitepress/`. Legacy Sphinx docs remain in `docs/` (with a deprecation banner pointing to the new site).

### Versioning Strategy

We maintain two versions:
- **stable** — `docs-vitepress/versions/0.13/` (released, maps to `v0.13.*` tags)
- **latest** — `docs-vitepress/versions/latest/` (in-development, tracks the `0.13.0dev` branch)

Adding a new version means:
1. Copy `latest/` → `versions/X.Y/`
2. Add the version to `docs-vitepress/.vitepress/config.ts` `versions` array
3. Tag the release

**Never edit pages that live under `versions/X.Y/` after that version is released.** Only `latest/` and the current active stable are changed.

### Site Structure

```
docs-vitepress/
  .vitepress/
    config.ts          # VitePress config, nav, sidebar, versions
    theme/
      index.ts         # Theme entry
      custom.css       # Custom styles
  public/              # Static assets (images, logo)
  versions/
    0.13/              # Stable version
      index.md
      guide/
      api/
    latest/            # Development version
      index.md
      guide/
      api/
  index.md             # Root — redirects to /versions/stable/
```

### Adding Pages

Each page starts with:

```yaml
---
title: Page Title
description: Brief description for search/preview
---
```

Use VitePress admonitions (`::: info`, `::: tip`, `::: warning`, `::: danger`, `::: details`).
Internal links: `[text](./path-to-page)` (relative, no `.html`).

---

## Documentation Writing Standards

### Writing Style

1. **Clear & Concise** — explain concepts without jargon
2. **Active Voice** — "you can rename columns" not "columns can be renamed"
3. **Examples First** — show before explaining
4. **Beginner-Friendly** — no assumed knowledge of ML hyperparameter tuning
5. **Consistent Terminology** — use terms from the glossary

### Page Structure

Every guide/tutorial page should have:

1. **Brief intro** — what you'll learn (2-3 sentences)
2. **Prerequisites** — what you need to know/have installed
3. **Step-by-step guide** — with code blocks and screenshots
4. **Examples** — working, self-contained Python code
5. **Tips & Gotchas** — common mistakes
6. **Next Steps** — links to related pages

### Code Examples

- All examples must run without modification
- Keep code readable over optimized
- Use `sklearn` built-in datasets (digits, iris, breast_cancer) so no CSV download is needed
- Show imports at the top of each standalone snippet
- Use `python` as the fenced code block language

### Front Matter

```yaml
---
title: Page Title
description: Brief description for search/preview
---
```

### Headings

```markdown
# Page Title (H1 — only one per page)

## Major Section (H2)

### Subsection (H3)
```

### Links

- **Internal:** `[text](./path/to/page)` — relative, no `.html`
- **External:** `[text](https://example.com)` — VitePress opens externals in new tab automatically
- **Anchors:** `[Jump to section](#section-heading)`

### Callouts

```markdown
:::info
Informational tip
:::

:::tip
Helpful advice
:::

:::warning
Important caution
:::

:::danger
Critical warning
:::

:::details Click to expand
Hidden content
:::
```

### Feature Documentation Template

```markdown
---
title: Feature Name
description: One sentence — what it does and when to use it.
---

# Feature Name

Brief one-liner.

## Use Cases

- Scenario 1
- Scenario 2

## How It Works

Explanation.

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| param1    | str  | `"x"`   | ...         |

## Examples

### Basic Usage

```python
# working code here
```

### Advanced Usage

```python
# working code here
```

## Tips & Gotchas

- Common mistake 1
- Common mistake 2

## Next Steps

- [Related Page](./related-page)
```

---

## Workflow Notes

- The GitHub Actions workflow at `.github/workflows/docs.yml` builds and deploys VitePress to `gh-pages` branch on push to `master` and on version tags.
- To preview locally: `cd docs-vitepress && npm run docs:dev`
- To build: `cd docs-vitepress && npm run docs:build`
- The legacy `readthedocs.yml` is kept so the RTD site continues to build, but every page now shows a deprecation banner pointing to the GitHub Pages URL.

<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## RTK Commands by Workflow

### Build & Compile (80-90% savings)
```bash
rtk cargo build         # Cargo build output
rtk cargo check         # Cargo check output
rtk cargo clippy        # Clippy warnings grouped by file (80%)
rtk tsc                 # TypeScript errors grouped by file/code (83%)
rtk lint                # ESLint/Biome violations grouped (84%)
rtk prettier --check    # Files needing format only (70%)
rtk next build          # Next.js build with route metrics (87%)
```

### Test (90-99% savings)
```bash
rtk cargo test          # Cargo test failures only (90%)
rtk vitest run          # Vitest failures only (99.5%)
rtk playwright test     # Playwright failures only (94%)
rtk test <cmd>          # Generic test wrapper - failures only
```

### Git (59-80% savings)
```bash
rtk git status          # Compact status
rtk git log             # Compact log (works with all git flags)
rtk git diff            # Compact diff (80%)
rtk git show            # Compact show (80%)
rtk git add             # Ultra-compact confirmations (59%)
rtk git commit          # Ultra-compact confirmations (59%)
rtk git push            # Ultra-compact confirmations
rtk git pull            # Ultra-compact confirmations
rtk git branch          # Compact branch list
rtk git fetch           # Compact fetch
rtk git stash           # Compact stash
rtk git worktree        # Compact worktree
```

Note: Git passthrough works for ALL subcommands, even those not explicitly listed.

### GitHub (26-87% savings)
```bash
rtk gh pr view <num>    # Compact PR view (87%)
rtk gh pr checks        # Compact PR checks (79%)
rtk gh run list         # Compact workflow runs (82%)
rtk gh issue list       # Compact issue list (80%)
rtk gh api              # Compact API responses (26%)
```

### JavaScript/TypeScript Tooling (70-90% savings)
```bash
rtk pnpm list           # Compact dependency tree (70%)
rtk pnpm outdated       # Compact outdated packages (80%)
rtk pnpm install        # Compact install output (90%)
rtk npm run <script>    # Compact npm script output
rtk npx <cmd>           # Compact npx command output
rtk prisma              # Prisma without ASCII art (88%)
```

### Files & Search (60-75% savings)
```bash
rtk ls <path>           # Tree format, compact (65%)
rtk read <file>         # Code reading with filtering (60%)
rtk grep <pattern>      # Search grouped by file (75%)
rtk find <pattern>      # Find grouped by directory (70%)
```

### Analysis & Debug (70-90% savings)
```bash
rtk err <cmd>           # Filter errors only from any command
rtk log <file>          # Deduplicated logs with counts
rtk json <file>         # JSON structure without values
rtk deps                # Dependency overview
rtk env                 # Environment variables compact
rtk summary <cmd>       # Smart summary of command output
rtk diff                # Ultra-compact diffs
```

### Infrastructure (85% savings)
```bash
rtk docker ps           # Compact container list
rtk docker images       # Compact image list
rtk docker logs <c>     # Deduplicated logs
rtk kubectl get         # Compact resource list
rtk kubectl logs        # Deduplicated pod logs
```

### Network (65-70% savings)
```bash
rtk curl <url>          # Compact HTTP responses (70%)
rtk wget <url>          # Compact download output (65%)
```

### Meta Commands
```bash
rtk gain                # View token savings statistics
rtk gain --history      # View command history with savings
rtk discover            # Analyze Claude Code sessions for missed RTK usage
rtk proxy <cmd>         # Run command without filtering (for debugging)
rtk init                # Add RTK instructions to CLAUDE.md
rtk init --global       # Add RTK to ~/.claude/CLAUDE.md
```

## Token Savings Overview

| Category | Commands | Typical Savings |
|----------|----------|-----------------|
| Tests | vitest, playwright, cargo test | 90-99% |
| Build | next, tsc, lint, prettier | 70-87% |
| Git | status, log, diff, add, commit | 59-80% |
| GitHub | gh pr, gh run, gh issue | 26-87% |
| Package Managers | pnpm, npm, npx | 70-90% |
| Files | ls, read, grep, find | 60-75% |
| Infrastructure | docker, kubectl | 85% |
| Network | curl, wget | 65-70% |

Overall average: **60-90% token reduction** on common development operations.
<!-- /rtk-instructions -->