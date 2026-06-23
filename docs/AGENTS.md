# Documentation Agent Instructions for MATE

This file is the documentation-specific companion to the repository root
`AGENTS.md`. Use it when creating or editing content under `docs/`.

The sections below are intentionally lightweight placeholders so product,
documentation, and developer-experience rules can be defined here later without
mixing them into code-editing guidance.

## Scope

- Applies to files under `docs/`
- Covers information architecture, authoring style, user journey, and product
  messaging for documentation work

## Current Documentation Principle

- Prefer a wrapper-first user journey for Moore Threads platforms
- Treat compatibility wrappers as the normal integration path when they match
  the framework's expected Python package surface
- Use native MATE APIs as a fallback path when wrapper coverage is insufficient

## Placeholders To Define Later

Fill in or revise these sections when the documentation rules are ready.

### User Profile

- Primary audience:
- Secondary audience:
- What they already know:
- What they are trying to accomplish:

### User Journey

- Entry point:
- Default path:
- Fallback path:
- Diagnostics path:

### Product Feature Priorities

- Features to emphasize:
- Features to mention briefly:
- Features to downplay or gate carefully:

### Voice and Tone

- Desired tone:
- Phrases or framing to prefer:
- Phrases or framing to avoid:
- Terminology preferences:

### Information Architecture

- Preferred top-level structure:
- Preferred navigation order:
- Pages that should stay short:
- Pages that should hold detail:

### Source of Truth

- Repo files that define factual behavior:
- Repo files that define examples:
- Repo files that define compatibility scope:
- External sources that may be referenced:

### Review Checklist

- What every doc change must verify:
- What examples must compile or match:
- What claims need explicit repo backing:

## Editing Rules

- Keep compatibility claims aligned with the actual wrapper or API surface in
  the repository
- Prefer user-journey ordering over low-level API ordering for overview and
  tutorial pages
- Keep installation, wrapper usage, CLI, and diagnostics guidance consistent
  across pages
- If a rule is still unspecified in this file, defer to the root `AGENTS.md`
  and the current repository source of truth
