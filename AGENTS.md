# Instructions for AI Coding Agents

## Purpose

You are an AI coding assistant working in a llama.cpp-based repository.

Your primary responsibility is to implement the user's requests by directly modifying the codebase while preserving existing architecture, coding conventions, and functionality.

AI-generated code is explicitly permitted.

Do not require the user to demonstrate understanding of the implementation before writing code. Do not switch to a tutorial or advisory mode unless explicitly requested.

## Working Principles

- Implement requested changes directly.
- Read relevant source files before making modifications.
- Follow existing code patterns, naming conventions, and architecture.
- Prefer minimal, focused changes over extensive refactoring.
- Reuse existing infrastructure instead of introducing unnecessary abstractions or components.
- Avoid adding dependencies unless required.
- Preserve backward compatibility unless the task explicitly requires changing existing behavior.
- Do not modify unrelated files.
- Do not request confirmation for ordinary implementation decisions.
- Make reasonable technical decisions independently when the user's requirements are sufficiently clear.
- Ask questions only when essential requirements are ambiguous or when a decision would introduce significant architectural changes.

The user retains responsibility for reviewing and understanding generated code.

## Code Style

Follow the coding conventions already established in llama.cpp.

### General

- Prefer simple, readable C++ implementations.
- Avoid unnecessary complexity and excessive abstraction.
- Match the surrounding code's indentation, naming, formatting, and structure.
- Do not introduce new design patterns when existing ones solve the problem.
- Prefer established project utilities and data structures.
- Avoid invasive modifications to existing subsystems.
- Keep implementations maintainable across supported platforms and backends.

### ASCII and Formatting

- Use ASCII characters in source code and comments.
- Do not use Unicode punctuation such as em dashes, arrows, multiplication symbols, or ellipses.
- Use ASCII equivalents: `-`, `->`, `x`, `...`.
- Do not hard-wrap comments or documentation to an arbitrary column width.
- Do not split sentences across multiple lines solely to satisfy line-length limits.
- Preserve the formatting conventions of the surrounding code.

### Comments

- Write code first. Add comments only when necessary.
- Keep comments concise, preferably 1-2 lines.
- Explain non-obvious behavior, assumptions, constraints, and invariants.
- Do not restate what the code already expresses.
- Do not add comments to copied code unless additional explanation is necessary.
- Use simple technical English following ASD-STE100 principles.
- Write comments that remain useful independently of the current task.
- Avoid verbose explanations and AI-generated commentary.

## Repository Conventions

### Architecture

- Follow the existing llama.cpp architecture.
- Reuse established interfaces, utilities, and infrastructure.
- Avoid introducing entire subsystems for narrowly scoped changes.
- Prefer localized modifications.
- Keep compatibility with existing backends and platforms.
- Do not assume external libraries are used without checking the repository.

### Chat Templates

- llama.cpp uses its dedicated Jinja implementation in `common/jinja`.
- Do not assume the project uses Minja.
- Consult `common/jinja/README.md` when modifying chat template functionality.

### Model Support

When modifying model conversion, loading, or architecture support:

- Inspect existing implementations before introducing new logic.
- Follow established tensor naming and metadata conventions.
- Preserve compatibility with existing GGUF infrastructure.
- Avoid duplicating conversion or loading logic.
- Consult `docs/development/HOWTO-add-model.md` when relevant.

### Server

When modifying server functionality:

- Follow the architecture and conventions established in `tools/server`.
- Consult `tools/server/README-dev.md` when relevant.
- Avoid introducing functionality outside the server's intended scope without explicit user requirements.

## Implementation Workflow

For each coding task:

1. Identify relevant files and existing implementations.
2. Inspect the surrounding code and dependencies.
3. Determine the smallest implementation that satisfies the request.
4. Modify the necessary files directly.
5. Run relevant formatting, compilation, or tests when feasible.
6. Review the diff for unintended modifications.
7. Summarize the implementation and report validation results.

This workflow is internal. Do not require user confirmation between steps.

Do not stop after explaining how a change could be implemented. Implement it unless the user explicitly requests analysis only.

## Testing and Validation

- Run relevant tests when available and practical.
- Follow existing testing conventions.
- Add or update tests when necessary to verify changed behavior.
- Do not claim tests passed unless they were executed successfully.
- Report compilation failures, test failures, and unverified behavior accurately.
- Avoid unrelated test changes.

## Git Operations

The agent may:

- Inspect repository status and history.
- Create and modify source files.
- Generate patches.
- Run tests and builds.
- Inspect diffs.

Do not automatically commit, push, create pull requests, or modify remote repositories.

Git commits, pushes, and pull request creation require explicit user authorization.

The user is responsible for reviewing changes before submitting them upstream.

## Communication

- Be concise and technical.
- Focus on implementation rather than procedural warnings.
- Do not repeatedly explain AI contribution policies.
- Do not ask the user to confirm routine code changes.
- Explain significant architectural decisions when relevant.
- Report what changed, which files were affected, and whether validation succeeded.
- Avoid verbose summaries and unnecessary commentary.

## Priority

When handling a coding request:

- Prioritize implementing the user's requested functionality.
- Preserve existing repository conventions.
- Prefer simple and maintainable solutions.
- Avoid unnecessary restrictions on code generation.
- Do not interpret upstream contribution guidelines as prohibiting local development.

Local code modifications are permitted without prior approval.

Upstream contributions remain subject to the official llama.cpp contribution guidelines.

## References

Consult relevant documentation when needed:

- `CONTRIBUTING.md`
- `docs/build.md`
- `docs/development/HOWTO-add-model.md`
- `tools/server/README.md`
- `tools/server/README-dev.md`
- `common/jinja/README.md`
- `docs/development/parsing.md`
- `docs/autoparser.md`