# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains one engineering design document: `EVA_AUDIO2FACE_PIPELINE_V1.1.md`. It describes the EVA robot’s speech-driven facial animation pipeline, calibration, learned adaptation, synchronization, and acceptance criteria. There are no source-code, test, or asset directories yet. Keep related documentation at the repository root until a larger structure is needed.

## Build, Test, and Development Commands

No build system, dependency manifest, application runtime, or automated test suite is present. Review changes in a Markdown preview that supports Mermaid diagrams and display mathematics.

Useful inspection commands:

- `rg --files --hidden`: list repository files, including hidden files.
- `rg -n '^##|^###' EVA_AUDIO2FACE_PIPELINE_V1.1.md`: inspect the document’s section hierarchy.

When working in a valid Git checkout, use `git diff --check` to check whitespace and `git diff` to review edits before committing.

## Coding Style & Naming Conventions

Preserve the main document’s Chinese prose, numbered sections, concise paragraphs, tables, and numbered reference list. Use fenced blocks for Mermaid diagrams, backticks for interface fields, and the existing mathematical notation for equations. Preserve identifiers such as `FaceMotion-52`, `schema_version`, and `timestamp_ms` consistently.

Follow the existing descriptive, uppercase, underscore-separated filename pattern for versioned pipeline documents. Update version metadata and the change summary together when publishing a new revision.

## Testing & Technical Review

Documentation review is the current validation process; no testing framework or coverage threshold is configured. Check heading order, table rendering, equations, Mermaid syntax, and reference numbering. Verify changed technical claims against the cited primary sources. Keep engineering targets distinct from measured results, and distinguish predicted shape error from observed hardware error. Preserve calibration, actuator arbitration, and safety constraints when revising the pipeline.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace, so no established commit convention can be confirmed. Use short, imperative messages, for example `docs: clarify calibration acceptance criteria`.

Pull requests should summarize the affected sections, explain the engineering rationale, identify supporting references or related issues, and state what was reviewed. Include preview screenshots when changing diagrams or complex formatting. Do not describe proposed hardware behavior as validated without measurement evidence.
