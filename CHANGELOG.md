# Changelog

All notable changes to this project should be recorded in this file.

The format follows Keep a Changelog, and version numbers are intended to follow Semantic Versioning.

## [0.1.1] - 2026-07-15

### Fixed

- Reconstructed native text-PDF tables as readable Markdown pipe tables instead of flattened text.
- Preserved complete numbered references and explicit source URLs from text PDFs.
- Kept visually separate bold label lines as separate Markdown paragraphs.
- Strengthened built-in coverage checks for text-PDF tables, formulas, headings, and references.

### Changed

- Consolidated regression checks into the main program and removed the legacy standalone `tests` directory.

## [0.1.0] - 2026-07-03

### Added

- Initial public project structure, local OCR pipeline, GUI, CLI, tests, and documentation.
- A single-source version number in `ptt/__init__.py`.
- A repeatable GitHub release workflow in `docs/release-process.md`.
