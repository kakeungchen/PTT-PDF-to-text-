# Changelog

All notable changes to this project should be recorded in this file.

The format follows Keep a Changelog, and version numbers are intended to follow Semantic Versioning.

## [0.1.2] - 2026-07-16

### Fixed

- 修复复杂跨页/跨列 OCR 表格被展平后造成的字段错关联和可读性问题。
- 增加算分示例表的指标、分子、分母、达标率、目标、权重、得分及融合核算结构化输出。
- 修复带括号的小节标题被章节覆盖审计误判为缺失的问题。
- 文本型 PDF 保留标准表格、参考文献编号和 URL；默认转换结果仍只输出单个 Markdown 文件。

### Changed

- 内建回归检查扩展至 175 项。

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
