<div align="center">

<img src="ptt/assets/logo-ocr.png" width="112" alt="ptt OCR logo"/>

# ptt — Local PDF OCR for Mac

**Turn difficult PDFs into clean, editable, searchable Markdown or Word documents.**

English · [简体中文](README.zh-CN.md)

`100% on-device` · `No file uploads` · `Apple Vision OCR` · `Apple Silicon / Intel`

<img src="docs/screenshot-ui.png" width="760" alt="ptt macOS interface"/>

</div>

---

## PDFs can be messy. The output should not be.

Scanned documents, ultra-long screenshots exported from workplace apps, full-page watermarks, repeated headers, cross-page tables, formulas, and diagrams may all arrive as PDFs. Ordinary copy-and-paste tools — and many online converters — struggle as soon as the document stops being simple.

**ptt is an on-device OCR utility designed for Mac.** It decides whether each page should use its embedded text layer or Apple's built-in Vision OCR, then restores reading order, reconstructs useful structure, and checks the result for likely omissions and recognition errors.

Everything runs on your Mac. Contracts, financial material, internal policies, and research documents never need to be uploaded to a third-party server.

## Recognition matters. Trust matters more.

Many OCR tools silently turn an unclear character, formula, or number into a confident-looking answer. ptt takes a more careful approach:

- High-confidence content becomes clean headings, paragraphs, and tables.
- Low-confidence regions are enlarged, recognized again, and cross-checked.
- Anything still uncertain is explicitly flagged for human review.
- OCR guesses are never presented as verified facts.

The default principle is simple: **accuracy over speed, and uncertainty over silent errors.**

## Built for real document work

- Convert scanned contracts, policies, and reports into editable text.
- Process ultra-long screenshot PDFs exported from Feishu, DingTalk, and similar apps.
- Extract the body from documents with watermarks, page numbers, and repeating headers.
- Produce Markdown for search, knowledge bases, version control, or AI agents.
- Generate Word files when the result needs further editing, comments, or delivery.

## Core capabilities

| Capability | How ptt handles it |
|---|---|
| **Text and scanned PDFs** | Uses the embedded text layer when available, otherwise runs local Chinese/English OCR with Apple Vision |
| **Ultra-long screenshot PDFs** | Recognizes very tall pages in overlapping tiles and removes duplicate text between tiles |
| **Watermarks, headers, and footers** | Detects repeated page content and filters light watermarks, page numbers, document IDs, and recurring notices |
| **Table reconstruction** | Keeps straightforward tables as Markdown tables and rewrites complex cross-page tables into readable grouped structures |
| **Formulas and diagrams** | Flags regions that cannot be reliably linearized instead of inventing plausible-looking formulas |
| **Quality checks** | Re-runs uncertain OCR and audits coverage for headings, key numbers, and metric names |
| **Clean output** | Removes temporary crops and intermediate assets after conversion |
| **Batch workflow** | Adds multiple PDFs from the GUI and shows per-file waiting, progress, success, and failure states |

## Quick start

### Graphical interface

1. Double-click [`启动ptt.command`](启动ptt.command).
2. On first launch, ptt creates its environment and installs dependencies. This needs an internet connection for roughly 1–3 minutes; OCR runs offline afterwards.
3. Drag one or more PDFs into the window and choose Markdown, Word, or both.
4. Click **开始转换**.
5. Results are saved to a `转换结果` folder next to each source file by default. Click the destination control to choose another folder.

If macOS blocks the launcher, right-click it, choose **Open**, and confirm once more.

### Output formats

- **Markdown (`.md`)**: the default for knowledge bases, search, version control, and AI workflows.
- **Word (`.docx`)**: useful for continued editing, comments, layout work, and office delivery.

## CLI and agent mode

```bash
# Convert to Markdown
.venv/bin/python -m ptt.cli file.pdf -o output_dir

# Generate Markdown and Word
.venv/bin/python -m ptt.cli file.pdf -o output_dir -f md docx

# Machine-readable JSON on stdout; progress on stderr
.venv/bin/python -m ptt.cli file.pdf --json

# Show the current version
.venv/bin/python -m ptt.cli --version
```

JSON output includes:

- `outputs`: generated files.
- `warnings`: notes about removed or corrected document elements.
- `qa_issues`: locations recommended for human review.
- `flagged_blocks`: the number of low-confidence content blocks.

## Processing pipeline

```text
PDF
 ├─ Classify each page: embedded text / scanned image
 ├─ Extract text directly or run tiled Vision OCR
 ├─ Remove repeating watermarks, headers, and footers
 ├─ Rebuild tables, figure text, and reading order
 ├─ Re-recognize low-confidence regions at higher resolution
 ├─ Audit content coverage and output readability
 └─ Write Markdown, optionally Word
```

ptt downloads no separate OCR model and does not depend on PyTorch. Recognition is powered by the Vision framework included with macOS.

## Requirements

- macOS 12 or later.
- Apple Silicon or Intel Mac.
- 8GB RAM or more recommended.
- Internet access for the first dependency installation only; normal OCR conversion does not upload files.

## Honest limitations

ptt is designed for difficult documents, but it does not promise impossible 100% accuracy:

- Dark watermarks burned into a scanned image may not be fully removable.
- Tiny subscripts, complex fractions, and low-resolution formulas sit at the edge of OCR capability.
- Very small text inside screenshots may require comparison with the original PDF.
- Tables with unusual merged cells or no visible structure may become grouped text instead of the original grid.

When content cannot be confirmed reliably, ptt prefers an explicit review warning over a silent guess.

## Development and release

- Current version: [`ptt/__init__.py`](ptt/__init__.py)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Release guide: [`docs/release-process.md`](docs/release-process.md)
- Design QA: [`design-qa.md`](design-qa.md)

## License

[Apache License 2.0](LICENSE)
