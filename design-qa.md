# macOS 27 GUI Design QA

## Evidence

- Source visual truth: `docs/design-target-macos27.png`
- Implementation screenshot: `docs/screenshot-ui.png`
- Empty-state screenshot: `docs/screenshot-empty.png`
- Compact-window screenshot: `docs/screenshot-compact.png`
- Full-view comparison: `docs/design-comparison.png` (source left, implementation right)
- Focused comparison: `docs/design-comparison-focus.png` (header and control dock; source left, implementation right)
- Viewport: 1120 × 800; compact resilience check at 860 × 680
- State: three PDFs — completed, processing at 62%, and waiting

## Findings

- No actionable P0, P1, or P2 differences remain.
- [P3] The implementation's primary button uses one line instead of the mock's title and secondary line. The shorter control keeps the dock readable at the supported 860 px minimum width and preserves the same primary-action hierarchy.
- [P3] The mock includes a trailing overflow control that is intentionally omitted because the current product has no real actions for it. This avoids empty chrome and keeps the core conversion path focused.
- [P3] Native file icons can differ slightly by macOS version and PDF handler. The implementation intentionally uses the macOS file icon provider instead of a fixed imitation.

## Required Fidelity Surfaces

- Fonts and typography: uses the macOS system UI font with PingFang SC fallback. Brand, primary action, file names, metadata, table headers, and status text retain the mock's hierarchy without truncation in either tested viewport.
- Spacing and layout rhythm: header, drop surface, grouped file queue, control dock, and status row preserve the source order and visual grouping. The compact viewport has no overlap, clipping, or off-screen primary action.
- Colors and visual tokens: warm neutral canvas, white standard-material surfaces, restrained hairlines, system blue actions, green success, gray waiting, and red failure states match the selected light direction. Translucent styling is limited to control/navigation surfaces.
- Image quality and asset fidelity: the generated 512 × 512 OCR app icon is used directly in the header, drop state, application icon, and README. It clearly depicts document scanning and remains sharp at UI sizes. No placeholder or code-drawn logo is used.
- Copy and content: all static Chinese copy is product-specific, concise, and consistent with local-only OCR. File rows use real filenames, page counts, sizes, and state labels.
- Icons and controls: app-specific identity uses the generated asset; standard actions and native file types use Qt/macOS system icons. Checkboxes, destination control, remove action, tabs, progress bars, and primary action all have visible states.
- Accessibility and resilience: primary controls have accessible names or descriptive text, focusable native widgets, clear disabled states, and sufficient contrast. The UI was checked at 1120 × 800 and its declared 860 × 680 minimum.

## Comparison History

### Pass 1 — blocked

- [P2] Segmented navigation used generic system glyphs whose meanings did not match conversion/history.
- [P2] Custom checkbox styling obscured the checked mark, weakening format selection clarity.
- [P2] The drop-state symbol was a generic blank document and did not reinforce the new OCR identity.

Fixes made:

- Removed ambiguous tab glyphs and retained clear text labels.
- Restored native checkbox indicators while keeping the refined container treatment.
- Reused the generated OCR app icon in the drop state.
- Switched file rows to the native macOS file icon provider.

### Pass 2 — passed

- Post-fix evidence: `docs/design-comparison.png` and `docs/design-comparison-focus.png`.
- Header identity, checked state, information hierarchy, status colors, dock controls, and primary action are now visibly clear.
- No actionable P0/P1/P2 mismatch remains in the full view or focused regions.

## Interaction Checks

- Add and deduplicate PDFs.
- Show and hide the queue as files are added or removed.
- Switch between conversion and history views.
- Select Markdown/Word output formats.
- Display waiting, processing, completed, and failed row states.
- Update overall progress and unlock controls after completion.
- Open custom/default output location controls.

## Follow-up Polish

- A future native Swift/AppKit port could adopt OS-level Liquid Glass APIs dynamically. The current PySide6 implementation approximates the selected material treatment while preserving cross-version compatibility.

final result: passed
