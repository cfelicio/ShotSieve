# Visual QA Checklist

Last updated: 2026-04-24

Use this checklist for a quick manual visual-usability pass on the local review UI after frontend changes. The goal is simple: good contrast, readable text, comfortable controls, sensible font sizing, and clear review states for photographers actively looking at images on screen.

## Suggested setup

- Browser: current Chromium build used for local QA
- Start from a fresh page load with a small test library so the `Review` tab has queue rows
- Check both normal desktop viewing and one narrow viewport (`390px` wide is enough)
- If something looks borderline, repeat once at browser zoom `125%`

## 1. Contrast and readable type

Goal: confirm the UI is easy to read at a glance during long culling sessions.

1. Open `Library`, `Compare`, `Review`, and `Settings` once each.
2. Check primary actions such as `Analyze`, `Run Comparison`, `✓ Keep`, and `✕ Reject`.
3. Check quieter text such as filter labels, score lines, helper hints, and queue metadata.
4. Confirm the current theme still gives enough contrast between text, panels, buttons, and backgrounds.
5. If needed, repeat at browser zoom `125%` and confirm the font sizes still feel balanced rather than cramped or tiny.

Expected result:

- Primary actions are visually obvious.
- Secondary text is still readable without squinting.
- Contrast stays comfortable in the active theme.

## 2. Review workflow clarity

Goal: confirm the culling workflow is visually clear while moving quickly between photos.

1. Open `Review` and move through several photos.
2. Confirm the current photo is easy to identify in both the queue and the detail pane.
3. Select one or more queue items and confirm the selected state is visually obvious.
4. Confirm the selection count, scope text (`Current photo` / `Selected photos`), and `Issues 0` summary are easy to notice.
5. Click the current photo once to open the lightbox, then close it.
6. Confirm score text, issue text, warning banners, and action buttons stay readable while the page updates.

Expected result:

- Current vs selected items are visually distinct.
- Keep/reject actions are easy to spot.
- Clicking the detail image opens the lightbox cleanly.
- Important state changes can be noticed without hunting around the screen.

## 3. Dialog and control comfort

Goal: confirm dialogs and controls feel comfortable rather than fiddly.

1. Open the folder browser from `Library`.
2. Confirm the path field, `Up`, `Choose Folder`, and `Close` controls are easy to read and click.
3. Open the export dialog from `Review`.
4. Confirm the destination field, `Browse`, mode selector, and `Export Now` button are clearly readable.
5. On a narrow viewport, confirm dialog content still fits without awkward clipping or overlapping controls.

Expected result:

- Dialog text is legible.
- Controls are large enough to hit comfortably.
- Nothing important is clipped or pushed off-screen.

## 4. Responsive spot check

Goal: confirm the UI still feels usable on small and medium widths.

1. Check the app around `390px`, `768px`, and a normal desktop width.
2. Confirm there is no distracting horizontal scroll.
3. Confirm important actions stay visible: `Analyze`, `Run Comparison`, `✓ Keep`, and the main settings heading.
4. Confirm queue rows, filters, and action bars wrap cleanly instead of collapsing into unreadable clutter.

Expected result:

- Layout stays readable across the target widths.
- Core actions remain visible without awkward scrolling.
- Text and controls do not collapse into a cramped mess.

## Recommended follow-up cadence

- Run this checklist before release candidates.
- Re-run after changes to layout, button styling, dialog sizing, tab structure, review-state presentation, or detail-image/lightbox behavior.
- Pair this manual pass with `tests/test_frontend_accessibility.py` so browser regressions and visual-usability regressions stay aligned.
