"""Exercise the actual result presenter with browser DOM and selection state."""
from __future__ import annotations


def test_cleanup_warnings_are_visible_and_unknown_status_preserves_selection(chromium_page):
    page, expect = chromium_page
    state = page.evaluate("""() => {
        const state = {selectedIds: new Set([1, 2]), bulkSelection: null};
        const presenter = window.ShotSieveWorkflowExport.createWorkflowExport({
            state, api: {}, busy: {}, notifications: {}, review: {}, ui: {}, workflowLibrary: {},
        });
        presenter.presentOperationResult({outcome: "unknown", job_status: "unknown"},
            {payload: {file_ids: [1, 2], mode: "move"}});
        const afterUnknown = [...state.selectedIds];
        const result = {
            action: "move", outcome: "success", completed_count: 1,
            items: [{file_id: 1, source: "photo.jpg", outcome: "success", stage: "catalog_update"}],
            warnings: [{file_id: 1, source: "photo.jpg", stage: "preview_cleanup", error_text: "Preview access denied"}],
        };
        presenter.presentOperationResult(result, {payload: {file_ids: [1], mode: "move"}});
        return {afterUnknown, afterSuccess: [...state.selectedIds], tone: presenter.operationTone(result)};
    }""")
    assert state == {"afterUnknown": [1, 2], "afterSuccess": [2], "tone": "warning"}
    expect(page.locator("#operation-result-counts")).to_contain_text("Cleanup warnings: 1")
    expect(page.locator("#operation-result-items")).to_contain_text("Preview access denied")
    expect(page.locator("#operation-result-panel")).to_have_attribute("data-outcome", "partial")
