from __future__ import annotations



from test_frontend_accessibility import (
    _open_compare_tab,
    _open_export_dialog,
    _open_folder_browser,
    _open_review_tab,
    _render_compare_results,
)


def test_active_library_scope_separates_totals_and_resets_review_state(scoped_chromium_page) -> None:
    chromium_page, expect, root_a, root_b = scoped_chromium_page

    def choose_library(root: str) -> None:
        chromium_page.evaluate(
            """
            (root) => {
                const input = document.getElementById("library-root-input");
                input.value = root;
                input.dispatchEvent(new Event("change", { bubbles: true }));
            }
            """,
            root,
        )
        chromium_page.wait_for_function(
            """
            (root) => document.getElementById("root-filter")?.value === root
                && document.getElementById("review-scope-context")?.textContent?.includes(root)
            """,
            arg=root,
        )

    choose_library(root_a)
    _open_review_tab(chromium_page)
    expect(chromium_page.locator("#summary-strip")).to_contain_text("This library")
    expect(chromium_page.locator("#summary-strip")).to_contain_text("61 scored")
    expect(chromium_page.locator("#summary-strip")).to_contain_text("All cached libraries")
    expect(chromium_page.locator("#summary-strip")).to_contain_text("62 scored")
    expect(chromium_page.locator("#page-info")).to_contain_text("1–60 of 61")

    chromium_page.locator("#select-all-matching-btn").click()
    expect(chromium_page.locator("#selection-label")).to_have_text("61 selected")
    chromium_page.locator("#page-next").click()
    expect(chromium_page.locator("#page-info")).to_contain_text("61–61 of 61")

    choose_library(root_b)
    expect(chromium_page.locator("#review-scope-context")).to_have_text(f"Reviewing this library: {root_b}")
    expect(chromium_page.locator("#selection-label")).to_have_text("0 selected")
    expect(chromium_page.locator("#page-info")).to_contain_text("1–1 of 1")
    expect(chromium_page.locator("#queue-list")).to_contain_text("b-001.jpg")

    chromium_page.locator("#root-filter").select_option("")
    expect(chromium_page.locator("#review-scope-context")).to_have_text("All libraries — global catalog view")
    expect(chromium_page.locator("#review-scope-context")).to_have_attribute("data-scope", "global")
    expect(chromium_page.locator("#page-info")).to_contain_text("1–60 of 62")


def test_deleting_last_review_page_clamps_back_to_previous_page(large_chromium_page) -> None:
    chromium_page, expect = large_chromium_page
    _open_review_tab(chromium_page)

    chromium_page.locator("#page-next").click()
    expect(chromium_page.locator("#page-info")).to_contain_text("61–65 of 65")

    chromium_page.evaluate("() => { window.confirm = () => true; }")
    chromium_page.locator("#select-all-btn").click()
    expect(chromium_page.locator("#selection-label")).to_have_text("5 selected")

    chromium_page.locator("#batch-delete-disk").click()

    chromium_page.wait_for_function(
        """
        () => {
            const pageInfo = document.getElementById('page-info')?.textContent || '';
            const reviewPosition = document.getElementById('review-position')?.textContent || '';
            return pageInfo.includes('1–60 of 60') && reviewPosition === '1 of 60';
        }
        """
    )

    expect(chromium_page.locator("#page-info")).to_contain_text("1–60 of 60")
    expect(chromium_page.locator("#review-position")).to_have_text("1 of 60")


def test_lightbox_modal_traps_and_restores_focus(chromium_page) -> None:
    chromium_page, _ = chromium_page
    _open_review_tab(chromium_page)

    detail_image = chromium_page.locator("#detail-image")
    detail_image.wait_for(state="visible")
    detail_image.click()

    chromium_page.wait_for_function("() => document.getElementById('lightbox-overlay')?.open === true")

    active_id = chromium_page.evaluate("() => document.activeElement?.id")
    chromium_page.keyboard.press("Tab")
    after_tab_id = chromium_page.evaluate("() => document.activeElement?.id")

    chromium_page.keyboard.press("Escape")
    chromium_page.wait_for_function("() => document.getElementById('lightbox-overlay')?.open === false")
    restored_id = chromium_page.evaluate("() => document.activeElement?.id")

    assert active_id == "lightbox-close"
    assert after_tab_id == "lightbox-close"
    assert restored_id == "detail-image"


def test_folder_browser_path_field_has_explicit_accessible_name(chromium_page) -> None:
    chromium_page, expect = chromium_page

    chromium_page.evaluate("() => document.getElementById('folder-browser')?.showModal()")
    expect(chromium_page.locator("#browser-path")).to_have_accessible_name("Current folder path")


def test_folder_browser_close_restores_focus_to_trigger(chromium_page) -> None:
    chromium_page, _ = chromium_page

    trigger = chromium_page.get_by_role("button", name="Browse for photo folder")
    trigger.focus()
    _open_folder_browser(chromium_page)
    chromium_page.locator("#folder-browser button[type='submit']").click()
    chromium_page.wait_for_function("() => document.getElementById('folder-browser')?.open === false")

    active_id = chromium_page.evaluate("() => document.activeElement?.id")

    assert active_id == "browse-library-root"


def test_folder_browser_choose_restores_focus_to_trigger(chromium_page) -> None:
    chromium_page, _ = chromium_page

    trigger = chromium_page.get_by_role("button", name="Browse for photo folder")
    trigger.focus()
    _open_folder_browser(chromium_page)
    chosen_path = chromium_page.locator("#browser-path").input_value()
    chromium_page.locator("#browser-choose").click()
    chromium_page.wait_for_function("() => document.getElementById('folder-browser')?.open === false")

    active_id = chromium_page.evaluate("() => document.activeElement?.id")
    selected_path = chromium_page.locator("#library-root-input").input_value()

    assert active_id == "browse-library-root"
    assert selected_path == chosen_path


def test_export_dialog_close_restores_focus_to_batch_move_trigger(chromium_page) -> None:
    chromium_page, _ = chromium_page

    batch_move = chromium_page.locator("#batch-move")
    _open_export_dialog(chromium_page)
    chromium_page.locator("#export-dialog button[type='submit']").click()
    chromium_page.wait_for_function("() => document.getElementById('export-dialog')?.open === false")

    active_id = chromium_page.evaluate("() => document.activeElement?.id")

    assert batch_move.is_visible()
    assert active_id == "batch-move"


def test_compare_failure_rendering_surfaces_warning_banner_and_failure_aware_summary(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _open_compare_tab(chromium_page)

    _render_compare_results(
        chromium_page,
        {
            "model_names": ["topiq_nr", "arniqa"],
            "rows": [
                {
                    "file_id": 0,
                    "path": "C:/photos/broken.jpg",
                    "topiq_nr_score": None,
                    "topiq_nr_confidence": None,
                    "topiq_nr_error": "Model weights missing",
                    "arniqa_score": 74.0,
                    "arniqa_confidence": 85.0,
                }
            ],
            "compare_failures": [],
            "files_considered": 1,
            "files_compared": 1,
            "files_skipped": 0,
            "files_failed": 1,
            "elapsed_seconds": 1.2,
            "model_timings_seconds": {"arniqa": 0.6},
        },
    )

    warning = chromium_page.locator("#compare-results-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("Some model runs failed:")
    expect(warning).to_contain_text("broken.jpg — TOPIQ (Recommended): Model weights missing")

    topiq_summary = chromium_page.locator("#compare-summary-cards .compare-summary-card", has_text="TOPIQ (Recommended)")
    expect(topiq_summary).to_be_visible()
    summary_text = topiq_summary.inner_text()
    assert "all failed" in summary_text
    assert "n/a" not in summary_text

    expect(chromium_page.locator("#compare-card-gallery .compare-result-card")).to_have_count(1)
    expect(chromium_page.locator("#compare-card-gallery .compare-model-error")).to_contain_text("Failed: Model weights missing")


def test_compare_results_default_to_topiq_sort_and_support_extreme_filters(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _open_compare_tab(chromium_page)

    _render_compare_results(
        chromium_page,
        {
            "model_names": ["topiq_nr", "arniqa"],
            "rows": [
                {
                    "file_id": 1,
                    "path": "C:/photos/lowest.jpg",
                    "topiq_nr_score": 10.0,
                    "topiq_nr_confidence": 90.0,
                    "arniqa_score": 50.0,
                    "arniqa_confidence": 80.0,
                },
                {
                    "file_id": 2,
                    "path": "C:/photos/middle.jpg",
                    "topiq_nr_score": 55.0,
                    "topiq_nr_confidence": 90.0,
                    "arniqa_score": 40.0,
                    "arniqa_confidence": 80.0,
                },
                {
                    "file_id": 3,
                    "path": "C:/photos/highest.jpg",
                    "topiq_nr_score": 95.0,
                    "topiq_nr_confidence": 90.0,
                    "arniqa_score": 60.0,
                    "arniqa_confidence": 80.0,
                },
            ],
            "compare_failures": [],
            "files_considered": 3,
            "files_compared": 3,
            "files_skipped": 0,
            "files_failed": 0,
            "elapsed_seconds": 1.0,
            "model_timings_seconds": {"topiq_nr": 0.5, "arniqa": 0.5},
        },
    )

    expect(chromium_page.locator("#compare-row-sort")).to_have_value("topiq_nr:desc")
    expect(chromium_page.locator("#compare-row-filter")).to_have_value("all")
    expect(chromium_page.locator("#compare-card-gallery .compare-result-card")).to_have_count(3)

    chromium_page.locator("#compare-row-filter").select_option("extremes")
    expect(chromium_page.locator("#compare-card-gallery .compare-result-card")).to_have_count(2)
    expect(chromium_page.locator("#compare-card-gallery")).to_contain_text("lowest.jpg")
    expect(chromium_page.locator("#compare-card-gallery")).to_contain_text("highest.jpg")
    expect(chromium_page.locator("#compare-card-gallery")).not_to_contain_text("middle.jpg")


def test_compare_setup_failure_without_rows_keeps_warning_and_empty_state_visible(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _open_compare_tab(chromium_page)

    _render_compare_results(
        chromium_page,
        {
            "model_names": ["topiq_nr", "arniqa"],
            "rows": [
                {
                    "file_id": 0,
                    "path": "C:/photos/previous-success.jpg",
                    "topiq_nr_score": 82.0,
                    "topiq_nr_confidence": 91.0,
                    "arniqa_score": 74.0,
                    "arniqa_confidence": 85.0,
                }
            ],
            "compare_failures": [],
            "files_considered": 1,
            "files_compared": 1,
            "files_skipped": 0,
            "files_failed": 0,
            "elapsed_seconds": 0.8,
            "model_timings_seconds": {"topiq_nr": 0.4, "arniqa": 0.4},
        },
    )
    expect(chromium_page.locator("#compare-card-gallery .compare-result-card")).to_have_count(1)

    _render_compare_results(
        chromium_page,
        {
            "model_names": ["topiq_nr", "arniqa"],
            "rows": [],
            "compare_failures": [
                {
                    "file_id": 3,
                    "path": "C:/photos/broken.heic",
                    "reason": "HEIF preview generation failed",
                    "stage": "preview_generation",
                }
            ],
            "files_considered": 1,
            "files_compared": 0,
            "files_skipped": 0,
            "files_failed": 1,
            "elapsed_seconds": 0.6,
            "model_timings_seconds": {},
        },
    )

    warning = chromium_page.locator("#compare-results-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("Some model runs failed:")
    expect(warning).to_contain_text("broken.heic — HEIF preview generation failed")

    empty_state = chromium_page.locator("#compare-empty")
    expect(empty_state).to_be_visible()
    expect(empty_state).to_contain_text("No comparable cached files were available")
    expect(empty_state).to_contain_text("1 file(s) failed during comparison setup.")

    results = chromium_page.locator("#compare-results")
    assert "hidden" in (results.get_attribute("class") or "")
    expect(chromium_page.locator("#compare-card-gallery .compare-result-card")).to_have_count(0)


def test_compare_truncation_warning_stays_visible_with_results(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _open_compare_tab(chromium_page)

    _render_compare_results(
        chromium_page,
        {
            "model_names": ["topiq_nr", "arniqa"],
            "rows": [
                {
                    "file_id": 0,
                    "path": "C:/photos/sample.jpg",
                    "topiq_nr_score": 82.0,
                    "topiq_nr_confidence": 91.0,
                    "arniqa_score": 74.0,
                    "arniqa_confidence": 85.0,
                }
            ],
            "compare_failures": [],
            "requested_rows_total": 32000,
            "processed_rows_total": 10000,
            "truncated": True,
            "max_rows": 10000,
            "files_considered": 10000,
            "files_compared": 10000,
            "files_skipped": 0,
            "files_failed": 0,
            "elapsed_seconds": 12.4,
            "model_timings_seconds": {"topiq_nr": 6.1, "arniqa": 6.3},
        },
    )

    warning = chromium_page.locator("#compare-results-warning")
    expect(warning).to_be_visible()
    expect(warning).to_contain_text("Comparing first 10,000 of 32,000 files.")
    expect(warning).to_contain_text("Narrow the root or apply filters for a full compare.")
    expect(chromium_page.locator("#compare-card-gallery .compare-result-card")).to_have_count(1)
