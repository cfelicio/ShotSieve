from __future__ import annotations

import re
from pathlib import Path

import pytest
from PIL import Image


def _create_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (160, 120), color=color)
    image.save(path, format="JPEG")


def _open_review_tab(page) -> None:
    page.get_by_role("tab", name="Review").click()
    page.wait_for_function(
        """
        () => {
          const rows = document.querySelectorAll('#queue-list .queue-item');
          const position = document.getElementById('review-position');
          return rows.length >= 3 && Boolean(position?.textContent?.trim());
        }
        """
    )


def _open_compare_tab(page) -> None:
    page.get_by_role("tab", name="Compare").click()
    page.wait_for_function(
        """
                () => {
                    const compareButton = document.getElementById('compare-run');
                    const modelCards = document.querySelectorAll('#compare-model-grid .compare-model-card').length;
                    return compareButton instanceof HTMLElement
                        && !compareButton.hidden
                        && compareButton.getClientRects().length > 0
                        && modelCards >= 1;
                }
        """
    )


def _render_compare_results(page, comparison: dict[str, object], *, root: str = "C:/photos") -> None:
    page.evaluate(
        """
        ({ comparison, root }) => {
            const utils = window.ShotSieveUtils;
            const stateModule = window.ShotSieveState;
            const workflowsModule = window.ShotSieveWorkflows;
            if (!utils || !stateModule || !workflowsModule?.createWorkflows) {
                throw new Error("ShotSieve compare renderer helpers are unavailable.");
            }

            document.getElementById("library-root-input").value = root;

            const state = stateModule.createState();
            state.options = {
                learned: {
                    model_catalog: [
                        {
                            canonical_id: "topiq_nr",
                            label: "TOPIQ (Recommended)",
                            description: "Fast, stable all-rounder for general photo-quality ranking.",
                        },
                        {
                            canonical_id: "clipiqa",
                            label: "CLIPIQA",
                            description: "CLIP-based quality scorer for a complementary second opinion.",
                        },
                    ],
                },
            };
            state.comparison = comparison;

            const workflows = workflowsModule.createWorkflows({
                state,
                api: {
                    fetchJson: async () => {
                        throw new Error("fetchJson should not run in compare render harness");
                    },
                    postJson: async () => {
                        throw new Error("postJson should not run in compare render harness");
                    },
                },
                busy: {
                    setBusyMessage() {},
                    setBusyPhaseProgress() {},
                    setBusyProgress() {},
                    withBusy: async (_message, fn) => fn(),
                },
                compare: {
                    compareBatchSize: () => 1,
                    compareProgressMessage: () => "",
                    compareProgressPercent: () => 0,
                    comparisonDefaults: () => [],
                    currentResourceProfile: () => "normal",
                    scanProgressMessage: () => "",
                    scanProgressPercent: () => 0,
                    scoreBatchSize: () => 1,
                    scoreProgressMessage: () => "",
                    scoreProgressPercent: () => 0,
                },
                formatting: {
                    escapeHtml: utils.escapeHtml,
                    formatDuration: utils.formatDuration,
                    formatFilesPerSecond: utils.formatFilesPerSecond,
                    formatNumber: utils.formatNumber,
                    getScoreColor: utils.getScoreColor,
                    mergeTimingTotals: utils.mergeTimingTotals,
                    pathLeaf: utils.pathLeaf,
                    sortComparisonRows: utils.sortComparisonRows,
                },
                notifications: {
                    showToast() {},
                },
                review: {
                    isAutoAdvanceEnabled: () => true,
                    loadQueue: async () => {},
                    refreshOverview: async () => {},
                    refreshWorkspace: async () => {},
                    reviewDecisions: stateModule.REVIEW_DECISIONS,
                    selectFile: async () => {},
                    syncReviewRoot() {},
                },
                ui: {
                    closeOverlay() {},
                    currentLibraryRoot: () => root,
                    saveUiState() {},
                    selectedComparisonModels: () => comparison.model_names || [],
                    setTab() {},
                },
            });

            workflows.renderComparisonResults();

            const compareRowSort = document.getElementById("compare-row-sort");
            const compareRowFilter = document.getElementById("compare-row-filter");
            if (compareRowSort) {
                compareRowSort.onchange = (event) => {
                    state.compareRowSort = event.target.value || "topiq_nr:desc";
                    state.compareRowSortInitialized = true;
                    workflows.renderComparisonResults();
                };
            }
            if (compareRowFilter) {
                compareRowFilter.onchange = (event) => {
                    state.compareRowFilter = event.target.value || "all";
                    workflows.renderComparisonResults();
                };
            }
        }
        """,
        {"comparison": comparison, "root": root},
    )


def _open_settings_tab(page) -> None:
    page.get_by_role("tab", name="Settings").click()
    page.wait_for_function(
        """
        () => {
          const hardwareCards = document.querySelectorAll('#hardware-cards .runtime-card').length;
          const runtimeCards = document.querySelectorAll('#runtime-cards .runtime-card').length;
          return hardwareCards >= 1 && runtimeCards >= 1;
        }
        """
    )


def _open_folder_browser(page) -> None:
    page.get_by_role("button", name="Browse for photo folder").click()
    page.wait_for_function(
        """
        () => {
          const dialog = document.getElementById('folder-browser');
          const pathField = document.getElementById('browser-path');
          const roots = document.getElementById('browser-roots');
          const list = document.getElementById('browser-list');
          return dialog?.open === true
            && Boolean(pathField?.value)
            && Boolean(roots?.textContent?.trim())
            && Boolean(list?.textContent?.trim());
        }
        """
    )


def _wait_for_shell_ready(page, timeout: float = 60000) -> None:
    page.wait_for_function(
        """
        () => {
          const modelOptions = document.querySelectorAll('#model-select option').length;
          const deviceOptions = document.querySelectorAll('#device-select option').length;
          return modelOptions >= 1 && deviceOptions >= 1;
        }
        """,
        timeout=timeout,
    )


def _open_export_dialog(page) -> str:
    _open_review_tab(page)
    first_row = page.locator("#queue-list .queue-item").first
    first_filename = first_row.locator(".queue-file").inner_text()
    first_row.get_by_role("checkbox", name=f"Select {first_filename}").click()
    page.locator("#batch-move").click()
    page.wait_for_function("() => document.getElementById('export-dialog')?.open === true")
    return first_filename


def _bounding_size(locator) -> dict[str, float]:
    locator.wait_for(state="visible")
    return locator.evaluate(
        """
        (node) => {
          const rect = node.getBoundingClientRect();
          return { width: rect.width, height: rect.height };
        }
        """
    )


def _assert_touch_target_floor(page, selector: str, *, label: str) -> None:
    size = _bounding_size(page.locator(selector).first)
    assert size["width"] >= 44, f"{label} width {size['width']}px is below 44px"
    assert size["height"] >= 44, f"{label} height {size['height']}px is below 44px"


RESPONSIVE_VIEWPORTS = [
    pytest.param({"width": 390, "height": 844}, id="mobile-390"),
    pytest.param({"width": 768, "height": 1024}, id="tablet-768"),
    pytest.param({"width": 1440, "height": 900}, id="desktop-1440"),
]


def _set_viewport(page, *, width: int, height: int) -> None:
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_timeout(150)


def _bounding_rect(locator) -> dict[str, float]:
    locator.wait_for(state="visible")
    return locator.evaluate(
        """
        (node) => {
          const rect = node.getBoundingClientRect();
          return {
            left: rect.left,
            right: rect.right,
            top: rect.top,
            bottom: rect.bottom,
            width: rect.width,
            height: rect.height,
          };
        }
        """
    )


def _assert_no_horizontal_overflow(page, *, label: str) -> None:
    metrics = page.evaluate(
        """
        () => ({
          innerWidth: window.innerWidth,
          scrollWidth: document.documentElement.scrollWidth,
        })
        """
    )
    assert metrics["scrollWidth"] <= metrics["innerWidth"] + 1, (
        f"{label} overflows horizontally: scrollWidth={metrics['scrollWidth']} "
        f"innerWidth={metrics['innerWidth']}"
    )


def _assert_within_viewport(page, locator, *, label: str) -> None:
    rect = _bounding_rect(locator)
    viewport = page.viewport_size
    assert viewport is not None
    assert rect["left"] >= -1, f"{label} starts left of the viewport: {rect}"
    assert rect["right"] <= viewport["width"] + 1, f"{label} extends past the right edge: {rect}"
    assert rect["top"] >= -1, f"{label} starts above the viewport: {rect}"
    assert rect["bottom"] <= viewport["height"] + 1, f"{label} extends below the viewport: {rect}"


def test_accessibility_checklist_stays_visual_usability_focused() -> None:
    checklist_text = (Path(__file__).resolve().parents[1] / "docs" / "accessibility-checklist.md").read_text(encoding="utf-8")
    html_text = (Path(__file__).resolve().parents[1] / "src" / "shotsieve" / "static" / "index.html").read_text(encoding="utf-8")
    review_js_text = (Path(__file__).resolve().parents[1] / "src" / "shotsieve" / "static" / "app-review.js").read_text(encoding="utf-8")

    assert "contrast" in checklist_text.casefold()
    assert "font" in checklist_text.casefold()
    assert not re.search(r"screen[- ]reader", checklist_text, re.IGNORECASE)
    assert not re.search(r"assistive[- ]technology", checklist_text, re.IGNORECASE)
    assert not re.search(r"voiceover|narrator|nvda", checklist_text, re.IGNORECASE)
    assert "`Current photo`" in checklist_text
    assert "`Selected photos`" in checklist_text
    assert "detail-open-lightbox" not in html_text
    assert "detail-modelline" not in html_text
    assert re.search(r"['\"`]Current photo['\"`]", review_js_text)
    assert re.search(r"['\"`]Selected photos['\"`]", review_js_text)


def test_keyboard_tab_activation_keeps_focus_on_active_tab(chromium_page) -> None:
    chromium_page, _ = chromium_page
    assert chromium_page.locator("#compare-overlay").count() == 0
    assert not chromium_page.locator("#lightbox-overlay").is_visible()

    compare_tab = chromium_page.get_by_role("tab", name="Compare")
    compare_tab.focus()
    chromium_page.keyboard.press("Enter")

    active_id = chromium_page.evaluate("() => document.activeElement?.id")

    assert active_id == "tab-compare-button"


def test_keyboard_tab_navigation_supports_wraparound_home_and_end(chromium_page) -> None:
    chromium_page, _ = chromium_page

    workspace_tab = chromium_page.get_by_role("tab", name="Library")
    workspace_tab.focus()
    chromium_page.keyboard.press("ArrowLeft")
    last_active_id = chromium_page.evaluate("() => document.activeElement?.id")

    chromium_page.keyboard.press("Home")
    home_active_id = chromium_page.evaluate("() => document.activeElement?.id")

    chromium_page.keyboard.press("End")
    end_active_id = chromium_page.evaluate("() => document.activeElement?.id")

    assert last_active_id == "tab-settings-button"
    assert home_active_id == "tab-workspace-button"
    assert end_active_id == "tab-settings-button"


def test_pointer_tab_click_preserves_arrow_navigation(chromium_page) -> None:
    chromium_page, _ = chromium_page
    _open_review_tab(chromium_page)

    starting_position = chromium_page.locator("#review-position").inner_text()
    chromium_page.keyboard.press("ArrowRight")
    chromium_page.wait_for_function(
        "expected => document.getElementById('review-position')?.textContent !== expected",
        arg=starting_position,
    )

    updated_position = chromium_page.locator("#review-position").inner_text()

    assert starting_position != updated_position


def test_reset_everything_restores_normal_resource_profile(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _open_settings_tab(chromium_page)

    profile_select = chromium_page.locator("#resource-profile-select")
    profile_select.select_option("aggressive")
    expect(profile_select).to_have_value("aggressive")

    chromium_page.evaluate("() => { window.confirm = () => true; }")
    chromium_page.locator("#clear-all-cache").click()

    expect(profile_select).to_have_value("normal")
    stored_profile = chromium_page.evaluate("() => window.localStorage.getItem('shotsieve_resource_profile')")
    assert stored_profile is None


def test_review_position_counts_globally_across_pages(large_chromium_page) -> None:
    chromium_page, expect = large_chromium_page
    _open_review_tab(chromium_page)

    expect(chromium_page.locator("#review-position")).to_have_text("1 of 65")
    expect(chromium_page.locator("#page-info")).to_contain_text("1–60 of 65")

    chromium_page.locator("#page-next").click()

    expect(chromium_page.locator("#page-info")).to_contain_text("61–65 of 65")
    expect(chromium_page.locator("#review-position")).to_have_text("61 of 65")

    chromium_page.locator("#next-item").click()
    expect(chromium_page.locator("#review-position")).to_have_text("62 of 65")


def test_folder_inputs_expose_clean_accessible_names(chromium_page) -> None:
    chromium_page, expect = chromium_page

    expect(chromium_page.locator("#library-root-input")).to_have_accessible_name("Photo Folder")
    expect(chromium_page.locator("#browse-library-root")).to_have_accessible_name("Browse for photo folder")

    chromium_page.evaluate("() => document.getElementById('export-dialog')?.showModal()")
    expect(chromium_page.locator("#export-destination")).to_have_accessible_name("Destination Folder")
    expect(chromium_page.locator("#browse-export-dir")).to_have_accessible_name("Browse for export destination")


def test_accessibility_smoke_exposes_named_shell_controls_and_dialogs(chromium_page) -> None:
    chromium_page, expect = chromium_page

    expect(chromium_page.get_by_role("tab", name="Library")).to_be_visible()
    expect(chromium_page.get_by_role("tab", name="Compare")).to_be_visible()
    expect(chromium_page.get_by_role("tab", name="Review")).to_be_visible()
    expect(chromium_page.get_by_role("tab", name="Settings")).to_be_visible()

    expect(chromium_page.locator("#theme-toggle")).to_have_accessible_name(re.compile(r"Switch to (light|dark) theme"))
    expect(chromium_page.locator("#refresh-all")).to_have_accessible_name("Refresh workspace")

    _open_folder_browser(chromium_page)
    expect(chromium_page.locator("#folder-browser")).to_have_accessible_name("Select a Directory")
    expect(chromium_page.locator("#browser-path")).to_have_accessible_name("Current directory")
    chromium_page.keyboard.press("Escape")


def test_review_keep_shortcut_matches_batch_scope_when_selection_exists(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _open_review_tab(chromium_page)

    first_row = chromium_page.locator("#queue-list .queue-item").first
    first_title = first_row.locator(".queue-file").inner_text()
    second_row = chromium_page.locator("#queue-list .queue-item").nth(1)
    second_title = second_row.locator(".queue-file").inner_text()

    first_row.get_by_role("checkbox", name=f"Select {first_title}").click()
    second_row.get_by_role("button", name=f"Open details for {second_title}").click()
    expect(chromium_page.locator("#batch-export-mark")).to_be_enabled()
    chromium_page.evaluate("() => document.activeElement?.blur()")
    chromium_page.keyboard.press("s")

    expect(first_row.locator(".badge-select")).to_be_visible()
    expect(chromium_page.locator("#detail-title")).to_have_text(second_title)


def test_review_action_buttons_expose_current_scope_to_assistive_tech(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _open_review_tab(chromium_page)

    keep_button = chromium_page.locator("#batch-export-mark")
    expect(keep_button).to_have_accessible_description("Current photo")

    first_row = chromium_page.locator("#queue-list .queue-item").first
    first_title = first_row.locator(".queue-file").inner_text()
    first_row.get_by_role("checkbox", name=f"Select {first_title}").click()

    expect(keep_button).to_have_accessible_description("Selected photos")
