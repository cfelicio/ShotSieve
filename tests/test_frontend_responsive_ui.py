from __future__ import annotations


import pytest

from test_frontend_accessibility import (
    RESPONSIVE_VIEWPORTS,
    _assert_no_horizontal_overflow,
    _assert_touch_target_floor,
    _assert_within_viewport,
    _open_compare_tab,
    _open_folder_browser,
    _open_review_tab,
    _open_settings_tab,
    _set_viewport,
)


def test_mobile_review_uses_toolbar_navigation_without_bottom_overlay(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _set_viewport(chromium_page, width=390, height=844)
    _open_review_tab(chromium_page)

    detail_prev = chromium_page.locator("#previous-item")
    detail_next = chromium_page.locator("#next-item")
    detail_image = chromium_page.locator("#detail-image")

    expect(chromium_page.locator("#compact-review-nav")).to_have_count(0)
    expect(detail_prev).to_be_visible()
    expect(detail_next).to_be_visible()
    expect(detail_prev).to_be_disabled()
    expect(detail_next).to_be_enabled()

    detail_image.focus()
    detail_next.click()

    expect(chromium_page.locator("#review-position")).to_have_text("2 of 3")
    expect(detail_prev).to_be_enabled()
    expect(detail_next).to_be_enabled()


def test_narrow_phone_prioritizes_compact_nav_space_and_larger_targets(chromium_page) -> None:
    chromium_page, expect = chromium_page
    _set_viewport(chromium_page, width=320, height=700)
    _open_review_tab(chromium_page)

    detail_prev = chromium_page.locator("#previous-item")
    detail_next = chromium_page.locator("#next-item")
    shortcut_strip = chromium_page.locator(".shortcut-strip")

    expect(chromium_page.locator("#compact-review-nav")).to_have_count(0)
    expect(detail_prev).to_be_visible()
    expect(detail_next).to_be_visible()
    expect(shortcut_strip).to_be_visible()

    visible_shortcuts = chromium_page.evaluate(
        """
        () => [...document.querySelectorAll('.shortcut-strip .shortcut-item')]
          .filter((node) => node.getClientRects().length > 0)
          .map((node) => node.textContent?.trim() || '')
        """
    )
    assert visible_shortcuts == ["Arrows Navigate"]


def test_mobile_touch_targets_expand_queue_hit_areas(mobile_chromium_page) -> None:
    chromium_page, _ = mobile_chromium_page

    assert chromium_page.evaluate("() => window.matchMedia('(pointer: coarse)').matches") is True
    _assert_touch_target_floor(chromium_page, ".checkbox-row", label="Recursive scan toggle")

    _open_review_tab(chromium_page)

    _assert_touch_target_floor(chromium_page, "#queue-list .queue-item .queue-select", label="Review queue checkbox target")


@pytest.mark.parametrize("viewport", RESPONSIVE_VIEWPORTS)
def test_responsive_layout_avoids_horizontal_overflow_and_keeps_tab_actions_visible(
    chromium_page,
    viewport: dict[str, int],
) -> None:
    chromium_page, _ = chromium_page
    _set_viewport(chromium_page, **viewport)

    chromium_page.get_by_role("tab", name="Library").click()
    _assert_no_horizontal_overflow(chromium_page, label=f"Library tab at {viewport['width']}px")
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#analyze-library"),
        label=f"Analyze button at {viewport['width']}px",
    )

    _open_compare_tab(chromium_page)
    _assert_no_horizontal_overflow(chromium_page, label=f"Compare tab at {viewport['width']}px")
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#compare-run"),
        label=f"Compare run button at {viewport['width']}px",
    )

    _open_review_tab(chromium_page)
    _assert_no_horizontal_overflow(chromium_page, label=f"Review tab at {viewport['width']}px")
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#batch-export-mark"),
        label=f"Review keep button at {viewport['width']}px",
    )

    _open_settings_tab(chromium_page)
    _assert_no_horizontal_overflow(chromium_page, label=f"Settings tab at {viewport['width']}px")
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#tab-settings h2").first,
        label=f"Settings heading at {viewport['width']}px",
    )


@pytest.mark.parametrize("viewport", RESPONSIVE_VIEWPORTS)
def test_responsive_dialog_flows_fit_inside_viewport(chromium_page, viewport: dict[str, int]) -> None:
    chromium_page, _ = chromium_page
    _set_viewport(chromium_page, **viewport)

    chromium_page.get_by_role("tab", name="Library").click()
    _open_folder_browser(chromium_page)
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#folder-browser"),
        label=f"Folder browser dialog at {viewport['width']}px",
    )
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#browser-choose"),
        label=f"Choose Folder button at {viewport['width']}px",
    )
    chromium_page.locator("#folder-browser button[type='submit']").click()
    chromium_page.wait_for_function("() => document.getElementById('folder-browser')?.open === false")

    _open_review_tab(chromium_page)
    first_row = chromium_page.locator("#queue-list .queue-item").first
    first_filename = first_row.locator(".queue-file").inner_text()
    first_row.get_by_role("checkbox", name=f"Select {first_filename}").click()
    chromium_page.locator("#batch-move").click()
    chromium_page.wait_for_function("() => document.getElementById('export-dialog')?.open === true")
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#export-dialog"),
        label=f"Export dialog at {viewport['width']}px",
    )
    _assert_within_viewport(
        chromium_page,
        chromium_page.locator("#export-confirm"),
        label=f"Export confirmation button at {viewport['width']}px",
    )
