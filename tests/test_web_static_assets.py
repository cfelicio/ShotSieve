"""Tests for static asset headers, HTML structure, CSS rules, and JS behavior."""
from __future__ import annotations

from html.parser import HTMLParser
from urllib.request import urlopen


def test_web_removes_compare_progress_compatibility_export() -> None:
    from shotsieve import web as web_module

    assert not hasattr(web_module, "CompareProgress")



class _SectionHeadingParser(HTMLParser):
    def __init__(self, *, section_id: str) -> None:
        super().__init__()
        self.section_id = section_id
        self._section_depth = 0
        self._current_heading_tag: str | None = None
        self._current_heading_text: list[str] = []
        self.headings: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = dict(attrs)
        if tag == "section":
            if self._section_depth == 0:
                if attr_map.get("id") == self.section_id:
                    self._section_depth = 1
            else:
                self._section_depth += 1

        if self._section_depth == 0:
            return

        if tag in {"h2", "h3"}:
            self._current_heading_tag = tag
            self._current_heading_text = []

    def handle_endtag(self, tag: str) -> None:
        if self._section_depth == 0:
            return

        if self._current_heading_tag == tag:
            heading_text = "".join(self._current_heading_text).strip()
            self.headings.append((tag, heading_text))
            self._current_heading_tag = None
            self._current_heading_text = []

        if tag == "section":
            self._section_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._current_heading_tag is not None:
            self._current_heading_text.append(data)


class _ElementIdParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.batch_action_button_ids: list[str] = []
        self._batch_actions_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attr_map = dict(attrs)
        element_id = attr_map.get("id")
        if element_id:
            self.ids.add(element_id)

        class_tokens = set(str(attr_map.get("class") or "").split())
        if tag == "div" and "batch-actions" in class_tokens:
            self._batch_actions_depth += 1

        if self._batch_actions_depth > 0 and tag == "button" and element_id:
            self.batch_action_button_ids.append(element_id)

    def handle_endtag(self, tag: str) -> None:
        if tag == "div" and self._batch_actions_depth > 0:
            self._batch_actions_depth -= 1


def _extract_media_block(body: str, media_query: str) -> str:
    media_start = body.index(media_query)
    block_start = body.index("{", media_start)
    depth = 0

    for index in range(block_start, len(body)):
        if body[index] == "{":
            depth += 1
        elif body[index] == "}":
            depth -= 1
            if depth == 0:
                return body[media_start : index + 1]

    raise AssertionError(f"Could not extract full media block for {media_query!r}")


class TestStaticAssetHeaders:
    @staticmethod
    def _combined_js(base_url: str) -> str:
        parts = [
            urlopen(f"{base_url}/app-state.js").read().decode("utf-8"),
            urlopen(f"{base_url}/app.js").read().decode("utf-8"),
            urlopen(f"{base_url}/app-utils.js").read().decode("utf-8"),
            urlopen(f"{base_url}/app-busy.js").read().decode("utf-8"),
            urlopen(f"{base_url}/app-review.js").read().decode("utf-8"),
            urlopen(f"{base_url}/app-workflow-polling.js").read().decode("utf-8"),
            urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8"),
            urlopen(f"{base_url}/app-events.js").read().decode("utf-8"),
        ]
        return "\n".join(parts)

    @staticmethod
    def _combined_css(base_url: str) -> str:
        parts = [
            urlopen(f"{base_url}/styles.css").read().decode("utf-8"),
            urlopen(f"{base_url}/styles-layout.css").read().decode("utf-8"),
            urlopen(f"{base_url}/styles-polish.css").read().decode("utf-8"),
        ]
        return "\n".join(parts)

    def test_static_html_includes_cache_control(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        headers = response.headers
        assert "Cache-Control" in headers
        assert "must-revalidate" in headers["Cache-Control"]

    def test_static_html_includes_basic_security_headers(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        headers = response.headers
        assert headers["X-Content-Type-Options"] == "nosniff"
        assert headers["X-Frame-Options"] == "DENY"

    def test_static_html_exposes_accessible_tabs(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'role="tablist"' in body
        assert 'id="tab-workspace-button"' in body
        assert 'id="tab-compare-button"' in body
        assert 'role="tab"' in body
        assert 'aria-controls="tab-workspace"' in body
        assert 'aria-controls="tab-compare"' in body
        assert 'aria-selected="true"' in body
        assert 'role="tabpanel"' in body
        assert 'aria-labelledby="tab-workspace-button"' in body
        assert 'aria-labelledby="tab-compare-button"' in body
        assert 'id="tab-review" role="tabpanel"' in body
        assert 'id="tab-compare" role="tabpanel"' in body
        assert 'id="tab-compare-button" class="tab-button" data-tab="compare"' in body
        assert 'id="tab-compare-button" class="tab-button" data-tab="compare" type="button" role="tab" aria-controls="tab-compare" aria-selected="false" tabindex="-1"' in body

    def test_static_html_labels_dialogs_and_lightbox_only(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'id="folder-browser" class="folder-dialog" aria-labelledby="folder-browser-title"' in body
        assert 'id="export-dialog" class="folder-dialog" aria-labelledby="export-dialog-title"' in body
        assert 'dialog id="compare-overlay"' not in body
        assert 'id="compare-close"' not in body
        assert 'id="compare-pick-left"' not in body
        assert 'id="compare-pick-right"' not in body
        assert 'dialog id="lightbox-overlay" class="lightbox-overlay overlay-closed" role="dialog" aria-modal="true" aria-label="Photo preview"' in body
        assert 'id="lightbox-close" type="button" class="lightbox-close" aria-label="Close lightbox" autofocus' in body

    def test_static_html_folder_browser_has_prominent_choose_folder_cta(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")
        assert '<div class="dialog-footer-actions">' in body
        assert 'id="browser-choose" type="button" class="primary-action">Choose Folder</button>' in body

    def test_static_html_labels_icon_only_header_actions(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'id="theme-toggle" type="button" class="ghost icon-btn" aria-label="Switch theme"' in body
        assert 'id="refresh-all" type="button" class="ghost icon-btn" data-busy-lock="true" aria-label="Refresh workspace"' in body

    def test_static_html_library_tab_has_level_two_heading_before_analysis_subsections(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")
        parser = _SectionHeadingParser(section_id="tab-workspace")
        parser.feed(body)

        assert parser.headings[0] == ("h2", "Library")
        assert ("h3", "File Discovery") in parser.headings
        assert ("h3", "Quality Scoring") in parser.headings
        assert 'id="preview-mode-select"' not in body
        assert "RAW Preview Quality" not in body
        assert "RAW previews use Auto quality by default." not in body

    def test_static_html_review_tab_has_stable_workspace_and_subsection_headings(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")
        parser = _SectionHeadingParser(section_id="tab-review")
        parser.feed(body)

        assert ("h3", "Filters & Sorting") not in parser.headings
        assert ("h3", "Batch Actions") in parser.headings

    def test_static_html_detail_preview_exposes_explicit_dialog_trigger(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")
        assert 'id="detail-image" alt="Selected photo preview" role="button" tabindex="0" aria-label="Open selected photo in lightbox"' in body
        assert 'id="detail-modelline"' not in body
        assert 'id="detail-open-lightbox"' not in body
        assert 'class="detail-media-actions"' not in body

    def test_static_html_review_tab_uses_issue_filter_and_reordered_detail_meta(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")

        assert 'id="queue-count"' not in body
        assert 'id="auto-advance-toggle"' not in body
        assert 'id="detail-issues"' not in body
        assert 'id="issues-filter"' in body
        assert '>Issues Only</option>' in body
        assert body.index('id="review-position"') < body.index('id="detail-scoreline"')
        assert body.index('id="detail-path"') < body.index('id="detail-scoreline"')

    def test_static_html_compare_tab_includes_extreme_filter_control(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")

        assert 'id="compare-row-filter"' in body
        assert '>All photos</option>' in body
        assert '>Highest &amp; lowest</option>' in body

    def test_static_html_review_tab_avoids_duplicate_refresh_labels(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")

        assert 'id="refresh-all" type="button" class="ghost icon-btn" data-busy-lock="true" aria-label="Refresh workspace"' in body
        assert 'id="refresh-button"' not in body
        assert '>Reset Filters</button>' in body
        assert '>Clear Mark</button>' in body

    def test_static_html_trims_redundant_compare_and_review_helper_copy(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")

        assert "Model Lab" not in body
        assert "Side By Side" not in body
        assert "Filters update automatically." not in body

    def test_static_css_review_detail_media_shell_does_not_force_oversized_min_height(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/styles.css").read().decode("utf-8")

        assert "min-height: clamp(560px, 78vh, 1100px);" not in body
        assert ".detail-issues-note" in body

    def test_static_html_uses_non_scissor_favicon(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert '<link rel="icon"' in body
        assert "data:image/svg+xml" in body
        assert "✂️" not in body

    def test_static_html_loads_app_utils_before_main_app_script(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert '<script src="/app-utils.js"></script>' in body
        assert '<script src="/app-review.js"></script>' in body
        assert '<script src="/app-workflows.js"></script>' in body
        assert '<script src="/app-events.js"></script>' in body
        assert '<script src="/app.js"></script>' in body

    def test_static_html_loads_frontend_state_busy_and_polling_modules_before_app_shell(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")

        state_index = body.index('<script src="/app-state.js"></script>')
        utils_index = body.index('<script src="/app-utils.js"></script>')
        busy_index = body.index('<script src="/app-busy.js"></script>')
        review_index = body.index('<script src="/app-review.js"></script>')
        polling_index = body.index('<script src="/app-workflow-polling.js"></script>')
        workflows_index = body.index('<script src="/app-workflows.js"></script>')
        events_index = body.index('<script src="/app-events.js"></script>')
        main_index = body.index('<script src="/app.js"></script>')

        assert state_index < utils_index < busy_index < review_index < polling_index < workflows_index < events_index < main_index

    def test_static_html_review_tab_uses_simplified_actions(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        parser = _ElementIdParser()
        parser.feed(body)

        assert "review-position" in parser.ids
        assert "auto-advance-toggle" not in parser.ids
        assert parser.batch_action_button_ids == [
            "batch-export-mark",
            "batch-delete-mark",
            "batch-move",
            "batch-delete-disk",
            "batch-clear-mark",
        ]
        assert "batch-remove-cache" not in parser.ids
        assert "mark-export" not in parser.ids
        assert "mark-delete" not in parser.ids
        assert "mark-export-next" not in parser.ids
        assert "mark-delete-next" not in parser.ids
        assert "clear-marks" not in parser.ids

    def test_static_html_omits_hidden_compatibility_containers(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/").read().decode("utf-8")
        parser = _ElementIdParser()
        parser.feed(body)

        for removed_id in (
            "workflow-profile",
            "scan-runs",
            "root-shortcuts",
            "action-log",
            "browse-preview-dir",
        ):
            assert removed_id not in parser.ids

        assert "hidden-filter" not in body

    def test_static_html_review_shortcuts_only_show_keep_reject_and_navigation(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert "<kbd>Arrows</kbd> Navigate" in body
        assert "<kbd>S</kbd> Keep" in body
        assert "<kbd>R</kbd> Reject" in body
        assert "Undecided" not in body
        assert "Close View" not in body

    def test_static_html_review_sort_prioritizes_ai_score_terms(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'value="learned_asc">Lowest Score</option>' in body
        assert 'value="learned_desc">Highest Score</option>' in body
        assert "Lowest AI Score" not in body
        assert "Highest AI Score" not in body
        assert 'value="score_asc"' not in body
        assert 'value="score_desc"' not in body

    def test_static_html_review_batch_bar_removes_compare_and_export_noise(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'id="batch-compare"' not in body
        assert 'id="batch-export"' not in body
        assert 'id="batch-move"' in body
        assert 'id="batch-delete-disk"' in body

    def test_static_html_compare_tab_includes_dedicated_progress_ui(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'id="compare-busy-panel"' in body
        assert 'id="compare-busy-indicator"' in body
        assert 'id="compare-busy-progress"' in body
        assert 'id="compare-busy-progress-fill"' in body

    def test_static_html_library_and_compare_use_single_progress_bar(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'id="busy-indicator"' in body
        assert 'id="busy-progress"' in body
        assert 'id="compare-busy-indicator"' in body
        assert 'id="compare-busy-progress"' in body
        assert 'id="busy-phase-indicator"' not in body
        assert 'id="busy-phase-progress"' not in body
        assert 'id="compare-busy-phase-indicator"' not in body
        assert 'id="compare-busy-phase-progress"' not in body

    def test_static_js_busy_state_uses_single_progress_bar_with_phase_message_and_no_eta(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "Phase " in body
        assert "ETA " not in body
        assert "setBusyPhaseProgress" in body
        assert "busy-phase-progress" not in body
        assert "compare-busy-phase-progress" not in body

    def test_static_js_compare_progress_uses_three_phases_and_simplified_copy(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "phaseCount: 3" in body
        assert "Model scoring" in body
        assert " · this model " not in body

    def test_static_js_compare_polling_always_fetches_result_payload_after_completion(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "fetchResult: fetchCompareJobResult" in body
        assert "return await fetchResult(jobId);" in body
        assert "return status.summary;" not in body

    def test_static_js_compare_summary_ignores_null_scores_in_averages(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "ShotSieveUtils.comparisonScoreNumber(row[`${modelName}_score`])" in body
        assert ".filter((value) => value !== null)" in body

    def test_static_js_compare_sort_treats_null_scores_as_missing(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/app-utils.js").read().decode("utf-8")
        assert "function comparisonScoreNumber(value)" in body
        assert "const leftValue = comparisonScoreNumber(left[scoreKey]);" in body
        assert "const leftMissing = leftValue === null;" in body
        assert "const rightMissing = rightValue === null;" in body

    def test_static_compare_tab_does_not_render_active_library_root_card(self, test_server):
        base_url, _, _ = test_server
        html_body = urlopen(f"{base_url}/").read().decode("utf-8")
        assert 'id="compare-root-value"' not in html_body
        assert 'class="compare-root-card"' not in html_body

        js_body = self._combined_js(base_url)
        assert "syncCompareRootLabel" not in js_body

    def test_static_html_compare_tab_includes_sort_controls(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'id="compare-row-sort"' in body
        assert 'id="compare-results-warning" class="compare-results-warning hidden"' in body
        assert 'id="compare-model-order"' not in body

    def test_static_html_settings_remove_advanced_model_controls(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/")
        body = response.read().decode("utf-8")
        assert 'id="advanced-settings-panel"' not in body
        assert 'id="advanced-reset-defaults"' not in body
        assert 'id="advanced-enable-compare"' not in body
        assert 'id="advanced-enable-musiq"' not in body
        assert 'id="advanced-enable-maniqa"' not in body

    def test_static_css_includes_content_length(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/styles.css")
        assert response.headers.get("Content-Length") is not None
        assert int(response.headers["Content-Length"]) > 0

    def test_static_css_supports_light_background_tokens_and_reduced_motion(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert "--page-glow:" in body
        assert "--page-ambient:" in body
        assert "--page-base:" in body
        assert "background:" in body
        assert "var(--page-glow)" in body
        assert "var(--page-ambient)" in body
        assert "var(--page-base)" in body
        assert "@media (prefers-reduced-motion: reduce)" in body

    def test_static_css_light_theme_overrides_card_surface_tokens(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert "[data-theme=\"light\"]" in body
        assert "--card-surface:" in body
        assert "--glass-border:" in body

    def test_static_css_honors_hidden_attribute_for_topbar_tabs(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert ".tab-button[hidden]" in body
        assert "display: none !important;" in body

    def test_static_css_hides_closed_lightbox_overlay(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert "dialog.compare-overlay:not([open])" not in body
        assert "dialog.lightbox-overlay:not([open])" in body
        assert "display: none;" in body

    def test_static_css_light_theme_reduces_overlay_haze(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert "[data-theme=\"light\"] .grain" in body
        assert "[data-theme=\"light\"] .bench-glow" in body
        assert "opacity:" in body

    def test_static_css_review_tab_applies_density_and_readability_refinements(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert ".review-sidebar > div" in body
        assert "position: sticky;" in body
        assert ".review-filter-header" in body
        assert ".review-layout" in body
        assert "grid-template-columns: minmax(340px, 30vw) minmax(0, 1fr);" in body
        assert "grid-template-columns: minmax(360px, 28vw) minmax(0, 1fr);" in body
        assert ".queue-list" in body
        assert "padding: 6px 6px 12px;" in body
        assert ".detail-toolbar" in body
        assert "line-height: 1.2;" in body

    def test_static_css_uses_documented_mobile_first_breakpoints(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert "@media (min-width: 48rem)" in body
        assert "@media (min-width: 64rem)" in body
        assert "@media (min-width: 75rem)" in body
        assert "@media (max-width:" not in body

    def test_static_split_css_files_keep_viewport_layout_rules_out_of_polish_sheet(self, test_server):
        base_url, _, _ = test_server
        layout_body = urlopen(f"{base_url}/styles-layout.css").read().decode("utf-8")
        polish_body = urlopen(f"{base_url}/styles-polish.css").read().decode("utf-8")

        assert "Contract: page layout, dialogs, and responsive breakpoints." in layout_body
        assert "Contract: visual polish, emphasis, and motion." in polish_body
        assert "@media (min-width: 48rem)" in layout_body
        assert "@media (min-width:" not in polish_body
        assert "@media (max-width:" not in polish_body

        for selector, expected_property in (
            (".shell {", "padding-bottom: 80px;"),
            (".topbar {", "flex-wrap: wrap;"),
            (".topbar-nav {", "overflow-x: auto;"),
            (".topbar-stats {", "width: 100%;"),
            (".library-layout {", "max-width: 800px;"),
            (".compare-model-grid {", "grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));"),
            (".review-layout {", "grid-template-columns: 1fr;"),
            (".detail-grid {", "grid-template-columns: minmax(0, 1fr);"),
            (".split-row {", "grid-template-columns: 1fr;"),
            (".sidebar-scroller {", "overflow: visible;"),
        ):
            assert selector in layout_body
            selector_index = layout_body.index(selector)
            selector_block = layout_body[selector_index : selector_index + 320]
            assert expected_property in selector_block

    def test_static_css_shortcut_strip_uses_theme_variable_background(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert ".shortcut-strip" in body
        assert "--shortcut-strip-bg:" in body
        assert "background: var(--shortcut-strip-bg);" in body

    def test_static_css_toast_region_is_offset_above_shortcut_strip(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert "--shortcut-strip-height:" in body
        assert ".toast-region" in body
        assert "bottom: calc(var(--shortcut-strip-height) + 18px);" in body

    def test_static_css_coarse_pointer_mode_expands_checkbox_and_toggle_hit_targets(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/styles-layout.css").read().decode("utf-8")

        media_block = _extract_media_block(body, "@media (pointer: coarse)")

        for selector in (
            ".checkbox-row",
            ".queue-select",
        ):
            assert selector in media_block

        assert "min-height: 44px;" in media_block

    def test_static_css_compare_cards_use_large_top_aligned_preview(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        card_start = body.index(".compare-result-card {")
        card_block = body[card_start : card_start + 260]
        assert "display: grid;" in card_block
        assert "grid-template-columns: 1fr;" in card_block

        photo_start = body.index(".compare-photo {")
        photo_block = body[photo_start : photo_start + 260]
        assert "height: clamp(" in photo_block

    def test_static_css_folder_browser_keeps_choose_folder_cta_visible_on_small_screens(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_css(base_url)
        assert ".dialog-shell" in body
        assert "grid-template-rows:" in body
        assert ".dialog-footer-actions" in body
        assert "position: sticky;" in body
        assert "env(safe-area-inset-bottom" in body

    def test_static_js_includes_content_length(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/app.js")
        assert response.headers.get("Content-Length") is not None
        assert int(response.headers["Content-Length"]) > 0

    def test_static_utils_js_includes_content_length(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/app-utils.js")
        assert response.headers.get("Content-Length") is not None
        assert int(response.headers["Content-Length"]) > 0

    def test_static_review_js_includes_content_length(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/app-review.js")
        assert response.headers.get("Content-Length") is not None
        assert int(response.headers["Content-Length"]) > 0

    def test_static_events_js_includes_content_length(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/app-events.js")
        assert response.headers.get("Content-Length") is not None
        assert int(response.headers["Content-Length"]) > 0

    def test_static_workflows_js_includes_content_length(self, test_server):
        base_url, _, _ = test_server
        response = urlopen(f"{base_url}/app-workflows.js")
        assert response.headers.get("Content-Length") is not None
        assert int(response.headers["Content-Length"]) > 0

    def test_static_frontend_support_modules_include_content_length(self, test_server):
        base_url, _, _ = test_server
        for route in ("/app-state.js", "/app-busy.js", "/app-workflow-polling.js"):
            response = urlopen(f"{base_url}{route}")
            assert response.headers.get("Content-Length") is not None
            assert int(response.headers["Content-Length"]) > 0

    def test_static_split_css_files_include_content_length(self, test_server):
        base_url, _, _ = test_server
        for route in ("/styles-layout.css", "/styles-polish.css"):
            response = urlopen(f"{base_url}{route}")
            assert response.headers.get("Content-Length") is not None
            assert int(response.headers["Content-Length"]) > 0

    def test_static_js_defaults_theme_from_system_preference(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "window.matchMedia(\"(prefers-color-scheme: dark)\")" in body
        assert "const preferredTheme" in body

    def test_static_js_app_shell_delegates_state_and_busy_control_to_support_modules(self, test_server):
        base_url, _, _ = test_server
        app_body = urlopen(f"{base_url}/app.js").read().decode("utf-8")
        state_body = urlopen(f"{base_url}/app-state.js").read().decode("utf-8")
        busy_body = urlopen(f"{base_url}/app-busy.js").read().decode("utf-8")
        polling_body = urlopen(f"{base_url}/app-workflow-polling.js").read().decode("utf-8")

        assert "window.ShotSieveState" in state_body
        assert "window.ShotSieveBusy" in busy_body
        assert "window.ShotSieveWorkflowPolling" in polling_body
        assert "const state = {" not in app_body
        assert "function setBusy(" not in app_body
        assert "async function withBusy(" not in app_body

    def test_static_js_omits_hidden_review_compare_overlay_logic(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "function openComparison()" not in body
        assert "Select exactly 2 items to compare." not in body
        assert 'closeOverlay("compare-overlay")' not in body
        assert '"#compare-overlay"' not in body

    def test_static_js_compare_includes_estimate_and_runtime_metrics(self, test_server):
        base_url, _, _ = test_server
        combined_body = self._combined_js(base_url)
        assert "/api/compare-estimate" in combined_body
        assert "model_timings_seconds" in combined_body
        assert "elapsed_seconds" in combined_body
        assert "files/s" in combined_body

    def test_static_js_compare_auto_runs_scan_only_when_cache_missing(self, test_server):
        base_url, _, _ = test_server
        combined_body = self._combined_js(base_url)
        assert "No cached photos found under this library root. Running Scan first." in combined_body
        compare_prereq_index = combined_body.index("No cached photos found under this library root. Running Scan first.")
        compare_prereq_block = combined_body[compare_prereq_index : compare_prereq_index + 1200]
        assert "await runScan(root, {" in compare_prereq_block
        assert "await runScore(root" not in compare_prereq_block

    def test_static_js_compare_surfaces_long_running_model_init_status(self, test_server):
        base_url, _, _ = test_server
        combined_body = self._combined_js(base_url)
        assert "Loading model weights" in combined_body
        assert "first run may take a while" in combined_body
        assert "elapsed" in combined_body

    def test_static_js_compare_uses_async_job_polling_routes(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "/api/compare-models/start" in body
        assert "/api/compare-models/status" in body
        assert "/api/compare-models/result" in body
        assert "compareJobId" in body

    def test_static_js_compare_supports_row_sorting_without_model_reordering(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "compare-row-sort" in body
        assert "compare-model-order" not in body
        assert "slowest" not in body

    def test_static_js_uses_state_catalog_module_and_direct_model_utility_calls(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        app_body = urlopen(f"{base_url}/app.js").read().decode("utf-8")

        assert 'const DEFAULT_MODEL_CATALOG = ["topiq_nr", "clipiqa", "qalign"]' in body
        assert 'const HIDDEN_MODEL_NAMES = ["arniqa", "arniqa-spaq"]' in body
        assert "availableLearnedModelsUtil(options, DEFAULT_MODEL_CATALOG, HIDDEN_MODEL_NAMES)" in app_body
        assert "function availableLearnedModels(options)" not in app_body
        assert "advanced-reset-defaults" not in body
        assert "applyRecommendedDefaults" not in body
        assert "applyCompareTabVisibility" not in body

    def test_static_js_scan_only_uses_fast_metadata_path(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "async function runScan(rootOverride = null, { generatePreviews = true, pipeline = null } = {})" in body
        assert "Scanning metadata only for faster discovery" in body
        assert "runScan(null, { generatePreviews: false })" in body
        assert 'preview_mode:' not in body

    def test_static_js_uses_backend_default_raw_preview_mode_without_selector(self, test_server):
        base_url, _, _ = test_server
        state_body = urlopen(f"{base_url}/app-state.js").read().decode("utf-8")
        app_body = urlopen(f"{base_url}/app.js").read().decode("utf-8")
        workflows_body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")
        events_body = urlopen(f"{base_url}/app-events.js").read().decode("utf-8")

        assert 'previewMode:' not in state_body
        assert 'currentPreviewMode' not in workflows_body
        assert 'preview_mode:' not in workflows_body
        assert 'preview-mode-select' not in events_body
        assert 'const reviewRoot = syncReviewRoot(root) || root;' in workflows_body
        assert 'rootFilter.add(new Option(root, root, true, true));' in workflows_body
        assert 'replace(/^(?:\\.\\/|~\\/)+/, "")' in app_body
        assert 'rootFilter.add(new Option(previous, previous, false, true));' in app_body

    def test_static_js_scan_uses_async_job_polling_routes(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "/api/scan/start" in body
        assert "/api/scan/status" in body
        assert "/api/scan/result" in body
        assert "scanJobId" in body

    def test_static_js_cancel_requests_server_side_job_cancellation(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "/api/scan/cancel" in body
        assert "/api/score/cancel" in body
        assert "/api/compare-models/cancel" in body

    def test_static_js_uses_async_job_polling_for_delete_export_and_cache_actions(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "/api/files/delete/start" in body
        assert "/api/files/export/start" in body
        assert "/api/cache/clear/start" in body
        assert "/api/operations/status" in body
        assert "/api/operations/result" in body
        assert "/api/operations/cancel" in body

    def test_static_js_uses_review_state_route_for_marked_batch_actions(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")
        assert "/api/review/file-ids" in body

    def test_static_js_cancel_button_disables_and_shows_cancelling_state(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "cancelPending" in body
        assert "Cancelling..." in body
        assert "cancelBtn.disabled = state.cancelPending" in body

    def test_static_js_bulk_actions_use_filter_backed_selection_for_all_matching(self, test_server):
        base_url, _, _ = test_server
        app_body = urlopen(f"{base_url}/app.js").read().decode("utf-8")
        workflows_body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")

        assert "bulkSelection" in app_body
        assert "selection:" in workflows_body
        assert "allIds.forEach((id) => state.selectedIds.add(id));" not in app_body

    def test_static_js_bulk_actions_require_effective_selected_count(self, test_server):
        base_url, _, _ = test_server
        events_body = urlopen(f"{base_url}/app-events.js").read().decode("utf-8")
        review_body = urlopen(f"{base_url}/app-review.js").read().decode("utf-8")
        workflows_body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")

        assert "Math.max(0, Number(state.bulkSelection.count || 0) - excludedCount)" in events_body
        assert "Math.max(0, Number(state.bulkSelection.count || 0) - excludedIds.length)" in workflows_body
        assert 'return "No photo selected";' in review_body

    def test_static_js_analyze_uses_fast_metadata_scan_before_scoring(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert 'setBusyMessage("Fast scan: indexing files without preview generation...")' in body
        assert "pipeline: { stepIndex: 1, totalSteps: 3 }" in body
        assert "pipeline: { stepIndex: 2, totalSteps: 3 }" in body

    def test_static_js_compare_model_cards_include_model_specific_descriptions(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert 'modelDescriptions[modelName] || "No detailed notes available for this model."' in body
        assert "Best all-rounder and recommended starting point" in body

    def test_static_js_compare_omits_confidence_na_placeholder(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "confidence n/a" not in body

    def test_static_js_compare_keeps_row_and_setup_failure_paths_wired(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")
        assert "comparisonFailureText(row, modelName)" in body
        assert "comparison.compare_failures" in body
        assert 'gallery.innerHTML = ""' in body

    def test_static_js_compare_warns_when_results_are_truncated(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")
        assert "comparison.truncated" in body
        assert "Comparing first ${processedRowsText} of ${requestedRowsText} files." in body
        assert "Narrow the root or apply filters for a full compare." in body

    def test_static_js_settings_surfaces_detailed_acceleration_info(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "Available Accelerators" in body
        assert "Auto Mode Priority" in body
        assert "Auto mode picks the first available runtime" in body

    def test_static_js_settings_omit_xpu_packaging_note(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "Intel XPU" not in body
        assert "Intel XPU remains source-install only today" not in body
        assert "not one of the packaged runtime downloads" not in body

    def test_static_html_includes_runtime_model_warning_placeholder(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/index.html").read().decode("utf-8")
        assert 'id="runtime-model-warning"' in body

    def test_static_js_surfaces_qalign_cpu_and_directml_unavailable_notice(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "Q-Align is unavailable for the active runtime" in body
        assert "Use TOPIQ or CLIPIQA" in body
        assert "works on CPU but is VERY slow there" not in body
        assert "runtime-model-warning" in body
        assert "default_runtime" in body

    def test_static_js_compare_defaults_to_topiq_sort_and_extreme_filter_support(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")

        assert 'state.compareRowSort = rowChoices[0]?.value || "input"' not in body
        assert 'topiq_nr:desc' in body
        assert 'compare-row-filter' in body
        assert 'extremes' in body

    def test_static_js_scopes_compare_progress_to_compare_operation(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "compare-busy-panel" in body
        assert "activeOperation" in body
        assert "operationType: \"compare\"" in body

    def test_static_js_error_toasts_are_sticky_until_dismissed(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "if (tone !== \"error\")" in body
        assert "toast-close" in body
        assert "node.remove()" in body

    def test_static_js_review_does_not_force_default_max_score_cutoff(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert 'document.getElementById("max-score").value = "50";' not in body

    def test_static_review_js_uses_ai_label_without_overall_score_cards(self, test_server):
        base_url, _, _ = test_server
        body = urlopen(f"{base_url}/app-review.js").read().decode("utf-8")
        assert "AI score" in body
        assert 'scoreCard("Overall"' not in body
        assert "Overall ${" not in body
        assert "detail-issues-note" in body

    def test_static_review_open_file_uses_local_open_endpoint(self, test_server):
        base_url, _, _ = test_server
        review_body = urlopen(f"{base_url}/app-review.js").read().decode("utf-8")
        workflows_body = urlopen(f"{base_url}/app-workflows.js").read().decode("utf-8")

        assert "openOriginalFile(detail.id)" in review_body
        assert "event.preventDefault();" in review_body
        assert 'postJson("/api/files/open", { file_id: Number(fileId) })' in workflows_body

    def test_static_js_compare_cards_use_header_rows_for_alignment(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "compare-model-card-head" in body
        assert "compare-model-card-copy" in body

    def test_static_js_review_auto_advance_handles_page_boundaries(self, test_server):
        base_url, _, _ = test_server
        body = self._combined_js(base_url)
        assert "shouldAdvancePage" in body
        assert "state.page += 1;" in body
        assert "await loadQueue();" in body

