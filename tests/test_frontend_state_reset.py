from __future__ import annotations

from urllib.request import urlopen

from test_frontend_accessibility import _open_review_tab


def test_reset_everything_clears_persisted_ui_state(frontend_server: str) -> None:
    state_body = urlopen(f"{frontend_server}/app-state.js").read().decode("utf-8")
    events_body = urlopen(f"{frontend_server}/app-events.js").read().decode("utf-8")

    assert "function clearUiState(" in state_body
    assert "storage.removeItem(UI_STATE_KEY);" in state_body
    assert "clearUiState({ immediate: true });" in events_body
    assert '"clear-all-cache"' in events_body
    assert "onSuccess: resetPersistedUiStateAfterFullReset" in events_body
    assert 'localStorage.removeItem("shotsieve_resource_profile");' in events_body
    assert 'profileSelect.value = "normal";' in events_body
    assert 'previewModeSelect.value = state.options?.default_preview_mode || "auto";' not in events_body


def test_ui_state_is_scoped_to_database_marker(frontend_server: str) -> None:
    state_body = urlopen(f"{frontend_server}/app-state.js").read().decode("utf-8")
    controller_body = urlopen(f"{frontend_server}/app-controller.js").read().decode("utf-8")

    assert "function currentDatabaseMarker()" in state_body
    assert "documentRef.body?.dataset?.databasePath" in state_body
    assert "if (!savedDatabase || savedDatabase !== expectedDatabase)" in state_body
    assert "database: currentDatabaseMarker()," in state_body
    assert "document.body.dataset.databasePath = options.database || \"\";" in controller_body


def test_frontend_boot_retries_a_transient_options_failure(chromium_page) -> None:
    page, _ = chromium_page
    attempts = 0

    def fail_first_options_request(route) -> None:
        nonlocal attempts
        if "/api/options" in route.request.url and attempts == 0:
            attempts += 1
            route.fulfill(
                status=503,
                headers={"Content-Type": "application/json"},
                body='{"error":"transient test failure"}',
            )
            return
        route.continue_()

    page.route("**/api/options*", fail_first_options_request)
    try:
        page.reload()
        page.wait_for_function(
            "() => document.body?.dataset?.appReady === 'true'",
            timeout=10000,
        )
        assert attempts == 1
        assert page.locator("#model-select option").count() >= 1
        assert page.locator("#device-select option").count() >= 1
    finally:
        page.unroute("**/api/options*", fail_first_options_request)


def test_public_workflow_facade_preserves_library_and_review_behavior(chromium_page) -> None:
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
          const calls = [];
          const state = {
            activeId: 7,
            abortController: null,
            bulkSelection: null,
            detail: { id: 7 },
            loadedReviewSelection: { selection: { root: "old-root" } },
            options: { default_scoring_mode: "topiq_nr" },
            page: 4,
            pageSize: 60,
            queue: [],
            selectedIds: new Set([7]),
            totalFiles: 1,
          };
          const postJson = async (url, payload) => {
            calls.push(["post", url]);
            if (url === "/api/scan/start") return { job_id: "scan-job" };
            if (url === "/api/score/start") return { job_id: "score-job" };
            return { id: payload?.file_id || null };
          };
          const fetchJson = async (url) => {
            calls.push(["fetch", url]);
            if (url.includes("/api/scan/status")) return { status: "completed" };
            if (url.includes("/api/scan/result")) {
              return { files_seen: 1, files_added: 1, files_updated: 0, files_removed: 0 };
            }
            if (url.includes("/api/score/status")) return { status: "completed" };
            if (url.includes("/api/score/result")) {
              return { files_scored: 1, learned_scored: 1, skipped: 0, failed: 0, elapsed_seconds: 0.1 };
            }
            return {};
          };
          const noop = () => {};
          const workflows = window.ShotSieveWorkflows.createWorkflows({
            state,
            api: { fetchJson, postJson },
            busy: {
              clearTrackedJob: noop,
              markTrackedJobUnknown: noop,
              setBusyMessage: noop,
              setBusyPhaseProgress: noop,
              setBusyProgress: noop,
              trackJob: noop,
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
              escapeHtml: (value) => String(value),
              formatDuration: () => "0s",
              formatFilesPerSecond: () => "0 files/s",
              formatNumber: (value) => String(value),
              getScoreColor: () => "",
              mergeTimingTotals: () => {},
              pathLeaf: (value) => String(value).split("/").pop(),
              sortComparisonRows: (rows) => rows,
            },
            notifications: {
              addLogEntry: (title) => calls.push(["log", title]),
              showToast: (message) => calls.push(["toast", message]),
            },
            review: {
              applyReviewUpdate: () => true,
              isAutoAdvanceEnabled: () => false,
              loadQueue: async () => calls.push(["review", "loadQueue"]),
              refreshOverview: async () => calls.push(["review", "refreshOverview"]),
              refreshWorkspace: async () => calls.push(["review", "refreshWorkspace"]),
              reviewDecisions: { keep: { marked: "keep" }, reject: { marked: "reject" }, reset: { marked: null } },
              renderPagination: () => calls.push(["review", "renderPagination"]),
              selectFile: async () => {},
              syncReviewRoot: (root) => root,
            },
            ui: {
              closeOverlay: noop,
              currentLibraryRoot: () => "C:/photos",
              openOverlay: noop,
              saveUiState: () => calls.push(["ui", "saveUiState"]),
              selectedComparisonModels: () => [],
              setTab: (tab) => calls.push(["ui", "setTab", tab]),
            },
          });

          for (const id of ["query-filter", "min-score", "max-score"]) {
            document.getElementById(id).value = "stale";
          }
          document.getElementById("sort-filter").value = "path_asc";
          document.getElementById("marked-filter").value = "keep";
          document.getElementById("issues-filter").value = "present";
          workflows.resetReviewFiltersForAnalyze("C:/photos");
          const resetState = {
            query: document.getElementById("query-filter").value,
            sort: document.getElementById("sort-filter").value,
            marked: document.getElementById("marked-filter").value,
            issues: document.getElementById("issues-filter").value,
            page: state.page,
            root: document.getElementById("root-filter").value,
          };

          await workflows.saveReview({ marked: "keep" });
          await workflows.openOriginalFile("9");
          await workflows.analyzeLibrary();

          return {
            resetState,
            hasPublicMethods: [
              "resetReviewFiltersForAnalyze",
              "analyzeLibrary",
              "saveReview",
              "openOriginalFile",
            ].every((name) => typeof workflows[name] === "function"),
            activeId: state.activeId,
            detail: state.detail,
            page: state.page,
            analyzeLoadIndex: calls.findIndex((entry) => entry[0] === "review" && entry[1] === "loadQueue"),
            analyzeTabIndex: calls.findIndex((entry) => entry[0] === "ui" && entry[1] === "setTab" && entry[2] === "review"),
            saveRefreshPresent: calls.some((entry) => entry[0] === "review" && entry[1] === "refreshOverview"),
            savePaginationPresent: calls.some((entry) => entry[0] === "review" && entry[1] === "renderPagination"),
            openedFile: calls.some((entry) => entry[0] === "post" && entry[1] === "/api/files/open"),
            scanStarted: calls.some((entry) => entry[0] === "post" && entry[1] === "/api/scan/start"),
            scoreStarted: calls.some((entry) => entry[0] === "post" && entry[1] === "/api/score/start"),
          };
        }
        """,
    )

    assert result["hasPublicMethods"] is True
    assert result["resetState"] == {
        "query": "",
        "sort": "learned_asc",
        "marked": "all",
        "issues": "all",
        "page": 0,
        "root": "C:/photos",
    }
    assert result["activeId"] is None
    assert result["detail"] is None
    assert result["page"] == 0
    assert result["analyzeLoadIndex"] >= 0
    assert result["analyzeTabIndex"] > result["analyzeLoadIndex"]
    assert result["saveRefreshPresent"] is True
    assert result["savePaginationPresent"] is True
    assert result["openedFile"] is True
    assert result["scanStarted"] is True
    assert result["scoreStarted"] is True


def test_open_original_review_action_uses_composed_workflow(chromium_page) -> None:
    page, _ = chromium_page
    opened: list[dict[str, object]] = []

    def fulfill_open(route) -> None:
        opened.append(route.request.post_data_json)
        route.fulfill(status=200, content_type="application/json", body='{"opened":true}')

    page.route("**/api/files/open", fulfill_open)
    try:
        _open_review_tab(page)
        with page.expect_response("**/api/files/open") as response_info:
            page.locator("#open-original").click()
        assert response_info.value.ok
    finally:
        page.unroute("**/api/files/open", fulfill_open)

    assert len(opened) == 1
    assert isinstance(opened[0].get("file_id"), int)


def test_full_reset_refreshes_resource_profile_detail(chromium_page) -> None:
    page, _ = chromium_page
    page.get_by_role("tab", name="Settings").click()
    page.evaluate(
        """
        () => {
          window.confirm = () => true;
          document.getElementById("resource-profile-detail").textContent = "stale profile detail";
        }
        """
    )

    page.locator("#clear-all-cache").click()
    page.wait_for_function(
        """
        () => document.getElementById("resource-profile-select")?.value === "normal"
          && document.getElementById("resource-profile-detail")?.textContent !== "stale profile detail"
        """,
        timeout=10000,
    )

    assert page.locator("#resource-profile-detail").inner_text() != "stale profile detail"


def test_load_queue_keeps_query_available_after_page_clamp_retry(frontend_server: str) -> None:
    grid_body = urlopen(f"{frontend_server}/app-grid.js").read().decode("utf-8")

    load_queue_index = grid_body.index("async function loadQueue()")
    load_queue_block = grid_body[load_queue_index : load_queue_index + 1800]

    assert "let query = null;" in load_queue_block
    assert "query = currentQuery();" in load_queue_block
    assert "reviewSelectionSnapshotFromQuery(query, data.total || 0)" in load_queue_block


def test_operation_results_and_decision_csv_controls_are_present(frontend_server: str) -> None:
    index_body = urlopen(f"{frontend_server}/").read().decode("utf-8")
    export_body = urlopen(f"{frontend_server}/app-workflow-export.js").read().decode("utf-8")
    export_ui_body = urlopen(f"{frontend_server}/app-workflow-export-ui.js").read().decode("utf-8")
    library_body = urlopen(f"{frontend_server}/app-workflow-library-operations.js").read().decode("utf-8")
    polling_body = urlopen(f"{frontend_server}/app-workflow-polling.js").read().decode("utf-8")

    assert 'id="operation-result-panel"' in index_body
    assert 'id="operation-result-download"' in index_body
    assert 'id="download-decisions-csv"' in index_body
    assert 'startPath: "/api/files/delete/start"' in export_ui_body
    assert 'result.safe_retry_ids' in export_body
    assert 'textContent' in export_body
    assert 'retainFailedResult: true' in library_body
    assert 'No progress for 30 seconds' in library_body
    assert 'retainFailedResult = false' in polling_body
