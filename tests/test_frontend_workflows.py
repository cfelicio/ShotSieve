from __future__ import annotations



from test_frontend_accessibility import (
    _open_compare_tab,
    _open_export_dialog,
    _open_folder_browser,
    _open_review_tab,
    _render_compare_results,
)


def test_rapid_photo_selection_keeps_detail_and_review_action_on_latest_id(chromium_page) -> None:
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
          const state = {
            activeId: null,
            detail: null,
            queue: [{ id: 1 }, { id: 2 }],
            selectedIds: new Set(),
            bulkSelection: null,
            loadedReviewSelection: null,
            page: 0,
            pageSize: 60,
            totalFiles: 2,
          };
          const pending = new Map();
          const posts = [];
          const renderedDetails = [];
          const deferred = (id) => new Promise((resolve) => pending.set(id, resolve));
          const grid = window.ShotSieveGrid.createGridController({
            state,
            ui: {},
            formatting: {
              escapeHtml: String,
              formatNumber: String,
              getScoreColor: () => "",
              pathDirectory: String,
              pathLeaf: String,
            },
            reviewModule: {
              renderDetail: ({ state: current }) => renderedDetails.push(current.detail?.id ?? null),
              renderQueue: () => {},
              updateSelectionState: () => {},
            },
            notifications: { showToast: () => {} },
            api: { fetchJson: (url) => deferred(Number(new URL(url, location.href).searchParams.get("id"))) },
            stateModule: {},
            appUtils: {},
            handleError: () => {},
            openOriginalFile: () => {},
          });
          const workflow = window.ShotSieveWorkflowExport.createWorkflowExport({
            state,
            api: {
              fetchJson: async () => ({}),
              postJson: async (_url, payload) => {
                posts.push(payload);
                return { id: payload.file_id };
              },
            },
            busy: {
              setBusyMessage: () => {},
              setBusyPhaseProgress: () => {},
              withBusy: async (_message, task) => task(),
            },
            notifications: { showToast: () => {} },
            review: {
              applyReviewUpdate: () => true,
              isAutoAdvanceEnabled: () => false,
              loadQueue: async () => {},
              refreshOverview: async () => {},
              refreshWorkspace: async () => {},
              reviewDecisions: { keep: { decision_state: "keep" } },
              selectFile: grid.selectFile,
              renderPagination: () => {},
            },
            ui: { handleError: () => {}, openBrowser: () => {} },
            workflowLibrary: {},
          });

          const first = grid.selectFile(1);
          const second = grid.selectFile(2);
          pending.get(2)({ id: 2, name: "photo-two.jpg" });
          await second;
          pending.get(1)({ id: 1, name: "photo-one.jpg" });
          await first;
          await workflow.saveReview({ decision_state: "keep" });

          return {
            activeId: state.activeId,
            detailId: state.detail?.id,
            posts,
            renderedDetails,
          };
        }
        """,
    )

    assert result["activeId"] == 2
    assert result["detailId"] == 2
    assert result["posts"] == [{"file_id": 2, "decision_state": "keep"}]


def test_review_shortcuts_require_review_idle_context(chromium_page) -> None:
    page, _ = chromium_page
    _open_review_tab(page)
    starting_position = page.locator("#review-position").inner_text()
    review_mutations: list[str] = []

    def record_request(request) -> None:
        if request.method == "POST" and "/api/review" in request.url:
            review_mutations.append(request.url)

    page.on("request", record_request)

    page.get_by_role("tab", name="Library").click()
    page.evaluate(
        """
        () => {
          for (const key of ['ArrowRight', 's', 'r']) {
            document.body.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true, cancelable: true }));
          }
        }
        """,
    )
    page.wait_for_timeout(100)
    assert page.locator("#review-position").inner_text() == starting_position
    assert review_mutations == []

    _open_review_tab(page)
    page.evaluate(
        """
        () => {
          const overlay = document.getElementById('lightbox-overlay');
          overlay.showModal();
          document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 's', bubbles: true, cancelable: true }));
          overlay.close();
          document.getElementById('refresh-all').click();
          document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'r', bubbles: true, cancelable: true }));
        }
        """,
    )
    page.wait_for_timeout(250)
    assert review_mutations == []

    with page.expect_request(lambda request: request.method == "POST" and request.url.endswith("/api/review")):
        page.evaluate(
            """
            () => document.body.dispatchEvent(
              new KeyboardEvent('keydown', { key: 's', bubbles: true, cancelable: true }),
            )
            """,
        )


def test_late_review_response_updates_submitted_row_without_reactivating_photo(chromium_page) -> None:
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
          const state = {
            activeId: null,
            detail: null,
            queue: [{ id: 1, name: "photo-one.jpg" }, { id: 2, name: "photo-two.jpg" }],
            selectedIds: new Set(),
            bulkSelection: null,
            loadedReviewSelection: null,
            page: 0,
            pageSize: 60,
            totalFiles: 2,
          };
          const details = new Map();
          let resolveReview;
          const grid = window.ShotSieveGrid.createGridController({
            state,
            ui: {},
            formatting: {
              escapeHtml: String,
              formatNumber: String,
              getScoreColor: () => "",
              pathDirectory: String,
              pathLeaf: String,
            },
            reviewModule: {
              renderDetail: () => {},
              renderQueue: () => {},
              updateSelectionState: () => {},
            },
            notifications: { showToast: () => {} },
            api: {
              fetchJson: (url) => {
                const id = Number(new URL(url, location.href).searchParams.get("id"));
                return new Promise((resolve) => details.set(id, resolve));
              },
            },
            stateModule: {},
            appUtils: {},
            handleError: () => {},
            openOriginalFile: () => {},
          });
          const workflow = window.ShotSieveWorkflowExport.createWorkflowExport({
            state,
            api: {
              fetchJson: async () => ({}),
              postJson: async (_url, payload) => new Promise((resolve) => {
                resolveReview = () => resolve({ id: payload.file_id, decision_state: "keep" });
              }),
            },
            busy: { withBusy: async (_message, task) => task() },
            notifications: { showToast: () => {} },
            review: {
              applyReviewUpdate: grid.applyReviewUpdate,
              isAutoAdvanceEnabled: () => false,
              loadQueue: async () => {},
              refreshOverview: async () => {},
              refreshWorkspace: async () => {},
              reviewDecisions: { keep: { decision_state: "keep" } },
              selectFile: grid.selectFile,
              renderPagination: () => {},
            },
            ui: { handleError: () => {}, openBrowser: () => {} },
            workflowLibrary: {},
          });

          const first = grid.selectFile(1);
          details.get(1)({ id: 1, name: "photo-one.jpg" });
          await first;
          const save = workflow.saveReview({ decision_state: "keep" });

          const second = grid.selectFile(2);
          details.get(2)({ id: 2, name: "photo-two.jpg" });
          await second;
          resolveReview();
          await save;

          return {
            activeId: state.activeId,
            detailId: state.detail?.id,
            firstRowDecision: state.queue[0].decision_state,
            secondRowDecision: state.queue[1].decision_state || null,
          };
        }
        """,
    )

    assert result == {
        "activeId": 2,
        "detailId": 2,
        "firstRowDecision": "keep",
        "secondRowDecision": None,
    }


def test_scope_refresh_ignores_stale_responses_across_a_b_a_transition(chromium_page) -> None:
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
          document.body.innerHTML = `
            <input id="library-root-input">
            <select id="root-filter"><option value="A">A</option><option value="B">B</option></select>
            <div id="analysis-diagnostics-summary"></div>
            <div id="analysis-diagnostics-list"></div>
          `;
          const state = { overview: null, options: null, reviewScopeInitialized: true };
          const requests = [];
          const queueLoads = [];
          const responseFor = (root, kind) => kind === "overview"
            ? {
                root_marker: root,
                roots: ["A", "B"],
                active_library: { total_files: root === "B" ? 2 : 1 },
                catalog: { total_files: 3 },
              }
            : { total: 1, items: [{ path: `${root}/diagnostic.jpg`, status: "failed", error: root }] };
          const controller = window.ShotSieveController.createController({
            state,
            uiStore: {
              clearUiState: () => {},
              currentLibraryRoot: () => document.getElementById("library-root-input").value.trim(),
              loadUiState: () => ({}),
              saveUiState: () => {},
            },
            appUtils: { escapeHtml: (value) => String(value) },
            stateModule: {},
            api: {
              fetchJson: (url) => {
                const parsed = new URL(url, location.href);
                const kind = parsed.pathname.endsWith("/overview") ? "overview" : "diagnostics";
                const root = parsed.searchParams.get("root") || "";
                return new Promise((resolve) => requests.push({ kind, root, resolve }));
              },
            },
            workflows: {},
            grid: {
              invalidateLoadedReviewSelection: () => {},
              renderReviewScope: () => {},
              renderSummary: () => {},
              loadQueue: async () => queueLoads.push(document.getElementById("root-filter").value),
            },
            notifications: { showToast: () => {} },
          });
          const resolveRequest = (root, kind, occurrence = 0) => {
            const matches = requests.filter((request) => request.root === root && request.kind === kind);
            matches[occurrence].resolve(responseFor(root, kind));
          };
          const activateFromInput = (root) => {
            document.getElementById("library-root-input").value = root;
            return controller.activateLibraryScope(root);
          };

          const firstA = activateFromInput("A");
          const b = activateFromInput("B");
          const secondA = activateFromInput("A");
          await Promise.resolve();
          resolveRequest("B", "overview");
          resolveRequest("B", "diagnostics");
          await b;
          resolveRequest("A", "overview", 1);
          resolveRequest("A", "diagnostics", 1);
          await secondA;
          resolveRequest("A", "overview", 0);
          resolveRequest("A", "diagnostics", 0);
          await firstA;

          return {
            overviewRoot: state.overview.root_marker,
            diagnosticsText: document.getElementById("analysis-diagnostics-list").textContent,
            selectedRoot: document.getElementById("root-filter").value,
            queueLoads,
          };
        }
        """,
    )

    assert result["overviewRoot"] == "A"
    assert "A/diagnostic.jpg" in result["diagnosticsText"]
    assert result["selectedRoot"] == "A"
    assert result["queueLoads"] == ["A"]


def test_workspace_refresh_ignores_stale_responses_after_scope_changes(chromium_page) -> None:
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
          const rootInput = document.getElementById("library-root-input");
          const rootFilter = document.getElementById("root-filter");
          rootInput.value = "A";
          rootFilter.innerHTML = `<option value="">All libraries (global)</option><option value="A">A</option><option value="B">B</option>`;
          rootFilter.value = "A";
          const state = { overview: null, options: null, reviewScopeInitialized: true };
          const requests = [];
          const queueLoads = [];
          const options = {
            default_scoring_mode: "topiq_nr",
            default_extensions: [".jpg"],
            default_max_decode_megapixels: 64,
            runtime_targets: ["cpu"],
            database: "test.db",
            preview_dir: "previews",
            preview_capabilities: { heif_decoder: "none", raw_decoder: "none" },
            learned_models: ["topiq_nr"],
            learned: {
              hardware: { cpu_count: 4, ram_mb: 1024, vram_mb: 0 },
              runtime_status: { cpu: "ready" },
              default_runtime: "cpu",
              model_catalog: [],
              model_preparation: {},
            },
          };
          const responseFor = (root, kind) => kind === "overview"
            ? { root_marker: root, roots: ["A", "B"], active_library: {}, catalog: {} }
            : { total: 1, items: [{ path: `${root}/diagnostic.jpg`, status: "failed", error: root }] };
          const controller = window.ShotSieveController.createController({
            state,
            uiStore: {
              clearUiState: () => {},
              currentLibraryRoot: () => rootInput.value.trim(),
              loadUiState: () => ({}),
              saveUiState: () => {},
            },
            appUtils: {
              availableLearnedModels: () => ["topiq_nr"],
              currentResourceProfile: () => "normal",
              escapeHtml: (value) => String(value),
              formatPhotoSupport: () => "",
              parseRuntimeStatusMap: (value) => value || {},
              runtimeDisplayName: (value) => String(value),
              runtimeStatusToken: () => "",
              summarizeAccelerators: () => "",
              summarizeAutoPriority: () => "",
            },
            stateModule: {},
            api: {
              fetchJson: (url) => {
                const parsed = new URL(url, location.href);
                if (parsed.pathname.endsWith("/options")) {
                  return Promise.resolve(options);
                }
                const kind = parsed.pathname.endsWith("/overview") ? "overview" : "diagnostics";
                const root = parsed.searchParams.get("root") || "";
                return new Promise((resolve) => requests.push({ kind, root, resolve }));
              },
            },
            workflows: {},
            grid: {
              invalidateLoadedReviewSelection: () => {},
              renderReviewScope: () => {},
              renderSummary: () => {},
              loadQueue: async () => queueLoads.push(rootInput.value),
            },
            notifications: { showToast: () => {} },
          });
          const waitForRequests = async (count) => {
            while (requests.length < count) {
              await new Promise((resolve) => window.setTimeout(resolve, 0));
            }
          };
          const resolveRequest = (root, kind) => {
            const request = requests.find((candidate) => candidate.root === root && candidate.kind === kind && !candidate.resolved);
            request.resolved = true;
            request.resolve(responseFor(root, kind));
          };

          const staleRefresh = controller.refreshWorkspace();
          await waitForRequests(2);
          const currentRefresh = controller.activateLibraryScope("B");
          await waitForRequests(4);
          resolveRequest("B", "overview");
          resolveRequest("B", "diagnostics");
          await currentRefresh;
          resolveRequest("A", "overview");
          resolveRequest("A", "diagnostics");
          await staleRefresh;

          return {
            overviewRoot: state.overview.root_marker,
            diagnosticsText: document.getElementById("analysis-diagnostics-list")?.textContent || "",
            queueLoads,
          };
        }
        """,
    )

    assert result["overviewRoot"] == "B"
    assert "B/diagnostic.jpg" in result["diagnosticsText"]
    assert "A/diagnostic.jpg" not in result["diagnosticsText"]
    assert result["queueLoads"] == ["B"]


def test_scan_partial_completion_keeps_refresh_behavior_and_warns_about_diagnostics(chromium_page) -> None:
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
          const messages = [];
          const toasts = [];
          const refreshes = [];
          let nextResult = {
            overall_status: "completed",
            files_seen: 3,
            files_failed: 0,
          };
          const analysis = window.ShotSieveWorkflowLibraryAnalysis.createWorkflowLibraryAnalysis({
            state: {},
            api: { postJson: async () => ({ rows_total: 3 }) },
            busy: {
              setBusyMessage: (message) => messages.push(message),
              setBusyPhaseProgress: () => {},
              setBusyProgress: () => {},
              trackJob: () => {},
            },
            compare: { currentResourceProfile: () => "normal", scoreBatchSize: () => 1 },
            notifications: { showToast: (message, tone) => toasts.push([message, tone]) },
            pollingModule: { pollScanJob: async () => ({}) },
            review: {
              loadQueue: async () => {},
              refreshWorkspace: async () => refreshes.push("workspace"),
              syncReviewRoot: (root) => root,
            },
            ui: { currentLibraryRoot: () => "C:/photos", saveUiState: () => {}, setTab: () => {} },
            workflowExport: {},
            operations: { runTrackedJob: async () => nextResult },
          });

          await analysis.runScan("C:/photos", { generatePreviews: false });
          const clean = { message: messages.at(-1), toast: toasts.at(-1) };
          nextResult = {
            overall_status: "completed_with_errors",
            files_seen: 3,
            files_failed: 2,
          };
          await analysis.runScan("C:/photos", { generatePreviews: false });
          return { clean, partial: { message: messages.at(-1), toast: toasts.at(-1) }, refreshes };
        }
        """,
    )

    assert result["clean"]["message"] == "Scan completed. Processed 3 file(s)."
    assert result["clean"]["toast"] == ["Scan completed.", None]
    assert "2 file(s) failed" in result["partial"]["message"]
    assert "Analysis Diagnostics" in result["partial"]["message"]
    assert result["partial"]["toast"][1] == "warning"
    assert result["refreshes"] == ["workspace", "workspace"]


def test_stale_scan_completion_does_not_resync_an_old_library_scope(chromium_page) -> None:
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
          let currentRoot = "B";
          const syncedRoots = [];
          const analysis = window.ShotSieveWorkflowLibraryAnalysis.createWorkflowLibraryAnalysis({
            state: {},
            api: { postJson: async () => ({ rows_total: 1 }) },
            busy: {
              setBusyMessage: () => {},
              setBusyPhaseProgress: () => {},
              setBusyProgress: () => {},
              trackJob: () => {},
            },
            compare: { currentResourceProfile: () => "normal", scoreBatchSize: () => 1 },
            notifications: { showToast: () => {} },
            pollingModule: { pollScanJob: async () => ({ files_seen: 1, files_failed: 0 }) },
            review: {
              loadQueue: async () => {},
              refreshWorkspace: async () => { currentRoot = "B"; },
              syncReviewRoot: (root) => syncedRoots.push(root),
            },
            ui: { currentLibraryRoot: () => currentRoot, saveUiState: () => {}, setTab: () => {} },
            workflowExport: {},
            operations: { runTrackedJob: async () => ({ files_seen: 1, files_failed: 0 }) },
          });

          await analysis.runScan("A", { generatePreviews: false });
          return syncedRoots;
        }
        """,
    )

    assert result == []


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


def test_export_dialog_browse_opens_folder_browser(chromium_page) -> None:
    chromium_page, _ = chromium_page

    _open_export_dialog(chromium_page)
    chromium_page.locator("#browse-export-dir").click()
    chromium_page.wait_for_function(
        """
        () => document.getElementById('folder-browser')?.open === true
            && Boolean(document.getElementById('browser-path')?.value)
            && Boolean(document.getElementById('browser-list')?.textContent?.trim())
        """
    )

    assert chromium_page.locator("#export-dialog").is_visible()


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
