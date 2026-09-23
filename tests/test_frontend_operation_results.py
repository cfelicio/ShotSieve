"""Exercise the actual result presenter with browser DOM and selection state."""
from __future__ import annotations


def _run_retry_scenario(page, scenario: dict[str, object]) -> dict[str, object]:
    return page.evaluate(
        """
        async (scenario) => {
            window.confirm = () => true;
            const ids = scenario.ids;
            const mode = scenario.mode;
            const selection = scenario.selection;
            const calls = [];
            const revisionUrls = [];
            let refreshes = 0;

            const state = {
                selectedIds: new Set(ids),
                bulkSelection: null,
                latestOperationResult: {
                    action: mode,
                    outcome: "partial",
                    items: ids.map((fileId) => ({
                        file_id: fileId,
                        id: fileId,
                        action: mode,
                        outcome: "unprocessed",
                        retry_safe: true,
                    })),
                    safe_retry_ids: [...ids],
                    warnings: [],
                },
                latestOperationRequest: {
                    startPath: "/api/files/operation/start",
                    payload: {
                        mode,
                        selection,
                        selection_revision: "stale-revision",
                        count: ids.length,
                    },
                    fallbackLabel: `${mode} files`,
                    failureMessage: `${mode} failed`,
                },
                recoveryJob: null,
                operationStatusUnknown: false,
                operationJobId: null,
            };

            const resultItems = (fileIds, outcome, retrySafe) => fileIds.map((fileId) => ({
                file_id: fileId,
                id: fileId,
                action: mode,
                outcome,
                retry_safe: retrySafe,
            }));

            const runTrackedOperation = async ({ payload }) => {
                calls.push(JSON.parse(JSON.stringify(payload)));
                if (scenario.cancelAfterFirst && calls.length === 2) {
                    const error = new Error("cancelled");
                    error.name = "AbortError";
                    throw error;
                }
                if (scenario.failSecond && calls.length === 2) {
                    return {
                        action: mode,
                        outcome: "failed",
                        job_status: "failed",
                        fatal_error: "second chunk failed",
                        items: resultItems(payload.file_ids, "failed", false),
                    };
                }
                return {
                    action: mode,
                    outcome: "success",
                    items: resultItems(payload.file_ids, "success", false),
                };
            };
            // Match production composition: the stable bridge is populated
            // after the export workflow has been constructed.
            const workflowLibrary = {};

            const busy = {
                withBusy: async (_message, task, options) => {
                    try {
                        return await task();
                    } catch (error) {
                        if (error.name !== "AbortError") throw error;
                        await options.onCancelled({
                            confirmed: true,
                            result: { action: mode, outcome: "cancelled", cancelled: true, items: [] },
                        });
                    }
                },
            };

            const workflow = window.ShotSieveWorkflowExport.createWorkflowExport({
                state,
                api: {
                    fetchJson: async (url) => {
                        revisionUrls.push(url);
                        if (scenario.failRevision) {
                            throw new Error("stale selection revision");
                        }
                        return { selection_revision: `fresh-${revisionUrls.length}` };
                    },
                    postJson: async () => ({}),
                },
                busy,
                notifications: { showToast: () => {} },
                review: {
                    refreshWorkspace: async () => { refreshes += 1; },
                },
                ui: { openBrowser: () => {}, handleError: () => {} },
                workflowLibrary,
            });
            workflowLibrary.runTrackedOperation = runTrackedOperation;

            await workflow.retrySafeOperation();
            return {
                calls,
                revisionUrls,
                refreshes,
                selectedIds: [...state.selectedIds],
                result: {
                    outcome: state.latestOperationResult.outcome,
                    job_status: state.latestOperationResult.job_status || null,
                    completed_count: state.latestOperationResult.completed_count || 0,
                    failed_count: state.latestOperationResult.failed_count || 0,
                    unprocessed_count: state.latestOperationResult.unprocessed_count || 0,
                    safe_retry_ids: state.latestOperationResult.safe_retry_ids || [],
                    itemIds: state.latestOperationResult.items.map((item) => item.file_id),
                },
            };
        }
        """,
        scenario,
    )


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
    expect(page.locator("#operation-result-counts")).to_contain_text("Warnings: 1")
    expect(page.locator("#operation-result-items")).to_contain_text("Preview access denied")
    expect(page.locator("#operation-result-panel")).to_have_attribute("data-outcome", "partial")


def test_retry_preserves_selected_and_rejected_scopes_with_fresh_revisions(chromium_page):
    page, _ = chromium_page
    selected = _run_retry_scenario(
        page,
        {
            "ids": list(range(1, 502)),
            "mode": "copy",
            "selection": {
                "scope": "review-browser",
                "root": "C:/photos",
                "marked": "export",
                "issues": "all",
                "query": "sunset",
                "formats": ["jpg", "png"],
                "min_mp": 2,
                "metadata": "missing",
            },
        },
    )
    rejected = _run_retry_scenario(
        page,
        {
            "ids": [700, 701],
            "mode": "delete",
            "selection": {"scope": "review-state", "marked": "delete", "root": "C:/photos"},
        },
    )

    assert len(selected["calls"]) == 2
    assert [len(call["file_ids"]) for call in selected["calls"]] == [500, 1]
    assert all("selection" not in call for call in selected["calls"])
    assert all(call["page_selection"]["query"] == "sunset" for call in selected["calls"])
    assert [call["selection_revision"] for call in selected["calls"]] == ["fresh-1", "fresh-2"]
    assert selected["revisionUrls"] == [
        "/api/files?root=C%3A%2Fphotos&marked=export&issues=all&query=sunset&min_mp=2&metadata=missing&formats=jpg%2Cpng&limit=1&offset=0",
        "/api/files?root=C%3A%2Fphotos&marked=export&issues=all&query=sunset&min_mp=2&metadata=missing&formats=jpg%2Cpng&limit=1&offset=0",
    ]
    assert selected["result"]["itemIds"] == list(range(1, 502))
    assert selected["result"]["completed_count"] == 501
    assert selected["result"]["failed_count"] == 0
    assert selected["selectedIds"] == []

    assert len(rejected["calls"]) == 1
    assert rejected["calls"][0]["page_selection"]["scope"] == "review-state"
    assert rejected["calls"][0]["selection_revision"] == "fresh-1"
    assert rejected["revisionUrls"] == [
        "/api/review/file-ids?marked=delete&limit=1&offset=0&root=C%3A%2Fphotos",
    ]


def test_retry_stops_after_failed_second_chunk_and_retains_all_rows(chromium_page):
    page, _ = chromium_page
    result = _run_retry_scenario(
        page,
        {
            "ids": list(range(1, 1002)),
            "mode": "copy",
            "failSecond": True,
            "selection": {"scope": "review-browser", "root": "C:/photos", "marked": "export"},
        },
    )

    assert len(result["calls"]) == 2
    assert [len(call["file_ids"]) for call in result["calls"]] == [500, 500]
    assert result["result"]["itemIds"] == list(range(1, 1002))
    assert result["result"]["completed_count"] == 500
    assert result["result"]["failed_count"] == 500
    assert result["result"]["unprocessed_count"] == 1
    assert result["result"]["safe_retry_ids"] == [1001]
    assert result["refreshes"] == 1


def test_retry_keeps_selection_when_scope_revision_cannot_be_verified(chromium_page):
    page, _ = chromium_page
    result = _run_retry_scenario(
        page,
        {
            "ids": [10, 11],
            "mode": "copy",
            "failRevision": True,
            "selection": {"scope": "review-browser", "root": "C:/photos", "marked": "export"},
        },
    )

    assert result["calls"] == []
    assert result["result"]["outcome"] == "partial"
    assert result["result"]["itemIds"] == [10, 11]
    assert result["result"]["safe_retry_ids"] == [10, 11]
    assert result["selectedIds"] == [10, 11]
    assert result["refreshes"] == 1


def test_retry_cancel_after_first_chunk_retains_pending_ids(chromium_page):
    page, _ = chromium_page
    result = _run_retry_scenario(
        page,
        {
            "ids": list(range(1, 1002)),
            "mode": "move",
            "cancelAfterFirst": True,
            "selection": {"scope": "review-browser", "root": "C:/photos", "marked": "export"},
        },
    )

    assert len(result["calls"]) == 2
    assert result["result"]["outcome"] == "cancelled"
    assert result["result"]["completed_count"] == 500
    assert result["result"]["unprocessed_count"] == 501
    assert result["result"]["itemIds"] == list(range(1, 1002))
    assert result["result"]["safe_retry_ids"] == list(range(501, 1002))


def test_unresolved_job_blocks_new_work_and_check_status_refreshes(chromium_page):
    page, _ = chromium_page
    result = page.evaluate(
        """
        async () => {
            const state = {
                selectedIds: new Set(),
                activeJob: null,
                recoveryJob: null,
                operationStatusUnknown: false,
                scanJobId: null,
                scoreJobId: null,
                compareJobId: null,
                modelPreparationJobId: null,
                operationJobId: null,
                operationStatusPath: null,
                operationCancelPath: null,
                isBusy: false,
            };
            const busy = window.ShotSieveBusy.createBusyController({
                state,
                api: {
                    fetchJson: async () => ({ status: "completed" }),
                    postJson: async () => ({}),
                },
                notify: { showToast: () => {} },
            });
            busy.trackJob({
                kind: "scan",
                jobId: "scan-1",
                statusPath: "/api/scan/status",
                resultPath: "/api/scan/result",
                cancelPath: "/api/scan/cancel",
                label: "Scan",
            });
            busy.markTrackedJobUnknown(new Error("status request lost"));
            const recoveryVisible = !document.getElementById("job-recovery-panel").classList.contains("hidden");
            let started = false;
            let blockedMessage = "";
            try {
                await busy.withBusy("should not start", async () => { started = true; });
            } catch (error) {
                blockedMessage = error.message;
            }

            let refreshes = 0;
            const library = window.ShotSieveWorkflowLibrary.createWorkflowLibrary({
                state,
                api: {
                    fetchJson: async (url) => url.includes("/status?")
                        ? { status: "completed" }
                        : { job_id: "scan-1", files_seen: 3 },
                    postJson: async () => ({}),
                },
                busy,
                compare: { currentResourceProfile: () => "normal", scoreBatchSize: () => 1 },
                formatting: { escapeHtml: (value) => value, formatDuration: () => "0s" },
                notifications: { showToast: () => {} },
                pollingModule: {
                    pollJob: async () => ({}),
                    pollScanJob: async () => ({}),
                    pollScoreJob: async () => ({}),
                    pollModelPreparationJob: async () => ({}),
                    createResultFetcher: () => async () => ({}),
                    createStatusFetcher: () => async () => ({ status: "completed" }),
                },
                review: { refreshWorkspace: async () => { refreshes += 1; } },
                ui: {
                    currentLibraryRoot: () => "",
                    saveUiState: () => {},
                    setTab: () => {},
                },
                workflowExport: {},
            });
            const checked = await library.checkTrackedJob();
            return {
                recoveryVisible,
                started,
                blockedMessage,
                checked: checked.files_seen,
                refreshes,
                recoveryCleared: state.recoveryJob === null,
                scanCleared: state.scanJobId === null,
            };
        }
        """,
    )

    assert result["recoveryVisible"] is True
    assert result["started"] is False
    assert "status is unresolved" in result["blockedMessage"]
    assert result["checked"] == 3
    assert result["refreshes"] == 1
    assert result["recoveryCleared"] is True
    assert result["scanCleared"] is True


def test_scan_and_score_share_job_lifecycle_without_losing_kind_specific_polling(chromium_page):
    page, _ = chromium_page
    results = page.evaluate(
        """
        async () => {
          async function runScenario(kind, outcome) {
            const calls = [];
            const state = {
              abortController: { signal: { aborted: outcome === "abort" } },
              activeJob: null,
              recoveryJob: null,
              options: {
                default_scoring_mode: "topiq_nr",
                learned_models: ["topiq_nr"],
                learned: { recommended_batch_sizes: { topiq_nr: 4 } },
              },
              scanJobId: null,
              scoreJobId: null,
            };
            let clearCalls = 0;
            let markCalls = 0;
            const busy = {
              clearTrackedJob: (jobId) => {
                clearCalls += 1;
                if (state.activeJob?.jobId === jobId) state.activeJob = null;
              },
              markTrackedJobUnknown: (error) => {
                markCalls += 1;
                const job = state.activeJob || state.recoveryJob;
                state.recoveryJob = { ...job, status: "unknown", error: error.message };
                state.activeJob = null;
              },
              setBusyMessage: () => {},
              setBusyPhaseProgress: () => {},
              setBusyProgress: () => {},
              trackJob: (job) => {
                calls.push(["track", job]);
                state.activeJob = { ...job };
                state.recoveryJob = null;
              },
            };
            const failIfNeeded = () => {
              if (outcome === "success") return;
              const error = new Error(`${kind} status request lost`);
              if (outcome === "abort") error.name = "AbortError";
              throw error;
            };
            const pollingModule = {
              pollJob: async () => ({}),
              pollScanJob: async (jobId, options) => {
                calls.push(["poll-scan", jobId, options.filesTotalRef.value, options.pipeline.stepIndex]);
                failIfNeeded();
                return { files_seen: 1, files_added: 1, files_updated: 0, files_removed: 0 };
              },
              pollScoreJob: async (jobId, options) => {
                calls.push(["poll-score", jobId, options.rowsTotal, options.pipeline.stepIndex]);
                failIfNeeded();
                return { rows_loaded: 1, files_scored: 1, learned_scored: 1, files_skipped: 0, files_failed: 0 };
              },
              pollModelPreparationJob: async () => ({}),
              createResultFetcher: () => async () => ({}),
              createStatusFetcher: () => async () => ({ status: "completed" }),
            };
            const library = window.ShotSieveWorkflowLibrary.createWorkflowLibrary({
              state,
              api: {
                fetchJson: async () => ({}),
                postJson: async (url) => {
                  calls.push(["post", url]);
                  if (url === "/api/score-estimate") return { rows_total: 3 };
                  return { job_id: `${kind}-job` };
                },
              },
              busy,
              compare: { currentResourceProfile: () => "normal", scoreBatchSize: () => 4 },
              formatting: { escapeHtml: (value) => value, formatDuration: () => "0s" },
              notifications: { showToast: () => {} },
              pollingModule,
              review: { refreshWorkspace: async () => {}, syncReviewRoot: () => {} },
              ui: { currentLibraryRoot: () => "C:/photos", saveUiState: () => {}, setTab: () => {} },
              workflowExport: {},
            });

            let errorName = null;
            try {
              if (kind === "scan") {
                await library.runScan("C:/photos", {
                  generatePreviews: false,
                  pipeline: { stepIndex: 1, totalSteps: 3 },
                });
              } else {
                await library.runScore("C:/photos", {
                  pipeline: { stepIndex: 2, totalSteps: 3 },
                });
              }
            } catch (error) {
              errorName = error.name;
            }
            return {
              kind,
              outcome,
              errorName,
              jobId: state[`${kind}JobId`],
              recoveryJobId: state.recoveryJob?.jobId || null,
              activeJobId: state.activeJob?.jobId || null,
              clearCalls,
              markCalls,
              tracked: calls.find((entry) => entry[0] === "track")?.[1] || null,
              poll: calls.find((entry) => entry[0] === `poll-${kind}`) || null,
            };
          }

          const scenarios = [];
          for (const kind of ["scan", "score"]) {
            for (const outcome of ["success", "abort", "recovery"]) {
              scenarios.push(await runScenario(kind, outcome));
            }
          }
          return scenarios;
        }
        """,
    )

    for result in results:
        assert result["tracked"]["kind"] == result["kind"]
        assert result["tracked"]["statusPath"] == f"/api/{result['kind']}/status"
        assert result["tracked"]["resultPath"] == f"/api/{result['kind']}/result"
        assert result["tracked"]["cancelPath"] == f"/api/{result['kind']}/cancel"
        assert result["poll"][1] == f"{result['kind']}-job"

        if result["kind"] == "scan":
            assert result["poll"][2:] == [3, 1]
        else:
            assert result["poll"][2:] == [3, 2]

        if result["outcome"] == "success":
            assert result["errorName"] is None
            assert result["jobId"] is None
            assert result["recoveryJobId"] is None
            assert result["activeJobId"] is None
            assert result["clearCalls"] == 1
            assert result["markCalls"] == 0
        elif result["outcome"] == "abort":
            assert result["errorName"] == "AbortError"
            assert result["jobId"] == f"{result['kind']}-job"
            assert result["recoveryJobId"] is None
            assert result["activeJobId"] == f"{result['kind']}-job"
            assert result["clearCalls"] == 0
            assert result["markCalls"] == 0
        else:
            assert result["errorName"] == "Error"
            assert result["jobId"] == f"{result['kind']}-job"
            assert result["recoveryJobId"] == f"{result['kind']}-job"
            assert result["activeJobId"] is None
            assert result["clearCalls"] == 0
            assert result["markCalls"] == 1


def test_analyze_stops_after_model_unavailable_or_failed_scoring(chromium_page):
    page, _ = chromium_page
    results = page.evaluate(
        """
        async () => {
          async function runScenario(noModel) {
            const calls = [];
            const messages = [];
            const state = {
              options: noModel ? {} : { default_scoring_mode: "topiq_nr", learned_models: ["topiq_nr"] },
              page: 0,
              loadedReviewSelection: null,
            };
            const select = document.getElementById("model-select");
            select.value = noModel ? "" : "topiq_nr";
            const workflow = window.ShotSieveWorkflowLibraryAnalysis.createWorkflowLibraryAnalysis({
              api: {
                postJson: async (url) => {
                  calls.push(["post", url]);
                  return url === "/api/score-estimate" ? { rows_total: 1 } : { job_id: url };
                },
              },
              busy: {
                setBusyMessage: () => {},
                setBusyPhaseProgress: () => {},
                setBusyProgress: () => {},
              },
              compare: { currentResourceProfile: () => "normal", scoreBatchSize: () => 1 },
              notifications: { showToast: (message, tone) => messages.push([message, tone]) },
              pollingModule: {},
              review: {
                loadQueue: async () => calls.push(["loadQueue"]),
                refreshWorkspace: async () => calls.push(["refreshWorkspace"]),
                syncReviewRoot: (root) => root,
              },
              state,
              ui: {
                currentLibraryRoot: () => "C:/photos",
                saveUiState: () => {},
                setTab: (tab) => calls.push(["setTab", tab]),
              },
              workflowExport: { clearActiveSelection: () => {} },
              operations: {
                runTrackedJob: async ({ kind }) => {
                  calls.push(["job", kind]);
                  if (kind === "scan") return { files_seen: 1 };
                  return {
                    job_status: "failed",
                    job_error: "score failed",
                    diagnostic: { cause: "Model backend failed", recovery_action: "Prepare the model and retry." },
                  };
                },
              },
            });
            await workflow.analyzeLibrary();
            return { calls, messages };
          }
          return [await runScenario(false), await runScenario(true)];
        }
        """,
    )

    for scenario in results:
        assert not any(call[0] == "setTab" and call[1] == "review" for call in scenario["calls"])
        assert not any(call[0] == "loadQueue" for call in scenario["calls"])
        assert not any("Analysis completed" in message[0] for message in scenario["messages"])
    assert any("Model backend failed" in message[0] for message in results[0]["messages"])
    assert any("No learned IQA model" in message[0] for message in results[1]["messages"])


def test_recovery_fetches_failed_score_and_compare_diagnostics(chromium_page):
    page, _ = chromium_page
    results = page.evaluate(
        """
        async () => {
          async function runScenario(kind) {
            const urls = [];
            const messages = [];
            const jobId = `${kind}-job`;
            const state = {
              recoveryJob: {
                kind,
                jobId,
                label: kind === "score" ? "Scoring" : "Compare",
                statusPath: `/api/${kind}/status`,
                resultPath: `/api/${kind}/result`,
              },
              comparison: null,
            };
            const workflow = window.ShotSieveWorkflowLibraryOperations.createWorkflowLibraryOperations({
              api: {
                fetchJson: async (url) => {
                  urls.push(url);
                  if (url.includes("/status?")) return { status: "failed", error: "stored failure" };
                  return {
                    diagnostic: { cause: "Retained model failure", recovery_action: "Check analysis diagnostics." },
                    error_report: { cause: "Retained model failure", recovery_action: "Check analysis diagnostics." },
                  };
                },
                postJson: async () => ({}),
              },
              busy: {
                clearTrackedJob: () => { state.recoveryJob = null; },
                markTrackedJobUnknown: () => { throw new Error("unexpected status loss"); },
              },
              formatting: { formatDuration: () => "0s" },
              notifications: { showToast: (message, tone) => messages.push([message, tone]) },
              pollingModule: {
                createResultFetcher: () => async () => ({}),
                createStatusFetcher: () => async () => ({}),
                pollJob: async () => ({}),
              },
              review: { refreshWorkspace: async () => {} },
              state,
            });
            const result = await workflow.checkTrackedJob();
            let summaryRendered = false;
            if (kind === "compare") {
              const compareWorkflow = window.ShotSieveWorkflowCompare.createWorkflowCompare({
                api: { postJson: async () => ({}) },
                busy: {},
                compare: {},
                formatting: {
                  escapeHtml: (value) => String(value),
                  formatDuration: () => "0s",
                  formatFilesPerSecond: () => "0 files/s",
                  formatNumber: (value) => String(value),
                  getScoreColor: () => "",
                  mergeTimingTotals: () => ({}),
                  pathLeaf: (value) => String(value),
                  sortComparisonRows: (rows) => rows,
                },
                notifications: { showToast: () => {} },
                pollingModule: {},
                state,
                ui: { currentLibraryRoot: () => null, selectedComparisonModels: () => [], setTab: () => {} },
                workflowLibrary: {},
              });
              compareWorkflow.renderComparisonSummary();
              summaryRendered = true;
            }
            return {
              urls,
              messages,
              recoveryCleared: state.recoveryJob === null,
              comparison: state.comparison,
              summaryRendered,
              result,
            };
          }
          return [await runScenario("score"), await runScenario("compare")];
        }
        """,
    )

    for scenario in results:
        assert any("/result?job_id=" in url for url in scenario["urls"])
        assert scenario["recoveryCleared"] is True
        assert scenario["messages"]
        assert "Retained model failure" in scenario["messages"][0][0]
    assert scenario["messages"][0][1] == "error"
    assert results[1]["comparison"] is None
    assert results[1]["summaryRendered"] is True
