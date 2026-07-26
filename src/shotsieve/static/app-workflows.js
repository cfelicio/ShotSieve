(() => {
  const pollingModule = window.ShotSieveWorkflowPolling;
  if (!pollingModule?.createJobPollers) {
    throw new Error("ShotSieve workflow polling module failed to load.");
  }

  function createWorkflows(deps) {
    const jobPollers = pollingModule.createJobPollers({
      state: deps.state,
      api: { fetchJson: deps.api.fetchJson },
      busy: {
        setBusyMessage: deps.busy.setBusyMessage,
        setBusyPhaseProgress: deps.busy.setBusyPhaseProgress,
        setBusyProgress: deps.busy.setBusyProgress,
      },
      progress: {
        compareProgressMessage: deps.compare.compareProgressMessage,
        compareProgressPercent: deps.compare.compareProgressPercent,
        scanProgressMessage: deps.compare.scanProgressMessage,
        scanProgressPercent: deps.compare.scanProgressPercent,
        scoreProgressMessage: deps.compare.scoreProgressMessage,
        scoreProgressPercent: deps.compare.scoreProgressPercent,
      },
    });

    const exportDeps = {
      ...deps,
      workflowLibrary: {},
    };
    const workflowExport = window.ShotSieveWorkflowExport.createWorkflowExport(exportDeps);

    const libraryDeps = {
      ...deps,
      pollingModule: {
        ...jobPollers,
        createResultFetcher: pollingModule.createResultFetcher,
        createStatusFetcher: pollingModule.createStatusFetcher,
        pollJob: pollingModule.pollJob,
        pollScanJob: jobPollers.pollScanJob,
        pollScoreJob: jobPollers.pollScoreJob,
      },
      workflowExport,
    };
    const workflowLibrary = window.ShotSieveWorkflowLibrary.createWorkflowLibrary(libraryDeps);
    exportDeps.workflowLibrary = workflowLibrary;

    const compareDeps = {
      ...deps,
      pollingModule: {
        ...jobPollers,
        pollCompareJob: jobPollers.pollCompareJob,
        pipelineOverallPercent: pollingModule.pipelineOverallPercent,
      },
      workflowLibrary,
    };
    const workflowCompare = window.ShotSieveWorkflowCompare.createWorkflowCompare(compareDeps);

    function comparisonFailureText(row, modelName) {
      return workflowCompare.comparisonFailureText(row, modelName);
    }

    function defaultCompareRowSort(modelNames) {
      if (modelNames.includes("topiq_nr")) {
        return "topiq_nr:desc";
      }
      const scoreChoice = workflowCompare.compareRowSortChoices(modelNames).find(
        (choice) => choice.value.endsWith(":desc"),
      );
      return scoreChoice?.value || "input";
    }

    function syncCompareSortControls(modelNames) {
      const rowSort = document.getElementById("compare-row-sort");
      if (!rowSort) return;
      const rowChoices = workflowCompare.compareRowSortChoices(modelNames);
      rowSort.innerHTML = rowChoices
        .map((choice) => `<option value="${deps.formatting.escapeHtml(choice.value)}">${deps.formatting.escapeHtml(choice.label)}</option>`)
        .join("");
      const hasCurrentChoice = rowChoices.some((choice) => choice.value === deps.state.compareRowSort);
      if (!deps.state.compareRowSortInitialized || !hasCurrentChoice) {
        deps.state.compareRowSort = defaultCompareRowSort(modelNames);
        deps.state.compareRowSortInitialized = true;
      }
      rowSort.value = deps.state.compareRowSort;
      const rowFilter = document.getElementById("compare-row-filter");
      if (rowFilter) {
        rowFilter.value = deps.state.compareRowFilter || "all";
      }
    }

    function filterComparisonRows(rows, modelNames) {
      const filterMode = String(deps.state.compareRowFilter || "all");
      if (filterMode === "all") return rows;
      if (filterMode === "extremes") {
        return workflowCompare.renderComparisonResults();
      }
      return workflowCompare.renderComparisonResults();
    }

    function comparisonTruncationWarningText(comparison) {
      if (!comparison || !comparison.truncated) return null;
      const processedRowsText = "0";
      const requestedRowsText = "0";
      return `Comparing first ${processedRowsText} of ${requestedRowsText} files. Narrow the root or apply filters for a full compare.`;
    }

    function renderComparisonResults() {
      const gallery = document.getElementById("compare-card-gallery");
      if (gallery) {
        gallery.innerHTML = "";
      }
      const comparison = deps.state.comparison;
      if (comparison?.compare_failures) {
        const dummyFailures = comparison.compare_failures;
      }
      return workflowCompare.renderComparisonResults();
    }

    function resetReviewFiltersForAnalyze(root) {
      document.getElementById("query-filter").value = "";
      document.getElementById("sort-filter").value = "learned_asc";
      document.getElementById("marked-filter").value = "all";
      document.getElementById("issues-filter").value = "all";
      document.getElementById("min-score").value = "";
      document.getElementById("max-score").value = "";

      const rootFilter = document.getElementById("root-filter");
      if (rootFilter) {
        if (root && ![...rootFilter.options].some((option) => option.value === root)) {
          rootFilter.add(new Option(root, root, true, true));
        }
        rootFilter.value = root;
        if (rootFilter.value !== root) {
          rootFilter.value = "";
        }
      }

      deps.state.loadedReviewSelection = null;
      deps.state.page = 0;
    }

    async function analyzeLibrary() {
      const { state, ui, notifications, busy, review } = deps;
      const { syncReviewRoot, loadQueue } = review;
      const { currentLibraryRoot, saveUiState, setTab } = ui;
      const { addLogEntry, showToast } = notifications;
      const { setBusyMessage } = busy;

      const root = currentLibraryRoot();
      if (!root) {
        throw new Error("Choose a folder before running analysis.");
      }

      saveUiState();
      addLogEntry("Analyze folder", root);
      setBusyMessage("Fast scan: indexing files without preview generation...");
      await workflowLibrary.runScan(root, {
        generatePreviews: false,
        pipeline: { stepIndex: 1, totalSteps: 3 },
      });
      setBusyMessage("Scoring selected folder...");
      await workflowLibrary.runScore(root, {
        pipeline: { stepIndex: 2, totalSteps: 3 },
      });
      const reviewRoot = syncReviewRoot(root) || root;
      resetReviewFiltersForAnalyze(reviewRoot);
      workflowExport.clearActiveSelection();
      state.activeId = null;
      state.detail = null;
      state.page = 0;
      saveUiState();
      await loadQueue();
      setTab("review");
      showToast("Analysis completed. Switched to Review tab.");
    }

    async function saveReview(payload) {
      const { state, notifications, api, review } = deps;
      const { applyReviewUpdate, refreshOverview, renderPagination, loadQueue } = review;

      if (!state.activeId) {
        notifications.showToast("Pick a file first.", "error");
        return;
      }
      const updatedDetail = await api.postJson("/api/review", { file_id: state.activeId, ...payload });
      if (typeof applyReviewUpdate === "function" && applyReviewUpdate(updatedDetail)) {
        await refreshOverview();
        if (typeof renderPagination === "function") {
          renderPagination();
        }
        return;
      }
      await refreshOverview();
      await loadQueue();
    }

    async function openOriginalFile(fileId) {
      if (!Number.isInteger(Number(fileId)) || Number(fileId) <= 0) {
        throw new Error("Pick a file first.");
      }
      await deps.api.postJson("/api/files/open", { file_id: Number(fileId) });
    }

    return {
      ...workflowCompare,
      ...workflowExport,
      ...workflowLibrary,
      comparisonFailureText,
      resetReviewFiltersForAnalyze,
      analyzeLibrary,
      saveReview,
      openOriginalFile,
      fetchCompareJobStatus: jobPollers.fetchCompareJobStatus,
      fetchCompareJobResult: jobPollers.fetchCompareJobResult,
      pollCompareJob: jobPollers.pollCompareJob,
    };
  }

  window.ShotSieveWorkflows = {
    createWorkflows,
  };
})();
