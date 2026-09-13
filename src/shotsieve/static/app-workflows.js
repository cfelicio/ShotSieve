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
        createResultFetcher: jobPollers.createResultFetcher,
        createStatusFetcher: jobPollers.createStatusFetcher,
        pollJob: jobPollers.pollJob,
        pollScanJob: jobPollers.pollScanJob,
        pollScoreJob: jobPollers.pollScoreJob,
        pollModelPreparationJob: jobPollers.pollModelPreparationJob,
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
        pipelineOverallPercent: jobPollers.pipelineOverallPercent || pollingModule?.pipelineOverallPercent,
      },
      workflowLibrary,
    };
    const workflowCompare = window.ShotSieveWorkflowCompare.createWorkflowCompare(compareDeps);

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
