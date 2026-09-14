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

    // Export and library workflows call each other, so compose them through
    // stable bridges rather than capturing a temporary empty module object.
    const workflowExport = {};
    const workflowLibrary = {};
    Object.assign(
      workflowExport,
      window.ShotSieveWorkflowExport.createWorkflowExport({
        ...deps,
        workflowLibrary,
      }),
    );
    Object.assign(
      workflowLibrary,
      window.ShotSieveWorkflowLibrary.createWorkflowLibrary({
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
      }),
    );

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

    // Stable public facade for app.js and app-events.js. Compare, export, and
    // library behavior is owned by their dedicated modules; this file only
    // wires their dependencies and preserves the historical facade names.
    return {
      ...workflowCompare,
      ...workflowExport,
      ...workflowLibrary,
      fetchCompareJobStatus: jobPollers.fetchCompareJobStatus,
      fetchCompareJobResult: jobPollers.fetchCompareJobResult,
      pollCompareJob: jobPollers.pollCompareJob,
    };
  }

  window.ShotSieveWorkflows = {
    createWorkflows,
  };
})();
