(() => {
  const operationsModule = window.ShotSieveWorkflowLibraryOperations;
  const analysisModule = window.ShotSieveWorkflowLibraryAnalysis;
  const browserModule = window.ShotSieveWorkflowLibraryBrowser;

  if (!operationsModule?.createWorkflowLibraryOperations
    || !analysisModule?.createWorkflowLibraryAnalysis
    || !browserModule?.createWorkflowLibraryBrowser) {
    throw new Error("ShotSieve library workflow modules failed to load.");
  }

  function createWorkflowLibrary(deps) {
    const operations = operationsModule.createWorkflowLibraryOperations(deps);
    const analysis = analysisModule.createWorkflowLibraryAnalysis({
      ...deps,
      operations,
    });
    const browser = browserModule.createWorkflowLibraryBrowser(deps);
    return {
      ...operations,
      ...analysis,
      ...browser,
    };
  }

  window.ShotSieveWorkflowLibrary = {
    createWorkflowLibrary,
  };
})();
