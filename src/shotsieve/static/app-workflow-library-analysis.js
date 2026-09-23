(() => {
  function createWorkflowLibraryAnalysis(deps) {
    const {
      api,
      busy,
      compare,
      notifications,
      pollingModule,
      review,
      state,
      ui,
      workflowExport,
      operations,
    } = deps;

    const { postJson } = api;
    const {
      markTrackedJobUnknown = () => {},
      clearTrackedJob = () => {},
      setBusyMessage,
      setBusyPhaseProgress,
      setBusyProgress,
      trackJob = () => {},
    } = busy;
    const { currentResourceProfile, scoreBatchSize } = compare;
    const { showToast } = notifications;
    const { loadQueue, refreshWorkspace, syncReviewRoot } = review;
    const { currentLibraryRoot, saveUiState, setTab } = ui;
    const { pollScanJob, pollScoreJob, pollModelPreparationJob } = pollingModule;
    const { runTrackedJob, runTrackedOperation } = operations;

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

      state.loadedReviewSelection = null;
      state.page = 0;
    }

    async function runScan(rootOverride = null, { generatePreviews = true, pipeline = null } = {}) {
      const root = rootOverride || currentLibraryRoot();
      if (!root) {
        throw new Error("Choose a folder before running analysis.");
      }

      if (pipeline) {
        setBusyPhaseProgress({
          percent: null,
          phaseIndex: pipeline.stepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Scanning library",
        });
      } else {
        setBusyPhaseProgress({ percent: null, phaseIndex: 1, phaseCount: 2, phaseLabel: "Indexing files" });
      }

      const filesTotalRef = { value: null };
      try {
        const estimate = await postJson("/api/score-estimate", { root });
        const cachedTotal = Number(estimate.rows_total || 0);
        filesTotalRef.value = cachedTotal > 0 ? cachedTotal : null;
      } catch {
        filesTotalRef.value = null;
      }

      setBusyProgress(filesTotalRef.value && filesTotalRef.value > 0 ? 0 : null);
      setBusyMessage(generatePreviews
        ? "Scanning and generating previews..."
        : "Scanning metadata only for faster discovery...");

      const scanPayload = {
        roots: root.split("|").map((r) => r.trim()).filter(Boolean),
        extensions: document.getElementById("extensions-input").value.trim() || null,
        max_decode_megapixels: document.getElementById("max-decode-megapixels")?.value || null,
        ignore_rules: (document.getElementById("ignore-rules-input")?.value || "")
          .split("\n")
          .map((rule) => rule.trim())
          .filter(Boolean),
        recursive: document.getElementById("recursive-toggle").checked,
        rescan_all: false,
        generate_previews: generatePreviews,
        files_total_hint: filesTotalRef.value,
        resource_profile: currentResourceProfile(),
      };
      const result = await runTrackedJob({
        startPath: "/api/scan/start",
        payload: scanPayload,
        kind: "scan",
        label: "Scan",
        startFailureMessage: "Scan job failed to start.",
        statusPath: "/api/scan/status",
        resultPath: "/api/scan/result",
        cancelPath: "/api/scan/cancel",
        stateKey: "scanJobId",
        poll: (jobId) => pollScanJob(jobId, { filesTotalRef, pipeline }),
      });

      if (pipeline) {
        const donePercent = (Number(pipeline.stepIndex) / Number(pipeline.totalSteps)) * 100;
        setBusyProgress(Math.min(100, Math.max(0, Math.round(donePercent))));
        setBusyPhaseProgress({
          percent: 100,
          phaseIndex: pipeline.stepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Scanning library",
        });
      } else {
        setBusyProgress(100);
        setBusyPhaseProgress({ percent: 100, phaseIndex: 2, phaseCount: 2, phaseLabel: "Scanning files" });
      }
      setBusyMessage(`Scan completed. Processed ${result.files_seen} file(s).`);

      showToast("Scan completed.");
      await refreshWorkspace();
      syncReviewRoot(root);
    }

    async function runScore(rootOverride = null, { pipeline = null } = {}) {
      const root = rootOverride || currentLibraryRoot() || null;

      const selectedModel = document.getElementById("model-select").value || state.options?.default_scoring_mode || state.options?.learned_models?.[0] || "";
      if (!selectedModel) {
        showToast("No learned IQA model is currently available. Check the runtime setup in Settings.", "error");
        return false;
      }
      const learnedBackend = selectedModel;
      const runtimeTarget = document.getElementById("device-select").value || "auto";
      const requestedBatchSize = scoreBatchSize(learnedBackend, runtimeTarget, state.options?.learned?.recommended_batch_sizes);
      let rowsTotal = null;

      if (pipeline) {
        setBusyPhaseProgress({
          percent: 0,
          phaseIndex: pipeline.stepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Loading model",
        });
      } else {
        setBusyPhaseProgress({ percent: 0, phaseIndex: 1, phaseCount: 3, phaseLabel: "Loading model" });
      }

      try {
        const estimate = await postJson("/api/score-estimate", { root });
        rowsTotal = Number(estimate.rows_total || 0);
        if (rowsTotal > 0) {
          setBusyProgress(0);
          setBusyMessage(`Scoring... 0/${rowsTotal} (0%)`);
        }
      } catch {
        rowsTotal = null;
      }

      const scorePayload = {
        root,
        learned_backend_name: learnedBackend,
        device: runtimeTarget || null,
        batch_size: requestedBatchSize,
        max_decode_megapixels: document.getElementById("max-decode-megapixels")?.value || null,
        force: false,
        resource_profile: currentResourceProfile(),
      };
      const result = await runTrackedJob({
        startPath: "/api/score/start",
        payload: scorePayload,
        kind: "score",
        label: "Scoring",
        startFailureMessage: "Score job failed to start.",
        statusPath: "/api/score/status",
        resultPath: "/api/score/result",
        cancelPath: "/api/score/cancel",
        stateKey: "scoreJobId",
        poll: (jobId) => pollScoreJob(jobId, { rowsTotal, pipeline }),
      });

      if (result?.job_status === "failed" || result?.diagnostic) {
        const diagnostic = result?.diagnostic || result?.error_report || {};
        const cause = diagnostic.cause || result?.job_error || "Scoring failed.";
        const recovery = diagnostic.recovery_action || "Open Settings and choose Prepare selected model before retrying.";
        showToast(`${cause} ${recovery}`, "error");
        await refreshWorkspace();
        return false;
      }

      if (pipeline) {
        const doneStepIndex = Math.min(Number(pipeline.totalSteps), Number(pipeline.stepIndex) + 1);
        const donePercent = (doneStepIndex / Number(pipeline.totalSteps)) * 100;
        setBusyProgress(Math.min(100, Math.max(0, Math.round(donePercent))));
        setBusyPhaseProgress({
          percent: 100,
          phaseIndex: doneStepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Model scoring complete",
        });
      } else {
        setBusyProgress(100);
        setBusyPhaseProgress({ percent: 100, phaseIndex: 3, phaseCount: 3, phaseLabel: "Model scoring complete" });
      }
      setBusyMessage(`Scoring completed. Processed ${result.rows_loaded || 0} row(s).`);

      showToast("Scoring completed.");
      await refreshWorkspace();
      syncReviewRoot(root);
      return true;
    }

    async function prepareSelectedModel() {
      const model = document.getElementById("model-select")?.value || state.options?.default_scoring_mode || "";
      if (!model) {
        throw new Error("No supported learned-IQA model is available to prepare.");
      }
      const catalogEntry = Array.isArray(state.options?.learned?.model_catalog)
        ? state.options.learned.model_catalog.find((entry) => entry?.canonical_id === model)
        : null;
      const supportedRuntimes = Array.isArray(catalogEntry?.supported_runtimes)
        ? catalogEntry.supported_runtimes.map((runtime) => String(runtime).toLowerCase())
        : ["cpu"];
      const acceleratorOnly = !supportedRuntimes.includes("cpu");
      const selectedDevice = document.getElementById("device-select")?.value || "auto";
      const preparationDevice = acceleratorOnly && supportedRuntimes.includes(String(selectedDevice).toLowerCase())
        ? selectedDevice
        : (acceleratorOnly ? "auto" : "cpu");
      setBusyPhaseProgress({ percent: null, phaseIndex: 1, phaseCount: 3, phaseLabel: "Preparing model" });
      setBusyProgress(0);
      setBusyMessage(`Preparing ${model} on ${preparationDevice.toUpperCase()}. First use may download model assets...`);
      const startPayload = await postJson("/api/models/prepare/start", { model, device: preparationDevice }, { signal: state.abortController?.signal });
      const jobId = String(startPayload?.job_id || "");
      if (!jobId) {
        throw new Error("Model preparation failed to start.");
      }
      state.modelPreparationJobId = jobId;
      trackJob({
        kind: "preparation",
        jobId,
        statusPath: "/api/models/prepare/status",
        resultPath: "/api/models/prepare/result",
        cancelPath: "/api/models/prepare/cancel",
        label: "Model preparation",
      });
      try {
        const result = await pollModelPreparationJob(jobId);
        clearTrackedJob(jobId);
        if (result?.job_status === "failed" || ["failed", "runtime_unavailable"].includes(String(result?.state || ""))) {
          const diagnostic = result?.error_report || result?.diagnostic || {};
          const detail = result?.error || diagnostic.cause || "Model preparation failed.";
          const recovery = result?.recovery_action || diagnostic.recovery_action || "Retry preparation from Settings.";
          showToast(`Model preparation failed: ${detail} ${recovery}`, "error");
          return result;
        }
        setBusyProgress(100);
        setBusyPhaseProgress({ percent: 100, phaseIndex: 3, phaseCount: 3, phaseLabel: "Model prepared" });
        const testedRuntime = String(result?.tested_runtime || result?.actual_runtime || preparationDevice).toUpperCase();
        showToast(`${model} is prepared and passed a ${testedRuntime} validation inference.`);
        return result;
      } catch (error) {
        if (error?.name !== "AbortError") {
          markTrackedJobUnknown(error);
        }
        throw error;
      } finally {
        if (!state.abortController?.signal?.aborted && !state.recoveryJob) {
          state.modelPreparationJobId = null;
        }
        await refreshWorkspace();
      }
    }

    async function analyzeLibrary() {
      const root = currentLibraryRoot();
      if (!root) {
        throw new Error("Choose a folder before running analysis.");
      }

      saveUiState();
      setBusyMessage("Fast scan: indexing files without preview generation...");
      await runScan(root, {
        generatePreviews: false,
        pipeline: { stepIndex: 1, totalSteps: 3 },
      });
      setBusyMessage("Scoring selected folder...");
      const scoreSucceeded = await runScore(root, {
        pipeline: { stepIndex: 2, totalSteps: 3 },
      });
      if (!scoreSucceeded) {
        return;
      }
      const reviewRoot = syncReviewRoot(root) || root;
      resetReviewFiltersForAnalyze(reviewRoot);
      if (workflowExport?.clearActiveSelection) {
        workflowExport.clearActiveSelection();
      }
      state.activeId = null;
      state.detail = null;
      state.page = 0;
      saveUiState();
      await loadQueue();
      setTab("review");
      showToast("Analysis completed. Switched to Review tab.");
    }

    return {
      resetReviewFiltersForAnalyze,
      runScan,
      runScore,
      prepareSelectedModel,
      analyzeLibrary,
    };
  }

  window.ShotSieveWorkflowLibraryAnalysis = {
    createWorkflowLibraryAnalysis,
  };
})();
