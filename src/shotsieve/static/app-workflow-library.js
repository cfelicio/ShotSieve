(() => {
  function createWorkflowLibrary(deps) {
    const {
      api,
      busy,
      compare,
      formatting,
      notifications,
      pollingModule,
      review,
      state,
      ui,
      workflowExport,
    } = deps;

    const { fetchJson, postJson } = api;
    const {
      clearTrackedJob = () => {},
      markTrackedJobUnknown = () => {},
      setBusyMessage,
      setBusyPhaseProgress,
      setBusyProgress,
      trackJob = () => {},
    } = busy;
    const { currentResourceProfile, scoreBatchSize } = compare;
    const { escapeHtml, formatDuration } = formatting;
    const { addLogEntry, showToast } = notifications;
    const { loadQueue, refreshWorkspace, selectFile, syncReviewRoot } = review;
    const { currentLibraryRoot, saveUiState, setTab } = ui;

    const { pollJob, pollScanJob, pollScoreJob, pollModelPreparationJob, createResultFetcher, createStatusFetcher } = pollingModule;

    const fetchOperationJobStatus = createStatusFetcher("/api/operations/status");
    const fetchOperationJobResult = createResultFetcher("/api/operations/result");

    function operationPhaseLabel(phase, fallbackLabel) {
      const normalized = String(phase || "").toLowerCase();
      const labels = {
        deleting_files: "Deleting files",
        exporting_files: "Exporting files",
        moving_files: "Moving files",
        clearing_cache: "Clearing cache",
      };
      return labels[normalized] || fallbackLabel;
    }

    function operationProgressMessage(progress, elapsedSeconds, fallbackLabel) {
      const label = operationPhaseLabel(progress?.phase, fallbackLabel);
      const processed = Number(progress?.files_processed || 0);
      const total = Number(progress?.files_total || 0);
      const countText = total > 0 ? ` (${processed}/${total})` : "";
      const elapsedText = elapsedSeconds >= 1 ? ` · ${formatDuration(elapsedSeconds)}` : "";
      const stagnantForSeconds = state.operationProgressChangedAt
        ? Math.floor((Date.now() - state.operationProgressChangedAt) / 1000)
        : 0;
      const warning = stagnantForSeconds >= 30
        ? "\nNo progress for 30 seconds. Check status or cancel if the operation is stuck."
        : "";
      return `${label}${countText}${elapsedText}${warning}`;
    }

    async function pollOperationJob(jobId, { fallbackLabel, failureMessage }) {
      return pollJob({
        jobId,
        fetchStatus: fetchOperationJobStatus,
        fetchResult: fetchOperationJobResult,
        progressMessage: (progress, elapsedSeconds) => operationProgressMessage(progress, elapsedSeconds, fallbackLabel),
        progressTotal: null,
        failureMessage,
        retainFailedResult: true,
        onProgress: ({ progress }) => {
          const progressSignature = JSON.stringify(progress || {});
          if (progressSignature !== state.operationProgressSignature) {
            state.operationProgressSignature = progressSignature;
            state.operationProgressChangedAt = Date.now();
          }
          const total = Number(progress?.files_total || 0);
          const processed = Number(progress?.files_processed || 0);
          const percent = total > 0
            ? Math.max(0, Math.min(100, (processed / total) * 100))
            : null;
          return {
            overallPercent: percent,
            phaseState: {
              percent,
              phaseIndex: 1,
              phaseCount: 1,
              phaseLabel: operationPhaseLabel(progress?.phase, fallbackLabel),
            },
          };
        },
      });
    }

    async function runTrackedOperation({ startPath, payload, fallbackLabel, failureMessage }) {
      const startPayload = await postJson(startPath, payload, { signal: state.abortController?.signal });
      const jobId = String(startPayload?.job_id || "");
      if (!jobId) {
        throw new Error(`${fallbackLabel} failed to start.`);
      }

      state.operationJobId = jobId;
      state.operationStatusPath = "/api/operations/status";
      state.operationCancelPath = "/api/operations/cancel";
      state.operationStatusUnknown = false;
      state.operationProgressSignature = null;
      state.operationProgressChangedAt = Date.now();
      state.latestOperationRequest = { startPath, payload: { ...payload }, fallbackLabel, failureMessage };
      trackJob({
        kind: "operation",
        jobId,
        statusPath: "/api/operations/status",
        resultPath: "/api/operations/result",
        cancelPath: "/api/operations/cancel",
        label: fallbackLabel,
      });

      try {
        const result = await pollOperationJob(jobId, { fallbackLabel, failureMessage });
        clearTrackedJob(jobId);
        return result;
      } catch (error) {
        if (error?.name !== "AbortError") {
          markTrackedJobUnknown(error);
          state.operationStatusUnknown = true;
          const unknownResult = {
            ...(state.latestOperationResult || {}),
            action: String(payload?.mode || "operation"),
            outcome: "unknown",
            job_status: "unknown",
            fatal_error: String(error?.message || error),
          };
          state.latestOperationResult = unknownResult;
          if (typeof state.operationResultHandler === "function") {
            state.operationResultHandler(unknownResult, state.latestOperationRequest);
          }
        }
        throw error;
      } finally {
        if (!state.abortController?.signal?.aborted && !state.recoveryJob && !state.operationStatusUnknown) {
          state.operationJobId = null;
          state.operationStatusPath = null;
          state.operationCancelPath = null;
        }
      }
    }

    async function checkTrackedJob() {
      const job = state.recoveryJob;
      if (!job?.jobId) {
        throw new Error("No unresolved job status is available to check.");
      }
      let status;
      try {
        status = await fetchJson(`${job.statusPath}?job_id=${encodeURIComponent(job.jobId)}`);
      } catch (error) {
        markTrackedJobUnknown(error);
        throw error;
      }
      const statusValue = String(status?.status || "").toLowerCase();
      if (statusValue === "running") {
        showToast(`${job.label || "The operation"} is still running. Check again shortly.`);
        return null;
      }
      if (!["completed", "failed"].includes(statusValue)) {
        const error = new Error(`The server returned an unrecognized job status: ${status?.status || "missing"}`);
        markTrackedJobUnknown(error);
        throw error;
      }

      let result = null;
      const shouldFetchResult = job.kind === "operation"
        || statusValue === "completed"
        || job.kind === "preparation";
      if (shouldFetchResult) {
        try {
          result = await fetchJson(`${job.resultPath}?job_id=${encodeURIComponent(job.jobId)}`);
        } catch (error) {
          markTrackedJobUnknown(error);
          throw error;
        }
      }

      clearTrackedJob(job.jobId);
      if (job.kind === "operation" && result) {
        state.latestOperationResult = result;
      }
      if (job.kind === "compare" && result) {
        state.comparison = result;
      }
      await refreshWorkspace();
      if (job.kind === "operation") {
        return result;
      }
      return result || {
        job_id: job.jobId,
        job_status: statusValue,
        job_error: status?.error || null,
        progress: status?.progress || null,
      };
    }

    async function checkTrackedOperation() {
      return checkTrackedJob();
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

      const scanJobStart = await postJson("/api/scan/start", {
        roots: root.split("|").map(r => r.trim()).filter(Boolean),
        extensions: document.getElementById("extensions-input").value.trim() || null,
        ignore_rules: (document.getElementById("ignore-rules-input")?.value || "")
          .split("\n")
          .map(rule => rule.trim())
          .filter(Boolean),
        recursive: document.getElementById("recursive-toggle").checked,
        rescan_all: false,
        generate_previews: generatePreviews,
        files_total_hint: filesTotalRef.value,
        resource_profile: currentResourceProfile(),
      }, { signal: state.abortController?.signal });

      const scanJobId = String(scanJobStart?.job_id || "");
      if (!scanJobId) {
        throw new Error("Scan job failed to start.");
      }

      state.scanJobId = scanJobId;
      trackJob({
        kind: "scan",
        jobId: scanJobId,
        statusPath: "/api/scan/status",
        resultPath: "/api/scan/result",
        cancelPath: "/api/scan/cancel",
        label: "Scan",
      });
      let result = null;
      try {
        result = await pollScanJob(scanJobId, { filesTotalRef, pipeline });
        clearTrackedJob(scanJobId);
      } catch (error) {
        if (error?.name !== "AbortError") {
          markTrackedJobUnknown(error);
        }
        throw error;
      } finally {
        if (!state.abortController?.signal?.aborted && !state.recoveryJob) {
          state.scanJobId = null;
        }
      }

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

      addLogEntry("Scan completed", `Seen ${result.files_seen}, added ${result.files_added}, updated ${result.files_updated}, removed ${result.files_removed}.`);
      showToast("Scan completed.");
      await refreshWorkspace();
      syncReviewRoot(root);
    }

    async function runScore(rootOverride = null, { pipeline = null } = {}) {
      const root = rootOverride || currentLibraryRoot() || null;

      const selectedModel = document.getElementById("model-select").value || state.options?.default_scoring_mode || state.options?.learned_models?.[0] || "";
      if (!selectedModel) {
        showToast("No learned IQA model is currently available. Check the runtime setup in Settings.", "error");
        return;
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

      const scoreJobStart = await postJson("/api/score/start", {
        root,
        learned_backend_name: learnedBackend,
        device: runtimeTarget || null,
        batch_size: requestedBatchSize,
        force: false,
        resource_profile: currentResourceProfile(),
      }, { signal: state.abortController?.signal });

      const scoreJobId = String(scoreJobStart?.job_id || "");
      if (!scoreJobId) {
        throw new Error("Score job failed to start.");
      }

      state.scoreJobId = scoreJobId;
      trackJob({
        kind: "score",
        jobId: scoreJobId,
        statusPath: "/api/score/status",
        resultPath: "/api/score/result",
        cancelPath: "/api/score/cancel",
        label: "Scoring",
      });
      let result = null;
      try {
        result = await pollScoreJob(scoreJobId, { rowsTotal, pipeline });
        clearTrackedJob(scoreJobId);
      } catch (error) {
        if (error?.name !== "AbortError") {
          markTrackedJobUnknown(error);
        }
        throw error;
      } finally {
        if (!state.abortController?.signal?.aborted && !state.recoveryJob) {
          state.scoreJobId = null;
        }
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

      addLogEntry("Score completed", `Scored ${result.files_scored || 0}, learned ${result.learned_scored || 0}, skipped ${result.files_skipped || 0}, failed ${result.files_failed || 0}.`);
      showToast("Scoring completed.");
      await refreshWorkspace();
      syncReviewRoot(root);
    }

    async function prepareSelectedModel() {
      const model = document.getElementById("model-select")?.value || state.options?.default_scoring_mode || "";
      if (!model) {
        throw new Error("No supported learned-IQA model is available to prepare.");
      }
      setBusyPhaseProgress({ percent: null, phaseIndex: 1, phaseCount: 3, phaseLabel: "Preparing model" });
      setBusyProgress(0);
      setBusyMessage(`Preparing ${model} on CPU. First use may download model assets...`);
      const startPayload = await postJson("/api/models/prepare/start", { model }, { signal: state.abortController?.signal });
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
          const detail = result?.error || result?.error_report?.cause || "Model preparation failed.";
          showToast(`Model preparation failed: ${detail}`, "error");
          addLogEntry("Model preparation failed", detail);
          return result;
        }
        setBusyProgress(100);
        setBusyPhaseProgress({ percent: 100, phaseIndex: 3, phaseCount: 3, phaseLabel: "Model prepared" });
        showToast(`${model} is prepared and passed a CPU validation inference.`);
        addLogEntry("Model prepared", `${model} is ready for use.`);
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
      addLogEntry("Analyze folder", root);
      setBusyMessage("Fast scan: indexing files without preview generation...");
      await runScan(root, {
        generatePreviews: false,
        pipeline: { stepIndex: 1, totalSteps: 3 },
      });
      setBusyMessage("Scoring selected folder...");
      await runScore(root, {
        pipeline: { stepIndex: 2, totalSteps: 3 },
      });
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

    function renderLibraryRoots() {
      const listContainer = document.getElementById("library-roots-list");
      const rootStr = currentLibraryRoot();
      const roots = rootStr.split("|").map(r => r.trim()).filter(Boolean);

      const decisionRoot = document.getElementById("decision-csv-root");
      if (decisionRoot) {
        const previousRoot = decisionRoot.value;
        decisionRoot.replaceChildren(new Option("Choose a library root", ""));
        roots.forEach((root) => decisionRoot.add(new Option(root, root)));
        if (roots.includes(previousRoot)) {
          decisionRoot.value = previousRoot;
        } else if (roots.length === 1) {
          decisionRoot.value = roots[0];
        }
      }

      if (!listContainer) return;

      if (roots.length === 0) {
        listContainer.innerHTML = `<p class="muted">No folders selected yet. Click "Add Folder" to add directories to your library.</p>`;
        return;
      }

      listContainer.innerHTML = roots.map((rootPath) => `
        <div class="library-root-item">
          <span class="library-root-path">${escapeHtml(rootPath)}</span>
          <button type="button" class="library-root-remove" data-path="${escapeHtml(rootPath)}" aria-label="Remove folder">✕</button>
        </div>
      `).join("");

      listContainer.querySelectorAll(".library-root-remove").forEach((button) => {
        button.addEventListener("click", () => {
          const pathToRemove = button.dataset.path;
          const updatedRoots = roots.filter(r => r !== pathToRemove);
          const hiddenInput = document.getElementById("library-root-input");
          if (hiddenInput) {
            hiddenInput.value = updatedRoots.join("|");
            hiddenInput.dispatchEvent(new Event("input", { bubbles: true }));
            hiddenInput.dispatchEvent(new Event("change", { bubbles: true }));
            renderLibraryRoots();
          }
        });
      });
    }

    async function downloadDecisionCsv() {
      const root = document.getElementById("decision-csv-root")?.value || "";
      const decision = document.getElementById("decision-csv-decision")?.value || "both";
      if (!root) {
        throw new Error("Choose a library root before downloading decisions.");
      }
      const response = await fetch(`/api/review/decisions.csv?root=${encodeURIComponent(root)}&decision=${encodeURIComponent(decision)}`);
      if (!response.ok) {
        let message = `Decision CSV request failed (${response.status}).`;
        try {
          const payload = await response.json();
          message = payload?.error || message;
        } catch {
          // Keep the status-based message for non-JSON server errors.
        }
        throw new Error(message);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "shotsieve-decisions.csv";
      link.click();
      URL.revokeObjectURL(url);
      showToast("Decision CSV downloaded.");
    }

    function installDecisionCsvEvents() {
      const button = document.getElementById("download-decisions-csv");
      if (!button || button.dataset.eventsInstalled === "true") return;
      button.dataset.eventsInstalled = "true";
      button.addEventListener("click", () => {
        downloadDecisionCsv().catch(handleError);
      });
    }

    async function clearCache(scope, message) {
      setBusyPhaseProgress({ percent: 0, phaseIndex: 1, phaseCount: 1, phaseLabel: "Clearing cache" });
      const result = await runTrackedOperation({
        startPath: "/api/cache/clear/start",
        payload: { scope },
        fallbackLabel: "Clearing cache",
        failureMessage: "Cache action failed.",
      });
      if (workflowExport?.presentOperationResult && (result?.outcome || Array.isArray(result?.items))) {
        workflowExport.presentOperationResult(result, state.latestOperationRequest);
      }
      addLogEntry("Cache action", `${message}: files ${result.files}, scores ${result.scores}, review ${result.review}.`);
      showToast(message, workflowExport?.operationTone ? workflowExport.operationTone(result) : "success");
      if (scope === "all") {
        if (workflowExport?.clearActiveSelection) {
          workflowExport.clearActiveSelection();
        }
        state.activeId = null;
        state.detail = null;
      }
      await refreshWorkspace();
    }

    function missingCleanupConfirmation(previews) {
      const lines = [
        "Review the cached entries that will be removed:",
        "",
        "Original files on disk will not be touched.",
        "",
      ];
      for (const preview of previews) {
        lines.push(`Root: ${preview.root}`);
        lines.push(`Cached entries: ${Number(preview.candidate_count || 0).toLocaleString()}`);
        lines.push(`Review decisions removed: ${Number(preview.affected_review_count || 0).toLocaleString()}`);
        for (const candidate of (preview.candidates || [])) {
          const decision = Number(candidate.review_count || 0) > 0
            ? ` [review: ${candidate.decision_state || "recorded"}]`
            : "";
          lines.push(`  - ${candidate.path}${decision}`);
        }
        lines.push("");
      }
      lines.push("Continue with this cleanup?");
      return lines.join("\n");
    }

    async function reviewMissingEntries() {
      const roots = (currentLibraryRoot() || "").split("|").map((root) => root.trim()).filter(Boolean);
      if (!roots.length) {
        throw new Error("Choose a folder before reviewing missing entries.");
      }

      setBusyMessage("Checking selected library for missing entries...");
      const previews = await Promise.all(roots.map((root) => fetchJson(
        `/api/cache/missing/preview?root=${encodeURIComponent(root)}`,
        { signal: state.abortController?.signal },
      )));
      const unknownPreview = previews.find((preview) => preview?.status === "unknown");
      if (unknownPreview) {
        throw new Error(unknownPreview.error || "The selected library could not be verified.");
      }

      const readyPreviews = previews.filter((preview) => preview?.status === "ready" && Number(preview.candidate_count || 0) > 0);
      if (!readyPreviews.length) {
        showToast("No missing cached entries found.");
        return;
      }
      if (!window.confirm(missingCleanupConfirmation(readyPreviews))) {
        return;
      }

      setBusyMessage("Applying confirmed missing-entry cleanup...");
      let removedCount = 0;
      let reviewRemovedCount = 0;
      for (const preview of readyPreviews) {
        const result = await postJson("/api/cache/missing/apply", {
          root: preview.root,
          token: preview.token,
          candidate_ids: (preview.candidates || []).map((candidate) => Number(candidate.id)),
        }, { signal: state.abortController?.signal });
        if (result?.status === "refresh_required") {
          throw new Error("The catalog changed after the preview. Review missing entries again before applying cleanup.");
        }
        if (result?.status === "unknown") {
          throw new Error(result.error || "The selected library could not be verified during cleanup.");
        }
        if (result?.status !== "applied") {
          throw new Error("Missing-entry cleanup did not complete.");
        }
        removedCount += Number(result.removed_count || 0);
        reviewRemovedCount += Number(result.review_removed_count || 0);
      }

      addLogEntry(
        "Missing-entry cleanup",
        `Removed ${removedCount} cached entr${removedCount === 1 ? "y" : "ies"} and ${reviewRemovedCount} review decision(s).`,
      );
      showToast(`Removed ${removedCount} missing cached entr${removedCount === 1 ? "y" : "ies"}.`);
      await refreshWorkspace();
    }

    async function deleteSelectedFiles() {
      const selectionRequest = workflowExport ? workflowExport.activeSelectionRequest() : { count: 0 };
      if (!selectionRequest.count) {
        throw new Error("Select one or more items first.");
      }
      if (!window.confirm(`Delete ${selectionRequest.count} file(s) from disk? This cannot be undone.`)) {
        return;
      }
      setBusyPhaseProgress({ percent: 0, phaseIndex: 1, phaseCount: 1, phaseLabel: "Deleting files" });
      const result = await runTrackedOperation({
        startPath: "/api/files/delete/start",
        payload: { ...selectionRequest, delete_from_disk: true, count: selectionRequest.count },
        fallbackLabel: "Deleting files",
        failureMessage: "Delete failed.",
      });
      if (workflowExport?.presentOperationResult) {
        workflowExport.presentOperationResult(result, state.latestOperationRequest);
      }
      addLogEntry("Disk delete", `Deleted ${result.deleted_count}, failed ${result.failed_count}.`);
      showToast(`Deleted ${result.deleted_count} files from disk.`, workflowExport?.operationTone ? workflowExport.operationTone(result) : "success");
      await refreshWorkspace();
    }

    async function navigateSelection(step) {
      if (!state.queue.length) return;
      const currentIndex = state.queue.findIndex((item) => item.id === state.activeId);
      if (currentIndex === -1) {
        await selectFile(state.queue[0].id);
        return;
      }

      const nextIndex = currentIndex + step;
      if (nextIndex >= 0 && nextIndex < state.queue.length) {
        await selectFile(state.queue[nextIndex].id);
        return;
      }

      if (step > 0 && ((state.page + 1) * state.pageSize) < state.totalFiles) {
        state.page += 1;
        await loadQueue();
        if (state.queue.length) {
          await selectFile(state.queue[0].id);
        }
        return;
      }

      if (step < 0 && state.page > 0) {
        state.page -= 1;
        await loadQueue();
        if (state.queue.length) {
          await selectFile(state.queue[state.queue.length - 1].id);
        }
      }
    }

    async function openOriginalFile(fileId) {
      if (!Number.isInteger(Number(fileId)) || Number(fileId) <= 0) {
        throw new Error("Pick a file first.");
      }
      await postJson("/api/files/open", { file_id: Number(fileId) });
    }

    async function openBrowser(targetId) {
      state.browserTarget = targetId;
      const dialog = document.getElementById("folder-browser");
      if (!dialog.open) {
        dialog.showModal();
      }
      const roots = await fetchJson("/api/fs/roots");
      const rootContainer = document.getElementById("browser-roots");
      rootContainer.innerHTML = roots.items.map((item) => `<button type="button" class="ghost browser-root" data-path="${escapeHtml(item.path)}">${escapeHtml(item.name)}</button>`).join("");
      rootContainer.querySelectorAll(".browser-root").forEach((button) => {
        button.addEventListener("click", () => browseDirectory(button.dataset.path).catch(handleError));
      });

      let startPath = "";
      const targetEl = document.getElementById(targetId);
      if (targetEl && targetEl.value) {
        startPath = targetEl.value;
      } else {
        const currentLibraryVal = document.getElementById("library-root-input")?.value;
        if (currentLibraryVal) {
          const libraryRoots = currentLibraryVal.split("|").map(r => r.trim()).filter(Boolean);
          if (libraryRoots.length > 0) {
            startPath = libraryRoots[libraryRoots.length - 1];
          }
        }
      }
      if (!startPath) {
        startPath = state.browserPath || roots.items[0]?.path || "/";
      }

      try {
        await browseDirectory(startPath);
      } catch (err) {
        console.warn("Failed to navigate to browser start path, falling back to root:", err);
        const fallback = roots.items[0]?.path || "/";
        await browseDirectory(fallback).catch(handleError);
      }
    }

    function buildBreadcrumbItems(rawPath) {
      const isUnc = rawPath.startsWith("\\\\") || rawPath.startsWith("//");
      const normPath = rawPath.replace(/\\/g, "/");

      if (isUnc) {
        const parts = normPath.slice(2).split("/").filter(Boolean);
        let accumulated = "\\\\";
        return parts.map((part, index) => {
          if (index === 0) {
            accumulated += part;
          } else {
            accumulated += "\\" + part;
          }
          return `<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(part)}</button>`;
        });
      }

      const isWindowsDrive = /^[a-zA-Z]:/.test(normPath);
      if (isWindowsDrive) {
        const driveLetter = normPath.slice(0, 2);
        const rest = normPath.slice(2).split("/").filter(Boolean);
        let accumulated = driveLetter + "\\";
        const crumbs = [
          `<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(driveLetter)}</button>`
        ];
        for (const part of rest) {
          accumulated += (accumulated.endsWith("\\") ? "" : "\\") + part;
          crumbs.push(`<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(part)}</button>`);
        }
        return crumbs;
      }

      const parts = normPath.split("/").filter(Boolean);
      let accumulated = "/";
      const crumbs = [
        `<button type="button" class="breadcrumb-item" data-path="/">${escapeHtml("/")}</button>`
      ];
      for (const part of parts) {
        accumulated += (accumulated.endsWith("/") ? "" : "/") + part;
        crumbs.push(`<button type="button" class="breadcrumb-item" data-path="${escapeHtml(accumulated)}">${escapeHtml(part)}</button>`);
      }
      return crumbs;
    }

    let activeBrowseSeq = 0;

    async function browseDirectory(path) {
      const currentSeq = ++activeBrowseSeq;
      const list = document.getElementById("browser-list");
      const pathInput = document.getElementById("browser-path");
      state.browserPath = null;
      if (pathInput) pathInput.value = path;

      if (list && !list.children.length) {
        list.innerHTML = `<p class="muted">Loading directory contents...</p>`;
      }

      try {
        const payload = await fetchJson(`/api/fs/list?path=${encodeURIComponent(path)}`);
        if (currentSeq !== activeBrowseSeq) {
          return;
        }

        state.browserPath = payload.path;
        if (pathInput) pathInput.value = payload.path;

        if (list) {
          list.innerHTML = payload.items.length
            ? payload.items.map((item) => `
                <button type="button" class="browser-item" data-path="${escapeHtml(item.path)}">
                  <strong>${escapeHtml(item.name)}</strong>
                  <span class="muted">${escapeHtml(item.path)}</span>
                </button>
              `).join("")
            : `<p class="muted">No subdirectories available.</p>`;

          list.querySelectorAll(".browser-item").forEach((button) => {
            button.addEventListener("click", () => browseDirectory(button.dataset.path).catch(handleError));
          });
        }

        const breadcrumbsContainer = document.getElementById("browser-breadcrumbs");
        if (breadcrumbsContainer) {
          breadcrumbsContainer.innerHTML = buildBreadcrumbItems(payload.path).join('<span class="breadcrumb-separator">/</span>');
          breadcrumbsContainer.querySelectorAll(".breadcrumb-item").forEach((btn) => {
            btn.addEventListener("click", () => browseDirectory(btn.dataset.path).catch(handleError));
          });
        }
      } catch (err) {
        if (currentSeq !== activeBrowseSeq) {
          return;
        }
        if (list) {
          list.innerHTML = `<p class="muted danger-text">Could not open folder: ${escapeHtml(err.message || "Access denied")}</p>`;
        }
      }
    }

    function chooseBrowserPath() {
      if (!state.browserTarget) return;
      const selectedPath = state.browserPath || document.getElementById("browser-path")?.value?.trim();
      if (!selectedPath) return;
      const targetInput = document.getElementById(state.browserTarget);
      if (!targetInput) {
        return;
      }
      targetInput.value = selectedPath;
      targetInput.dispatchEvent(new Event("input", { bubbles: true }));
      targetInput.dispatchEvent(new Event("change", { bubbles: true }));
      document.getElementById("folder-browser").close();
    }

    function handleError(error) {
      console.error(error);
      let message = error?.message || "Unexpected error";
      if (message === "Failed to fetch") {
        message = "The local server request failed. If an analysis is still running, wait for completion before retrying.";
      }
      showToast(message, "error");
      addLogEntry("Error", message);
    }

    return {
      runTrackedOperation,
      checkTrackedJob,
      checkTrackedOperation,
      resetReviewFiltersForAnalyze,
      runScan,
      runScore,
      prepareSelectedModel,
      analyzeLibrary,
      renderLibraryRoots,
      downloadDecisionCsv,
      installDecisionCsvEvents,
      clearCache,
      reviewMissingEntries,
      deleteSelectedFiles,
      navigateSelection,
      openOriginalFile,
      openBrowser,
      browseDirectory,
      chooseBrowserPath,
      handleError,
    };
  }

  window.ShotSieveWorkflowLibrary = {
    createWorkflowLibrary,
  };
})();
