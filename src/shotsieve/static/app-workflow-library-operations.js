(() => {
  function createWorkflowLibraryOperations(deps) {
    const {
      api,
      busy,
      formatting,
      notifications,
      pollingModule,
      review,
      state,
      workflowExport,
    } = deps;

    const { fetchJson, postJson } = api;
    const {
      clearTrackedJob = () => {},
      markTrackedJobUnknown = () => {},
      setBusyMessage,
      setBusyPhaseProgress,
      trackJob = () => {},
    } = busy;
    const { formatDuration } = formatting;
    const { showToast } = notifications;
    const { refreshWorkspace } = review;
    const { pollJob, createResultFetcher, createStatusFetcher } = pollingModule;
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

    async function runTrackedJob({
      startPath,
      payload,
      kind,
      label,
      startFailureMessage,
      statusPath,
      resultPath,
      cancelPath,
      stateKey,
      poll,
      onStarted,
      onUnknown,
      shouldFinish,
      onFinished,
    }) {
      const startPayload = await postJson(startPath, payload, { signal: state.abortController?.signal });
      const jobId = String(startPayload?.job_id || "");
      if (!jobId) {
        throw new Error(startFailureMessage || `${label} failed to start.`);
      }

      state[stateKey] = jobId;
      if (typeof onStarted === "function") {
        onStarted(jobId);
      }
      trackJob({ kind, jobId, statusPath, resultPath, cancelPath, label });

      try {
        const result = await poll(jobId);
        clearTrackedJob(jobId);
        return result;
      } catch (error) {
        if (error?.name !== "AbortError") {
          markTrackedJobUnknown(error);
          if (typeof onUnknown === "function") {
            onUnknown(error, jobId);
          }
        }
        throw error;
      } finally {
        if (!state.abortController?.signal?.aborted
          && !state.recoveryJob
          && (typeof shouldFinish !== "function" || shouldFinish())) {
          state[stateKey] = null;
          if (typeof onFinished === "function") {
            onFinished();
          }
        }
      }
    }

    async function runTrackedOperation({ startPath, payload, fallbackLabel, failureMessage }) {
      return runTrackedJob({
        startPath,
        payload,
        kind: "operation",
        label: fallbackLabel,
        startFailureMessage: `${fallbackLabel} failed to start.`,
        statusPath: "/api/operations/status",
        resultPath: "/api/operations/result",
        cancelPath: "/api/operations/cancel",
        stateKey: "operationJobId",
        poll: (jobId) => pollOperationJob(jobId, { fallbackLabel, failureMessage }),
        onStarted: () => {
          state.operationStatusPath = "/api/operations/status";
          state.operationCancelPath = "/api/operations/cancel";
          state.operationStatusUnknown = false;
          state.operationProgressSignature = null;
          state.operationProgressChangedAt = Date.now();
          state.latestOperationRequest = { startPath, payload: { ...payload }, fallbackLabel, failureMessage };
        },
        onUnknown: (error) => {
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
        },
        shouldFinish: () => !state.operationStatusUnknown,
        onFinished: () => {
          state.operationStatusPath = null;
          state.operationCancelPath = null;
        },
      });
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
        || ["preparation", "score", "compare"].includes(job.kind);
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
      if (job.kind === "compare" && statusValue === "completed" && result) {
        state.comparison = result;
      }
      if (statusValue === "failed" && ["score", "compare"].includes(job.kind)) {
        const diagnostic = result?.diagnostic || result?.error_report || {};
        const detail = diagnostic.cause || result?.job_error || status?.error || `${job.label || "Analysis"} failed.`;
        const recovery = diagnostic.recovery_action || "Review the analysis diagnostics and retry when the issue is resolved.";
        showToast(`${detail} ${recovery}`, "error");
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
      const roots = (deps.ui.currentLibraryRoot() || "").split("|").map((root) => root.trim()).filter(Boolean);
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
      showToast(`Deleted ${result.deleted_count} files from disk.`, workflowExport?.operationTone ? workflowExport.operationTone(result) : "success");
      await refreshWorkspace();
    }

    return {
      runTrackedJob,
      runTrackedOperation,
      checkTrackedJob,
      checkTrackedOperation,
      clearCache,
      reviewMissingEntries,
      deleteSelectedFiles,
    };
  }

  window.ShotSieveWorkflowLibraryOperations = {
    createWorkflowLibraryOperations,
  };
})();
