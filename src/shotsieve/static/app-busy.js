(() => {
  function createBusyController({ state, api, notify, documentRef = document } = {}) {
    const { fetchJson, postJson } = api;
    const { addLogEntry, showToast } = notify;

    function jobKindLabel(kind) {
      const labels = {
        compare: "Model comparison",
        operation: "File operation",
        preparation: "Model preparation",
        scan: "Scan",
        score: "Scoring",
      };
      return labels[String(kind || "").toLowerCase()] || "Operation";
    }

    function trackJob(job) {
      if (!job?.jobId) {
        return;
      }
      state.activeJob = { ...job };
      state.recoveryJob = null;
      if (job.kind === "operation") {
        state.operationStatusUnknown = false;
      }
      renderBusyState();
    }

    function clearTrackedJob(jobId) {
      if (!jobId) {
        return;
      }
      const job = state.activeJob?.jobId === jobId
        ? state.activeJob
        : state.recoveryJob?.jobId === jobId
          ? state.recoveryJob
          : null;
      if (!job) {
        return;
      }
      state.activeJob = null;
      state.recoveryJob = null;
      if (job.kind === "operation") {
        state.operationStatusUnknown = false;
      }
      if (state.compareJobId === jobId) state.compareJobId = null;
      if (state.scoreJobId === jobId) state.scoreJobId = null;
      if (state.scanJobId === jobId) state.scanJobId = null;
      if (state.modelPreparationJobId === jobId) state.modelPreparationJobId = null;
      if (state.operationJobId === jobId) {
        state.operationJobId = null;
        state.operationStatusPath = null;
        state.operationCancelPath = null;
      }
      renderBusyState();
    }

    function markTrackedJobUnknown(error) {
      const job = state.activeJob || state.recoveryJob;
      if (!job?.jobId) {
        return null;
      }
      const message = String(error?.message || error || "The latest job status could not be confirmed.");
      state.recoveryJob = { ...job, status: "unknown", error: message };
      state.activeJob = null;
      if (job.kind === "operation") {
        state.operationStatusUnknown = true;
      }
      renderBusyState();
      return state.recoveryJob;
    }

    function renderRecoveryState() {
      const panel = documentRef.getElementById("job-recovery-panel");
      const messageNode = documentRef.getElementById("job-recovery-message");
      if (!panel || !messageNode) {
        return;
      }
      const job = state.recoveryJob;
      const visible = Boolean(job?.jobId) && job.kind !== "operation";
      panel.classList.toggle("hidden", !visible);
      if (!visible) {
        messageNode.textContent = "";
        return;
      }
      const detail = job.error ? ` ${job.error}` : "";
      messageNode.textContent = `${jobKindLabel(job.kind)} status is unresolved.${detail} Use Check status before starting another operation.`;
    }

    function formatBusyStatusMessage(baseMessage) {
      const lines = [];
      const phaseCount = Number(state.busyPhaseCount || 0);
      const phaseIndex = Number(state.busyPhaseIndex || 0);
      const phaseLabel = String(state.busyPhaseLabel || "").trim();

      if (phaseCount > 0 && phaseIndex > 0) {
        lines.push(phaseLabel
          ? `Phase ${phaseIndex}/${phaseCount} · ${phaseLabel}`
          : `Phase ${phaseIndex}/${phaseCount}`);
      }

      const trimmedBase = String(baseMessage || "").trim();
      if (trimmedBase) {
        lines.push(trimmedBase);
      }

      return lines.length ? lines.join("\n") : "Working...";
    }

    function renderCompareBusyState() {
      const panel = documentRef.getElementById("compare-busy-panel");
      const indicator = documentRef.getElementById("compare-busy-indicator");
      const progressBar = documentRef.getElementById("compare-busy-progress");
      const progressFill = documentRef.getElementById("compare-busy-progress-fill");
      const cancelBtn = documentRef.getElementById("compare-cancel-operation");
      if (!panel || !indicator || !progressBar || !progressFill || !cancelBtn) {
        return;
      }

      const isCompareBusy = state.isBusy && state.activeOperation === "compare";
      if (!isCompareBusy) {
        panel.classList.add("hidden");
        indicator.textContent = "";
        progressBar.classList.add("hidden");
        progressFill.style.width = "0%";
        cancelBtn.classList.add("hidden");
        cancelBtn.disabled = false;
        cancelBtn.textContent = "Cancel";
        return;
      }

      panel.classList.remove("hidden");
      indicator.textContent = formatBusyStatusMessage(state.busyMessage || "Comparing models...");

      if (state.busyPercent !== null) {
        progressBar.classList.remove("hidden");
        progressBar.removeAttribute("aria-hidden");
        progressBar.setAttribute("aria-valuenow", String(Math.round(state.busyPercent)));
        progressFill.style.width = `${state.busyPercent}%`;
      } else {
        progressBar.classList.add("hidden");
        progressBar.setAttribute("aria-hidden", "true");
        progressBar.setAttribute("aria-valuenow", "0");
        progressFill.style.width = "0%";
      }

      if (state.abortController) {
        cancelBtn.classList.remove("hidden");
        cancelBtn.disabled = state.cancelPending;
        cancelBtn.textContent = state.cancelPending ? "Cancelling..." : "Cancel";
      } else {
        cancelBtn.classList.add("hidden");
        cancelBtn.disabled = false;
        cancelBtn.textContent = "Cancel";
      }
    }

    function renderBusyState() {
      const busyContainer = documentRef.getElementById("busy-status-container");
      const indicator = documentRef.getElementById("busy-indicator");
      const progressBar = documentRef.getElementById("busy-progress");
      const progressFill = documentRef.getElementById("busy-progress-fill");
      const cancelBtn = documentRef.getElementById("cancel-operation");
      const showPrimaryBusy = state.isBusy && state.activeOperation !== "compare";

      if (busyContainer) {
        busyContainer.classList.toggle("hidden", !showPrimaryBusy);
      }
      if (indicator) {
        if (showPrimaryBusy) {
          indicator.classList.remove("hidden");
          indicator.removeAttribute("aria-hidden");
          indicator.textContent = formatBusyStatusMessage(state.busyMessage || "Working...");
        } else {
          indicator.classList.add("hidden");
          indicator.setAttribute("aria-hidden", "true");
          indicator.textContent = "";
        }
      }

      if (cancelBtn) {
        if (showPrimaryBusy && state.abortController) {
          cancelBtn.classList.remove("hidden");
          cancelBtn.disabled = state.cancelPending;
          cancelBtn.textContent = state.cancelPending ? "Cancelling..." : "Cancel";
        } else {
          cancelBtn.classList.add("hidden");
          cancelBtn.disabled = false;
          cancelBtn.textContent = "Cancel";
        }
      }

      if (progressBar && progressFill) {
        if (showPrimaryBusy && state.busyPercent !== null) {
          progressBar.classList.remove("hidden");
          progressBar.removeAttribute("aria-hidden");
          progressBar.setAttribute("aria-valuenow", String(Math.round(state.busyPercent)));
          progressFill.style.width = `${state.busyPercent}%`;
        } else {
          progressBar.classList.add("hidden");
          progressBar.setAttribute("aria-hidden", "true");
          progressBar.setAttribute("aria-valuenow", "0");
          progressFill.style.width = "0%";
        }
      }

      documentRef.querySelectorAll("[data-busy-lock='true']").forEach((node) => {
        if (node.id === "cancel-operation") {
          return;
        }
        node.disabled = state.isBusy;
      });

      renderCompareBusyState();
      renderRecoveryState();
    }

    function setBusy(isBusy, message = "", operationType = null) {
      state.isBusy = isBusy;
      state.activeOperation = isBusy ? operationType : null;
      state.busyMessage = isBusy ? (message || "Working...") : "";
      state.busyPercent = null;
      state.busyPhasePercent = null;
      state.busyPhaseIndex = 0;
      state.busyPhaseCount = 0;
      state.busyPhaseLabel = "";
      state.cancelPending = false;
      if (isBusy) {
        state.busyStartTime = Date.now();
        state.abortController = new AbortController();
        state.controlAbortController = new AbortController();
      } else {
        state.busyStartTime = null;
        state.abortController = null;
        const recoveryJobId = state.recoveryJob?.jobId || null;
        if (state.compareJobId !== recoveryJobId) state.compareJobId = null;
        if (state.scoreJobId !== recoveryJobId) state.scoreJobId = null;
        if (state.scanJobId !== recoveryJobId) state.scanJobId = null;
        if (state.modelPreparationJobId !== recoveryJobId) state.modelPreparationJobId = null;
        if (state.operationJobId !== recoveryJobId) {
          state.operationJobId = null;
          state.operationStatusPath = null;
          state.operationCancelPath = null;
        }
        state.controlAbortController = null;
      }
      renderBusyState();
    }

    function setBusyMessage(message) {
      if (!state.isBusy) {
        return;
      }
      state.busyMessage = message;
      renderBusyState();
    }

    function setBusyProgress(percent) {
      if (!state.isBusy) {
        return;
      }
      if (percent === null || percent === undefined) {
        state.busyPercent = null;
      } else {
        state.busyPercent = Math.max(0, Math.min(100, Math.round(percent)));
      }
      renderBusyState();
    }

    function setBusyPhaseProgress({ percent = null, phaseIndex = 0, phaseCount = 0, phaseLabel = "" } = {}) {
      if (!state.isBusy) {
        return;
      }

      if (percent === null || percent === undefined || Number.isNaN(Number(percent))) {
        state.busyPhasePercent = null;
      } else {
        state.busyPhasePercent = Math.max(0, Math.min(100, Math.round(Number(percent))));
      }

      const normalizedCount = Math.max(0, Number(phaseCount || 0));
      const normalizedIndex = Math.max(0, Number(phaseIndex || 0));
      state.busyPhaseCount = normalizedCount;
      state.busyPhaseIndex = normalizedCount > 0 ? Math.min(normalizedCount, normalizedIndex || 1) : 0;
      state.busyPhaseLabel = String(phaseLabel || "").trim();
      renderBusyState();
    }

    function sleep(ms) {
      return new Promise((resolve) => window.setTimeout(resolve, ms));
    }

    async function waitForJobToStop(statusPath, jobId, timeoutMs = 12000) {
      if (!jobId) {
        return true;
      }
      const deadline = Date.now() + timeoutMs;
      while (Date.now() < deadline) {
        try {
          const status = await fetchJson(`${statusPath}?job_id=${encodeURIComponent(jobId)}`, {
            signal: state.controlAbortController?.signal,
          });
          const statusValue = String(status?.status || "").toLowerCase();
          if (["completed", "failed", "cancelled"].includes(statusValue)) {
            return true;
          }
          if (statusValue && statusValue !== "running") return false;
        } catch {
          // A failed status request is not proof that the worker stopped.
          continue;
        }
        await sleep(250);
      }
      return false;
    }

    async function cancelServerJob(jobId, cancelPath, statusPath) {
      if (!jobId) {
        return true;
      }
      if (!cancelPath || !statusPath) {
        return false;
      }
      await postJson(
        `${cancelPath}?job_id=${encodeURIComponent(jobId)}`,
        {},
        { signal: state.controlAbortController?.signal },
      ).catch(() => {});
      return waitForJobToStop(statusPath, jobId).catch(() => false);
    }

    async function requestServerCancellation() {
      const trackedJob = state.activeJob || state.recoveryJob;
      const operationJobId = state.operationJobId;
      const jobs = [
        [state.scanJobId, "/api/scan/cancel", "/api/scan/status"],
        [state.scoreJobId, "/api/score/cancel", "/api/score/status"],
        [state.compareJobId, "/api/compare-models/cancel", "/api/compare-models/status"],
        [state.modelPreparationJobId, "/api/models/prepare/cancel", "/api/models/prepare/status"],
        [state.operationJobId, state.operationCancelPath, state.operationStatusPath],
      ].filter(([jobId]) => jobId);
      const results = await Promise.allSettled(jobs.map(([jobId, cancelPath, statusPath]) =>
        cancelServerJob(jobId, cancelPath, statusPath)));
      const allStopped = results.every((result) => result.status === "fulfilled" && result.value === true);
      if (!allStopped) {
        const recovery = markTrackedJobUnknown(new Error(
          "Cancellation was requested, but the server has not confirmed a terminal state.",
        ));
        if (recovery?.kind === "operation") {
          const unknownResult = {
            ...(state.latestOperationResult || {}),
            action: String(state.latestOperationRequest?.payload?.mode || "operation"),
            outcome: "unknown",
            job_status: "unknown",
            fatal_error: recovery.error,
          };
          state.latestOperationResult = unknownResult;
          if (typeof state.operationResultHandler === "function") {
            state.operationResultHandler(unknownResult, state.latestOperationRequest);
          }
        }
        return false;
      }

      if (trackedJob?.kind === "operation" && trackedJob.jobId === operationJobId) {
        try {
          const result = await fetchJson(`${trackedJob.resultPath}?job_id=${encodeURIComponent(operationJobId)}`, {
            signal: state.controlAbortController?.signal,
          });
          state.latestOperationResult = result;
          if (typeof state.operationResultHandler === "function") {
            state.operationResultHandler(result, state.latestOperationRequest);
          }
        } catch (error) {
          markTrackedJobUnknown(error);
          return false;
        }
      }

      if (trackedJob?.jobId) {
        clearTrackedJob(trackedJob.jobId);
      }
      return true;
    }

    async function withBusy(message, task, options = {}) {
      if (state.recoveryJob?.jobId) {
        throw new Error(`${jobKindLabel(state.recoveryJob.kind)} status is unresolved. Use Check status before starting another operation.`);
      }
      if (state.isBusy) {
        throw new Error("Another operation is already running. Please wait for it to finish.");
      }

      setBusy(true, message, options.operationType || null);
      try {
        return await task();
      } catch (error) {
        if (error?.name === "AbortError") {
          state.cancelPending = true;
          setBusyMessage("Cancelling...");
          showToast("Cancellation requested. Stopping the current operation...", "error");
          addLogEntry("Cancelled", message);
          const cancellationConfirmed = await requestServerCancellation();
          if (typeof options.onCancelled === "function") {
            await options.onCancelled({
              confirmed: cancellationConfirmed,
              result: state.latestOperationResult,
            });
          }
          if (!cancellationConfirmed) {
            if (state.operationJobId && !state.latestOperationResult?.fatal_error) {
              const unknownResult = {
                ...(state.latestOperationResult || {}),
                action: String(state.latestOperationRequest?.payload?.mode || "operation"),
                outcome: "unknown",
                job_status: "unknown",
                fatal_error: "Cancellation was requested, but the server has not confirmed a terminal state.",
              };
              state.latestOperationResult = unknownResult;
              if (typeof state.operationResultHandler === "function") {
                state.operationResultHandler(unknownResult, state.latestOperationRequest);
              }
            }
            showToast("Cancellation was requested, but the server status is unknown. Use Check status before starting another operation.", "error");
          }
          return;
        }
        throw error;
      } finally {
        setBusy(false);
      }
    }

    return {
      formatBusyStatusMessage,
      clearTrackedJob,
      markTrackedJobUnknown,
      renderBusyState,
      renderRecoveryState,
      setBusy,
      setBusyMessage,
      setBusyPhaseProgress,
      setBusyProgress,
      trackJob,
      withBusy,
    };
  }

  window.ShotSieveBusy = {
    createBusyController,
  };
})();
