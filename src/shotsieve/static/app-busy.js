(() => {
  function createBusyController({ state, api, notify, documentRef = document } = {}) {
    const { fetchJson, postJson } = api;
    const { addLogEntry, showToast } = notify;

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
      } else {
        state.busyStartTime = null;
        state.abortController = null;
        state.compareJobId = null;
        state.scoreJobId = null;
        state.scanJobId = null;
        state.operationJobId = null;
        state.operationStatusPath = null;
        state.operationCancelPath = null;
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
          const status = await fetchJson(`${statusPath}?job_id=${encodeURIComponent(jobId)}`);
          const statusValue = String(status?.status || "").toLowerCase();
          if (statusValue && statusValue !== "running") {
            return true;
          }
        } catch {
          return true;
        }
        await sleep(250);
      }
      return false;
    }

    async function cancelServerJob(jobId, cancelPath, statusPath) {
      if (!jobId) {
        return;
      }
      await postJson(`${cancelPath}?job_id=${encodeURIComponent(jobId)}`, {}).catch(() => {});
      await waitForJobToStop(statusPath, jobId).catch(() => {});
    }

    async function requestServerCancellation() {
      await Promise.allSettled([
        cancelServerJob(state.scanJobId, "/api/scan/cancel", "/api/scan/status"),
        cancelServerJob(state.scoreJobId, "/api/score/cancel", "/api/score/status"),
        cancelServerJob(state.compareJobId, "/api/compare-models/cancel", "/api/compare-models/status"),
        cancelServerJob(state.operationJobId, state.operationCancelPath, state.operationStatusPath),
      ]);
    }

    async function withBusy(message, task, options = {}) {
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
          await requestServerCancellation();
          return;
        }
        throw error;
      } finally {
        setBusy(false);
      }
    }

    return {
      formatBusyStatusMessage,
      renderBusyState,
      setBusy,
      setBusyMessage,
      setBusyPhaseProgress,
      setBusyProgress,
      withBusy,
    };
  }

  window.ShotSieveBusy = {
    createBusyController,
  };
})();
