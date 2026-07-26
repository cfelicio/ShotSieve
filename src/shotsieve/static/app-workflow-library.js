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
    const { setBusyMessage, setBusyPhaseProgress, setBusyProgress } = busy;
    const { currentResourceProfile, scoreBatchSize } = compare;
    const { escapeHtml, formatDuration } = formatting;
    const { addLogEntry, showToast } = notifications;
    const { loadQueue, refreshWorkspace, selectFile, syncReviewRoot } = review;
    const { currentLibraryRoot, saveUiState, setTab } = ui;

    const { pollJob, pollScanJob, pollScoreJob, createResultFetcher, createStatusFetcher } = pollingModule;

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
      return `${label}${countText}${elapsedText}`;
    }

    async function pollOperationJob(jobId, { fallbackLabel, failureMessage }) {
      return pollJob({
        jobId,
        fetchStatus: fetchOperationJobStatus,
        fetchResult: fetchOperationJobResult,
        progressMessage: (progress, elapsedSeconds) => operationProgressMessage(progress, elapsedSeconds, fallbackLabel),
        progressTotal: null,
        failureMessage,
        onProgress: ({ progress }) => {
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

      try {
        return await pollOperationJob(jobId, { fallbackLabel, failureMessage });
      } finally {
        if (!state.abortController?.signal?.aborted) {
          state.operationJobId = null;
          state.operationStatusPath = null;
          state.operationCancelPath = null;
        }
      }
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
      let result = null;
      try {
        result = await pollScanJob(scanJobId, { filesTotalRef, pipeline });
      } finally {
        if (!state.abortController?.signal?.aborted) {
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

      const selectedModel = document.getElementById("model-select").value || state.options?.default_scoring_mode || state.options?.learned_models?.[0] || "topiq_nr";
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
      let result = null;
      try {
        result = await pollScoreJob(scoreJobId, { rowsTotal, pipeline });
      } finally {
        if (!state.abortController?.signal?.aborted) {
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
      if (!listContainer) return;

      const rootStr = currentLibraryRoot();
      const roots = rootStr.split("|").map(r => r.trim()).filter(Boolean);

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

    async function clearCache(scope, message) {
      setBusyPhaseProgress({ percent: 0, phaseIndex: 1, phaseCount: 1, phaseLabel: "Clearing cache" });
      const result = await runTrackedOperation({
        startPath: "/api/cache/clear/start",
        payload: { scope },
        fallbackLabel: "Clearing cache",
        failureMessage: "Cache action failed.",
      });
      addLogEntry("Cache action", `${message}: files ${result.files}, scores ${result.scores}, review ${result.review}.`);
      showToast(message);
      if (scope === "all") {
        if (workflowExport?.clearActiveSelection) {
          workflowExport.clearActiveSelection();
        }
        state.activeId = null;
        state.detail = null;
      }
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
      addLogEntry("Disk delete", `Deleted ${result.deleted_count}, failed ${result.failed_count}.`);
      if (workflowExport?.clearActiveSelection) {
        workflowExport.clearActiveSelection();
      }
      showToast(`Deleted ${result.deleted_count} files from disk.`);
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
      if (!state.browserTarget || !state.browserPath) return;
      const targetInput = document.getElementById(state.browserTarget);
      if (!targetInput) {
        return;
      }
      targetInput.value = state.browserPath;
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
      resetReviewFiltersForAnalyze,
      runScan,
      runScore,
      analyzeLibrary,
      renderLibraryRoots,
      clearCache,
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
