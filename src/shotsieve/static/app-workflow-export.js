(() => {
  function createWorkflowExport(deps) {
    const {
      api,
      busy,
      notifications,
      review,
      state,
      workflowLibrary,
    } = deps;

    const { fetchJson, postJson } = api;
    const { withBusy } = busy;
    const { addLogEntry, showToast } = notifications;
    const {
      applyReviewUpdate,
      isAutoAdvanceEnabled,
      loadQueue,
      refreshOverview,
      refreshWorkspace,
      reviewDecisions,
      selectFile,
      renderPagination,
    } = review;
    const operationResults = deps.operationResults || window.ShotSieveWorkflowResults;
    if (!operationResults) {
      throw new Error("ShotSieve operation-result module failed to load.");
    }
    const {
      operationActionLabel,
      operationItems,
      operationTone,
      retainOperationSelection,
      mergeOperationResults,
      retrySafeOperation: runSafeRetry,
    } = operationResults;

    async function saveReview(payload) {
      if (!state.activeId) {
        showToast("Pick a file first.", "error");
        return;
      }
      const updatedDetail = await postJson("/api/review", { file_id: state.activeId, ...payload });
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

    function reviewDecisionPayload(action) {
      const payload = reviewDecisions[action];
      if (!payload) {
        throw new Error(`Unknown review action: ${action}`);
      }
      return payload;
    }

    function hasActiveSelection() {
      const excludedCount = state.bulkSelection?.excludedIds instanceof Set
        ? state.bulkSelection.excludedIds.size
        : 0;
      const effectiveSelectionCount = state.bulkSelection
        ? Math.max(0, Number(state.bulkSelection.count || 0) - excludedCount)
        : state.selectedIds.size;
      return effectiveSelectionCount > 0;
    }

    function clearActiveSelection() {
      state.bulkSelection = null;
      state.selectedIds.clear();
      state.lastSelectionAnchorIndex = -1;
    }

    function currentSelectionRevision() {
      return state.loadedReviewSelection?.selectionRevision || null;
    }

    async function fetchReviewStateSelectionRevision(marked, root = null, query = null) {
      const params = new URLSearchParams();
      params.set("marked", marked);
      params.set("limit", "1");
      params.set("offset", "0");
      if (root) {
        params.set("root", root);
      }
      if (query) {
        params.set("query", query);
      }
      const data = await fetchJson(`/api/review/file-ids?${params.toString()}`);
      return data.selection_revision || null;
    }

    async function fetchSelectionRevision(selection) {
      const scope = String(selection?.scope || "review-browser");
      if (scope === "review-state") {
        return fetchReviewStateSelectionRevision(selection.marked, selection.root, selection.query);
      }

      const params = new URLSearchParams();
      for (const name of [
        "root", "marked", "issues", "query", "min_score", "max_score", "min_mp", "max_mp",
        "min_width", "max_width", "min_height", "max_height", "min_edge", "max_edge", "min_size",
        "max_size", "metadata",
      ]) {
        const value = selection?.[name];
        if (value !== null && value !== undefined && value !== "") {
          params.set(name, String(value));
        }
      }
      if (Array.isArray(selection?.formats) && selection.formats.length) {
        params.set("formats", selection.formats.join(","));
      }
      params.set("limit", "1");
      params.set("offset", "0");
      const data = await fetchJson(`/api/files?${params.toString()}`);
      return data.selection_revision || null;
    }

    function activeSelectionRequest() {
      if (state.bulkSelection) {
        const excludedIds = [...(state.bulkSelection.excludedIds || new Set())];
        const effectiveSelectionCount = Math.max(0, Number(state.bulkSelection.count || 0) - excludedIds.length);
        const request = {
          selection: state.bulkSelection.selection,
          selection_revision: state.bulkSelection.selectionRevision,
          count: effectiveSelectionCount,
        };

        if (excludedIds.length) {
          request.exclude_file_ids = excludedIds;
        }

        return request;
      }
      const fileIds = [...state.selectedIds];
      return {
        file_ids: fileIds,
        count: fileIds.length,
        selection_revision: currentSelectionRevision(),
        page_selection: state.loadedReviewSelection?.selection || null,
      };
    }

    async function saveReviewDecision(action) {
      const shouldAdvance = action !== "reset" && isAutoAdvanceEnabled();
      await saveReviewDecisionWithOptions(action, { advance: shouldAdvance });
    }

    function nextReviewCandidateId(currentId) {
      const currentIndex = state.queue.findIndex((item) => item.id === currentId);
      if (currentIndex < 0) {
        return null;
      }
      if (state.queue[currentIndex + 1]) {
        return state.queue[currentIndex + 1].id;
      }
      if (state.queue[currentIndex - 1]) {
        return state.queue[currentIndex - 1].id;
      }
      return null;
    }

    async function saveReviewDecisionWithOptions(action, { advance } = { advance: false }) {
      const currentId = state.activeId;
      const currentIndex = currentId ? state.queue.findIndex((item) => item.id === currentId) : -1;
      const candidateId = advance && currentIndex >= 0 && state.queue[currentIndex + 1]
        ? state.queue[currentIndex + 1].id
        : null;
      const shouldAdvancePage = Boolean(
        advance
        && currentIndex >= 0
        && currentIndex === state.queue.length - 1
        && ((state.page + 1) * state.pageSize) < state.totalFiles,
      );

      await saveReview(reviewDecisionPayload(action));

      if (!advance) {
        return;
      }

      if (candidateId && state.queue.some((item) => item.id === candidateId)) {
        await selectFile(candidateId);
        return;
      }

      if (shouldAdvancePage) {
        state.page += 1;
        await loadQueue();
        if (state.queue.length) {
          await selectFile(state.queue[0].id);
        }
        return;
      }

      const fallbackIndex = Math.min(Math.max(currentIndex, 0), state.queue.length - 1);
      if (state.queue[fallbackIndex]) {
        await selectFile(state.queue[fallbackIndex].id);
      }
    }

    async function runBatchReview(payload, message) {
      const selectionRequest = activeSelectionRequest();
      if (!selectionRequest.count) {
        showToast("Select at least one result first.", "error");
        return;
      }
      await postJson("/api/review/batch", { ...selectionRequest, ...payload });
      addLogEntry("Batch review update", `${message} on ${selectionRequest.count} items.`);
      showToast(`${message} (${selectionRequest.count} items).`);
      clearActiveSelection();
      await refreshWorkspace();
    }

    async function runBatchReviewDecision(action, message) {
      await runBatchReview(reviewDecisionPayload(action), message);
    }

    async function fetchMarkedFileIds(marked) {
      const fileIds = [];
      let offset = 0;
      const limit = 500;

      while (true) {
        const params = new URLSearchParams();
        params.set("marked", marked);
        params.set("limit", String(limit));
        params.set("offset", String(offset));
        const data = await fetchJson(`/api/review/file-ids?${params.toString()}`);
        const ids = Array.isArray(data.ids) ? data.ids : [];
        fileIds.push(...ids);
        if (ids.length < limit) {
          break;
        }
        offset += limit;
      }

      return fileIds;
    }

    function summarizeExportResult(result) {
      const parts = [];
      if (result.copied) parts.push(`${result.copied} copied`);
      if (result.moved) parts.push(`${result.moved} moved`);
      if (result.failed?.length) parts.push(`${result.failed.length} failed`);
      if (result.warnings?.length) parts.push(`${result.warnings.length} cleanup warning(s)`);
      return parts.join(", ");
    }

    // Operation result shape, retry safety, and selection reconciliation live in
    // the injected domain utility so this module can focus on review/export UI.

    function appendOperationLine(container, label, value) {
      const line = document.createElement("p");
      const labelNode = document.createElement("strong");
      labelNode.textContent = `${label}: `;
      line.appendChild(labelNode);
      const valueNode = document.createElement("span");
      valueNode.textContent = String(value ?? "");
      line.appendChild(valueNode);
      container.appendChild(line);
    }


    function presentOperationResult(result, request = null) {
      if (!result || typeof result !== "object") {
        return;
      }
      state.latestOperationResult = result;
      if (request) {
        state.latestOperationRequest = request;
      }
      retainOperationSelection(state, result, request);

      const panel = document.getElementById("operation-result-panel");
      if (!panel) return;
      const title = document.getElementById("operation-result-title");
      const summary = document.getElementById("operation-result-summary");
      const counts = document.getElementById("operation-result-counts");
      const itemsContainer = document.getElementById("operation-result-items");
      const status = String(result.outcome || (result.job_status === "failed" ? "failed" : "success"));
      const actionLabel = operationActionLabel(result, request);
      const items = operationItems(result);
      const requestedCount = Number(request?.payload?.count || 0);
      const total = Number(result.completed_count || 0)
        + Number(result.partial_count || 0)
        + Number(result.failed_count || 0)
        + Number(result.unprocessed_count || 0);

      panel.classList.remove("hidden");
      panel.dataset.outcome = status === "success" && result.warnings?.length ? "partial" : status;
      title.textContent = status === "noop" || (!items.length && !total && !result.copied && !result.moved && !result.deleted_count)
        ? `No matching files for ${actionLabel.toLowerCase()}`
        : `${actionLabel} ${status === "success" ? "complete" : "results"}`;
      summary.textContent = result.job_status === "failed"
        ? `The job failed: ${result.job_error || result.fatal_error || "see details below"}`
        : result.job_status === "unknown"
          ? `Job status is unknown: ${result.fatal_error || "the last status request failed"}. Use Check status to resume this same job.`
        : result.cancelled
          ? "The operation was cancelled. Completed and unprocessed files are shown below."
          : requestedCount && !total && !result.copied && !result.moved && !result.deleted_count
            ? `No files matched the ${requestedCount.toLocaleString()} selected item(s).`
            : "Review each outcome before retrying any file.";

      counts.replaceChildren();
      const countValues = [
        ["Completed", result.completed_count ?? (Number(result.copied || 0) + Number(result.moved || 0) + Number(result.deleted_count || 0))],
        ["Partial", result.partial_count || 0],
        ["Failed", result.failed_count ?? (Array.isArray(result.failed) ? result.failed.length : 0)],
        ["Unprocessed", result.unprocessed_count || 0],
        ["Cleanup warnings", result.warnings?.length || 0],
      ];
      for (const [label, value] of countValues) {
        const count = document.createElement("span");
        count.className = "operation-result-count";
        count.textContent = `${label}: ${Number(value || 0).toLocaleString()}`;
        counts.appendChild(count);
      }

      itemsContainer.replaceChildren();
      const detailItems = [
        ...(result.warnings || []).map((item) => ({ ...item, outcome: "cleanup warning" })),
        ...items.filter((item) => item.outcome !== "success"),
        ...items.filter((item) => item.outcome === "success"),
      ];
      const boundedItems = detailItems.slice(0, 50);
      for (const item of boundedItems) {
        const itemNode = document.createElement("li");
        appendOperationLine(itemNode, "Outcome", item.outcome || "unknown");
        appendOperationLine(itemNode, "Path", item.source || item.path || "(path unavailable)");
        if (item.destination) appendOperationLine(itemNode, "Destination", item.destination);
        appendOperationLine(itemNode, "Stage", item.stage || "unknown");
        if (item.error_text || item.error) appendOperationLine(itemNode, "Details", item.error_text || item.error);
        if (item.errno !== undefined && item.errno !== null) appendOperationLine(itemNode, "OS error", item.errno);
        if (item.winerror !== undefined && item.winerror !== null) appendOperationLine(itemNode, "Windows error", item.winerror);
        itemsContainer.appendChild(itemNode);
      }
      if (detailItems.length > boundedItems.length) {
        const more = document.createElement("li");
        more.className = "muted";
        more.textContent = `${detailItems.length - boundedItems.length} more result(s) available in Download JSON.`;
        itemsContainer.appendChild(more);
      }

      const retryButton = document.getElementById("operation-result-retry");
      if (retryButton) {
        retryButton.classList.toggle(
          "hidden",
          Boolean(state.recoveryJob) || !(Array.isArray(result.safe_retry_ids) && result.safe_retry_ids.length),
        );
      }
      const checkButton = document.getElementById("operation-result-check-status");
      if (checkButton) {
        checkButton.classList.toggle("hidden", !state.operationStatusUnknown || !state.operationJobId);
      }
    }

    function retrySafeOperation() {
      return runSafeRetry({
        state,
        withBusy,
        fetchSelectionRevision,
        // The library workflow is populated after export construction. Keep
        // this callback late-bound so retries use the completed bridge.
        runTrackedOperation: (...args) => workflowLibrary.runTrackedOperation(...args),
        refreshWorkspace,
        presentResult: presentOperationResult,
        showToast,
        confirmRetry: (message) => confirm(message),
      });
    }

    state.operationResultHandler = presentOperationResult;

    const workflowExport = {
      saveReview,
      reviewDecisionPayload,
      hasActiveSelection,
      clearActiveSelection,
      currentSelectionRevision,
      fetchReviewStateSelectionRevision,
      activeSelectionRequest,
      saveReviewDecision,
      nextReviewCandidateId,
      saveReviewDecisionWithOptions,
      runBatchReview,
      runBatchReviewDecision,
      fetchMarkedFileIds,
      summarizeExportResult,
      presentOperationResult,
      operationTone,
      mergeOperationResults,
      retrySafeOperation,
      fetchSelectionRevision,
    };
    const exportUi = deps.exportUi || window.ShotSieveWorkflowExportUi;
    if (exportUi?.createWorkflowExportUi) {
      Object.assign(workflowExport, exportUi.createWorkflowExportUi({
        ...deps,
        workflowExport,
        operationResults,
      }));
    }
    return workflowExport;
  }

  window.ShotSieveWorkflowExport = {
    createWorkflowExport,
  };
})();
