(() => {
  function createWorkflowExport(deps) {
    const {
      api,
      busy,
      notifications,
      review,
      state,
      ui,
      workflowLibrary,
    } = deps;

    const { fetchJson, postJson } = api;
    const { setBusyMessage, setBusyPhaseProgress, withBusy } = busy;
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
    const { openBrowser, handleError } = ui;

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

    async function fetchReviewStateSelectionRevision(marked, root = null) {
      const params = new URLSearchParams();
      params.set("marked", marked);
      params.set("limit", "1");
      params.set("offset", "0");
      if (root) {
        params.set("root", root);
      }
      const data = await fetchJson(`/api/review/file-ids?${params.toString()}`);
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

    function operationActionLabel(result, request = null) {
      const rawAction = String(result?.action || "").toLowerCase();
      const action = rawAction === "export"
        ? String(request?.payload?.mode || "copy").toLowerCase()
        : (rawAction || String(request?.payload?.mode || "operation").toLowerCase());
      return action === "copy" ? "Copy" : action === "move" ? "Move" : action === "delete" ? "Delete" : "Operation";
    }

    function operationItems(result) {
      return Array.isArray(result?.items) ? result.items.filter((item) => item && typeof item === "object") : [];
    }

    function operationRequestIds(request) {
      const rawIds = request?.payload?.file_ids;
      return Array.isArray(rawIds)
        ? rawIds.map(Number).filter((fileId) => Number.isInteger(fileId) && fileId > 0)
        : [];
    }

    function retainOperationSelection(result, request = null) {
      if (result?.job_status === "unknown" || result?.outcome === "unknown") return;
      const items = operationItems(result);
      const operatedIds = new Set(operationRequestIds(request));
      items.forEach((item) => {
        const fileId = Number(item.file_id || item.id);
        if (Number.isInteger(fileId) && fileId > 0) operatedIds.add(fileId);
      });
      const preservedIds = [...state.selectedIds].filter((fileId) => !operatedIds.has(Number(fileId)));
      if (!items.length) {
        state.bulkSelection = null;
        state.selectedIds = request?.payload?.selection ? new Set() : new Set(preservedIds);
        state.lastSelectionAnchorIndex = -1;
        return;
      }

      const remainingIds = items
        .filter((item) => String(item.outcome || "") !== "success")
        .map((item) => Number(item.file_id || item.id))
        .filter((fileId) => Number.isInteger(fileId) && fileId > 0);
      state.bulkSelection = null;
      state.selectedIds = new Set([...preservedIds, ...remainingIds]);
      state.lastSelectionAnchorIndex = -1;
    }

    function operationTone(result) {
      const outcome = String(result?.outcome || "").toLowerCase();
      if (!outcome) {
        if (Array.isArray(result?.failed) && result.failed.length) return "error";
        return Number(result?.copied || 0) || Number(result?.moved || 0) || Number(result?.deleted_count || 0)
          ? "success"
          : "warning";
      }
      if (outcome === "success") return result.warnings?.length ? "warning" : "success";
      if (outcome === "partial") return "warning";
      if (outcome === "noop") return "warning";
      return "error";
    }

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

    function operationDetailsText(result) {
      return JSON.stringify(result || {}, null, 2);
    }

    function presentOperationResult(result, request = null) {
      if (!result || typeof result !== "object") {
        return;
      }
      state.latestOperationResult = result;
      if (request) {
        state.latestOperationRequest = request;
      }
      retainOperationSelection(result, request);

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
        retryButton.classList.toggle("hidden", !(Array.isArray(result.safe_retry_ids) && result.safe_retry_ids.length));
      }
      const checkButton = document.getElementById("operation-result-check-status");
      if (checkButton) {
        checkButton.classList.toggle("hidden", !state.operationStatusUnknown || !state.operationJobId);
      }
    }

    async function retrySafeOperation() {
      const result = state.latestOperationResult;
      const request = state.latestOperationRequest;
      const safeIds = [...new Set((result?.safe_retry_ids || []).map(Number))].filter((id) => Number.isInteger(id) && id > 0);
      if (!safeIds.length || !request) {
        showToast("There are no safely retryable files.", "error");
        return;
      }

      const payload = { ...(request.payload || {}) };
      const originalSelection = payload.selection;
      delete payload.selection;
      delete payload.exclude_file_ids;
      delete payload.page_selection;
      payload.file_ids = safeIds;
      payload.count = safeIds.length;
      if (originalSelection?.scope === "review-state") {
        payload.selection_revision = await fetchReviewStateSelectionRevision(originalSelection.marked, originalSelection.root);
      } else {
        payload.selection_revision = currentSelectionRevision();
      }
      if (!payload.selection_revision) {
        throw new Error("The review selection changed. Refresh the results before retrying.");
      }
      const mode = String(payload.mode || result.action || "").toLowerCase();
      if ((mode === "move" || mode === "delete") && !confirm(`Retry ${mode} for ${safeIds.length} file(s)?`)) {
        return;
      }

      await withBusy(`Retrying ${safeIds.length} file(s)...`, async () => {
        const retryResults = [];
        for (let offset = 0; offset < safeIds.length; offset += 500) {
          const chunkIds = safeIds.slice(offset, offset + 500);
          const chunkPayload = { ...payload, file_ids: chunkIds, count: chunkIds.length };
          if (originalSelection?.scope === "review-state") {
            chunkPayload.selection_revision = await fetchReviewStateSelectionRevision(originalSelection.marked, originalSelection.root);
          } else {
            chunkPayload.selection_revision = currentSelectionRevision();
          }
          retryResults.push(await workflowLibrary.runTrackedOperation({
            startPath: request.startPath,
            payload: chunkPayload,
            fallbackLabel: request.fallbackLabel,
            failureMessage: request.failureMessage,
          }));
        }
        const retryResult = retryResults.reduce((aggregate, next) => {
          if (!aggregate) return { ...next };
          aggregate.items = [...(aggregate.items || []), ...(next.items || [])];
          aggregate.failed = [...(aggregate.failed || []), ...(next.failed || [])];
          aggregate.warnings = [...(aggregate.warnings || []), ...(next.warnings || [])];
          aggregate.safe_retry_ids = [...new Set([...(aggregate.safe_retry_ids || []), ...(next.safe_retry_ids || [])])];
          for (const key of ["copied", "moved", "deleted_count", "completed_count", "failed_count", "partial_count", "unprocessed_count"]) {
            aggregate[key] = Number(aggregate[key] || 0) + Number(next[key] || 0);
          }
          aggregate.outcome = [aggregate.outcome, next.outcome].includes("cancelled")
            ? "cancelled"
            : [aggregate.outcome, next.outcome].some((outcome) => ["partial", "failed", "unknown"].includes(outcome))
              ? (aggregate.completed_count ? "partial" : "failed")
              : "success";
          return aggregate;
        }, null);
        const retainedRequest = { ...request, payload: { ...payload, file_ids: safeIds, count: safeIds.length } };
        presentOperationResult(retryResult, retainedRequest);
        await refreshWorkspace();
      }, { operationType: "operation" });
    }

    function installOperationResultEvents() {
      const panel = document.getElementById("operation-result-panel");
      if (!panel || panel.dataset.eventsInstalled === "true") return;
      panel.dataset.eventsInstalled = "true";
      document.getElementById("operation-result-dismiss")?.addEventListener("click", () => {
        panel.classList.add("hidden");
      });
      document.getElementById("operation-result-copy")?.addEventListener("click", async () => {
        try {
          const details = operationDetailsText(state.latestOperationResult);
          if (navigator.clipboard?.writeText) {
            await navigator.clipboard.writeText(details);
          } else {
            const fallback = document.createElement("textarea");
            fallback.value = details;
            fallback.setAttribute("readonly", "true");
            fallback.style.position = "fixed";
            fallback.style.opacity = "0";
            document.body.appendChild(fallback);
            fallback.select();
            if (!document.execCommand("copy")) throw new Error("copy command failed");
            fallback.remove();
          }
          showToast("Operation details copied.");
        } catch {
          showToast("Could not copy operation details.", "error");
        }
      });
      document.getElementById("operation-result-download")?.addEventListener("click", () => {
        const blob = new Blob([operationDetailsText(state.latestOperationResult)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "shotsieve-operation-result.json";
        link.click();
        URL.revokeObjectURL(url);
      });
      document.getElementById("operation-result-retry")?.addEventListener("click", () => {
        retrySafeOperation().catch(handleError);
      });
      document.getElementById("operation-result-check-status")?.addEventListener("click", async () => {
        try {
          const checked = await workflowLibrary.checkTrackedOperation();
          if (checked) presentOperationResult(checked, state.latestOperationRequest);
        } catch (error) {
          handleError(error);
        }
      });
    }

    state.operationResultHandler = presentOperationResult;

    function buildSelectedExportRequest(mode) {
      return {
        mode,
        resolveRequest: async () => activeSelectionRequest(),
        busyMessage: (count) => `${mode === "move" ? "Moving" : "Copying"} ${count} files...`,
        successPrefix: mode === "move" ? "Move complete" : "Copy complete",
        logTitle: "Export",
        emptyResultMessage: "Select at least one file to export.",
      };
    }

    function openExportDialog(mode, emptySelectionMessage, request = null) {
      if (!request && !hasActiveSelection()) {
        showToast(emptySelectionMessage, "error");
        return;
      }
      state.pendingExport = request || buildSelectedExportRequest(mode);
      document.getElementById("export-mode").value = mode;
      document.getElementById("export-dialog").showModal();
    }

    function installExportDialogEvents() {
      document.getElementById("browse-export-dir").addEventListener("click", () => openBrowser("export-destination").catch(handleError));
      document.getElementById("export-confirm").addEventListener("click", () => {
        const destination = document.getElementById("export-destination").value.trim();
        const request = state.pendingExport || buildSelectedExportRequest(document.getElementById("export-mode").value);
        if (!destination) {
          showToast("Choose a destination folder.", "error");
          return;
        }
        document.getElementById("export-dialog").close();
        state.pendingExport = null;

        withBusy("Preparing export...", async () => {
          const selectionRequest = await request.resolveRequest();
          if (!selectionRequest.count) {
            showToast(request.emptyResultMessage, "error");
            return;
          }

          if (request.mode === "move") {
            const msg = `Move ${selectionRequest.count} file(s) to ${destination}?\n\nThis will remove the original files and replace them at the new location.`;
            if (!confirm(msg)) return;
          }

          const phaseLabel = request.mode === "move" ? "Moving files" : "Exporting files";
          setBusyMessage(request.busyMessage(selectionRequest.count));
          setBusyPhaseProgress({ percent: 0, phaseIndex: 1, phaseCount: 1, phaseLabel });
          const result = await workflowLibrary.runTrackedOperation({
            startPath: "/api/files/export/start",
            payload: {
              ...selectionRequest,
              destination,
              mode: request.mode,
              count: selectionRequest.count,
            },
            fallbackLabel: phaseLabel,
            failureMessage: `${phaseLabel} failed.`,
          });
          presentOperationResult(result, {
            startPath: "/api/files/export/start",
            payload: {
              ...selectionRequest,
              destination,
              mode: request.mode,
              count: selectionRequest.count,
            },
            fallbackLabel: phaseLabel,
            failureMessage: `${phaseLabel} failed.`,
          });
          const summary = summarizeExportResult(result);
          const resultLabel = operationTone(result) === "success"
            ? request.successPrefix
            : `${request.mode === "move" ? "Move" : "Copy"} results`;
          showToast(`${resultLabel}: ${summary || "no matching files"}.`, operationTone(result));
          addLogEntry(request.logTitle, `${request.mode} to ${destination}: ${summary}`);
          await refreshWorkspace();
        }).catch(handleError);
      });
    }

    function installRejectedActionEvents() {
      document.getElementById("delete-all-rejected").addEventListener("click", () => {
        const root = document.getElementById("root-filter")?.value || "";
        const rejectedCount = Number(state.overview?.active_library?.delete_marked || state.overview?.summary?.delete_marked || 0);
        if (!root) {
          showToast("Choose a library before deleting rejected photos. The All libraries view is global.", "error");
          return;
        }
        if (!rejectedCount) {
          showToast("No rejected photos to delete.", "error");
          return;
        }
        const msg = `Permanently delete ${rejectedCount} rejected photo${rejectedCount !== 1 ? "s" : ""} in this library from disk?\n\nLibrary: ${root}\n\nThis cannot be undone. The original files will be removed from your computer.`;
        if (!confirm(msg)) return;
        withBusy(`Deleting ${rejectedCount} rejected files in this library...`, async () => {
          const selectionRevision = await fetchReviewStateSelectionRevision("delete", root);
          if (!selectionRevision) {
            showToast("Review results are refreshing. Try again in a moment.", "error");
            return;
          }
          const selection = { scope: "review-state", marked: "delete", root };
          if (!rejectedCount) {
            showToast("No rejected files found.", "error");
            return;
          }
          const operationRequest = {
            startPath: "/api/files/delete/start",
            payload: {
              selection,
              selection_revision: selectionRevision,
              delete_from_disk: true,
              count: rejectedCount,
            },
            fallbackLabel: "Deleting rejected files",
            failureMessage: "Delete rejected files failed.",
          };
          const result = await workflowLibrary.runTrackedOperation({
            startPath: operationRequest.startPath,
            payload: operationRequest.payload,
            fallbackLabel: operationRequest.fallbackLabel,
            failureMessage: operationRequest.failureMessage,
          });
          presentOperationResult(result, operationRequest);
          addLogEntry("Delete rejected in library", `Deleted ${result.deleted_count} files from ${root}, ${result.failed_count} failed.`);
          showToast(`Deleted ${result.deleted_count || 0} rejected files from this library.`, operationTone(result));
          await refreshWorkspace();
        }).catch(handleError);
      });

      document.getElementById("move-all-rejected").addEventListener("click", () => {
        const root = document.getElementById("root-filter")?.value || "";
        const rejectedCount = Number(state.overview?.active_library?.delete_marked || state.overview?.summary?.delete_marked || 0);
        if (!root) {
          showToast("Choose a library before moving rejected photos. The All libraries view is global.", "error");
          return;
        }
        if (!rejectedCount) {
          showToast("No rejected photos to move.", "error");
          return;
        }
        openExportDialog("move", "No rejected photos to move.", {
          mode: "move",
          resolveRequest: async () => {
            const selectionRevision = await fetchReviewStateSelectionRevision("delete", root);
            if (!selectionRevision) {
              throw new Error("Review results are refreshing. Try again in a moment.");
            }
            return {
              selection: { scope: "review-state", marked: "delete", root },
              selection_revision: selectionRevision,
              count: rejectedCount,
            };
          },
          busyMessage: (count) => `Moving ${count} rejected files in this library...`,
          successPrefix: "Move complete",
          logTitle: "Move rejected",
          emptyResultMessage: "No rejected files found.",
        });
      });
      installOperationResultEvents();
    }

    return {
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
      buildSelectedExportRequest,
      openExportDialog,
      installExportDialogEvents,
      installRejectedActionEvents,
      presentOperationResult,
      operationTone,
      installOperationResultEvents,
    };
  }

  window.ShotSieveWorkflowExport = {
    createWorkflowExport,
  };
})();
