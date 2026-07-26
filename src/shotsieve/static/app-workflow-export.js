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
      return parts.join(", ");
    }

    function buildSelectedExportRequest(mode) {
      return {
        mode,
        resolveRequest: async () => activeSelectionRequest(),
        busyMessage: (count) => `Exporting ${count} files...`,
        successPrefix: "Export complete",
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
          const summary = summarizeExportResult(result);
          showToast(`${request.successPrefix}: ${summary}.`);
          addLogEntry(request.logTitle, `${request.mode} to ${destination}: ${summary}`);
          clearActiveSelection();
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
          const result = await postJson("/api/files/delete", {
            selection,
            selection_revision: selectionRevision,
            delete_from_disk: true,
          });
          addLogEntry("Delete rejected in library", `Deleted ${result.deleted_count} files from ${root}, ${result.failed_count} failed.`);
          showToast(`Deleted ${result.deleted_count} rejected files from this library.`);
          clearActiveSelection();
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
    };
  }

  window.ShotSieveWorkflowExport = {
    createWorkflowExport,
  };
})();
