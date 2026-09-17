(() => {
  function createWorkflowExportUi(deps) {
    const {
      api,
      busy,
      notifications,
      state,
      ui,
      workflowExport,
      workflowLibrary,
      operationResults = window.ShotSieveWorkflowResults,
    } = deps;

    const { fetchJson } = api;
    const { setBusyMessage, setBusyPhaseProgress, withBusy } = busy;
    const { addLogEntry, showToast } = notifications;
    const { refreshWorkspace } = deps.review;
    const { handleError } = ui;
    // The export and library workflows are composed in dependency order, so
    // the browser function is added to the stable library bridge afterwards.
    // Resolve it when the button is used instead of capturing an undefined
    // value during export workflow construction.
    const openBrowser = ui.openBrowser || ((...args) => workflowLibrary.openBrowser(...args));
    const { operationTone, operationDetailsText } = operationResults;

    function buildSelectedExportRequest(mode) {
      return {
        mode,
        resolveRequest: async () => workflowExport.activeSelectionRequest(),
        busyMessage: (count) => `${mode === "move" ? "Moving" : "Copying"} ${count} files...`,
        successPrefix: mode === "move" ? "Move complete" : "Copy complete",
        logTitle: "Export",
        emptyResultMessage: "Select at least one file to export.",
      };
    }

    function openExportDialog(mode, emptySelectionMessage, request = null) {
      if (!request && !workflowExport.hasActiveSelection()) {
        showToast(emptySelectionMessage, "error");
        return;
      }
      state.pendingExport = request || buildSelectedExportRequest(mode);
      document.getElementById("export-mode").value = mode;
      document.getElementById("export-dialog").showModal();
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
        workflowExport.retrySafeOperation().catch(handleError);
      });
      document.getElementById("operation-result-check-status")?.addEventListener("click", async () => {
        try {
          const checked = await workflowLibrary.checkTrackedOperation();
          if (checked) workflowExport.presentOperationResult(checked, state.latestOperationRequest);
        } catch (error) {
          handleError(error);
        }
      });
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
          const operationRequest = {
            startPath: "/api/files/export/start",
            payload: {
              ...selectionRequest,
              destination,
              mode: request.mode,
              count: selectionRequest.count,
            },
            fallbackLabel: phaseLabel,
            failureMessage: `${phaseLabel} failed.`,
          };
          const result = await workflowLibrary.runTrackedOperation(operationRequest);
          workflowExport.presentOperationResult(result, operationRequest);
          const summary = workflowExport.summarizeExportResult(result);
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
          const selectionRevision = await workflowExport.fetchReviewStateSelectionRevision("delete", root);
          if (!selectionRevision) {
            showToast("Review results are refreshing. Try again in a moment.", "error");
            return;
          }
          const selection = { scope: "review-state", marked: "delete", root };
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
          const result = await workflowLibrary.runTrackedOperation(operationRequest);
          workflowExport.presentOperationResult(result, operationRequest);
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
            const selectionRevision = await workflowExport.fetchReviewStateSelectionRevision("delete", root);
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
      buildSelectedExportRequest,
      openExportDialog,
      installOperationResultEvents,
      installExportDialogEvents,
      installRejectedActionEvents,
    };
  }

  window.ShotSieveWorkflowExportUi = {
    createWorkflowExportUi,
  };
})();
