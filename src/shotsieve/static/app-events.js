(function () {
  function createEvents(deps) {
    const {
      state,
      withBusy,
      refreshWorkspace,
      loadQueue,
      saveUiState,
      saveReviewDecision,
      saveReviewDecisionWithOptions,
      navigateSelection,
      runBatchReviewDecision,
      openExportDialog,
      deleteSelectedFiles,
      installExportDialogEvents,
      analyzeLibrary,
      runScan,
      runScore,
      runModelComparison,
      renderComparisonResults,
      clearCache,
      openBrowser,
      browseDirectory,
      chooseBrowserPath,
      handleError,
      showToast,
      installRejectedActionEvents,
      applyTheme,
      clearUiState,
      closeOverlay,
      loadUiState,
      setTab,
      openOverlay,
      renderBusyState,
      selectAll,
      selectNone,
      selectAllMatching,
      invalidateLoadedReviewSelection,
      activateLibraryScope,
      resetReviewToActiveLibrary,
      setReviewScope,
      runPreflight,
      renderLibraryRoots,
      loadAnalysisDiagnostics,
    } = deps;

    const TAB_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);
    let filterTimer = null;
    let preflightTimer = null;

    function triggerPreflightDebounced(delay = 250) {
      window.clearTimeout(preflightTimer);
      preflightTimer = window.setTimeout(() => {
        if (typeof runPreflight === "function") {
          runPreflight().catch(handleError);
        }
      }, delay);
    }

    function moveTabFocus(currentButton, direction) {
      const buttons = [...document.querySelectorAll(".tab-button")].filter((button) => !button.hidden);
      const currentIndex = buttons.indexOf(currentButton);
      if (currentIndex === -1) {
        return;
      }
      let nextIndex = currentIndex;
      if (direction === "start") {
        nextIndex = 0;
      } else if (direction === "end") {
        nextIndex = buttons.length - 1;
      } else {
        nextIndex = (currentIndex + direction + buttons.length) % buttons.length;
      }
      const nextButton = buttons[nextIndex];
      setTab(nextButton.dataset.tab, { focusButton: true });
    }

    function isShortcutTarget(target) {
      return target instanceof HTMLInputElement
        || target instanceof HTMLTextAreaElement
        || target instanceof HTMLSelectElement
        || target instanceof HTMLButtonElement
        || target instanceof HTMLAnchorElement
        || target?.closest?.("dialog")
        || target?.closest?.("[role='tablist']");
    }

    function confirmAndRun(buttonId, busyMsg, cacheKey, successMsg, { onSuccess } = {}) {
      document.getElementById(buttonId).addEventListener("click", () => {
        const btn = document.getElementById(buttonId);
        const msg = btn.dataset.confirm || "Are you sure?";
        if (!confirm(msg)) {
          return;
        }
        withBusy(busyMsg, async () => {
          await clearCache(cacheKey, successMsg);
          if (typeof onSuccess === "function") {
            await onSuccess();
          }
        }).catch(handleError);
      });
    }

    function resetPersistedUiStateAfterFullReset() {
      localStorage.removeItem("shotsieve_resource_profile");

      const libraryRootInput = document.getElementById("library-root-input");
      if (libraryRootInput) {
        libraryRootInput.value = "";
      }

      const extensionsInput = document.getElementById("extensions-input");
      if (extensionsInput) {
        const defaults = Array.isArray(state.options?.default_extensions)
          ? state.options.default_extensions.join(",")
          : "";
        extensionsInput.value = defaults;
      }

      const recursiveToggle = document.getElementById("recursive-toggle");
      if (recursiveToggle) {
        recursiveToggle.checked = true;
      }

      const modelSelect = document.getElementById("model-select");
      if (modelSelect) {
        const defaultModel = state.options?.default_scoring_mode || modelSelect.options?.[0]?.value || "";
        modelSelect.value = defaultModel;
      }

      const deviceSelect = document.getElementById("device-select");
      if (deviceSelect) {
        const hasAuto = [...deviceSelect.options].some((option) => option.value === "auto");
        deviceSelect.value = hasAuto ? "auto" : (deviceSelect.options?.[0]?.value || "");
      }

      const profileSelect = document.getElementById("resource-profile-select");
      if (profileSelect) {
        profileSelect.value = "normal";
        const hw = state.options?.learned?.hardware;
        if (typeof updateResourceProfileDetail === "function") {
          updateResourceProfileDetail(hw);
        }
      }

      const queryFilter = document.getElementById("query-filter");
      if (queryFilter) {
        queryFilter.value = "";
      }
      const rootFilter = document.getElementById("root-filter");
      if (rootFilter) {
        rootFilter.value = "";
      }
      const markedFilter = document.getElementById("marked-filter");
      if (markedFilter) {
        markedFilter.value = "all";
      }
      const issuesFilter = document.getElementById("issues-filter");
      if (issuesFilter) {
        issuesFilter.value = "all";
      }
      const minScore = document.getElementById("min-score");
      if (minScore) {
        minScore.value = "";
      }
      const maxScore = document.getElementById("max-score");
      if (maxScore) {
        maxScore.value = "";
      }

      invalidateLoadedReviewSelection({ clearActiveSelection: true });
      state.page = 0;
      clearUiState({ immediate: true });
    }

    function hasBatchSelection() {
      const excludedCount = state.bulkSelection?.excludedIds instanceof Set
        ? state.bulkSelection.excludedIds.size
        : 0;
      const effectiveSelectionCount = state.bulkSelection
        ? Math.max(0, Number(state.bulkSelection.count || 0) - excludedCount)
        : state.selectedIds.size;
      return effectiveSelectionCount > 0;
    }

    function runReviewToolbarAction(action, busyMessage, batchMessage) {
      if (hasBatchSelection()) {
        withBusy(busyMessage, () => runBatchReviewDecision(action, batchMessage)).catch(handleError);
        return;
      }
      withBusy(busyMessage, () => saveReviewDecision(action)).catch(handleError);
    }

    function requestOperationCancel() {
      if (!state.abortController || state.cancelPending) {
        return;
      }
      state.cancelPending = true;
      renderBusyState();
      state.abortController.abort();
    }

    function openLightboxFromDetailImage(sourceImage) {
      if (!(sourceImage instanceof HTMLImageElement) || !sourceImage.src) {
        return;
      }
      if (typeof sourceImage.focus === "function") {
        sourceImage.focus();
      }
      const lightboxImg = document.getElementById("lightbox-image");
      lightboxImg.src = sourceImage.src;
      const detailTitle = document.getElementById("detail-title");
      lightboxImg.alt = detailTitle ? `Full-size preview of ${detailTitle.textContent}` : "Full-size photo";
      deps.openOverlay("lightbox-overlay");
    }

    return function installEvents() {
      document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", (event) => {
          setTab(button.dataset.tab);
          if (event.detail > 0) {
            button.blur();
          }
        });
        button.addEventListener("keydown", (event) => {
          if (!TAB_KEYS.has(event.key)) {
            return;
          }
          event.preventDefault();
          if (event.key === "Home") {
            moveTabFocus(button, "start");
            return;
          }
          if (event.key === "End") {
            moveTabFocus(button, "end");
            return;
          }
          const direction = event.key === "ArrowLeft" || event.key === "ArrowUp" ? -1 : 1;
          moveTabFocus(button, direction);
        });
      });

      document.getElementById("refresh-all").addEventListener("click", () => withBusy("Refreshing workspace...", () => refreshWorkspace()).catch(handleError));
      document.getElementById("clear-filters").addEventListener("click", () => {
        ["query-filter", "min-score", "max-score", "filter-min-mp", "filter-max-mp", "filter-min-size", "filter-max-size"].forEach((id) => {
          const el = document.getElementById(id);
          if (el) el.value = "";
        });
        ["marked-filter", "issues-filter"].forEach((id) => {
          const el = document.getElementById(id);
          if (el) el.value = "all";
        });
        const metaStatus = document.getElementById("filter-metadata-status");
        if (metaStatus) metaStatus.value = "all";
        document.getElementById("sort-filter").value = "learned_asc";
        document.querySelectorAll("input[name='format-filter']").forEach((checkbox) => { checkbox.checked = true; });
        resetReviewToActiveLibrary();
        loadQueue().catch(handleError);
      });

      const toggleAdvancedBtn = document.getElementById("toggle-advanced-filters");
      if (toggleAdvancedBtn) {
        toggleAdvancedBtn.addEventListener("click", () => {
          const panel = document.getElementById("advanced-filters-panel");
          if (panel.classList.contains("hidden")) {
            panel.classList.remove("hidden");
            toggleAdvancedBtn.textContent = "Hide Metadata Filters";
          } else {
            panel.classList.add("hidden");
            toggleAdvancedBtn.textContent = "Show Metadata Filters";
          }
        });
      }

      ["root-filter", "sort-filter", "marked-filter", "issues-filter", "filter-metadata-status"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener("change", () => {
            (async () => {
            if (id === "root-filter") {
              const root = document.getElementById("root-filter").value;
              if (root) {
                await activateLibraryScope(root);
                return;
              }
              setReviewScope("");
              await refreshWorkspace();
              return;
            }
            invalidateLoadedReviewSelection({ clearActiveSelection: true });
            state.page = 0;
            saveUiState();
            await loadQueue();
            })().catch(handleError);
          });
        }
      });

      document.querySelectorAll("input[name='format-filter']").forEach((checkbox) => {
        checkbox.addEventListener("change", () => {
          invalidateLoadedReviewSelection({ clearActiveSelection: true });
          state.page = 0;
          saveUiState();
          loadQueue().catch(handleError);
        });
      });

      ["min-score", "max-score", "filter-min-mp", "filter-max-mp", "filter-min-size", "filter-max-size"].forEach((id) => {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener("change", () => { 
            invalidateLoadedReviewSelection({ clearActiveSelection: true });
            state.page = 0; 
            saveUiState(); 
            loadQueue().catch(handleError); 
          });
          el.addEventListener("input", () => {
            window.clearTimeout(filterTimer);
            invalidateLoadedReviewSelection({ clearActiveSelection: true });
            filterTimer = window.setTimeout(() => { 
              state.page = 0; 
              saveUiState(); 
              loadQueue().catch(handleError); 
            }, 250);
          });
        }
      });

      [
        "library-root-input",
        "extensions-input",
        "recursive-toggle",
        "model-select",
        "device-select",
        "ignore-rules-input",
      ].forEach((id) => {
        const el = document.getElementById(id);
        if (el) {
          el.addEventListener("change", () => {
            saveUiState();
          });
        }
      });

      document.getElementById("library-root-input").addEventListener("input", () => {
        saveUiState();
      });

      document.getElementById("library-root-input").addEventListener("change", () => {
        const val = document.getElementById("library-root-input").value;
        activateLibraryScope(val).catch(handleError);
        renderLibraryRoots();
        if (typeof runPreflight === "function") {
          runPreflight().catch(handleError);
        }
      });

      document.getElementById("ignore-rules-input")?.addEventListener("input", () => {
        saveUiState();
        triggerPreflightDebounced(350);
      });
      document.getElementById("ignore-rules-input")?.addEventListener("change", () => {
        saveUiState();
        triggerPreflightDebounced(50);
      });

      document.getElementById("recursive-toggle")?.addEventListener("change", () => {
        saveUiState();
        triggerPreflightDebounced(50);
      });

      document.getElementById("extensions-input")?.addEventListener("input", () => {
        saveUiState();
        triggerPreflightDebounced(350);
      });
      document.getElementById("extensions-input")?.addEventListener("change", () => {
        saveUiState();
        triggerPreflightDebounced(50);
      });

      const addRootBtn = document.getElementById("browse-library-root");
      if (addRootBtn) {
        addRootBtn.addEventListener("click", () => {
          openBrowser("add-root-helper-input").catch(handleError);
        });
      }

      const addRootHelper = document.getElementById("add-root-helper-input");
      if (addRootHelper) {
        addRootHelper.addEventListener("change", () => {
          const newPath = addRootHelper.value.trim();
          if (!newPath) return;

          const currentVal = document.getElementById("library-root-input").value;
          const roots = currentVal.split("|").map(r => r.trim()).filter(Boolean);

          if (!roots.includes(newPath)) {
            roots.push(newPath);
            const hiddenInput = document.getElementById("library-root-input");
            hiddenInput.value = roots.join("|");
            hiddenInput.dispatchEvent(new Event("input", { bubbles: true }));
            hiddenInput.dispatchEvent(new Event("change", { bubbles: true }));
          }
          addRootHelper.value = "";
        });
      }

      const excludeFolderBtn = document.getElementById("exclude-library-folder");
      if (excludeFolderBtn) {
        excludeFolderBtn.addEventListener("click", () => {
          openBrowser("exclude-root-helper-input").catch(handleError);
        });
      }

      const excludeRootHelper = document.getElementById("exclude-root-helper-input");
      if (excludeRootHelper) {
        excludeRootHelper.addEventListener("change", () => {
          const newPath = excludeRootHelper.value.trim();
          if (!newPath) return;

          const textarea = document.getElementById("ignore-rules-input");
          if (textarea) {
            let currentVal = textarea.value.trim();
            const lines = currentVal ? currentVal.split("\n").map(l => l.trim()).filter(Boolean) : [];
            if (!lines.includes(newPath)) {
              lines.push(newPath);
              textarea.value = lines.join("\n");
              textarea.dispatchEvent(new Event("input", { bubbles: true }));
              textarea.dispatchEvent(new Event("change", { bubbles: true }));
            }
          }
          excludeRootHelper.value = "";
        });
      }

      document.getElementById("query-filter").addEventListener("input", () => {
        window.clearTimeout(filterTimer);
        invalidateLoadedReviewSelection({ clearActiveSelection: true });
        filterTimer = window.setTimeout(() => { 
          state.page = 0; 
          saveUiState();
          loadQueue().catch(handleError); 
        }, 250);
      });

      document.getElementById("select-all-btn").addEventListener("click", selectAll);
      document.getElementById("select-none-btn").addEventListener("click", selectNone);
      document.getElementById("select-all-matching-btn").addEventListener("click", () => selectAllMatching().catch(handleError));

      document.querySelectorAll("[data-review-nav='prev']").forEach((button) => {
        button.addEventListener("click", () => navigateSelection(-1).catch(handleError));
      });
      document.querySelectorAll("[data-review-nav='next']").forEach((button) => {
        button.addEventListener("click", () => navigateSelection(1).catch(handleError));
      });

      document.getElementById("batch-delete-mark").addEventListener("click", () => runReviewToolbarAction("reject", "Applying review mark...", "Marked delete"));
      document.getElementById("batch-export-mark").addEventListener("click", () => runReviewToolbarAction("keep", "Applying review mark...", "Marked export"));
      document.getElementById("batch-clear-mark").addEventListener("click", () => runReviewToolbarAction("reset", "Clearing review mark...", "Cleared marks"));
      document.getElementById("batch-move").addEventListener("click", () => openExportDialog("move", "Select at least one file to move."));

      document.getElementById("batch-delete-disk").addEventListener("click", () => {
        if (!hasBatchSelection()) {
          showToast("Select at least one file to delete.", "error");
          return;
        }
        withBusy("Deleting selected files...", () => deleteSelectedFiles()).catch(handleError);
      });

      installExportDialogEvents();

      document.getElementById("analyze-library").addEventListener("click", () => withBusy("Analyzing selected folder...", () => analyzeLibrary()).catch(handleError));

      document.getElementById("page-prev").addEventListener("click", () => {
        if (state.page > 0) { state.page--; loadQueue().catch(handleError); }
      });
      document.getElementById("page-next").addEventListener("click", () => {
        if ((state.page + 1) * state.pageSize < state.totalFiles) { state.page++; loadQueue().catch(handleError); }
      });
      document.getElementById("scan-library").addEventListener("click", () => withBusy("Scanning selected folder...", () => runScan(null, { generatePreviews: false })).catch(handleError));
      document.getElementById("score-library").addEventListener("click", () => withBusy("Scoring selected folder...", () => runScore()).catch(handleError));
      document.getElementById("compare-run").addEventListener("click", () => withBusy("Comparing learned models...", () => runModelComparison(), { operationType: "compare" }).catch(handleError));
      document.getElementById("compare-row-sort").addEventListener("change", (event) => {
        state.compareRowSort = event.target.value || "topiq_nr:desc";
        state.compareRowSortInitialized = true;
        renderComparisonResults();
      });

      document.getElementById("compare-row-filter").addEventListener("change", (event) => {
        state.compareRowFilter = event.target.value || "all";
        renderComparisonResults();
      });

      document.getElementById("cancel-operation").addEventListener("click", requestOperationCancel);

      document.getElementById("compare-cancel-operation").addEventListener("click", requestOperationCancel);

      document.getElementById("refresh-analysis-diagnostics")?.addEventListener("click", () => {
        loadAnalysisDiagnostics().catch(handleError);
      });

      confirmAndRun("prune-missing-cache", "Cleaning up missing files...", "missing", "Cleaned up missing files");
      confirmAndRun("clear-scores", "Clearing scores...", "scores", "Cleared score cache");
      confirmAndRun("clear-review", "Clearing review marks...", "review", "Cleared review marks");
      confirmAndRun(
        "clear-all-cache",
        "Clearing entire cache...",
        "all",
        "Cleared entire cache",
        { onSuccess: resetPersistedUiStateAfterFullReset },
      );

      // Resource profile selector
      const profileSelect = document.getElementById("resource-profile-select");
      if (profileSelect) {
        profileSelect.addEventListener("change", () => {
          localStorage.setItem("shotsieve_resource_profile", profileSelect.value);
          const hw = state.options?.learned?.hardware;
          if (typeof updateResourceProfileDetail === "function") {
            updateResourceProfileDetail(hw);
          }
        });
      }

      document.getElementById("browser-up").addEventListener("click", () => {
        if (!state.browserPath) return;
        const norm = state.browserPath.replace(/\\/g, "/").replace(/\/+$/, "");
        const isUnc = state.browserPath.startsWith("\\\\") || state.browserPath.startsWith("//");
        
        let parent = "";
        if (isUnc) {
          const parts = norm.slice(2).split("/").filter(Boolean);
          if (parts.length <= 2) {
            parent = "\\\\" + parts.join("\\");
          } else {
            parts.pop();
            parent = "\\\\" + parts.join("\\");
          }
        } else if (/^[a-zA-Z]:/.test(norm)) {
          const driveLetter = norm.slice(0, 2);
          const rest = norm.slice(2).split("/").filter(Boolean);
          if (rest.length <= 1) {
            parent = driveLetter + "\\";
          } else {
            rest.pop();
            parent = driveLetter + "\\" + rest.join("\\");
          }
        } else {
          const parts = norm.split("/").filter(Boolean);
          if (parts.length <= 1) {
            parent = "/";
          } else {
            parts.pop();
            parent = "/" + parts.join("/");
          }
        }

        browseDirectory(parent).catch(handleError);
      });
      document.getElementById("browser-choose").addEventListener("click", chooseBrowserPath);

      const browserPathInput = document.getElementById("browser-path");
      if (browserPathInput) {
        browserPathInput.addEventListener("keydown", (event) => {
          if (event.key === "Enter") {
            event.preventDefault();
            browseDirectory(browserPathInput.value.trim()).catch(handleError);
          }
        });
      }

      document.addEventListener("keydown", (event) => {
        if (isShortcutTarget(event.target)) {
          return;
        }
        if (event.key === "ArrowDown" || event.key === "ArrowRight") {
          event.preventDefault();
          navigateSelection(1).catch(handleError);
        }
        if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
          event.preventDefault();
          navigateSelection(-1).catch(handleError);
        }
        if (event.key === "s" || event.key === "S") {
          event.preventDefault();
          runReviewToolbarAction("keep", "Marking keep...", "Marked export");
        }
        if (event.key === "r" || event.key === "R") {
          event.preventDefault();
          runReviewToolbarAction("reject", "Marking reject...", "Marked delete");
        }
        if (event.key === "Escape") {
          closeOverlay("lightbox-overlay");
        }
      });

      document.getElementById("detail-image").addEventListener("click", (event) => {
        event.preventDefault();
        openLightboxFromDetailImage(event.target);
      });

      document.getElementById("detail-image").addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") {
          return;
        }
        event.preventDefault();
        openLightboxFromDetailImage(event.currentTarget);
      });

      document.getElementById("lightbox-close").addEventListener("click", () => {
        closeOverlay("lightbox-overlay");
      });

      document.getElementById("lightbox-overlay").addEventListener("click", (event) => {
        if (event.target === event.currentTarget) {
          closeOverlay("lightbox-overlay");
        }
      });

      installRejectedActionEvents();

      const preferredTheme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
      const savedTheme = localStorage.getItem("shotsieve-theme") || preferredTheme;
      applyTheme(savedTheme);
      document.getElementById("theme-toggle").addEventListener("click", () => {
        const current = document.documentElement.getAttribute("data-theme") || preferredTheme;
        const next = current === "dark" ? "light" : "dark";
        applyTheme(next);
        localStorage.setItem("shotsieve-theme", next);
      });

    };
  }

  window.ShotSieveEvents = { createEvents };
})();