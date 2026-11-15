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
      prepareSelectedModel,
      runModelComparison,
      renderComparisonResults,
      clearCache,
      reviewMissingEntries,
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
      checkTrackedJob,
      selectAll,
      selectNone,
      selectAllMatching,
      invalidateLoadedReviewSelection,
      activateLibraryScope,
      resetReviewToActiveLibrary,
      setReviewScope,
      renderLibraryRoots,
      installDecisionCsvEvents,
      loadAnalysisDiagnostics,
      updateResourceProfileDetail,
    } = deps;

    const TAB_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End"]);
    let filterTimer = null;

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

    function hasOpenOverlay() {
      return [...document.querySelectorAll("dialog, [role='dialog']")].some((overlay) => {
        if (overlay instanceof HTMLDialogElement) {
          return overlay.open;
        }
        return overlay.hasAttribute("open") && !overlay.classList.contains("overlay-closed");
      });
    }

    function reviewShortcutsEnabled() {
      return state.tab === "review" && !state.isBusy && !hasOpenOverlay();
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

      const maxDecodeMegapixels = document.getElementById("max-decode-megapixels");
      if (maxDecodeMegapixels) {
        maxDecodeMegapixels.value = String(state.options?.default_max_decode_megapixels || 64);
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

    const BROWSER_RENDERABLE_EXTS = new Set(["jpg", "jpeg", "png", "webp", "gif"]);
    let lightboxZoom = 1;
    let lightboxPanX = 0;
    let lightboxPanY = 0;
    let isDraggingLightbox = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let dragInitialPanX = 0;
    let dragInitialPanY = 0;
    let dragMoved = false;
    let currentLightboxDetailId = null;

    function clampPan(x, y, scale) {
      const img = document.getElementById("lightbox-image");
      if (!img || scale <= 1.01) {
        return { x: 0, y: 0 };
      }
      const scaledW = (img.offsetWidth || window.innerWidth) * scale;
      const scaledH = (img.offsetHeight || window.innerHeight) * scale;
      const maxPanX = Math.max(0, (scaledW - window.innerWidth) / 2 + 60);
      const maxPanY = Math.max(0, (scaledH - window.innerHeight) / 2 + 60);
      return {
        x: Math.max(-maxPanX, Math.min(maxPanX, x)),
        y: Math.max(-maxPanY, Math.min(maxPanY, y)),
      };
    }

    function updateLightboxToolbar() {
      const label = document.getElementById("lightbox-zoom-level");
      if (label) {
        const isFit = Math.abs(lightboxZoom - 1) < 0.05;
        label.textContent = isFit ? "Fit" : `${Math.round(lightboxZoom * 100)}%`;
        label.title = isFit ? "Zoom to 250%" : "Reset zoom to fit window";
      }
      const zoomOutBtn = document.getElementById("lightbox-zoom-out");
      if (zoomOutBtn) {
        zoomOutBtn.disabled = lightboxZoom <= 1.01;
      }
      const zoomInBtn = document.getElementById("lightbox-zoom-in");
      if (zoomInBtn) {
        zoomInBtn.disabled = lightboxZoom >= 4.95;
      }
    }

    function applyLightboxTransform(animate = false) {
      const img = document.getElementById("lightbox-image");
      if (!img) return;
      img.style.transition = animate ? "transform 0.12s ease-out" : "none";
      if (Math.abs(lightboxZoom - 1) < 0.01 && lightboxPanX === 0 && lightboxPanY === 0) {
        img.style.transform = "";
        img.classList.remove("is-zoomed");
      } else {
        img.style.transform = `translate(${lightboxPanX}px, ${lightboxPanY}px) scale(${lightboxZoom})`;
        img.classList.add("is-zoomed");
      }
      updateLightboxToolbar();
    }

    function resetLightboxZoom() {
      lightboxZoom = 1;
      lightboxPanX = 0;
      lightboxPanY = 0;
      isDraggingLightbox = false;
      dragMoved = false;
      const img = document.getElementById("lightbox-image");
      if (img) {
        img.classList.remove("is-dragging");
      }
      applyLightboxTransform(false);
    }

    function setLightboxZoom(newScale, focalX, focalY, animate = true) {
      const oldScale = lightboxZoom;
      const clampedScale = Math.max(1, Math.min(5, newScale));
      if (Math.abs(clampedScale - 1) < 0.01) {
        lightboxZoom = 1;
        lightboxPanX = 0;
        lightboxPanY = 0;
        applyLightboxTransform(animate);
        return;
      }
      if (typeof focalX === "number" && typeof focalY === "number" && oldScale > 0) {
        const centerX = window.innerWidth / 2;
        const centerY = window.innerHeight / 2;
        const ratio = clampedScale / oldScale;
        const newPanX = focalX - centerX - ratio * (focalX - centerX - lightboxPanX);
        const newPanY = focalY - centerY - ratio * (focalY - centerY - lightboxPanY);
        const clamped = clampPan(newPanX, newPanY, clampedScale);
        lightboxPanX = clamped.x;
        lightboxPanY = clamped.y;
      } else {
        const clamped = clampPan(lightboxPanX, lightboxPanY, clampedScale);
        lightboxPanX = clamped.x;
        lightboxPanY = clamped.y;
      }
      lightboxZoom = clampedScale;
      applyLightboxTransform(animate);
    }

    function toggleLightboxZoom(clientX, clientY) {
      if (lightboxZoom > 1.05) {
        resetLightboxZoom();
      } else {
        const fx = typeof clientX === "number" ? clientX : window.innerWidth / 2;
        const fy = typeof clientY === "number" ? clientY : window.innerHeight / 2;
        setLightboxZoom(2.5, fx, fy, true);
      }
    }

    function openLightboxFromDetailImage(sourceImage) {
      if (!(sourceImage instanceof HTMLImageElement) || !sourceImage.src) {
        return;
      }
      if (typeof sourceImage.focus === "function") {
        sourceImage.focus();
      }
      resetLightboxZoom();
      const lightboxImg = document.getElementById("lightbox-image");
      lightboxImg.src = sourceImage.src;
      const detailTitle = document.getElementById("detail-title");
      lightboxImg.alt = detailTitle ? `Full-size preview of ${detailTitle.textContent}` : "Full-size photo";
      deps.openOverlay("lightbox-overlay");

      const detail = state.detail;
      if (detail && detail.id) {
        const detailId = detail.id;
        currentLightboxDetailId = detailId;
        let fullResId = null;
        const primaryFormat = String(detail.format || "").toLowerCase();
        if (BROWSER_RENDERABLE_EXTS.has(primaryFormat)) {
          fullResId = detailId;
        } else if (detail.is_pair && detail.companion) {
          const companionFormat = String(detail.companion.format || "").toLowerCase();
          if (BROWSER_RENDERABLE_EXTS.has(companionFormat)) {
            fullResId = detail.companion.id;
          }
        }
        if (fullResId) {
          const fullResUrl = `/api/media/source?id=${fullResId}`;
          const preloadImg = new Image();
          preloadImg.onload = () => {
            const overlay = document.getElementById("lightbox-overlay");
            const isOpen = overlay instanceof HTMLDialogElement ? overlay.open : overlay?.hasAttribute("open");
            if (isOpen && currentLightboxDetailId === detailId && lightboxImg) {
              lightboxImg.src = fullResUrl;
            }
          };
          preloadImg.src = fullResUrl;
        }
      }
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
        ["query-filter", "min-score", "max-score", "filter-min-mp", "filter-max-mp", "filter-min-edge", "filter-max-edge", "filter-min-size", "filter-max-size"].forEach((id) => {
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

      function syncFormatChipsState() {
        const checkboxes = document.querySelectorAll("input[name='format-filter']");
        if (!checkboxes.length) return;
        const allChecked = Array.from(checkboxes).every((cb) => cb.checked);

        const allChip = document.querySelector("[data-chip-format='all']") || document.querySelector("[data-format-chip='all']");
        if (allChip) {
          allChip.classList.toggle("active", allChecked);
        }

        checkboxes.forEach((cb) => {
          const chip = document.querySelector(`[data-chip-format='${cb.value}']`) || document.querySelector(`[data-format-chip='${cb.value}']`);
          if (chip) {
            chip.classList.toggle("active", cb.checked);
          }
        });
      }

      document.querySelectorAll("[data-chip-format], [data-format-chip]").forEach((chip) => {
        chip.addEventListener("click", () => {
          const format = chip.dataset.chipFormat || chip.dataset.formatChip;
          const checkboxes = document.querySelectorAll("input[name='format-filter']");
          if (format === "all") {
            const allChecked = Array.from(checkboxes).every((cb) => cb.checked);
            checkboxes.forEach((cb) => { cb.checked = !allChecked; });
          } else {
            const cb = document.querySelector(`input[name='format-filter'][value='${format}']`);
            if (cb) {
              cb.checked = !cb.checked;
            }
          }
          syncFormatChipsState();
          invalidateLoadedReviewSelection({ clearActiveSelection: true });
          state.page = 0;
          saveUiState();
          loadQueue().catch(handleError);
        });
      });

      document.querySelectorAll("input[name='format-filter']").forEach((checkbox) => {
        checkbox.addEventListener("change", () => {
          syncFormatChipsState();
          invalidateLoadedReviewSelection({ clearActiveSelection: true });
          state.page = 0;
          saveUiState();
          loadQueue().catch(handleError);
        });
      });

      // Interactive Resizable Splitter Bar
      const splitter = document.getElementById("review-splitter");
      const layout = document.getElementById("review-layout");
      if (splitter && layout) {
        const savedWidth = localStorage.getItem("shotsieve_review_sidebar_width");
        if (savedWidth) {
          layout.style.setProperty("--sidebar-width", `${savedWidth}px`);
        }

        let isDragging = false;

        const onPointerMove = (e) => {
          if (!isDragging) return;
          const rect = layout.getBoundingClientRect();
          const offsetX = e.clientX - rect.left;
          const minW = 280;
          const maxW = Math.min(800, window.innerWidth - 350);
          const clampedW = Math.max(minW, Math.min(maxW, offsetX));
          layout.style.setProperty("--sidebar-width", `${clampedW}px`);
          localStorage.setItem("shotsieve_review_sidebar_width", clampedW);
        };

        const onPointerUp = () => {
          if (!isDragging) return;
          isDragging = false;
          splitter.classList.remove("dragging");
          document.body.style.cursor = "";
          document.body.style.userSelect = "";
          window.removeEventListener("pointermove", onPointerMove);
          window.removeEventListener("pointerup", onPointerUp);
        };

        splitter.addEventListener("pointerdown", (e) => {
          isDragging = true;
          splitter.classList.add("dragging");
          document.body.style.cursor = "col-resize";
          document.body.style.userSelect = "none";
          window.addEventListener("pointermove", onPointerMove);
          window.addEventListener("pointerup", onPointerUp);
        });
      }

      ["min-score", "max-score", "filter-min-mp", "filter-max-mp", "filter-min-edge", "filter-max-edge", "filter-min-size", "filter-max-size"].forEach((id) => {
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
        "max-decode-megapixels",
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
      });

      document.getElementById("ignore-rules-input")?.addEventListener("input", () => {
        saveUiState();
      });
      document.getElementById("ignore-rules-input")?.addEventListener("change", () => {
        saveUiState();
      });

      document.getElementById("recursive-toggle")?.addEventListener("change", () => {
        saveUiState();
      });

      document.getElementById("extensions-input")?.addEventListener("input", () => {
        saveUiState();
      });
      document.getElementById("extensions-input")?.addEventListener("change", () => {
        saveUiState();
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

      const decisionKeepBtn = document.getElementById("decision-keep-btn");
      if (decisionKeepBtn) {
        decisionKeepBtn.addEventListener("click", () => runReviewToolbarAction("keep", "Applying review mark...", "Marked export"));
      }

      const decisionRejectBtn = document.getElementById("decision-reject-btn");
      if (decisionRejectBtn) {
        decisionRejectBtn.addEventListener("click", () => runReviewToolbarAction("reject", "Applying review mark...", "Marked delete"));
      }

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
      document.getElementById("prepare-model")?.addEventListener("click", () => withBusy("Preparing selected model...", () => prepareSelectedModel()).catch(handleError));
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

      document.getElementById("job-recovery-check-status")?.addEventListener("click", () => {
        checkTrackedJob().then((result) => {
          if (result?.model_names && result?.rows) {
            renderComparisonResults();
          }
        }).catch(handleError);
      });

      document.getElementById("refresh-analysis-diagnostics")?.addEventListener("click", () => {
        loadAnalysisDiagnostics().catch(handleError);
      });

      confirmAndRun("clear-scores", "Clearing scores...", "scores", "Cleared score cache");
      confirmAndRun("clear-review", "Clearing review marks...", "review", "Cleared review marks");
      confirmAndRun(
        "clear-all-cache",
        "Clearing entire cache...",
        "all",
        "Cleared entire cache",
        { onSuccess: resetPersistedUiStateAfterFullReset },
      );

      document.getElementById("review-missing-entries")?.addEventListener("click", () => {
        withBusy("Checking selected library for missing entries...", () => reviewMissingEntries()).catch(handleError);
      });

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
        const overlay = document.getElementById("lightbox-overlay");
        const isLightboxOpen = overlay instanceof HTMLDialogElement
          ? overlay.open
          : (overlay && overlay.hasAttribute("open") && !overlay.classList.contains("overlay-closed"));

        if (isLightboxOpen) {
          if (event.key === "Escape") {
            closeOverlay("lightbox-overlay");
            return;
          }
          if (event.key === "+" || event.key === "=") {
            event.preventDefault();
            setLightboxZoom(lightboxZoom * 1.3, window.innerWidth / 2, window.innerHeight / 2, true);
            return;
          }
          if (event.key === "-" || event.key === "_") {
            event.preventDefault();
            setLightboxZoom(lightboxZoom / 1.3, window.innerWidth / 2, window.innerHeight / 2, true);
            return;
          }
          if (event.key === "0" || event.key === "f" || event.key === "F") {
            event.preventDefault();
            resetLightboxZoom();
            return;
          }
          if (lightboxZoom > 1.01) {
            const step = 60;
            if (event.key === "ArrowLeft") {
              event.preventDefault();
              const clamped = clampPan(lightboxPanX + step, lightboxPanY, lightboxZoom);
              lightboxPanX = clamped.x;
              lightboxPanY = clamped.y;
              applyLightboxTransform(true);
              return;
            }
            if (event.key === "ArrowRight") {
              event.preventDefault();
              const clamped = clampPan(lightboxPanX - step, lightboxPanY, lightboxZoom);
              lightboxPanX = clamped.x;
              lightboxPanY = clamped.y;
              applyLightboxTransform(true);
              return;
            }
            if (event.key === "ArrowUp") {
              event.preventDefault();
              const clamped = clampPan(lightboxPanX, lightboxPanY + step, lightboxZoom);
              lightboxPanX = clamped.x;
              lightboxPanY = clamped.y;
              applyLightboxTransform(true);
              return;
            }
            if (event.key === "ArrowDown") {
              event.preventDefault();
              const clamped = clampPan(lightboxPanX, lightboxPanY - step, lightboxZoom);
              lightboxPanX = clamped.x;
              lightboxPanY = clamped.y;
              applyLightboxTransform(true);
              return;
            }
          }
          return;
        }

        if (event.key === "Escape") {
          closeOverlay("lightbox-overlay");
          return;
        }
        if (isShortcutTarget(event.target)) {
          return;
        }
        if (!reviewShortcutsEnabled()) {
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

      const lightboxImg = document.getElementById("lightbox-image");
      if (lightboxImg) {
        lightboxImg.addEventListener("pointerdown", (event) => {
          if (event.button !== 0) return;
          if (lightboxZoom <= 1.01) return;
          isDraggingLightbox = true;
          dragMoved = false;
          dragStartX = event.clientX;
          dragStartY = event.clientY;
          dragInitialPanX = lightboxPanX;
          dragInitialPanY = lightboxPanY;
          lightboxImg.classList.add("is-dragging");
          if (typeof lightboxImg.setPointerCapture === "function") {
            try {
              lightboxImg.setPointerCapture(event.pointerId);
            } catch (_) {}
          }
        });

        lightboxImg.addEventListener("pointermove", (event) => {
          if (!isDraggingLightbox) return;
          const dx = event.clientX - dragStartX;
          const dy = event.clientY - dragStartY;
          if (!dragMoved && (Math.abs(dx) > 3 || Math.abs(dy) > 3)) {
            dragMoved = true;
          }
          if (dragMoved) {
            const clamped = clampPan(dragInitialPanX + dx, dragInitialPanY + dy, lightboxZoom);
            lightboxPanX = clamped.x;
            lightboxPanY = clamped.y;
            applyLightboxTransform(false);
          }
        });

        const endPointerDrag = (event) => {
          if (!isDraggingLightbox) return;
          isDraggingLightbox = false;
          lightboxImg.classList.remove("is-dragging");
          if (typeof lightboxImg.releasePointerCapture === "function" && event?.pointerId) {
            try {
              lightboxImg.releasePointerCapture(event.pointerId);
            } catch (_) {}
          }
        };

        lightboxImg.addEventListener("pointerup", endPointerDrag);
        lightboxImg.addEventListener("pointercancel", endPointerDrag);

        lightboxImg.addEventListener("click", (event) => {
          if (dragMoved) {
            dragMoved = false;
            return;
          }
          toggleLightboxZoom(event.clientX, event.clientY);
        });

        lightboxImg.addEventListener("wheel", (event) => {
          event.preventDefault();
          const delta = -event.deltaY;
          const factor = delta > 0 ? 1.25 : 0.8;
          setLightboxZoom(lightboxZoom * factor, event.clientX, event.clientY, false);
        }, { passive: false });
      }

      const zoomInBtn = document.getElementById("lightbox-zoom-in");
      if (zoomInBtn) {
        zoomInBtn.addEventListener("click", (event) => {
          event.stopPropagation();
          setLightboxZoom(lightboxZoom * 1.3, window.innerWidth / 2, window.innerHeight / 2, true);
        });
      }

      const zoomOutBtn = document.getElementById("lightbox-zoom-out");
      if (zoomOutBtn) {
        zoomOutBtn.addEventListener("click", (event) => {
          event.stopPropagation();
          setLightboxZoom(lightboxZoom / 1.3, window.innerWidth / 2, window.innerHeight / 2, true);
        });
      }

      const zoomLevelBtn = document.getElementById("lightbox-zoom-level");
      if (zoomLevelBtn) {
        zoomLevelBtn.addEventListener("click", (event) => {
          event.stopPropagation();
          toggleLightboxZoom(window.innerWidth / 2, window.innerHeight / 2);
        });
      }

      const zoomResetBtn = document.getElementById("lightbox-zoom-reset");
      if (zoomResetBtn) {
        zoomResetBtn.addEventListener("click", (event) => {
          event.stopPropagation();
          resetLightboxZoom();
        });
      }

      const lightboxToolbar = document.getElementById("lightbox-toolbar");
      if (lightboxToolbar) {
        lightboxToolbar.addEventListener("click", (event) => {
          event.stopPropagation();
        });
      }

      document.getElementById("lightbox-close").addEventListener("click", () => {
        closeOverlay("lightbox-overlay");
      });

      const lightboxOverlay = document.getElementById("lightbox-overlay");
      if (lightboxOverlay) {
        lightboxOverlay.addEventListener("close", () => {
          resetLightboxZoom();
        });
        lightboxOverlay.addEventListener("click", (event) => {
          if (event.target === event.currentTarget || event.target.classList.contains("lightbox-content")) {
            closeOverlay("lightbox-overlay");
          }
        });
      }

      installRejectedActionEvents();
      installDecisionCsvEvents();

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
