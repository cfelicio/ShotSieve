(() => {
  function createController(deps) {
    const {
      state,
      uiStore,
      appUtils,
      stateModule,
      api,
      workflows,
      grid,
      notifications,
    } = deps;

    const {
      availableLearnedModels: availableLearnedModelsUtil,
      currentResourceProfile,
      escapeHtml,
      formatPhotoSupport,
      getScoreColor,
      parseRuntimeStatusMap,
      runtimeDisplayName,
      runtimeStatusToken,
      summarizeAccelerators,
      summarizeAutoPriority,
    } = appUtils;

    const { fetchJson } = api;
    const { clearUiState, currentLibraryRoot, loadUiState, saveUiState } = uiStore;
    const { addLogEntry, showToast } = notifications;

    const overlayFocusReturn = new Map();
    const OVERLAY_SELECTORS = ["#lightbox-overlay"];
    const SCORE_TOOLTIPS = {
      "AI Score": "AI aesthetic quality prediction (0–100). Higher is better.",
    };

    function scoreCard(label, value, hint = "") {
      if (value === null || value === undefined) {
        return "";
      }
      const tooltip = SCORE_TOOLTIPS[label] || "";
      const colorClass = getScoreColor(value);
      return `
        <article class="score-card ${colorClass}" title="${tooltip}">
          <span>${label}</span>
          <strong>${Number(value).toFixed(1)}</strong>
          ${hint ? `<span class="muted">${hint}</span>` : ""}
        </article>
      `;
    }

    function statusPill(label) {
      return `<span class="status-pill">${escapeHtml(label)}</span>`;
    }

    function overlayNodes() {
      return OVERLAY_SELECTORS
        .map((selector) => document.querySelector(selector))
        .filter((node) => node instanceof HTMLElement);
    }

    function supportsNativeDialog(overlay) {
      return typeof window.HTMLDialogElement !== "undefined"
        && overlay instanceof HTMLDialogElement
        && typeof overlay.showModal === "function"
        && typeof overlay.close === "function";
    }

    function overlayIsOpen(overlay) {
      if (!(overlay instanceof HTMLElement)) {
        return false;
      }
      if (supportsNativeDialog(overlay)) {
        return overlay.open;
      }
      return overlay.hasAttribute("open") && !overlay.classList.contains("overlay-closed");
    }

    function backgroundModalRoots() {
      return [...document.querySelectorAll("[data-modal-root]")].filter((node) => node instanceof HTMLElement);
    }

    function setBackgroundModalState(isHidden) {
      backgroundModalRoots().forEach((node) => {
        if (!(node instanceof HTMLElement)) {
          return;
        }

        if (isHidden) {
          if (!node.hasAttribute("data-modal-aria-hidden")) {
            node.setAttribute("data-modal-aria-hidden", node.getAttribute("aria-hidden") ?? "");
          }
          node.inert = true;
          node.setAttribute("aria-hidden", "true");
          return;
        }

        node.inert = false;
        const previousAriaHidden = node.getAttribute("data-modal-aria-hidden");
        if (previousAriaHidden === null) {
          return;
        }
        if (previousAriaHidden) {
          node.setAttribute("aria-hidden", previousAriaHidden);
        } else {
          node.removeAttribute("aria-hidden");
        }
        node.removeAttribute("data-modal-aria-hidden");
      });
    }

    function focusFirstOverlayControl(overlay) {
      const [target] = overlayFocusableElements(overlay);
      if (target instanceof HTMLElement) {
        target.focus();
        return;
      }
      overlay.focus();
    }

    function overlayFocusableElements(overlay) {
      return [...overlay.querySelectorAll(
        "[data-overlay-initial-focus], [autofocus], button:not([disabled]), [href], input:not([disabled]):not([type='hidden']), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex='-1'])",
      )].filter((node) => {
        if (!(node instanceof HTMLElement)) {
          return false;
        }
        return !node.hidden && node.getClientRects().length > 0;
      });
    }

    function restoreOverlayState(overlayId) {
      const hasOpenOverlay = overlayNodes().some((overlay) => overlayIsOpen(overlay));
      if (!hasOpenOverlay) {
        setBackgroundModalState(false);
      }

      const returnTarget = overlayFocusReturn.get(overlayId);
      overlayFocusReturn.delete(overlayId);
      if (
        !(returnTarget instanceof HTMLElement)
        || !returnTarget.isConnected
        || typeof returnTarget.focus !== "function"
        || returnTarget.hasAttribute("disabled")
      ) {
        return;
      }

      const restoreFocus = () => {
        if (!returnTarget.isConnected || returnTarget.hasAttribute("disabled")) {
          return;
        }
        returnTarget.focus();
      };

      if (typeof window.requestAnimationFrame === "function") {
        window.requestAnimationFrame(restoreFocus);
        return;
      }

      window.setTimeout(restoreFocus, 0);
    }

    function bindOverlayLifecycle(overlay) {
      if (!(overlay instanceof HTMLElement) || overlay.dataset.lifecycleBound === "true") {
        return;
      }

      if (supportsNativeDialog(overlay)) {
        overlay.addEventListener("cancel", (event) => {
          event.preventDefault();
          closeOverlay(overlay.id);
        });
      }

      overlay.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !supportsNativeDialog(overlay)) {
          event.preventDefault();
          closeOverlay(overlay.id);
          return;
        }

        if (event.key !== "Tab") {
          return;
        }

        const focusable = overlayFocusableElements(overlay);
        if (!focusable.length) {
          event.preventDefault();
          overlay.focus();
          return;
        }

        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        const activeElement = document.activeElement;

        if (focusable.length === 1) {
          event.preventDefault();
          first.focus();
          return;
        }

        if (event.shiftKey) {
          if (activeElement === first || activeElement === overlay) {
            event.preventDefault();
            last.focus();
          }
          return;
        }

        if (activeElement === last || activeElement === overlay) {
          event.preventDefault();
          first.focus();
        }
      });

      if (supportsNativeDialog(overlay)) {
        overlay.addEventListener("close", () => {
          overlay.classList.add("overlay-closed");
          restoreOverlayState(overlay.id);
        });
      }

      overlay.dataset.lifecycleBound = "true";
    }

    function setTab(tab, { focusButton = false } = {}) {
      state.tab = tab;
      document.querySelectorAll(".tab-button").forEach((button) => {
        const isActive = button.dataset.tab === tab;
        button.classList.toggle("active", isActive);
        button.setAttribute("aria-selected", isActive ? "true" : "false");
        button.tabIndex = isActive ? 0 : -1;
        if (isActive && focusButton) {
          button.focus();
        }
      });
      document.querySelectorAll(".tab-panel").forEach((panel) => {
        const isActive = panel.id === `tab-${tab}`;
        panel.classList.toggle("active", isActive);
        panel.hidden = !isActive;
      });
    }

    function openOverlay(overlayId) {
      const overlay = document.getElementById(overlayId);
      if (!(overlay instanceof HTMLElement)) {
        return;
      }

      bindOverlayLifecycle(overlay);

      const activeElement = document.activeElement;
      overlayFocusReturn.set(
        overlayId,
        activeElement instanceof HTMLElement && !overlay.contains(activeElement) ? activeElement : null,
      );

      overlay.classList.remove("overlay-closed");

      if (supportsNativeDialog(overlay)) {
        overlay.classList.remove("overlay-fallback-active");
        if (!overlay.open) {
          overlay.showModal();
        }
      } else if (!overlay.hasAttribute("open")) {
        overlay.classList.add("overlay-fallback-active");
        overlay.setAttribute("open", "");
      }

      setBackgroundModalState(true);
      focusFirstOverlayControl(overlay);
    }

    function closeOverlay(overlayId) {
      const overlay = document.getElementById(overlayId);
      if (!(overlay instanceof HTMLElement) || !overlayIsOpen(overlay)) {
        return;
      }

      if (supportsNativeDialog(overlay)) {
        overlay.close();
        return;
      }

      overlay.classList.add("overlay-closed");
      overlay.classList.remove("overlay-fallback-active");
      overlay.removeAttribute("open");
      restoreOverlayState(overlayId);
    }

    function applyTheme(theme) {
      document.documentElement.setAttribute("data-theme", theme);
      const themeToggle = document.getElementById("theme-toggle");
      if (!themeToggle) {
        return;
      }
      const nextThemeLabel = theme === "dark" ? "light" : "dark";
      themeToggle.textContent = theme === "dark" ? "🌙" : "☀️";
      themeToggle.setAttribute("aria-label", `Switch to ${nextThemeLabel} theme`);
      themeToggle.title = `Switch to ${nextThemeLabel} theme`;
    }

    function syncReviewRoot(root) {
      if (!root) {
        return "";
      }
      const rootFilter = document.getElementById("root-filter");
      if (!rootFilter) {
        return "";
      }

      const normalizedRoot = String(root)
        .replace(/[\\/]+/g, "/")
        .replace(/\/$/, "")
        .replace(/^(?:\.\/|~\/)+/, "")
        .toLowerCase();
      const options = [...rootFilter.options].map((option) => option.value).filter(Boolean);
      const exactMatch = options.find((option) => option === root);
      const suffixMatches = exactMatch
        ? [exactMatch]
        : options.filter((option) => {
          const normalizedOption = String(option)
            .replace(/[\\/]+/g, "/")
            .replace(/\/$/, "")
            .toLowerCase();
          return normalizedOption === normalizedRoot || normalizedOption.endsWith(`/${normalizedRoot}`);
        });
      const resolvedRoot = exactMatch || (suffixMatches.length === 1 ? suffixMatches[0] : "");
      if (resolvedRoot) {
        rootFilter.value = resolvedRoot;
      }
      return resolvedRoot;
    }

    function setReviewScope(root, { clearActiveSelection = true } = {}) {
      const rootFilter = document.getElementById("root-filter");
      if (!rootFilter) {
        return;
      }

      const scopeRoot = String(root || "").trim();
      if (scopeRoot && ![...rootFilter.options].some((option) => option.value === scopeRoot)) {
        rootFilter.add(new Option(scopeRoot, scopeRoot, false, false));
      }
      rootFilter.value = scopeRoot;
      if (rootFilter.value !== scopeRoot) {
        rootFilter.value = "";
      }

      state.reviewScopeInitialized = true;
      if (grid?.invalidateLoadedReviewSelection) {
        grid.invalidateLoadedReviewSelection({ clearActiveSelection });
      }
      state.activeId = null;
      state.detail = null;
      state.page = 0;
      if (grid?.renderReviewScope) {
        grid.renderReviewScope();
      }
    }

    async function activateLibraryScope(root) {
      const normalizedRoot = String(root || "").trim();
      const libraryRootInput = document.getElementById("library-root-input");
      if (libraryRootInput) {
        libraryRootInput.value = normalizedRoot;
      }
      setReviewScope(normalizedRoot);
      saveUiState();
      await refreshOverview();
      if (grid?.loadQueue) {
        await grid.loadQueue();
      }
    }

    function resetReviewToActiveLibrary() {
      setReviewScope(currentLibraryRoot());
    }

    function populateRootFilters() {
      const roots = state.overview?.roots || [];
      const rootFilter = document.getElementById("root-filter");
      const previous = rootFilter.value;
      rootFilter.innerHTML = [`<option value="">All libraries (global)</option>`]
        .concat(roots.map((root) => `<option value="${escapeHtml(root)}">${escapeHtml(root)}</option>`))
        .join("");
      if (previous && roots.includes(previous)) {
        rootFilter.value = previous;
        return;
      }
      if (previous) {
        rootFilter.add(new Option(previous, previous, false, true));
        rootFilter.value = previous;
      }
      if (!state.reviewScopeInitialized && currentLibraryRoot()) {
        setReviewScope(currentLibraryRoot(), { clearActiveSelection: false });
      }
      if (grid?.renderReviewScope) {
        grid.renderReviewScope();
      }
    }

    function renderOptions() {
      const options = state.options;
      if (!options) return;
      if (document.body?.dataset) {
        document.body.dataset.databasePath = options.database || "";
      }
      const persisted = loadUiState();

      const modelSelect = document.getElementById("model-select");
      const previousModel = modelSelect.value;
      const scoringModes = availableLearnedModelsUtil(options, stateModule.DEFAULT_MODEL_CATALOG, stateModule.HIDDEN_MODEL_NAMES);
      modelSelect.innerHTML = scoringModes.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(stateModule.MODEL_DISPLAY_NAMES[model] || model)}</option>`)
        .join("");

      const preferredModel = previousModel || persisted.model || options.default_scoring_mode || scoringModes[0] || "topiq_nr";
      modelSelect.value = scoringModes.includes(preferredModel)
        ? preferredModel
        : (options.default_scoring_mode && scoringModes.includes(options.default_scoring_mode)
          ? options.default_scoring_mode
          : (scoringModes[0] || "topiq_nr"));

      const deviceSelect = document.getElementById("device-select");
      const previousDevice = deviceSelect.value;
      deviceSelect.innerHTML = options.runtime_targets
        .map((runtime) => `<option value="${escapeHtml(runtime)}">${escapeHtml(runtimeDisplayName(runtime))}</option>`)
        .join("");
      deviceSelect.value = previousDevice || persisted.device || "auto";

      document.getElementById("extensions-input").value = persisted.extensions || options.default_extensions.join(",");
      document.getElementById("recursive-toggle").checked = persisted.recursive ?? true;
      document.getElementById("ignore-rules-input").value = persisted.ignoreRules || "";
      if (!currentLibraryRoot() && persisted.libraryRoot) {
        document.getElementById("library-root-input").value = persisted.libraryRoot;
      }
      if (persisted.maxScore !== undefined && persisted.maxScore !== "") {
        document.getElementById("max-score").value = persisted.maxScore;
      } else {
        document.getElementById("max-score").value = "";
      }
      if (persisted.minScore !== undefined && persisted.minScore !== "") {
        document.getElementById("min-score").value = persisted.minScore;
      } else {
        document.getElementById("min-score").value = "";
      }
      document.getElementById("issues-filter").value = persisted.issues || "all";

      const savedFormats = persisted.formats || ["jpeg", "png", "tiff", "heif", "raw", "other"];
      document.querySelectorAll("input[name='format-filter']").forEach((input) => {
        input.checked = savedFormats.includes(input.value);
      });
      document.getElementById("filter-min-mp").value = persisted.minMp || "";
      document.getElementById("filter-max-mp").value = persisted.maxMp || "";
      document.getElementById("filter-min-size").value = persisted.minSize || "";
      document.getElementById("filter-max-size").value = persisted.maxSize || "";
      document.getElementById("filter-metadata-status").value = persisted.metadataStatus || "all";

      modelSelect.onchange = () => {
        const val = modelSelect.value;
        document.getElementById("model-detail-hint").textContent = stateModule.MODEL_DESCRIPTIONS[val] || "No detailed notes available for this model.";
      };
      modelSelect.onchange();

      const runtimeModelWarning = document.getElementById("runtime-model-warning");
      if (runtimeModelWarning) {
        const activeRuntime = String(options.learned?.default_runtime || "").toLowerCase();
        const modelSet = new Set(options.learned_models || []);
        const qalignBlockedRuntimes = new Set(["cpu", "directml"]);
        const qalignUnavailableOnActiveRuntime = qalignBlockedRuntimes.has(activeRuntime) && !modelSet.has("qalign");
        if (qalignUnavailableOnActiveRuntime) {
          runtimeModelWarning.textContent = "Q-Align is unavailable for the active runtime. Use TOPIQ or CLIPIQA, or switch to another supported accelerator runtime.";
          runtimeModelWarning.classList.remove("hidden");
        } else {
          runtimeModelWarning.textContent = "";
          runtimeModelWarning.classList.add("hidden");
        }
      }

      if (workflows?.renderComparisonModelOptions) {
        workflows.renderComparisonModelOptions(options, scoringModes, persisted);
      }
      if (workflows?.renderComparisonSummary) {
        workflows.renderComparisonSummary();
      }

      const statusMap = parseRuntimeStatusMap(options.learned.runtime_status);
      const activeRuntime = String(options.learned.default_runtime || "cpu").toLowerCase();
      const activeStatus = statusMap[activeRuntime] || "unknown";
      const activeRuntimeLabel = `${runtimeDisplayName(activeRuntime)} ${runtimeStatusToken(activeStatus)} ${activeStatus}`;
      const heifOk = options.preview_capabilities.heif_decoder && options.preview_capabilities.heif_decoder !== "none";
      const rawOk = options.preview_capabilities.raw_decoder && options.preview_capabilities.raw_decoder !== "none";
      const hw = options.learned.hardware || {};
      const cpuLabel = hw.cpu_count ? `${hw.cpu_count} cores` : "Unknown";
      const ramLabel = hw.ram_mb ? `${(hw.ram_mb / 1024).toFixed(1)} GB` : "Unknown";
      const vramLabel = hw.vram_mb ? `${(hw.vram_mb / 1024).toFixed(1)} GB` : "Not detected";

      const hwCards = [
        ["CPU Cores", cpuLabel],
        ["System RAM", ramLabel],
        ["GPU VRAM", vramLabel],
      ];
      const hwEl = document.getElementById("hardware-cards");
      if (hwEl) {
        hwEl.innerHTML = hwCards.map(([label, value]) => `
          <article class="runtime-card">
            <p class="eyebrow">${escapeHtml(label)}</p>
            <strong>${escapeHtml(value)}</strong>
          </article>
        `).join("");
      }

      const runtimeCards = [
        ["Active Runtime", activeRuntimeLabel],
        ["Available Accelerators", summarizeAccelerators(statusMap)],
        ["Auto Mode Priority", summarizeAutoPriority(options)],
        ["Photo Support", formatPhotoSupport(heifOk, rawOk)],
      ];
      document.getElementById("runtime-cards").innerHTML = runtimeCards.map(([label, value]) => `
        <article class="runtime-card">
          <p class="eyebrow">${escapeHtml(label)}</p>
          <strong>${escapeHtml(value)}</strong>
        </article>
      `).join("");

      const profileSelect = document.getElementById("resource-profile-select");
      if (profileSelect) {
        const savedProfile = localStorage.getItem("shotsieve_resource_profile") || "normal";
        profileSelect.value = savedProfile;
        updateResourceProfileDetail(hw);
      }

      const rawExts = options.default_extensions.filter(e => [".3fr",".arw",".cr2",".cr3",".dng",".nef",".orf",".raf",".rw2"].includes(e));
      const heifExts = options.default_extensions.filter(e => [".heic",".heif"].includes(e));
      const stdExts = options.default_extensions.filter(e => [".jpg",".jpeg",".png",".tif",".tiff"].includes(e));
      const systemInfo = [
        ["Standard Formats", stdExts.map(e => e.replace(".","").toUpperCase()).join(", ") || "None"],
        ["RAW Formats", rawExts.map(e => e.replace(".","").toUpperCase()).join(", ") || "None"],
        ["HEIF Formats", heifExts.map(e => e.replace(".","").toUpperCase()).join(", ") || "None"],
        ["Database", options.database],
        ["Preview Cache", options.preview_dir],
      ];
      document.getElementById("system-info").innerHTML = systemInfo.map(([label, value]) => `
        <div class="system-info-card">
          <div class="system-info-label">${escapeHtml(label)}</div>
          <div class="system-info-value" title="${escapeHtml(value)}">${escapeHtml(value)}</div>
        </div>
      `).join("");

      if (workflows?.renderLibraryRoots) {
        workflows.renderLibraryRoots();
      }
    }

    function updateResourceProfileDetail(hw) {
      const hint = document.getElementById("resource-profile-detail");
      const select = document.getElementById("resource-profile-select");
      if (!hint || !select) return;
      const cores = hw?.cpu_count || 0;
      const vramGb = hw?.vram_mb ? (hw.vram_mb / 1024).toFixed(1) : null;
      const profile = select.value || "normal";
      const descriptions = {
        aggressive: `Uses ${Math.max(4, cores - 2)} of ${cores} CPU threads${vramGb ? `, ~80% of ${vramGb} GB VRAM` : ""}`,
        normal: `Uses ${Math.max(4, Math.floor(cores / 2))} of ${cores} CPU threads${vramGb ? `, ~50% of ${vramGb} GB VRAM` : ""}`,
        low: `Uses ${Math.max(2, Math.floor(cores / 4))} of ${cores} CPU threads${vramGb ? `, ~30% of ${vramGb} GB VRAM` : ""}`,
      };
      hint.textContent = descriptions[profile] || "";
    }

    async function refreshOverview() {
      const root = currentLibraryRoot();
      const query = root ? `?root=${encodeURIComponent(root)}` : "";
      state.overview = await fetchJson(`/api/overview${query}`);
      if (grid?.renderSummary) {
        grid.renderSummary();
      }
      populateRootFilters();
    }

    async function loadAnalysisDiagnostics() {
      const root = currentLibraryRoot();
      const query = new URLSearchParams({ limit: "100" });
      if (root) {
        query.set("root", root);
      }
      const payload = await fetchJson(`/api/analysis-diagnostics?${query.toString()}`);
      const total = Number(payload.total || 0);
      const items = Array.isArray(payload.items) ? payload.items : [];
      const summary = document.getElementById("analysis-diagnostics-summary");
      const list = document.getElementById("analysis-diagnostics-list");
      if (!summary || !list) {
        return;
      }

      if (!total) {
        summary.textContent = "All discovered photos in the current library have a quality score.";
        list.innerHTML = "";
        return;
      }

      const scope = root ? "this library" : "all cached libraries";
      summary.textContent = `${total.toLocaleString()} photo${total === 1 ? "" : "s"} in ${scope} ${total === 1 ? "needs" : "need"} attention.`;
      list.innerHTML = items.map((item) => {
        const status = String(item.status || "pending");
        return `
          <article class="analysis-diagnostic-item" data-status="${escapeHtml(status)}">
            <span class="analysis-diagnostic-status">${escapeHtml(status)}</span>
            <strong class="analysis-diagnostic-path">${escapeHtml(String(item.path || "Unknown file"))}</strong>
            <span class="analysis-diagnostic-error">${escapeHtml(String(item.error || "No diagnostic detail is available."))}</span>
          </article>
        `;
      }).join("");
    }

    async function loadOptions() {
      const profile = currentResourceProfile();
      state.options = await fetchJson(`/api/options?resource_profile=${encodeURIComponent(profile)}`);
      renderOptions();
    }

    async function refreshWorkspace() {
      await loadOptions();
      await refreshOverview();
      await loadAnalysisDiagnostics();

      if (grid?.renderSummary) {
        grid.renderSummary();
      }
      if (workflows?.renderComparisonSummary) {
        workflows.renderComparisonSummary();
      }
      if (grid?.loadQueue) {
        await grid.loadQueue();
      }
    }

    return {
      scoreCard,
      statusPill,
      openOverlay,
      closeOverlay,
      applyTheme,
      syncReviewRoot,
      setReviewScope,
      activateLibraryScope,
      resetReviewToActiveLibrary,
      populateRootFilters,
      renderOptions,
      updateResourceProfileDetail,
      refreshOverview,
      loadAnalysisDiagnostics,
      loadOptions,
      refreshWorkspace,
      setTab,
    };
  }

  window.ShotSieveController = {
    createController,
  };
})();
