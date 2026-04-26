const stateModule = window.ShotSieveState;
if (!stateModule?.createState || !stateModule?.createUiStateStore) {
  throw new Error("ShotSieve state module failed to load.");
}

const busyModule = window.ShotSieveBusy;
if (!busyModule?.createBusyController) {
  throw new Error("ShotSieve busy module failed to load.");
}

const {
  DEFAULT_MODEL_CATALOG,
  HIDDEN_MODEL_NAMES,
  MODEL_DESCRIPTIONS,
  MODEL_DISPLAY_NAMES,
  REVIEW_DECISIONS,
  createState,
  createUiStateStore,
} = stateModule;

const state = createState();
const {
  clearUiState,
  currentLibraryRoot,
  isAutoAdvanceEnabled,
  loadUiState,
  saveUiState,
  selectedComparisonModels,
} = createUiStateStore();

let queueAbortController = null;
const overlayFocusReturn = new Map();

const OVERLAY_SELECTORS = ["#lightbox-overlay"];


const SCORE_TOOLTIPS = {
  "AI Score": "AI aesthetic quality prediction (0–100). Higher is better.",
};

const appUtils = window.ShotSieveUtils;
if (!appUtils) {
  throw new Error("ShotSieve utility module failed to load.");
}

const {
  availableLearnedModels: availableLearnedModelsUtil,
  compareBatchSize,
  comparisonDefaults: comparisonDefaultsUtil,
  compareProgressMessage,
  compareProgressPercent,
  currentResourceProfile,
  escapeHtml,
  formatDuration,
  formatFilesPerSecond,
  getScoreColor,
  mergeTimingTotals,
  parseRuntimeStatusMap,
  pathDirectory,
  pathLeaf,
  formatPhotoSupport,
  runtimeDisplayName,
  runtimeStatusToken,
  summarizeAccelerators,
  summarizeAutoPriority,
  scoreBatchSize,
  scoreProgressMessage,
  scoreProgressPercent,
  scanProgressMessage,
  scanProgressPercent,
  sortComparisonRows,
  fetchJson,
  postJson,
  formatNumber,
} = appUtils;

const reviewModule = window.ShotSieveReview;
if (!reviewModule) {
  throw new Error("ShotSieve review module failed to load.");
}

const {
  getSortRelevantScore,
  renderDetail: renderDetailView,
  renderQueue: renderQueueView,
  updateSelectionState: updateSelectionStateView,
} = reviewModule;

const workflowsModule = window.ShotSieveWorkflows;
if (!workflowsModule?.createWorkflows) {
  throw new Error("ShotSieve workflows module failed to load.");
}

const {
  renderBusyState,
  setBusyMessage,
  setBusyPhaseProgress,
  setBusyProgress,
  withBusy,
} = busyModule.createBusyController({
  state,
  api: { fetchJson, postJson },
  notify: { addLogEntry, showToast },
});

const {
  renderComparisonSummary,
  renderComparisonResults,
  renderComparisonModelOptions,
  runModelComparison,
  saveReviewDecision,
  saveReviewDecisionWithOptions,
  runBatchReviewDecision,
  openExportDialog,
  installExportDialogEvents,
  installRejectedActionEvents,
  runScan,
  runScore,
  analyzeLibrary,
  clearCache,
  deleteSelectedFiles,
  navigateSelection,
  openOriginalFile,
  openBrowser,
  browseDirectory,
  chooseBrowserPath,
  handleError,
} = workflowsModule.createWorkflows({
  state,
  api: { fetchJson, postJson },
  busy: {
    setBusyMessage,
    setBusyPhaseProgress,
    setBusyProgress,
    withBusy,
  },
  compare: {
    compareBatchSize,
    compareProgressMessage,
    compareProgressPercent,
    comparisonDefaults: comparisonDefaultsUtil,
    currentResourceProfile,
    modelDescriptions: MODEL_DESCRIPTIONS,
    modelDisplayNames: MODEL_DISPLAY_NAMES,
    scanProgressMessage,
    scanProgressPercent,
    scoreBatchSize,
    scoreProgressMessage,
    scoreProgressPercent,
  },
  formatting: {
    escapeHtml,
    formatDuration,
    formatFilesPerSecond,
    formatNumber,
    getScoreColor,
    mergeTimingTotals,
    pathLeaf,
    sortComparisonRows,
  },
  notifications: {
    addLogEntry,
    showToast,
  },
  review: {
    isAutoAdvanceEnabled,
    loadQueue,
    refreshOverview,
    refreshWorkspace,
    reviewDecisions: REVIEW_DECISIONS,
    selectFile,
    syncReviewRoot,
  },
  ui: {
    closeOverlay,
    currentLibraryRoot,
    openOverlay,
    saveUiState,
    selectedComparisonModels,
    setTab,
  },
});

function selectAll() {
  state.bulkSelection = null;
  state.queue.forEach((item) => state.selectedIds.add(item.id));
  updateSelectionState();
}

function selectNone() {
  state.bulkSelection = null;
  state.selectedIds.clear();
  state.lastSelectionAnchorIndex = -1;
  updateSelectionState();
}

function invalidateLoadedReviewSelection({ clearActiveSelection = false } = {}) {
  state.loadedReviewSelection = null;
  if (clearActiveSelection) {
    state.bulkSelection = null;
    state.selectedIds.clear();
    state.lastSelectionAnchorIndex = -1;
    updateSelectionState();
  }
}

function selectionKey(selection) {
  return JSON.stringify(selection);
}

function reviewSelectionSnapshotFromQuery(params, totalFiles) {
  const selection = {
    scope: "review-browser",
    marked: params.get("marked") || "all",
    issues: params.get("issues") || "all",
    root: params.get("root") || null,
    query: params.get("query") || null,
    min_score: params.get("min_score") ? Number(params.get("min_score")) : null,
    max_score: params.get("max_score") ? Number(params.get("max_score")) : null,
  };

  return {
    selection,
    count: Number(totalFiles || 0),
    queryKey: selectionKey(selection),
  };
}

async function selectAllMatching() {
  if (state.totalFiles <= 0) {
    showToast("No files match the current filters.", "error");
    return;
  }
  if (state.totalFiles > 5000) {
    if (!confirm(`Select all ${state.totalFiles.toLocaleString()} matching files? This may take a moment.`)) {
      return;
    }
  }
  if (!state.loadedReviewSelection) {
    showToast("Review results are still loading. Try again in a moment.", "error");
    return;
  }
  state.selectedIds.clear();
  state.lastSelectionAnchorIndex = -1;
  state.bulkSelection = {
    selection: state.loadedReviewSelection.selection,
    count: state.loadedReviewSelection.count,
    queryKey: state.loadedReviewSelection.queryKey,
    excludedIds: new Set(),
    selectionRevision: state.loadedReviewSelection.selectionRevision,
  };
  updateSelectionState();
  showToast(`Selected ${state.totalFiles.toLocaleString()} files across all pages.`);
}

const eventsModule = window.ShotSieveEvents;
if (!eventsModule?.createEvents) {
  throw new Error("ShotSieve events module failed to load.");
}

const installEvents = eventsModule.createEvents({
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
});

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

function addLogEntry(title, detail) {
  void title;
  void detail;
}

function showToast(message, tone = "success") {
  const region = document.getElementById("toast-region");
  const node = document.createElement("div");
  node.className = `toast ${tone}`;
  node.setAttribute("role", tone === "error" ? "alert" : "status");
  node.setAttribute("aria-live", tone === "error" ? "assertive" : "polite");

  const text = document.createElement("span");
  text.className = "toast-message";
  text.textContent = message;

  const close = document.createElement("button");
  close.type = "button";
  close.className = "toast-close";
  close.setAttribute("aria-label", "Dismiss notification");
  close.textContent = "×";

  const removeToast = () => node.remove();
  close.addEventListener("click", removeToast);

  node.appendChild(text);
  node.appendChild(close);
  region.appendChild(node);
  if (tone !== "error") {
    window.setTimeout(removeToast, 3800);
  }
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
    return;
  }
  const rootFilter = document.getElementById("root-filter");
  if ([...rootFilter.options].some((option) => option.value === root)) {
    rootFilter.value = root;
  }
}

function currentQuery() {
  const params = new URLSearchParams();
  const query = document.getElementById("query-filter").value.trim();
  const root = document.getElementById("root-filter").value;
  const sort = document.getElementById("sort-filter").value;
  const marked = document.getElementById("marked-filter").value;
  const issues = document.getElementById("issues-filter").value;
  const minScore = document.getElementById("min-score").value;
  const maxScore = document.getElementById("max-score").value;

  if (query) params.set("query", query);
  if (root) params.set("root", root);
  if (sort) params.set("sort", sort);
  if (marked) params.set("marked", marked);
  if (issues && issues !== "all") params.set("issues", issues);
  if (minScore) params.set("min_score", minScore);
  if (maxScore) params.set("max_score", maxScore);
  params.set("limit", String(state.pageSize));
  params.set("offset", String(state.page * state.pageSize));
  return params;
}

function hasActiveReviewFilters() {
  const query = document.getElementById("query-filter")?.value?.trim() || "";
  const root = document.getElementById("root-filter")?.value || "";
  const marked = document.getElementById("marked-filter")?.value || "all";
  const issues = document.getElementById("issues-filter")?.value || "all";
  const minScore = document.getElementById("min-score")?.value?.trim() || "";
  const maxScore = document.getElementById("max-score")?.value?.trim() || "";
  return Boolean(query || root || minScore || maxScore || marked !== "all" || issues !== "all");
}

function renderSummary() {
  const summary = state.overview?.summary || { total_files: 0, scored_files: 0, delete_marked: 0, export_marked: 0 };
  document.getElementById("summary-strip").innerHTML = [
    [`<span class="stat-value">${summary.scored_files}</span> scored`],
    [`<span class="stat-value">${summary.delete_marked}</span> rejected`],
    [`<span class="stat-value">${summary.export_marked}</span> selected`],
    [`<span class="stat-value">${summary.total_files}</span> total`],
  ].map(([text]) => `<span class="stat-item">${text}</span>`).join("");
}

function populateRootFilters() {
  const roots = state.overview?.roots || [];
  const rootFilter = document.getElementById("root-filter");
  const previous = rootFilter.value;
  rootFilter.innerHTML = [`<option value="">All scanned roots</option>`]
    .concat(roots.map((root) => `<option value="${escapeHtml(root)}">${escapeHtml(root)}</option>`))
    .join("");
  if (previous && roots.includes(previous)) {
    rootFilter.value = previous;
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
  const scoringModes = availableLearnedModelsUtil(options, DEFAULT_MODEL_CATALOG, HIDDEN_MODEL_NAMES);
  modelSelect.innerHTML = scoringModes.map((model) => `<option value="${escapeHtml(model)}">${escapeHtml(MODEL_DISPLAY_NAMES[model] || model)}</option>`)
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
  const previewModeSelect = document.getElementById("preview-mode-select");
  if (previewModeSelect) {
    const availablePreviewModes = Array.isArray(options.preview_modes) && options.preview_modes.length
      ? options.preview_modes
      : ["fast", "auto", "high-quality"];
    const preferredPreviewMode = persisted.previewMode || options.default_preview_mode || "auto";
    previewModeSelect.value = availablePreviewModes.includes(preferredPreviewMode)
      ? preferredPreviewMode
      : (options.default_preview_mode || "auto");
  }
  document.getElementById("recursive-toggle").checked = persisted.recursive ?? true;
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

  modelSelect.onchange = () => {
    const val = modelSelect.value;
    document.getElementById("model-detail-hint").textContent = MODEL_DESCRIPTIONS[val] || "No detailed notes available for this model.";
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

  renderComparisonModelOptions(options, scoringModes, persisted);
  renderComparisonSummary();

  // Human-readable system info
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
  const xpuPackagingNote = "Intel XPU remains source-install only today and is not one of the packaged runtime downloads.";

  // Hardware cards (detected hardware)
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

  // Runtime cards (software / runtime info)
  const runtimeCards = [
    ["Active Runtime", activeRuntimeLabel],
    ["Available Accelerators", summarizeAccelerators(statusMap)],
    ["Auto Mode Priority", summarizeAutoPriority(options)],
    ["Intel XPU", xpuPackagingNote],
    ["Photo Support", formatPhotoSupport(heifOk, rawOk)],
  ];
  document.getElementById("runtime-cards").innerHTML = runtimeCards.map(([label, value]) => `
    <article class="runtime-card">
      <p class="eyebrow">${escapeHtml(label)}</p>
      <strong>${escapeHtml(value)}</strong>
    </article>
  `).join("");

  // Resource profile dropdown - restore from localStorage
  const profileSelect = document.getElementById("resource-profile-select");
  if (profileSelect) {
    const savedProfile = localStorage.getItem("shotsieve_resource_profile") || "normal";
    profileSelect.value = savedProfile;
    updateResourceProfileDetail(hw);
  }

  // Grouped format list
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


function renderQueue() {
  renderQueueView({
    state,
    renderDetail,
    formatNumber,
    getSortRelevantScore,
    getScoreColor,
    escapeHtml,
    pathLeaf,
    pathDirectory,
    selectFile,
    handleError,
  });
}

function updateSelectionState(options) {
  updateSelectionStateView({ state }, options);
}

function renderDetail() {
  renderDetailView({
    state,
    modelDisplayNames: MODEL_DISPLAY_NAMES,
    pathLeaf,
    escapeHtml,
    formatNumber,
    scoreCard,
    statusPill,
    openOriginalFile,
    handleError,
  });
}

async function refreshOverview() {
  state.overview = await fetchJson("/api/overview");
  renderSummary();
  populateRootFilters();
}

async function loadOptions() {
  const profile = currentResourceProfile();
  state.options = await fetchJson(`/api/options?resource_profile=${encodeURIComponent(profile)}`);
  renderOptions();
}

async function loadQueue() {
  if (queueAbortController) {
    queueAbortController.abort();
  }
  queueAbortController = new AbortController();
  const signal = queueAbortController.signal;

  try {
    const query = currentQuery();
    const data = await fetchJson(`/api/files?${query.toString()}`, { signal });

    if (signal.aborted) return;
    
    state.queue = data.items;
    state.totalFiles = Number(data.total || 0);
    const nextLoadedReviewSelection = {
      ...reviewSelectionSnapshotFromQuery(query, data.total || 0),
      selectionRevision: data.selection_revision || null,
    };
    state.loadedReviewSelection = nextLoadedReviewSelection;
    if (
      state.bulkSelection
      && (
        state.bulkSelection.queryKey !== nextLoadedReviewSelection.queryKey
        || state.bulkSelection.selectionRevision !== nextLoadedReviewSelection.selectionRevision
      )
    ) {
      state.bulkSelection = null;
    }
    state.lastSelectionAnchorIndex = -1;
    state.selectedIds = new Set([...state.selectedIds].filter((fileId) => state.queue.some((item) => item.id === fileId)));
    renderQueue();
    renderPagination();

     if (!state.queue.length) {
      return;
    }

    if (state.activeId && state.queue.some((item) => item.id === state.activeId)) {
      state.detail = await fetchJson(`/api/file?id=${state.activeId}`, { signal });
      if (signal.aborted) return;
      renderDetail();
      return;
    }

    await selectFile(state.queue[0].id);
  } catch (err) {
    if (err.name === "AbortError") return;
    throw err;
  } finally {
    if (queueAbortController?.signal === signal) {
      queueAbortController = null;
    }
  }
}

function renderPagination() {
  const hasResults = state.totalFiles > 0;
  const start = hasResults ? state.page * state.pageSize + 1 : 0;
  const end = hasResults ? Math.min(start + state.queue.length - 1, state.totalFiles) : 0;
  const summary = state.overview?.summary || {};
  const totalScoredInLibrary = Number(summary.scored_files || 0);
  const filtered = hasActiveReviewFilters();
  let label;
  if (hasResults) {
    if (filtered) {
      label = `Showing ${start}–${end} of ${state.totalFiles.toLocaleString()} matching photos`;
      if (totalScoredInLibrary > 0) {
        label += ` (${totalScoredInLibrary.toLocaleString()} scored in library)`;
      }
    } else if (totalScoredInLibrary > 0) {
      label = `Showing ${start}–${end} of ${totalScoredInLibrary.toLocaleString()} scored photos`;
    } else {
      label = `Showing ${start}–${end} of ${state.totalFiles.toLocaleString()} photos`;
    }
  } else {
    label = filtered
      ? `No photos match current filters${totalScoredInLibrary > 0 ? ` (${totalScoredInLibrary.toLocaleString()} scored in library)` : ""}`
      : "No scored photos yet";
  }
  document.getElementById("page-info").textContent = label;
  document.getElementById("page-prev").disabled = state.page === 0;
  document.getElementById("page-next").disabled = ((state.page + 1) * state.pageSize) >= state.totalFiles;

  // Update 'Select All (N)' button count
  const selectAllCount = document.getElementById("select-all-matching-count");
  if (selectAllCount) {
    selectAllCount.textContent = state.totalFiles > 0 ? `(${state.totalFiles.toLocaleString()})` : "";
  }

  // Show/hide rejected actions bar
  const rejectedCount = summary.delete_marked || 0;
  const rejectedBar = document.getElementById("rejected-actions");
  if (rejectedCount > 0) {
    rejectedBar.classList.remove("hidden");
    document.getElementById("rejected-label").textContent = `${rejectedCount} photo${rejectedCount !== 1 ? "s" : ""} rejected`;
  } else {
    rejectedBar.classList.add("hidden");
  }
}

async function selectFile(fileId) {
  state.activeId = fileId;
  
  const detailPromise = fetchJson(`/api/file?id=${fileId}`);
  
  // Update sidebar immediately without full re-render if possible
  if (state.queue.length > 0) {
    updateSelectionState({ scrollActive: true });
  } else {
    renderQueue();
  }
  
  state.detail = await detailPromise;
  renderDetail();
}

async function refreshWorkspace() {
  let optionsError = null;

  const [, optionsResult] = await Promise.allSettled([
    refreshOverview(),
    loadOptions(),
  ]);

  if (optionsResult.status === "rejected") {
    optionsError = optionsResult.reason;
  }

  renderSummary();
  renderComparisonSummary();
  await loadQueue();

  if (optionsError) {
    throw optionsError;
  }
}

async function boot() {
  installEvents();
  renderBusyState();

  await refreshWorkspace();
  renderComparisonResults();
}

boot().catch(handleError);