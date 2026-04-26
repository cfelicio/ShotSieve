(() => {
  const DEFAULT_MODEL_CATALOG = ["topiq_nr", "clipiqa", "qalign"];
  const HIDDEN_MODEL_NAMES = ["arniqa", "arniqa-spaq"];
  const UI_STATE_KEY = "shotsieve-ui-state-v4";

  const REVIEW_DECISIONS = {
    keep: { delete_marked: false, export_marked: true, decision_state: "export" },
    reject: { delete_marked: true, export_marked: false, decision_state: "delete" },
    reset: { delete_marked: false, export_marked: false, decision_state: "pending" },
  };

  const MODEL_DESCRIPTIONS = {
    topiq_nr: "Best all-rounder and recommended starting point - fast, stable, and strong quality ranking for most libraries.",
    "topiq_nr-flive": "TOPIQ trained for in-the-wild photo quality; useful for mixed consumer libraries.",
    "topiq_nr-spaq": "TOPIQ variant tuned on smartphone photo aesthetics and perceptual quality.",
    tres: "Transformer-based perceptual ranking; worth testing on varied or stylized image sets.",
    clipiqa: "Fast CLIP-based model that can catch outliers TOPIQ misses.",
    qualiclip: "Newer CLIP-based quality model; useful when human taste matters more than strict distortions.",
    qalign: "Outlier-sensitive vision-language scorer that can be helpful when you want a second opinion on unusual or polarizing images.",
  };

  const MODEL_DISPLAY_NAMES = {
    topiq_nr: "TOPIQ (Recommended)",
    "topiq_nr-flive": "TOPIQ FLIVE",
    "topiq_nr-spaq": "TOPIQ SPAQ",
    tres: "TReS",
    clipiqa: "CLIPIQA",
    qualiclip: "QualiCLIP",
    qalign: "Q-Align",
  };

  function createState() {
    return {
      tab: "workspace",
      options: null,
      overview: null,
      comparison: null,
      queue: [],
      detail: null,
      loadedReviewSelection: null,
      selectedIds: new Set(),
      bulkSelection: null,
      activeId: null,
      lastSelectedIndex: -1,
      lastSelectionAnchorIndex: -1,
      page: 0,
      pageSize: 60,
      totalFiles: 0,
      browserPath: null,
      browserTarget: null,
      isBusy: false,
      activeOperation: null,
      busyMessage: "",
      busyPercent: null,
      busyPhasePercent: null,
      busyPhaseIndex: 0,
      busyPhaseCount: 0,
      busyPhaseLabel: "",
      busyStartTime: null,
      abortController: null,
      cancelPending: false,
      compareJobId: null,
      scoreJobId: null,
      scanJobId: null,
      pendingExport: null,
      compareRowSort: "topiq_nr:desc",
      compareRowSortInitialized: false,
      compareRowFilter: "all",
    };
  }

  function createUiStateStore({ storage = window.localStorage, documentRef = document } = {}) {
    let saveUiTimer = null;

    function currentDatabaseMarker() {
      const marker = documentRef.body?.dataset?.databasePath;
      return typeof marker === "string" ? marker : "";
    }

    function loadUiState() {
      try {
        const raw = storage.getItem(UI_STATE_KEY);
        const parsed = raw ? JSON.parse(raw) : {};
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          return {};
        }

        const expectedDatabase = currentDatabaseMarker();
        if (expectedDatabase) {
          const savedDatabase = typeof parsed.database === "string" ? parsed.database : "";
          if (!savedDatabase || savedDatabase !== expectedDatabase) {
            return {};
          }
        }

        return parsed;
      } catch {
        return {};
      }
    }

    function buildUiStatePayload(overrides = {}) {
      return {
        database: currentDatabaseMarker(),
        libraryRoot: documentRef.getElementById("library-root-input")?.value || "",
        extensions: documentRef.getElementById("extensions-input")?.value || "",
        previewMode: documentRef.getElementById("preview-mode-select")?.value || "auto",
        recursive: documentRef.getElementById("recursive-toggle")?.checked ?? true,
        model: documentRef.getElementById("model-select")?.value || "",
        device: documentRef.getElementById("device-select")?.value || "auto",
        compareModels: [...documentRef.querySelectorAll("#compare-model-grid input[type='checkbox']:checked")].map((input) => input.value),
        minScore: documentRef.getElementById("min-score")?.value || "",
        maxScore: documentRef.getElementById("max-score")?.value || "",
        issues: documentRef.getElementById("issues-filter")?.value || "all",
        ...overrides,
      };
    }

    function persistUiState() {
      const payload = buildUiStatePayload();
      storage.setItem(UI_STATE_KEY, JSON.stringify(payload));
    }

    function saveUiState(options = {}) {
      if (options.immediate) {
        window.clearTimeout(saveUiTimer);
        saveUiTimer = null;
        persistUiState();
        return;
      }

      window.clearTimeout(saveUiTimer);
      saveUiTimer = window.setTimeout(() => {
        persistUiState();
      }, 200);
    }

    function clearUiState(options = {}) {
      window.clearTimeout(saveUiTimer);
      saveUiTimer = null;
      storage.removeItem(UI_STATE_KEY);
      if (options.immediate) {
        persistUiState();
      }
    }

    function isAutoAdvanceEnabled() {
      return true;
    }

    function currentLibraryRoot() {
      return documentRef.getElementById("library-root-input")?.value.trim() || "";
    }

    function selectedComparisonModels() {
      return [...documentRef.querySelectorAll("#compare-model-grid input[type='checkbox']:checked")].map((input) => input.value);
    }

    return {
      buildUiStatePayload,
      clearUiState,
      currentLibraryRoot,
      isAutoAdvanceEnabled,
      loadUiState,
      saveUiState,
      selectedComparisonModels,
    };
  }

  window.ShotSieveState = {
    DEFAULT_MODEL_CATALOG,
    HIDDEN_MODEL_NAMES,
    MODEL_DESCRIPTIONS,
    MODEL_DISPLAY_NAMES,
    REVIEW_DECISIONS,
    UI_STATE_KEY,
    createState,
    createUiStateStore,
  };
})();
