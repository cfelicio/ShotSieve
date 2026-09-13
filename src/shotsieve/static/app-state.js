(() => {
  const UI_STATE_KEY = "shotsieve-ui-state-v4";

  const REVIEW_DECISIONS = {
    keep: { delete_marked: false, export_marked: true, decision_state: "export" },
    reject: { delete_marked: true, export_marked: false, decision_state: "delete" },
    reset: { delete_marked: false, export_marked: false, decision_state: "pending" },
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
      reviewScopeInitialized: false,
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
      controlAbortController: null,
      cancelPending: false,
      compareJobId: null,
      scoreJobId: null,
      scanJobId: null,
      modelPreparationJobId: null,
      operationJobId: null,
      operationStatusPath: null,
      operationCancelPath: null,
      operationStatusUnknown: false,
      activeJob: null,
      recoveryJob: null,
      latestOperationResult: null,
      latestOperationRequest: null,
      operationResultHandler: null,
      operationProgressSignature: null,
      operationProgressChangedAt: null,
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
      const formatFilters = [...documentRef.querySelectorAll("input[name='format-filter']:checked")].map((i) => i.value);
      return {
        database: currentDatabaseMarker(),
        libraryRoot: documentRef.getElementById("library-root-input")?.value || "",
        extensions: documentRef.getElementById("extensions-input")?.value || "",
        recursive: documentRef.getElementById("recursive-toggle")?.checked ?? true,
        model: documentRef.getElementById("model-select")?.value || "",
        device: documentRef.getElementById("device-select")?.value || "auto",
        compareModels: [...documentRef.querySelectorAll("#compare-model-grid input[type='checkbox']:checked")].map((input) => input.value),
        minScore: documentRef.getElementById("min-score")?.value || "",
        maxScore: documentRef.getElementById("max-score")?.value || "",
        issues: documentRef.getElementById("issues-filter")?.value || "all",
        ignoreRules: documentRef.getElementById("ignore-rules-input")?.value || "",
        formats: formatFilters,
        minMp: documentRef.getElementById("filter-min-mp")?.value || "",
        maxMp: documentRef.getElementById("filter-max-mp")?.value || "",
        minSize: documentRef.getElementById("filter-min-size")?.value || "",
        maxSize: documentRef.getElementById("filter-max-size")?.value || "",
        metadataStatus: documentRef.getElementById("filter-metadata-status")?.value || "all",
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
    REVIEW_DECISIONS,
    UI_STATE_KEY,
    createState,
    createUiStateStore,
  };
})();
