(() => {
  const pollingModule = window.ShotSieveWorkflowPolling;
  if (!pollingModule?.createJobPollers) {
    throw new Error("ShotSieve workflow polling module failed to load.");
  }

  function createWorkflows(deps) {
    const {
      api,
      busy,
      compare,
      formatting,
      notifications,
      review,
      state,
      ui,
    } = deps;

    const { fetchJson, postJson } = api;
    const {
      setBusyMessage,
      setBusyPhaseProgress,
      setBusyProgress,
      withBusy,
    } = busy;
    const {
      compareBatchSize,
      compareProgressMessage,
      compareProgressPercent,
      comparisonDefaults,
      currentResourceProfile,
      modelDescriptions,
      modelDisplayNames,
      scanProgressMessage,
      scanProgressPercent,
      scoreBatchSize,
      scoreProgressMessage,
      scoreProgressPercent,
    } = compare;
    const {
      escapeHtml,
      formatDuration,
      formatFilesPerSecond,
      formatNumber,
      getScoreColor,
      mergeTimingTotals,
      pathLeaf,
      sortComparisonRows,
    } = formatting;
    const { addLogEntry, showToast } = notifications;
    const {
      isAutoAdvanceEnabled,
      loadQueue,
      refreshOverview,
      refreshWorkspace,
      reviewDecisions,
      selectFile,
      syncReviewRoot,
    } = review;
    const {
      closeOverlay,
      currentLibraryRoot,
      saveUiState,
      selectedComparisonModels,
      setTab,
    } = ui;

    const {
      fetchCompareJobResult,
      fetchCompareJobStatus,
      fetchScanJobResult,
      fetchScanJobStatus,
      fetchScoreJobResult,
      fetchScoreJobStatus,
      pipelineOverallPercent,
      pollCompareJob,
      pollScanJob,
      pollScoreJob,
    } = pollingModule.createJobPollers({
      state,
      api: { fetchJson },
      busy: {
        setBusyMessage,
        setBusyPhaseProgress,
        setBusyProgress,
      },
      progress: {
        compareProgressMessage,
        compareProgressPercent,
        scanProgressMessage,
        scanProgressPercent,
        scoreProgressMessage,
        scoreProgressPercent,
      },
    });

    function compareRowSortChoices(modelNames) {
      const choices = [
        { value: "input", label: "Original order" },
        { value: "path_asc", label: "Path (A–Z)" },
      ];

      for (const modelName of modelNames) {
        const label = modelDisplayNames[modelName] || modelName;
        choices.push({ value: `${modelName}:desc`, label: `${label} score (high → low)` });
        choices.push({ value: `${modelName}:asc`, label: `${label} score (low → high)` });
      }

      return choices;
    }

    function defaultCompareRowSort(modelNames) {
      if (modelNames.includes("topiq_nr")) {
        return "topiq_nr:desc";
      }

      const scoreChoice = compareRowSortChoices(modelNames).find(
        (choice) => choice.value.endsWith(":desc"),
      );
      return scoreChoice?.value || "input";
    }

    function compareFilterModelName(modelNames) {
      const [sortModelName, direction] = String(state.compareRowSort || "").split(":");
      if (direction && modelNames.includes(sortModelName)) {
        return sortModelName;
      }
      if (modelNames.includes("topiq_nr")) {
        return "topiq_nr";
      }
      return modelNames[0] || null;
    }

    function filterComparisonRows(rows, modelNames) {
      const filterMode = String(state.compareRowFilter || "all");
      if (filterMode === "all") {
        return rows;
      }

      const modelName = compareFilterModelName(modelNames);
      if (!modelName) {
        return rows;
      }

      const scoreKey = `${modelName}_score`;
      const scoredRows = rows.filter((row) => ShotSieveUtils.comparisonScoreNumber(row?.[scoreKey]) !== null);
      if (!scoredRows.length) {
        return [];
      }

      const values = scoredRows.map((row) => ShotSieveUtils.comparisonScoreNumber(row[scoreKey]));
      const minValue = Math.min(...values);
      const maxValue = Math.max(...values);

      if (filterMode === "min") {
        return scoredRows.filter((row) => ShotSieveUtils.comparisonScoreNumber(row[scoreKey]) === minValue);
      }

      if (filterMode === "max") {
        return scoredRows.filter((row) => ShotSieveUtils.comparisonScoreNumber(row[scoreKey]) === maxValue);
      }

      if (filterMode === "extremes") {
        return rows.filter((row) => {
          const value = ShotSieveUtils.comparisonScoreNumber(row?.[scoreKey]);
          return value === minValue || value === maxValue;
        });
      }

      return rows;
    }

    function syncCompareSortControls(modelNames) {
      const rowSort = document.getElementById("compare-row-sort");
      if (!rowSort) {
        return;
      }

      const rowChoices = compareRowSortChoices(modelNames);
      rowSort.innerHTML = rowChoices
        .map((choice) => `<option value="${escapeHtml(choice.value)}">${escapeHtml(choice.label)}</option>`)
        .join("");

      const hasCurrentChoice = rowChoices.some((choice) => choice.value === state.compareRowSort);
      if (!state.compareRowSortInitialized || !hasCurrentChoice) {
        state.compareRowSort = defaultCompareRowSort(modelNames);
        state.compareRowSortInitialized = true;
      }
      rowSort.value = state.compareRowSort;

      const rowFilter = document.getElementById("compare-row-filter");
      if (rowFilter) {
        rowFilter.value = state.compareRowFilter || "all";
      }
    }

    function comparisonFailureText(row, modelName) {
      const rawError = row?.[`${modelName}_error`];
      return typeof rawError === "string" ? rawError.trim() : "";
    }

    function comparisonFailureDetails(row, modelNames) {
      const failures = [];

      for (const modelName of modelNames) {
        const errorText = comparisonFailureText(row, modelName);
        if (!errorText) {
          continue;
        }
        failures.push({
          modelName,
          label: modelDisplayNames[modelName] || modelName,
          errorText,
        });
      }

      return failures;
    }

    function comparisonSetupFailureDetails(comparison) {
      const compareFailures = Array.isArray(comparison?.compare_failures)
        ? comparison.compare_failures
        : [];
      const failures = [];

      for (const failure of compareFailures) {
        if (!failure || typeof failure !== "object") {
          continue;
        }
        const path = typeof failure.path === "string" ? failure.path.trim() : "";
        const reason = typeof failure.reason === "string" ? failure.reason.trim() : "";
        if (!path || !reason) {
          continue;
        }
        failures.push({
          path,
          filename: pathLeaf(path) || path,
          reason,
          stage: typeof failure.stage === "string" ? failure.stage : "preparing_comparison",
        });
      }

      return failures;
    }

    function comparisonFailureSummaryText(comparison) {
      const modelNames = Array.isArray(comparison?.model_names)
        ? comparison.model_names
        : [];
      const rows = Array.isArray(comparison?.rows)
        ? comparison.rows
        : [];
      const entries = [];

      for (const failure of comparisonSetupFailureDetails(comparison)) {
        entries.push(`${failure.filename} — ${failure.reason}`);
      }

      for (const row of rows) {
        const filename = pathLeaf(row?.path || "") || String(row?.path || "Unknown file");
        const failures = comparisonFailureDetails(row, modelNames);
        for (const failure of failures) {
          entries.push(`${filename} — ${failure.label}: ${failure.errorText}`);
        }
      }

      if (!entries.length) {
        return null;
      }

      const preview = entries.slice(0, 5).join("; ");
      if (entries.length <= 5) {
        return preview;
      }
      return `${preview}; +${entries.length - 5} more`;
    }

    function comparisonTruncationWarningText(comparison) {
      if (!comparison || !comparison.truncated) {
        return null;
      }

      const processedRows = Number(comparison.processed_rows_total || comparison.files_considered || 0);
      const requestedRows = Number(comparison.requested_rows_total || processedRows || 0);
      if (!Number.isFinite(processedRows) || !Number.isFinite(requestedRows) || processedRows <= 0 || requestedRows <= 0 || processedRows >= requestedRows) {
        return null;
      }

      const processedRowsText = Math.max(0, Math.trunc(processedRows)).toLocaleString();
      const requestedRowsText = Math.max(0, Math.trunc(requestedRows)).toLocaleString();
      return `Comparing first ${processedRowsText} of ${requestedRowsText} files. Narrow the root or apply filters for a full compare.`;
    }

    function renderComparisonWarnings(comparison) {
      const warning = document.getElementById("compare-results-warning");
      if (!warning) {
        return;
      }

      const notices = [];
      const truncationText = comparisonTruncationWarningText(comparison);
      if (truncationText) {
        notices.push(truncationText);
      }

      const summaryText = comparisonFailureSummaryText(comparison);
      if (summaryText) {
        notices.push(`Some model runs failed: ${summaryText}`);
      }

      if (!notices.length) {
        warning.textContent = "";
        warning.classList.add("hidden");
        return;
      }

      warning.textContent = notices.join(" ");
      warning.classList.remove("hidden");
    }

    function renderComparisonSummary() {
      const summaryCards = document.getElementById("compare-summary-cards");
      if (!summaryCards) {
        return;
      }

      const comparison = state.comparison;
      if (!comparison) {
        summaryCards.innerHTML = "";
        return;
      }

      const cards = [
        ["Files Considered", `${comparison.files_considered || 0}`],
        ["Files Compared", `${comparison.files_compared}`],
        ["Models", `${comparison.model_names.length}`],
      ];
      if (comparison.files_failed) {
        cards.push(["Files Failed", `${comparison.files_failed}`]);
      }
      if (comparison.elapsed_seconds !== null && comparison.elapsed_seconds !== undefined) {
        cards.push(["Total Runtime", formatDuration(comparison.elapsed_seconds)]);
        cards.push(["Overall Speed", formatFilesPerSecond(comparison.files_compared, comparison.elapsed_seconds)]);
      }
      const orderedModelNames = Array.isArray(comparison.model_names)
        ? [...comparison.model_names]
        : [];

      for (const modelName of orderedModelNames) {
        const values = comparison.rows
          .map((row) => ShotSieveUtils.comparisonScoreNumber(row[`${modelName}_score`]))
          .filter((value) => value !== null);
        const average = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
        const failureCount = comparison.rows.reduce(
          (count, row) => count + (comparisonFailureText(row, modelName) ? 1 : 0),
          0,
        );
        const runtime = comparison.model_timings_seconds?.[modelName];
        const parts = [];
        if (average !== null) {
          parts.push(`${average.toFixed(1)} avg`);
        }
        if (failureCount) {
          parts.push(average === null && failureCount === comparison.rows.length ? "all failed" : `${failureCount} failed`);
        }
        if (runtime !== null && runtime !== undefined) {
          parts.push(`${formatDuration(runtime)} total`);
          parts.push(formatFilesPerSecond(comparison.files_compared, runtime));
        }
        cards.push([modelDisplayNames[modelName] || modelName, parts.length ? parts.join(" · ") : "n/a"]);
      }

      summaryCards.innerHTML = cards.map(([label, value]) => `
        <article class="runtime-card compare-summary-card">
          <p class="eyebrow">${escapeHtml(label)}</p>
          <strong>${escapeHtml(value)}</strong>
        </article>
      `).join("");
    }

    function renderComparisonResults() {
      const empty = document.getElementById("compare-empty");
      const results = document.getElementById("compare-results");
      const gallery = document.getElementById("compare-card-gallery");
      const galleryEmpty = document.getElementById("compare-gallery-empty");
      const comparison = state.comparison;
      const defaultEmptyMessage = "Choose a library root, select at least two models, and run a comparison to see ranked results here.";

      if (!comparison) {
        empty.textContent = defaultEmptyMessage;
        empty.classList.remove("hidden");
        results.classList.add("hidden");
        if (gallery) {
          gallery.innerHTML = "";
        }
        if (galleryEmpty) {
          galleryEmpty.classList.add("hidden");
        }
        renderComparisonWarnings(null);
        renderComparisonSummary();
        return;
      }

      if (!comparison.rows?.length) {
        const failureNote = comparison.files_failed
          ? ` ${comparison.files_failed} file(s) failed during comparison setup.`
          : "";
        empty.textContent = comparison.files_considered
          ? `No comparable cached files were available under ${currentLibraryRoot() || "the selected library root"}.${comparison.files_skipped ? ` ${comparison.files_skipped} file(s) were skipped.` : ""}${failureNote}`
          : "No cached photos were found under the selected library root. Run Analyze first.";
        empty.classList.remove("hidden");
        results.classList.add("hidden");
        if (gallery) {
          gallery.innerHTML = "";
        }
        if (galleryEmpty) {
          galleryEmpty.classList.add("hidden");
        }
        renderComparisonWarnings(comparison);
        renderComparisonSummary();
        return;
      }

      empty.classList.add("hidden");
      results.classList.remove("hidden");
      document.getElementById("compare-results-title").textContent = `${comparison.files_compared} file${comparison.files_compared !== 1 ? "s" : ""} compared across ${comparison.model_names.length} model${comparison.model_names.length !== 1 ? "s" : ""}`;
      const subtitleParts = [currentLibraryRoot() || "Current library root"];
      if (comparison.elapsed_seconds) {
        subtitleParts.push(`Total runtime ${formatDuration(comparison.elapsed_seconds)}`);
        subtitleParts.push(formatFilesPerSecond(comparison.files_compared, comparison.elapsed_seconds));
      }
      if (comparison.files_skipped) {
        subtitleParts.push(`${comparison.files_skipped} skipped`);
      }
      document.getElementById("compare-results-subtitle").textContent = subtitleParts.join(" · ");
      renderComparisonWarnings(comparison);
      syncCompareSortControls(comparison.model_names);

      const orderedModelNames = Array.isArray(comparison.model_names)
        ? [...comparison.model_names]
        : [];

      const sortedRows = sortComparisonRows(comparison.rows, state.compareRowSort);
      const visibleRows = filterComparisonRows(sortedRows, orderedModelNames);

      if (!gallery || !galleryEmpty) {
        renderComparisonSummary();
        return;
      }

      if (!visibleRows.length) {
        gallery.innerHTML = "";
        galleryEmpty.classList.remove("hidden");
        renderComparisonSummary();
        return;
      }

      galleryEmpty.classList.add("hidden");
      gallery.innerHTML = visibleRows.map((row, index) => {
        const filename = pathLeaf(row.path || "");
        const fileId = Number(row.file_id);
        const hasFileId = Number.isInteger(fileId) && fileId > 0;
        const photoMarkup = hasFileId
          ? `<img class="compare-photo" src="/api/media/preview?id=${fileId}" alt="${escapeHtml(filename)}">`
          : `<div class="compare-photo-fallback">No preview</div>`;

        const scoreMarkup = orderedModelNames.map((modelName) => {
          const score = row[`${modelName}_score`];
          const confidence = row[`${modelName}_confidence`];
          const errorText = comparisonFailureText(row, modelName);
          return `
            <li class="compare-model-score ${getScoreColor(score)}">
              <div>
                <strong>${escapeHtml(modelDisplayNames[modelName] || modelName)}</strong>
              </div>
              <div class="compare-model-score-values">
                <span class="compare-score-value">${formatNumber(score)}</span>
                ${confidence === null || confidence === undefined ? "" : `<span class="compare-confidence-value">confidence ${formatNumber(confidence)}</span>`}
                ${errorText ? `<span class="field-hint compare-model-error">Failed: ${escapeHtml(errorText)}</span>` : ""}
              </div>
            </li>
          `;
        }).join("");

        return `
          <article class="compare-result-card">
            <div class="compare-result-media">
              ${photoMarkup}
            </div>
            <div class="compare-result-main">
              <div class="compare-file-cell">
                <strong>#${index + 1} · ${escapeHtml(filename)}</strong>
                <span>${escapeHtml(row.path || "")}</span>
              </div>
              <ul class="compare-model-score-list">
                ${scoreMarkup}
              </ul>
            </div>
          </article>
        `;
      }).join("");

      renderComparisonSummary();
    }

    function renderComparisonModelOptions(options, allowedModels, persisted) {
      const target = document.getElementById("compare-model-grid");
      if (!target) {
        return;
      }

      const selected = new Set(comparisonDefaults(options, persisted, allowedModels));
      target.innerHTML = allowedModels.map((modelName) => `
        <label class="compare-model-card ${selected.has(modelName) ? "selected" : ""}">
          <div class="compare-model-card-head">
            <input type="checkbox" value="${escapeHtml(modelName)}" ${selected.has(modelName) ? "checked" : ""}>
            <span class="compare-model-card-copy">
              <span class="compare-model-name">${escapeHtml(modelDisplayNames[modelName] || modelName)}</span>
              <span class="field-hint">${escapeHtml(modelDescriptions[modelName] || "No detailed notes available for this model.")}</span>
            </span>
          </div>
        </label>
      `).join("");

      target.querySelectorAll("input[type='checkbox']").forEach((input) => {
        input.addEventListener("change", () => {
          const checked = selectedComparisonModels();
          if (!input.checked && checked.length < 2) {
            input.checked = true;
            showToast("Keep at least two models selected for comparison.", "error");
            return;
          }
          target.querySelectorAll(".compare-model-card").forEach((card) => {
            const checkbox = card.querySelector("input[type='checkbox']");
            card.classList.toggle("selected", checkbox?.checked);
          });
          saveUiState();
        });
      });
    }

    function currentPreviewMode() {
      return document.getElementById("preview-mode-select")?.value || "auto";
    }

    async function runModelComparison() {
      const root = currentLibraryRoot();
      if (!root) {
        showToast("Pick a library root first.", "error");
        setTab("workspace");
        return;
      }

      const models = selectedComparisonModels();
      const runtimeTarget = document.getElementById("device-select").value || "auto";
      const requestedBatchSize = compareBatchSize(models, runtimeTarget, state.options?.learned?.recommended_batch_sizes);
      if (models.length < 2) {
        showToast("Select at least two models to compare.", "error");
        return;
      }

      let rowsTotal = null;

      const comparePipeline = { totalSteps: 3 };

      setBusyPhaseProgress({
        percent: 0,
        phaseIndex: 1,
        phaseCount: comparePipeline.totalSteps,
        phaseLabel: "Preparing comparison",
      });

      const result = {
        model_names: [...models],
        rows: [],
        compare_failures: [],
        requested_rows_total: 0,
        processed_rows_total: 0,
        truncated: false,
        max_rows: 0,
        files_considered: 0,
        files_compared: 0,
        files_skipped: 0,
        files_failed: 0,
        elapsed_seconds: 0,
        model_timings_seconds: {},
      };

      try {
        const estimate = await postJson("/api/compare-estimate", { root });
        rowsTotal = Number(estimate.rows_total || 0);
        if (rowsTotal > 0) {
          setBusyProgress(0);
          setBusyMessage(`Comparing... 0/${rowsTotal} (0%) across ${models.length} model${models.length !== 1 ? "s" : ""}`);
          const prepDonePercent = pipelineOverallPercent(100, { stepIndex: 1, totalSteps: comparePipeline.totalSteps });
          setBusyProgress(prepDonePercent === null ? 0 : Math.round(prepDonePercent));
          setBusyPhaseProgress({
            percent: 100,
            phaseIndex: 1,
            phaseCount: comparePipeline.totalSteps,
            phaseLabel: "Preparing comparison",
          });
        }
      } catch {
        rowsTotal = null;
      }

      if (rowsTotal === 0) {
        const prerequisiteMessage = "No cached photos found under this library root. Running Scan first.";
        addLogEntry("Comparison prerequisites", prerequisiteMessage);
        showToast(prerequisiteMessage);
        setBusyMessage(prerequisiteMessage);

        await runScan(root, {
          generatePreviews: false,
          pipeline: { stepIndex: 1, totalSteps: comparePipeline.totalSteps },
        });

        const refreshedEstimate = await postJson("/api/compare-estimate", { root });
        rowsTotal = Number(refreshedEstimate.rows_total || 0);
        if (rowsTotal <= 0) {
          state.comparison = result;
          renderComparisonResults();
          addLogEntry("Model comparison skipped", "No comparable files were found after prerequisite scan.");
          showToast("No comparable photos found under this library root after scanning.", "error");
          return;
        }

        setBusyMessage(`Prerequisites complete. Comparing... 0/${rowsTotal} (0%) across ${models.length} model${models.length !== 1 ? "s" : ""}`);
      }

      setBusyPhaseProgress({
        percent: 0,
        phaseIndex: 2,
        phaseCount: comparePipeline.totalSteps,
        phaseLabel: "Loading models",
      });
      const loadingStartPercent = pipelineOverallPercent(0, { stepIndex: 2, totalSteps: comparePipeline.totalSteps });
      setBusyProgress(loadingStartPercent === null ? 0 : Math.round(loadingStartPercent));

      const compareJobStart = await postJson("/api/compare-models/start", {
        root,
        models,
        device: runtimeTarget || null,
        batch_size: requestedBatchSize,
        preview_mode: currentPreviewMode(),
        resource_profile: currentResourceProfile(),
      }, { signal: state.abortController?.signal });
      const compareJobId = String(compareJobStart?.job_id || "");
      if (!compareJobId) {
        throw new Error("Compare job failed to start.");
      }

      state.compareJobId = compareJobId;
      let summary = null;
      try {
        summary = await pollCompareJob(compareJobId, { rowsTotal, pipeline: comparePipeline });
      } finally {
        if (!state.abortController?.signal?.aborted) {
          state.compareJobId = null;
        }
      }

      result.model_names = Array.isArray(summary.model_names) && summary.model_names.length ? summary.model_names : result.model_names;
      result.rows = Array.isArray(summary.rows) ? summary.rows : [];
      result.compare_failures = Array.isArray(summary.compare_failures) ? summary.compare_failures : [];
      result.requested_rows_total = Number(summary.requested_rows_total || 0);
      result.processed_rows_total = Number(summary.processed_rows_total || 0);
      result.truncated = Boolean(summary.truncated);
      result.max_rows = Number(summary.max_rows || 0);
      result.files_considered = Number(summary.files_considered || 0);
      result.files_compared = Number(summary.files_compared || 0);
      result.files_skipped = Number(summary.files_skipped || 0);
      result.files_failed = Number(summary.files_failed || 0);
      result.elapsed_seconds = Number(summary.elapsed_seconds || 0);
      mergeTimingTotals(result.model_timings_seconds, summary.model_timings_seconds);

      result.elapsed_seconds = Number(result.elapsed_seconds.toFixed(4));
      for (const modelName of Object.keys(result.model_timings_seconds)) {
        result.model_timings_seconds[modelName] = Number(result.model_timings_seconds[modelName].toFixed(4));
      }

      state.comparison = result;
      renderComparisonResults();
      setBusyProgress(100);
      setBusyPhaseProgress({
        percent: 100,
        phaseIndex: 3,
        phaseCount: comparePipeline.totalSteps,
        phaseLabel: "Model scoring complete",
      });
      setBusyMessage(`Comparison completed in ${formatDuration(result.elapsed_seconds)}.`);
      addLogEntry("Model comparison completed", `Compared ${result.files_compared} file(s) across ${result.model_names.length} model(s) in ${formatDuration(result.elapsed_seconds)} at ${formatFilesPerSecond(result.files_compared, result.elapsed_seconds)}.`);
      showToast("Model comparison completed.");
    }

    async function saveReview(payload) {
      if (!state.activeId) {
        showToast("Pick a file first.", "error");
        return;
      }
      await postJson("/api/review", { file_id: state.activeId, ...payload });
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

    async function fetchReviewStateSelectionRevision(marked) {
      const params = new URLSearchParams();
      params.set("marked", marked);
      params.set("limit", "1");
      params.set("offset", "0");
      const data = await fetchJson(`/api/review/file-ids?${params.toString()}`);
      return data.selection_revision || null;
    }

    function activeSelectionRequest() {
      if (state.bulkSelection) {
        const excludedIds = [...(state.bulkSelection.excludedIds || new Set())];
        const effectiveSelectionCount = Math.max(0, Number(state.bulkSelection.count || 0) - excludedIds.length);
        return {
          selection: state.bulkSelection.selection,
          selection_revision: state.bulkSelection.selectionRevision,
          exclude_file_ids: excludedIds,
          count: effectiveSelectionCount,
        };
      }
      const fileIds = [...state.selectedIds];
      return {
        file_ids: fileIds,
        count: fileIds.length,
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

          setBusyMessage(request.busyMessage(selectionRequest.count));
          const result = await postJson("/api/files/export", {
            ...selectionRequest,
            destination,
            mode: request.mode,
          });
          const summary = summarizeExportResult(result);
          showToast(`${request.successPrefix}: ${summary}.`);
          addLogEntry(request.logTitle, `${request.mode} to ${destination}: ${summary}`);
          clearActiveSelection();
          await loadQueue();
        }).catch(handleError);
      });
    }

    function installRejectedActionEvents() {
      document.getElementById("delete-all-rejected").addEventListener("click", () => {
        const rejectedCount = state.overview?.summary?.delete_marked || 0;
        if (!rejectedCount) {
          showToast("No rejected photos to delete.", "error");
          return;
        }
        const msg = `Permanently delete ${rejectedCount} rejected photo${rejectedCount !== 1 ? "s" : ""} from disk?\n\nThis cannot be undone. The original files will be removed from your computer.`;
        if (!confirm(msg)) return;
        withBusy(`Deleting ${rejectedCount} rejected files...`, async () => {
          const selectionRevision = await fetchReviewStateSelectionRevision("delete");
          if (!selectionRevision) {
            showToast("Review results are refreshing. Try again in a moment.", "error");
            return;
          }
          const selection = { scope: "review-state", marked: "delete" };
          if (!rejectedCount) {
            showToast("No rejected files found.", "error");
            return;
          }
          const result = await postJson("/api/files/delete", {
            selection,
            selection_revision: selectionRevision,
            delete_from_disk: true,
          });
          addLogEntry("Delete rejected", `Deleted ${result.deleted_count} files, ${result.failed_count} failed.`);
          showToast(`Deleted ${result.deleted_count} rejected files from disk.`);
          clearActiveSelection();
          await refreshWorkspace();
        }).catch(handleError);
      });

      document.getElementById("move-all-rejected").addEventListener("click", () => {
        const rejectedCount = state.overview?.summary?.delete_marked || 0;
        if (!rejectedCount) {
          showToast("No rejected photos to move.", "error");
          return;
        }
        openExportDialog("move", "No rejected photos to move.", {
          mode: "move",
          resolveRequest: async () => {
            const selectionRevision = await fetchReviewStateSelectionRevision("delete");
            if (!selectionRevision) {
              throw new Error("Review results are refreshing. Try again in a moment.");
            }
            return {
              selection: { scope: "review-state", marked: "delete" },
              selection_revision: selectionRevision,
              count: Number(state.overview?.summary?.delete_marked || 0),
            };
          },
          busyMessage: (count) => `Moving ${count} rejected files...`,
          successPrefix: "Move complete",
          logTitle: "Move rejected",
          emptyResultMessage: "No rejected files found.",
        });
      });
    }

    async function runScan(rootOverride = null, { generatePreviews = true, pipeline = null } = {}) {
      const root = rootOverride || currentLibraryRoot();
      if (!root) {
        throw new Error("Choose a folder before running analysis.");
      }

      if (pipeline) {
        setBusyPhaseProgress({
          percent: null,
          phaseIndex: pipeline.stepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Scanning library",
        });
      } else {
        setBusyPhaseProgress({ percent: null, phaseIndex: 1, phaseCount: 2, phaseLabel: "Indexing files" });
      }

      const filesTotalRef = { value: null };
      try {
        // Lightweight hint from the existing cache (DB-only), avoids an expensive filesystem pre-walk.
        const estimate = await postJson("/api/score-estimate", { root });
        const cachedTotal = Number(estimate.rows_total || 0);
        filesTotalRef.value = cachedTotal > 0 ? cachedTotal : null;
      } catch {
        filesTotalRef.value = null;
      }

      setBusyProgress(filesTotalRef.value && filesTotalRef.value > 0 ? 0 : null);
      setBusyMessage(generatePreviews
        ? "Scanning and generating previews..."
        : "Scanning metadata only for faster discovery...");

      const scanJobStart = await postJson("/api/scan/start", {
        roots: [root],
        extensions: document.getElementById("extensions-input").value.trim() || null,
        preview_mode: currentPreviewMode(),
        recursive: document.getElementById("recursive-toggle").checked,
        rescan_all: false,
        generate_previews: generatePreviews,
        files_total_hint: filesTotalRef.value,
        resource_profile: currentResourceProfile(),
      }, { signal: state.abortController?.signal });

      const scanJobId = String(scanJobStart?.job_id || "");
      if (!scanJobId) {
        throw new Error("Scan job failed to start.");
      }

      state.scanJobId = scanJobId;
      let result = null;
      try {
        result = await pollScanJob(scanJobId, { filesTotalRef, pipeline });
      } finally {
        if (!state.abortController?.signal?.aborted) {
          state.scanJobId = null;
        }
      }

      if (pipeline) {
        const donePercent = (Number(pipeline.stepIndex) / Number(pipeline.totalSteps)) * 100;
        setBusyProgress(Math.min(100, Math.max(0, Math.round(donePercent))));
        setBusyPhaseProgress({
          percent: 100,
          phaseIndex: pipeline.stepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Scanning library",
        });
      } else {
        setBusyProgress(100);
        setBusyPhaseProgress({ percent: 100, phaseIndex: 2, phaseCount: 2, phaseLabel: "Scanning files" });
      }
      setBusyMessage(`Scan completed. Processed ${result.files_seen} file(s).`);

      addLogEntry("Scan completed", `Seen ${result.files_seen}, added ${result.files_added}, updated ${result.files_updated}, removed ${result.files_removed}.`);
      showToast("Scan completed.");
      await refreshWorkspace();
      syncReviewRoot(root);
    }

    async function runScore(rootOverride = null, { pipeline = null } = {}) {
      const root = rootOverride || currentLibraryRoot() || null;

      const selectedModel = document.getElementById("model-select").value || state.options?.default_scoring_mode || state.options?.learned_models?.[0] || "topiq_nr";
      const learnedBackend = selectedModel;
      const runtimeTarget = document.getElementById("device-select").value || "auto";
      const requestedBatchSize = scoreBatchSize(learnedBackend, runtimeTarget, state.options?.learned?.recommended_batch_sizes);
      let rowsTotal = null;

      if (pipeline) {
        setBusyPhaseProgress({
          percent: 0,
          phaseIndex: pipeline.stepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Loading model",
        });
      } else {
        setBusyPhaseProgress({ percent: 0, phaseIndex: 1, phaseCount: 3, phaseLabel: "Loading model" });
      }

      try {
        const estimate = await postJson("/api/score-estimate", { root });
        rowsTotal = Number(estimate.rows_total || 0);
        if (rowsTotal > 0) {
          setBusyProgress(0);
          setBusyMessage(`Scoring... 0/${rowsTotal} (0%)`);
        }
      } catch {
        rowsTotal = null;
      }

      const scoreJobStart = await postJson("/api/score/start", {
        root,
        learned_backend_name: learnedBackend,
        device: runtimeTarget || null,
        batch_size: requestedBatchSize,
        preview_mode: currentPreviewMode(),
        force: false,
        resource_profile: currentResourceProfile(),
      }, { signal: state.abortController?.signal });

      const scoreJobId = String(scoreJobStart?.job_id || "");
      if (!scoreJobId) {
        throw new Error("Score job failed to start.");
      }

      state.scoreJobId = scoreJobId;
      let result = null;
      try {
        result = await pollScoreJob(scoreJobId, { rowsTotal, pipeline });
      } finally {
        if (!state.abortController?.signal?.aborted) {
          state.scoreJobId = null;
        }
      }

      if (pipeline) {
        const doneStepIndex = Math.min(Number(pipeline.totalSteps), Number(pipeline.stepIndex) + 1);
        const donePercent = (doneStepIndex / Number(pipeline.totalSteps)) * 100;
        setBusyProgress(Math.min(100, Math.max(0, Math.round(donePercent))));
        setBusyPhaseProgress({
          percent: 100,
          phaseIndex: doneStepIndex,
          phaseCount: pipeline.totalSteps,
          phaseLabel: "Model scoring complete",
        });
      } else {
        setBusyProgress(100);
        setBusyPhaseProgress({ percent: 100, phaseIndex: 3, phaseCount: 3, phaseLabel: "Model scoring complete" });
      }
      setBusyMessage(`Scoring completed. Processed ${result.rows_loaded || 0} row(s).`);

      addLogEntry("Score completed", `Scored ${result.files_scored || 0}, learned ${result.learned_scored || 0}, skipped ${result.files_skipped || 0}, failed ${result.files_failed || 0}.`);
      showToast("Scoring completed.");
      await refreshWorkspace();
      syncReviewRoot(root);
    }

    async function analyzeLibrary() {
      const root = currentLibraryRoot();
      if (!root) {
        throw new Error("Choose a folder before running analysis.");
      }

      saveUiState();
      addLogEntry("Analyze folder", root);
      setBusyMessage("Fast scan: indexing files without preview generation...");
      await runScan(root, {
        generatePreviews: false,
        pipeline: { stepIndex: 1, totalSteps: 3 },
      });
      setBusyMessage("Scoring selected folder...");
      await runScore(root, {
        pipeline: { stepIndex: 2, totalSteps: 3 },
      });
      syncReviewRoot(root);
      await loadQueue();
      setTab("review");
      showToast("Analysis completed. Switched to Review tab.");
    }

    async function clearCache(scope, message) {
      const result = await postJson("/api/cache/clear", { scope });
      addLogEntry("Cache action", `${message}: files ${result.files}, scores ${result.scores}, review ${result.review}.`);
      showToast(message);
      if (scope === "all") {
        clearActiveSelection();
        state.activeId = null;
        state.detail = null;
      }
      await refreshWorkspace();
    }

    async function deleteSelectedFiles() {
      const selectionRequest = activeSelectionRequest();
      if (!selectionRequest.count) {
        throw new Error("Select one or more items first.");
      }
      if (!window.confirm(`Delete ${selectionRequest.count} file(s) from disk? This cannot be undone.`)) {
        return;
      }
      const result = await postJson("/api/files/delete", { ...selectionRequest, delete_from_disk: true });
      addLogEntry("Disk delete", `Deleted ${result.deleted_count}, failed ${result.failed_count}.`);
      clearActiveSelection();
      showToast(`Deleted ${result.deleted_count} files from disk.`);
      await refreshWorkspace();
    }

    async function navigateSelection(step) {
      if (!state.queue.length) return;
      const currentIndex = state.queue.findIndex((item) => item.id === state.activeId);
      if (currentIndex === -1) {
        await selectFile(state.queue[0].id);
        return;
      }

      const nextIndex = currentIndex + step;
      if (nextIndex >= 0 && nextIndex < state.queue.length) {
        await selectFile(state.queue[nextIndex].id);
        return;
      }

      if (step > 0 && ((state.page + 1) * state.pageSize) < state.totalFiles) {
        state.page += 1;
        await loadQueue();
        if (state.queue.length) {
          await selectFile(state.queue[0].id);
        }
        return;
      }

      if (step < 0 && state.page > 0) {
        state.page -= 1;
        await loadQueue();
        if (state.queue.length) {
          await selectFile(state.queue[state.queue.length - 1].id);
        }
      }
    }

    async function openOriginalFile(fileId) {
      if (!Number.isInteger(Number(fileId)) || Number(fileId) <= 0) {
        throw new Error("Pick a file first.");
      }
      await postJson("/api/files/open", { file_id: Number(fileId) });
    }

    async function openBrowser(targetId) {
      state.browserTarget = targetId;
      const dialog = document.getElementById("folder-browser");
      if (!dialog.open) {
        dialog.showModal();
      }
      const roots = await fetchJson("/api/fs/roots");
      const rootContainer = document.getElementById("browser-roots");
      rootContainer.innerHTML = roots.items.map((item) => `<button type="button" class="ghost browser-root" data-path="${escapeHtml(item.path)}">${escapeHtml(item.name)}</button>`).join("");
      rootContainer.querySelectorAll(".browser-root").forEach((button) => {
        button.addEventListener("click", () => browseDirectory(button.dataset.path).catch(handleError));
      });
      await browseDirectory(document.getElementById(targetId).value || roots.items[0]?.path || "/");
    }

    async function browseDirectory(path) {
      const payload = await fetchJson(`/api/fs/list?path=${encodeURIComponent(path)}`);
      state.browserPath = payload.path;
      document.getElementById("browser-path").value = payload.path;
      const list = document.getElementById("browser-list");
      list.innerHTML = payload.items.length
        ? payload.items.map((item) => `
            <button type="button" class="browser-item" data-path="${escapeHtml(item.path)}">
              <strong>${escapeHtml(item.name)}</strong>
              <span class="muted">${escapeHtml(item.path)}</span>
            </button>
          `).join("")
        : `<p class="muted">No subdirectories available.</p>`;

      list.querySelectorAll(".browser-item").forEach((button) => {
        button.addEventListener("click", () => browseDirectory(button.dataset.path).catch(handleError));
      });
    }

    function chooseBrowserPath() {
      if (!state.browserTarget || !state.browserPath) return;
      const targetInput = document.getElementById(state.browserTarget);
      if (!targetInput) {
        return;
      }
      targetInput.value = state.browserPath;
      targetInput.dispatchEvent(new Event("input", { bubbles: true }));
      targetInput.dispatchEvent(new Event("change", { bubbles: true }));
      document.getElementById("folder-browser").close();
    }

    function handleError(error) {
      console.error(error);
      let message = error?.message || "Unexpected error";
      if (message === "Failed to fetch") {
        message = "The local server request failed. If an analysis is still running, wait for completion before retrying.";
      }
      showToast(message, "error");
      addLogEntry("Error", message);
    }

    return {
      compareRowSortChoices,
      syncCompareSortControls,
      fetchCompareJobStatus,
      fetchCompareJobResult,
      pollCompareJob,
      comparisonFailureText,
      comparisonFailureDetails,
      comparisonFailureSummaryText,
      renderComparisonWarnings,
      renderComparisonSummary,
      renderComparisonResults,
      renderComparisonModelOptions,
      runModelComparison,
      saveReview,
      reviewDecisionPayload,
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
    };
  }

  window.ShotSieveWorkflows = {
    createWorkflows,
  };
})();
