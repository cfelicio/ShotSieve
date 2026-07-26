(() => {
  function createWorkflowCompare(deps) {
    const {
      api,
      busy,
      compare,
      formatting,
      notifications,
      pollingModule,
      state,
      ui,
      workflowLibrary,
    } = deps;

    const { postJson } = api;
    const { setBusyMessage, setBusyPhaseProgress, setBusyProgress } = busy;
    const {
      compareBatchSize,
      comparisonDefaults,
      currentResourceProfile,
      modelDescriptions,
      modelDisplayNames,
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
    const { currentLibraryRoot, selectedComparisonModels, setTab } = ui;

    const {
      pollCompareJob,
      pipelineOverallPercent,
    } = pollingModule;

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

      const modelsToRender = Array.isArray(allowedModels) && allowedModels.length ? allowedModels : (options?.learned_models?.length ? options.learned_models : ["topiq_nr", "clipiqa"]);
      const selected = new Set(comparisonDefaults(options, persisted, modelsToRender));
      target.innerHTML = modelsToRender.map((modelName) => `
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

        const { runScan } = workflowLibrary;
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

    return {
      compareRowSortChoices,
      syncCompareSortControls,
      comparisonFailureText,
      comparisonFailureDetails,
      comparisonFailureSummaryText,
      renderComparisonWarnings,
      renderComparisonSummary,
      renderComparisonResults,
      renderComparisonModelOptions,
      runModelComparison,
    };
  }

  window.ShotSieveWorkflowCompare = {
    createWorkflowCompare,
  };
})();
