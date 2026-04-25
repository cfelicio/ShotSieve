(() => {
  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function formatNumber(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) {
      return "n/a";
    }
    return Number(value).toFixed(1);
  }

  function getScoreColor(value) {
    if (value === null || value === undefined) return "";
    const score = Number(value);
    if (score >= 80) return "score-high";
    if (score >= 50) return "score-mid";
    return "score-low";
  }

  function pathLeaf(path) {
    return String(path).split(/[\\/]/).filter(Boolean).pop() || String(path);
  }

  function pathDirectory(path) {
    const parts = String(path).split(/[\\/]/).filter(Boolean);
    if (parts.length <= 1) {
      return String(path);
    }
    return parts.slice(0, -1).join(" / ");
  }

  function formatDuration(seconds) {
    if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) {
      return "n/a";
    }

    const totalSeconds = Math.max(0, Number(seconds));
    if (totalSeconds >= 60) {
      const minutes = Math.floor(totalSeconds / 60);
      const remainder = totalSeconds - (minutes * 60);
      const remainderText = remainder >= 10 ? remainder.toFixed(0) : remainder.toFixed(1);
      return `${minutes}m ${remainderText}s`;
    }

    return totalSeconds >= 10 ? `${totalSeconds.toFixed(0)}s` : `${totalSeconds.toFixed(1)}s`;
  }

  function formatFilesPerSecond(fileCount, seconds) {
    const numericCount = Number(fileCount);
    const numericSeconds = Number(seconds);
    if (Number.isNaN(numericCount) || Number.isNaN(numericSeconds) || numericCount <= 0 || numericSeconds <= 0) {
      return "n/a";
    }

    const rate = numericCount / numericSeconds;
    if (rate >= 100) {
      return `${rate.toFixed(0)} files/s`;
    }
    if (rate >= 10) {
      return `${rate.toFixed(1)} files/s`;
    }
    return `${rate.toFixed(2)} files/s`;
  }

  function mergeTimingTotals(target, source) {
    if (!source || typeof source !== "object") {
      return;
    }

    for (const [modelName, seconds] of Object.entries(source)) {
      const numericSeconds = Number(seconds);
      if (Number.isNaN(numericSeconds)) {
        continue;
      }
      target[modelName] = (target[modelName] || 0) + numericSeconds;
    }
  }

  function comparisonScoreNumber(value) {
    if (value === null || value === undefined) {
      return null;
    }
    const numericValue = Number(value);
    return Number.isNaN(numericValue) ? null : numericValue;
  }

  function sortComparisonRows(rows, sortMode) {
    const sortedRows = [...rows];
    if (sortMode === "input") {
      return sortedRows;
    }

    if (sortMode === "path_asc") {
      return sortedRows.sort((left, right) => String(left.path || "").localeCompare(String(right.path || ""), undefined, { sensitivity: "base" }));
    }

    const [modelName, direction] = String(sortMode || "").split(":");
    if (!modelName || (direction !== "asc" && direction !== "desc")) {
      return sortedRows;
    }

    const scoreKey = `${modelName}_score`;
    return sortedRows.sort((left, right) => {
      const leftValue = comparisonScoreNumber(left[scoreKey]);
      const rightValue = comparisonScoreNumber(right[scoreKey]);
      const leftMissing = leftValue === null;
      const rightMissing = rightValue === null;

      if (leftMissing !== rightMissing) {
        return leftMissing ? 1 : -1;
      }
      if (leftMissing && rightMissing) {
        return String(left.path || "").localeCompare(String(right.path || ""), undefined, { sensitivity: "base" });
      }

      if (leftValue !== rightValue) {
        return direction === "desc" ? rightValue - leftValue : leftValue - rightValue;
      }

      return String(left.path || "").localeCompare(String(right.path || ""), undefined, { sensitivity: "base" });
    });
  }

  function compareChunkSize(models, runtime) {
    const normalizedRuntime = String(runtime || "").toLowerCase();
    if (normalizedRuntime === "cpu" && models.some((modelName) => modelName === "qalign")) {
      return 1;
    }
    return models.some((modelName) => ["qalign", "qualiclip", "tres"].includes(modelName)) ? 10 : 20;
  }

  function isAcceleratedRuntime(runtime) {
    return ["auto", "cuda", "xpu", "directml", "nvidia", "intel", "amd", "apple", "mps"].includes(String(runtime || "").toLowerCase());
  }

  function scoreBatchSize(modelName, runtime, serverRecommendations) {
    // Use server-detected hardware-aware recommendation when available
    if (serverRecommendations && serverRecommendations[modelName]) {
      return Math.max(1, Number(serverRecommendations[modelName]));
    }
    // Fallback: conservative defaults when VRAM is unknown
    const heavy = ["qalign", "qualiclip", "clipiqa", "tres"].includes(modelName);
    if (heavy) {
      return 2;
    }
    return isAcceleratedRuntime(runtime) ? 12 : 4;
  }

  function compareBatchSize(models, runtime, serverRecommendations) {
    // The backend now computes per-model optimal batch sizes using
    // recommended_batch_size(), so we no longer need to compute a
    // global min here. Return null to let the backend use its own
    // per-model hardware-aware defaults.
    if (serverRecommendations) {
      return null;
    }
    // Fallback: conservative defaults when server recommendations unavailable
    const hasHeavyModel = models.some((modelName) => ["qalign", "qualiclip", "clipiqa", "tres"].includes(modelName));
    if (hasHeavyModel) {
      return 2;
    }
    return isAcceleratedRuntime(runtime) ? 12 : 4;
  }

  function compareProgressPercent(progress) {
    if (!progress || typeof progress !== "object") {
      return null;
    }

    // During preview generation the progress bar reflects preview completion only.
    if (progress.phase === "generating_previews") {
      const previewTotal = Math.max(1, Number(progress.files_total || 1));
      const previewDone = Math.max(0, Math.min(previewTotal, Number(progress.files_processed || 0)));
      // Reserve the first 10% of the bar for preview generation.
      return (previewDone / previewTotal) * 10;
    }

    const modelCount = Math.max(1, Number(progress.model_count || 1));
    const modelIndex = Math.max(1, Math.min(modelCount, Number(progress.model_index || 1)));
    const filesTotal = Math.max(1, Number(progress.files_total || 1));
    const filesProcessed = Math.max(0, Math.min(filesTotal, Number(progress.files_processed || 0)));
    const overallTotal = modelCount * filesTotal;
    const overallDone = Math.min(overallTotal, ((modelIndex - 1) * filesTotal) + filesProcessed);
    // Scoring occupies the remaining 10–100% range.
    return 10 + (overallDone / overallTotal) * 90;
  }

  function compareProgressMessage(progress, elapsedSeconds, rowsTotal) {
    const elapsedText = formatDuration(elapsedSeconds);
    if (!progress || typeof progress !== "object") {
      return `Comparing… Loading model weights (first run may take a while) · ${elapsedText} elapsed`;
    }

    const modelCount = Math.max(1, Number(progress.model_count || 1));
    const modelIndex = Math.max(1, Math.min(modelCount, Number(progress.model_index || 1)));
    const modelName = progress.model_name || "model";

    if (progress.phase === "loading") {
      return `Loading model ${modelIndex}/${modelCount}: ${modelName} · ${elapsedText} elapsed`;
    }

    // Preview-generation sub-phase
    if (progress.phase === "generating_previews") {
      const previewTotal = Math.max(0, Number(progress.files_total || 0));
      const previewDone = Math.max(0, Number(progress.files_processed || 0));
      if (previewTotal > 0) {
        const pct = Math.min(99, Math.round((previewDone / previewTotal) * 100));
        return `Generating previews… ${previewDone}/${previewTotal} (${pct}%) · ${elapsedText} elapsed`;
      }
      return `Generating previews… · ${elapsedText} elapsed`;
    }

    const filesTotal = rowsTotal && rowsTotal > 0 ? rowsTotal : Math.max(0, Number(progress.files_total || 0));
    const filesProcessed = Math.max(0, Number(progress.files_processed || 0));

    if (filesTotal > 0) {
      const percent = Math.min(99, compareProgressPercent(progress) ?? 0);
      const overallTotal = modelCount * filesTotal;
      const overallProcessed = Math.min(overallTotal, ((modelIndex - 1) * filesTotal) + filesProcessed);
      return `Model ${modelIndex}/${modelCount}: ${modelName} · Files ${filesProcessed}/${filesTotal} · Overall ${overallProcessed}/${overallTotal} (${Math.round(percent)}%) · ${elapsedText} elapsed`;
    }

    return `Model ${modelIndex}/${modelCount}: ${modelName} (${filesProcessed} processed) · ${elapsedText} elapsed`;
  }

  function scoreProgressPercent(progress) {
    if (!progress || typeof progress !== "object") {
      return null;
    }

    if (progress.phase === "loading") {
      return 0;
    }

    // During preview generation the bar reflects preview completion only.
    if (progress.phase === "generating_previews") {
      const previewTotal = Math.max(1, Number(progress.files_total || 1));
      const previewDone = Math.max(0, Math.min(previewTotal, Number(progress.files_processed || 0)));
      // Reserve the first 10% for preview generation.
      return (previewDone / previewTotal) * 10;
    }

    const filesTotal = Math.max(1, Number(progress.files_total || 1));
    const filesProcessed = Math.max(0, Math.min(filesTotal, Number(progress.files_processed || 0)));
    // Scoring occupies the remaining 10–100% range.
    return 10 + (filesProcessed / filesTotal) * 90;
  }

  function scoreProgressMessage(progress, elapsedSeconds, rowsTotal) {
    const elapsedText = formatDuration(elapsedSeconds);
    if (!progress || typeof progress !== "object") {
      return `Scoring… Loading model weights (first run may take a while) · ${elapsedText} elapsed`;
    }

    if (progress.phase === "loading") {
      return `Scoring… Loading model weights (first run may take a while) · ${elapsedText} elapsed`;
    }

    // Preview-generation sub-phase
    if (progress.phase === "generating_previews") {
      const previewTotal = Math.max(0, Number(progress.files_total || 0));
      const previewDone = Math.max(0, Number(progress.files_processed || 0));
      if (previewTotal > 0) {
        const pct = Math.min(99, Math.round((previewDone / previewTotal) * 100));
        return `Generating previews… ${previewDone}/${previewTotal} (${pct}%) · ${elapsedText} elapsed`;
      }
      return `Generating previews… · ${elapsedText} elapsed`;
    }

    const filesTotal = rowsTotal && rowsTotal > 0 ? rowsTotal : Math.max(0, Number(progress.files_total || 0));
    const filesProcessed = Math.max(0, Number(progress.files_processed || 0));

    if (filesTotal > 0) {
      const percent = Math.min(99, scoreProgressPercent(progress) ?? 0);
      return `Scoring… ${filesProcessed}/${filesTotal} (${Math.round(percent)}%) · ${elapsedText} elapsed`;
    }

    return `Scoring… ${filesProcessed} processed · ${elapsedText} elapsed`;
  }

  function scanProgressPercent(progress, filesTotalEstimate = null) {
    if (!progress || typeof progress !== "object") {
      return null;
    }

    const reportedTotal = Number(progress.files_total || 0);
    const estimateTotal = Number(filesTotalEstimate || 0);
    const filesTotal = reportedTotal > 0 ? reportedTotal : (estimateTotal > 0 ? estimateTotal : 0);
    if (filesTotal <= 0) {
      return null;
    }

    const filesProcessed = Math.max(0, Math.min(filesTotal, Number(progress.files_processed || 0)));
    return (filesProcessed / filesTotal) * 100;
  }

  function scanProgressMessage(progress, elapsedSeconds, filesTotalEstimate) {
    const elapsedText = formatDuration(elapsedSeconds);
    if (!progress || typeof progress !== "object") {
      return `Scanning… Discovering files · ${elapsedText} elapsed`;
    }

    const filesTotal = filesTotalEstimate && filesTotalEstimate > 0
      ? filesTotalEstimate
      : Math.max(0, Number(progress.files_total || 0));
    const filesProcessed = Math.max(0, Number(progress.files_processed || 0));
    const phase = String(progress.phase || "scanning").toLowerCase();

    if (filesTotal > 0) {
      const percent = Math.min(99, scanProgressPercent({ files_total: filesTotal, files_processed: filesProcessed }) ?? 0);
      if (phase === "failed") {
        return `Scanning failed at ${filesProcessed}/${filesTotal} (${Math.round(percent)}%) · ${elapsedText} elapsed`;
      }
      return `Scanning… ${filesProcessed}/${filesTotal} (${Math.round(percent)}%) · ${elapsedText} elapsed`;
    }

    if (phase === "failed") {
      return `Scanning failed after ${filesProcessed} processed · ${elapsedText} elapsed`;
    }
    return `Scanning… ${filesProcessed} processed · ${elapsedText} elapsed`;
  }

  function parseRuntimeStatusMap(runtimeStatusText) {
    const statusMap = {};
    for (const entry of String(runtimeStatusText || "").split(",")) {
      const [runtime, status] = entry.split(":");
      const runtimeKey = String(runtime || "").trim();
      const statusValue = String(status || "").trim();
      if (runtimeKey) {
        statusMap[runtimeKey] = statusValue || "unknown";
      }
    }
    return statusMap;
  }

  function runtimeDisplayName(runtime) {
    const labels = {
      cuda: "CUDA",
      xpu: "XPU",
      directml: "DirectML",
      mps: "MPS",
      cpu: "CPU",
      auto: "Auto",
    };
    return labels[String(runtime || "").toLowerCase()] || String(runtime || "unknown").toUpperCase();
  }

  function runtimeStatusToken(status) {
    const normalized = String(status || "unknown").toLowerCase();
    if (normalized === "available") {
      return "✅";
    }
    if (normalized === "not-installed") {
      return "⬇️";
    }
    if (normalized === "unsupported") {
      return "🚫";
    }
    if (normalized === "unavailable") {
      return "❌";
    }
    return "❔";
  }

  function summarizeAccelerators(statusMap) {
    const orderedAccelerators = ["cuda", "xpu", "directml", "mps"];
    return orderedAccelerators
      .map((runtime) => {
        const status = statusMap[runtime] || "unknown";
        return `${runtimeDisplayName(runtime)} ${runtimeStatusToken(status)} ${status}`;
      })
      .join(" · ");
  }

  function summarizeAutoPriority(options) {
    const priorityRaw = String(options?.learned?.auto_runtime_priority || "cuda,xpu,directml,cpu");
    const priority = priorityRaw
      .split(",")
      .map((runtime) => runtime.trim())
      .filter(Boolean)
      .map((runtime) => runtimeDisplayName(runtime))
      .join(" → ");
    return `${priority}. Auto mode picks the first available runtime in this executable build.`;
  }

  function formatPhotoSupport(heifOk, rawOk) {
    const parts = [];
    parts.push(heifOk ? "HEIF \u2705" : "HEIF \u274c");
    parts.push(rawOk ? "RAW \u2705" : "RAW \u274c");
    return parts.join(" \u00b7 ");
  }

  function availableLearnedModels(options, defaultModelCatalog, excludedModels = []) {
    const excluded = new Set((excludedModels || []).map((model) => String(model).toLowerCase()));
    const available = (options.learned_models || []).filter((model) => !excluded.has(String(model).toLowerCase()));
    const preferred = available.filter((model) => defaultModelCatalog.includes(model));
    return preferred;
  }

  function comparisonDefaults(options, persisted, allowedModels) {
    const preferred = [options.default_scoring_mode, "clipiqa", "qalign"].filter(Boolean);
    const source = Array.isArray(persisted.compareModels) && persisted.compareModels.length ? persisted.compareModels : preferred;
    const selected = source.filter((model, index, items) => allowedModels.includes(model) && items.indexOf(model) === index);
    if (selected.length >= 2) {
      return selected;
    }
    return allowedModels.slice(0, Math.min(3, allowedModels.length));
  }

  async function fetchJson(url, options = {}) {
    const signal = options.signal || null;
    const response = await fetch(url, { ...options, signal });
    const contentType = response.headers.get("Content-Type") || "";
    const payload = contentType.includes("application/json") ? await response.json() : null;

    if (!response.ok) {
      const message = payload?.error || `Request failed: ${response.status}`;
      throw new Error(message);
    }

    return payload;
  }

  function postJson(url, payload, { signal } = {}) {
    return fetchJson(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
  }

  function currentResourceProfile() {
    return localStorage.getItem("shotsieve_resource_profile") || "normal";
  }

  window.ShotSieveUtils = {
    availableLearnedModels,
    compareBatchSize,
    compareChunkSize,
    comparisonDefaults,
    compareProgressMessage,
    compareProgressPercent,
    currentResourceProfile,
    escapeHtml,
    fetchJson,
    postJson,
    formatDuration,
    formatFilesPerSecond,
    formatNumber,
    formatPhotoSupport,
    getScoreColor,
    isAcceleratedRuntime,
    mergeTimingTotals,
    comparisonScoreNumber,
    parseRuntimeStatusMap,
    pathDirectory,
    pathLeaf,
    runtimeDisplayName,
    runtimeStatusToken,
    summarizeAccelerators,
    summarizeAutoPriority,
    scoreProgressMessage,
    scoreProgressPercent,
    scanProgressMessage,
    scanProgressPercent,
    scoreBatchSize,
    sortComparisonRows,
  };
})();
