(() => {
  function createGridController(deps) {
    const {
      state,
      ui,
      formatting,
      reviewModule,
      notifications,
      api,
      stateModule,
      appUtils,
    } = deps;

    const { escapeHtml, formatNumber, getScoreColor, pathDirectory, pathLeaf } = formatting;
    const { getSortRelevantScore, renderDetail: renderDetailView, renderQueue: renderQueueView, updateSelectionState: updateSelectionStateView } = reviewModule;
    const { showToast } = notifications;
    const { fetchJson } = api;

    let queueAbortController = null;

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

      const formatFilters = [...document.querySelectorAll("input[name='format-filter']:checked")].map((i) => i.value);
      if (formatFilters.length < 6) {
        params.set("formats", formatFilters.length > 0 ? formatFilters.join(",") : "none");
      }

      const minMp = document.getElementById("filter-min-mp").value;
      if (minMp) params.set("min_mp", minMp);

      const maxMp = document.getElementById("filter-max-mp").value;
      if (maxMp) params.set("max_mp", maxMp);

      const minEdge = document.getElementById("filter-min-edge")?.value;
      if (minEdge) params.set("min_edge", minEdge);

      const maxEdge = document.getElementById("filter-max-edge")?.value;
      if (maxEdge) params.set("max_edge", maxEdge);

      const minSize = document.getElementById("filter-min-size").value;
      if (minSize) params.set("min_size", String(Math.round(parseFloat(minSize) * 1000000)));

      const maxSize = document.getElementById("filter-max-size").value;
      if (maxSize) params.set("max_size", String(Math.round(parseFloat(maxSize) * 1000000)));

      const metadataStatus = document.getElementById("filter-metadata-status").value;
      if (metadataStatus && metadataStatus !== "all") params.set("metadata", metadataStatus);

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

      const formatFilters = [...document.querySelectorAll("input[name='format-filter']:checked")].map((i) => i.value);
      const minMp = document.getElementById("filter-min-mp")?.value || "";
      const maxMp = document.getElementById("filter-max-mp")?.value || "";
      const minEdge = document.getElementById("filter-min-edge")?.value || "";
      const maxEdge = document.getElementById("filter-max-edge")?.value || "";
      const minSize = document.getElementById("filter-min-size")?.value || "";
      const maxSize = document.getElementById("filter-max-size")?.value || "";
      const metadataStatus = document.getElementById("filter-metadata-status")?.value || "all";

      return Boolean(
        query || root || minScore || maxScore || marked !== "all" || issues !== "all"
        || formatFilters.length < 6 || minMp || maxMp || minEdge || maxEdge || minSize || maxSize || metadataStatus !== "all"
      );
    }

    function renderSummary() {
      const emptyTotals = { total_files: 0, scored_files: 0, delete_marked: 0, export_marked: 0 };
      const activeLibrary = state.overview?.active_library || state.overview?.summary || emptyTotals;
      const catalog = state.overview?.catalog || state.overview?.summary || emptyTotals;
      const totalsMarkup = (totals) => [
        [`<span class="stat-value">${Number(totals.scored_files || 0).toLocaleString()}</span> scored`],
        [`<span class="stat-value">${Number(totals.delete_marked || 0).toLocaleString()}</span> rejected`],
        [`<span class="stat-value">${Number(totals.export_marked || 0).toLocaleString()}</span> selected`],
        [`<span class="stat-value">${Number(totals.total_files || 0).toLocaleString()}</span> discovered`],
      ].map(([text]) => `<span class="stat-item">${text}</span>`).join("");

      document.getElementById("summary-strip").innerHTML = [
        `<span class="summary-scope summary-scope-active"><strong>This library</strong>${totalsMarkup(activeLibrary)}</span>`,
        `<span class="summary-scope summary-scope-catalog"><strong>All cached libraries</strong>${totalsMarkup(catalog)}</span>`,
      ].join("");
    }

    function renderReviewScope() {
      const scopeContext = document.getElementById("review-scope-context");
      const root = document.getElementById("root-filter")?.value || "";
      if (!scopeContext) {
        return;
      }
      if (root) {
        scopeContext.textContent = `Reviewing this library: ${root}`;
        scopeContext.dataset.scope = "library";
        return;
      }
      scopeContext.textContent = "All libraries — global catalog view";
      scopeContext.dataset.scope = "global";
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
        handleError: deps.handleError,
      });
    }

    function updateSelectionState(options) {
      updateSelectionStateView({ state }, options);
    }

    function renderDetail() {
      renderDetailView({
        state,
        modelDisplayNames: stateModule.MODEL_DISPLAY_NAMES,
        pathLeaf,
        escapeHtml,
        formatNumber,
        scoreCard: deps.scoreCard,
        statusPill: deps.statusPill,
        openOriginalFile: deps.openOriginalFile,
        handleError: deps.handleError,
      });
    }

    async function loadQueue() {
      if (queueAbortController) {
        queueAbortController.abort();
      }
      queueAbortController = new AbortController();
      const signal = queueAbortController.signal;

      try {
        let query = null;
        let data = null;
        while (true) {
          query = currentQuery();
          data = await fetchJson(`/api/files?${query.toString()}`, { signal });

          if (signal.aborted) return;

          const totalFiles = Number(data.total || 0);
          const maxPage = totalFiles > 0
            ? Math.max(0, Math.ceil(totalFiles / state.pageSize) - 1)
            : 0;
          if (state.page > maxPage) {
            state.page = maxPage;
            continue;
          }
          break;
        }

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
      const activeLibrary = state.overview?.active_library || state.overview?.summary || {};
      const catalog = state.overview?.catalog || state.overview?.summary || {};
      const reviewRoot = document.getElementById("root-filter")?.value || "";
      const scopeSummary = reviewRoot ? activeLibrary : catalog;
      const scopeLabel = reviewRoot ? "this library" : "All cached libraries";
      const totalScoredInScope = Number(scopeSummary.scored_files || 0);
      const filtered = hasActiveReviewFilters();
      let label;
      if (hasResults) {
        if (filtered) {
          label = `Showing ${start}–${end} of ${state.totalFiles.toLocaleString()} matching photos`;
          if (totalScoredInScope > 0) {
            label += ` (${totalScoredInScope.toLocaleString()} scored in ${scopeLabel})`;
          }
        } else if (totalScoredInScope > 0) {
          label = `Showing ${start}–${end} of ${totalScoredInScope.toLocaleString()} scored photos in ${scopeLabel}`;
        } else {
          label = `Showing ${start}–${end} of ${state.totalFiles.toLocaleString()} photos`;
        }
      } else {
        label = filtered
          ? `No photos match current filters${totalScoredInScope > 0 ? ` (${totalScoredInScope.toLocaleString()} scored in ${scopeLabel})` : ""}`
          : "No scored photos yet";
      }
      document.getElementById("page-info").textContent = label;
      document.getElementById("page-prev").disabled = state.page === 0;
      document.getElementById("page-next").disabled = ((state.page + 1) * state.pageSize) >= state.totalFiles;

      const selectAllCount = document.getElementById("select-all-matching-count");
      if (selectAllCount) {
        selectAllCount.textContent = state.totalFiles > 0 ? `(${state.totalFiles.toLocaleString()})` : "";
      }

      const rejectedCount = Number(activeLibrary.delete_marked || 0);
      const rejectedBar = document.getElementById("rejected-actions");
      if (reviewRoot && rejectedCount > 0) {
        rejectedBar.classList.remove("hidden");
        document.getElementById("rejected-label").textContent = `${rejectedCount} photo${rejectedCount !== 1 ? "s" : ""} rejected in this library`;
      } else {
        rejectedBar.classList.add("hidden");
      }
    }

    async function selectFile(fileId) {
      state.activeId = fileId;

      const detailPromise = fetchJson(`/api/file?id=${fileId}`);

      if (state.queue.length > 0) {
        updateSelectionState({ scrollActive: true });
      } else {
        renderQueue();
      }

      state.detail = await detailPromise;
      renderDetail();
    }

    function applyReviewUpdate(updatedDetail) {
      const fileId = Number(updatedDetail?.id);
      if (!Number.isInteger(fileId) || fileId <= 0) {
        return false;
      }

      const queueIndex = state.queue.findIndex((item) => item.id === fileId);
      if (queueIndex >= 0) {
        state.queue[queueIndex] = {
          ...state.queue[queueIndex],
          ...updatedDetail,
        };
      }

      state.activeId = fileId;
      state.detail = state.detail?.id === fileId
        ? { ...state.detail, ...updatedDetail }
        : updatedDetail;
      renderQueue();
      renderDetail();
      return queueIndex >= 0;
    }

    return {
      selectAll,
      selectNone,
      selectAllMatching,
      invalidateLoadedReviewSelection,
      selectionKey,
      reviewSelectionSnapshotFromQuery,
      currentQuery,
      hasActiveReviewFilters,
      renderSummary,
      renderReviewScope,
      renderQueue,
      updateSelectionState,
      renderDetail,
      loadQueue,
      renderPagination,
      selectFile,
      applyReviewUpdate,
    };
  }

  window.ShotSieveGrid = {
    createGridController,
  };
})();
