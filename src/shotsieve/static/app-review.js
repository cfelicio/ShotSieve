(() => {
  function isCompactReviewLayout() {
    return Boolean(window.matchMedia?.("(max-width: 63.9375rem)").matches);
  }

  function updateCompactReviewNavigation(state, { queueIndex = -1 } = {}) {
    const hasPrev = queueIndex > 0 || state.page > 0;
    const hasNext = queueIndex >= 0
      && (
        queueIndex < state.queue.length - 1
        || ((state.page + 1) * state.pageSize) < state.totalFiles
      );

    document.querySelectorAll("[data-review-nav='prev']").forEach((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        return;
      }
      button.disabled = !hasPrev;
      button.setAttribute("aria-disabled", hasPrev ? "false" : "true");
    });

    document.querySelectorAll("[data-review-nav='next']").forEach((button) => {
      if (!(button instanceof HTMLButtonElement)) {
        return;
      }
      button.disabled = !hasNext;
      button.setAttribute("aria-disabled", hasNext ? "false" : "true");
    });
  }

  function getSortRelevantScore(item) {
    const sort = document.getElementById("sort-filter")?.value || "";
    const map = {
      learned_asc: { label: "AI", value: item.learned_score_normalized },
      learned_desc: { label: "AI", value: item.learned_score_normalized },
    };
    return map[sort] || null;
  }

  function displayScore(item) {
    if (item?.learned_score_normalized !== null && item?.learned_score_normalized !== undefined) {
      return item.learned_score_normalized;
    }
    return item?.overall_score;
  }

  function setSelectionRange(state, startIndex, endIndex, checked) {
    const start = Math.min(startIndex, endIndex);
    const end = Math.max(startIndex, endIndex);
    for (let i = start; i <= end; i++) {
      const fileId = state.queue[i]?.id;
      if (fileId === undefined) {
        continue;
      }
      setFileSelection(state, fileId, checked);
    }
  }

  function setFileSelection(state, fileId, checked) {
    if (state.bulkSelection) {
      const excludedIds = state.bulkSelection.excludedIds || new Set();
      if (checked) {
        excludedIds.delete(fileId);
      } else {
        excludedIds.add(fileId);
      }
      state.bulkSelection.excludedIds = excludedIds;
      return;
    }

    if (checked) {
      state.selectedIds.add(fileId);
    } else {
      state.selectedIds.delete(fileId);
    }
  }

  function selectionCount(state) {
    if (state.bulkSelection) {
      const excludedCount = state.bulkSelection.excludedIds instanceof Set
        ? state.bulkSelection.excludedIds.size
        : 0;
      return Math.max(0, Number(state.bulkSelection.count || 0) - excludedCount);
    }
    return state.selectedIds.size;
  }

  function isFileSelected(state, fileId) {
    if (state.bulkSelection) {
      return !state.bulkSelection.excludedIds?.has(fileId);
    }
    return state.selectedIds.has(fileId);
  }

  function selectionScopeText(state) {
    if (state.bulkSelection) {
      if (selectionCount(state) === 0) {
        return "No photo selected";
      }
      if (state.bulkSelection.excludedIds?.size) {
        return "All matching photos except unchecked";
      }
      return "All matching photos";
    }
    if (state.selectedIds.size) {
      return "Selected photos";
    }
    if (state.activeId) {
      return "Current photo";
    }
    return "No photo selected";
  }

  function updateSelectionSummary(state) {
    const countText = `${selectionCount(state)} selected`;

    const labelEl = document.getElementById("selection-label");
    if (labelEl) labelEl.textContent = countText;

    const scopeEl = document.getElementById("selection-scope-label");
    if (scopeEl) scopeEl.textContent = selectionScopeText(state);
  }

  let _thumbObserver = null;

  function updateSelectionState(deps, { scrollActive = false } = {}) {
    const { state } = deps;
    updateSelectionSummary(state);
    const queueList = document.getElementById("queue-list");
    if (!queueList) return;

    let activeNode = null;
    queueList.querySelectorAll(".queue-item").forEach((node, index) => {
      const fileId = Number(node.dataset.id);
      const isActive = state.activeId === fileId;
      const checkbox = node.querySelector("[data-select-id]");
      if (checkbox) checkbox.checked = isFileSelected(state, fileId);
      node.classList.toggle("active", isActive);
      if (isActive) {
        activeNode = node;
        state.lastSelectedIndex = index;
      }
    });

    if (activeNode && scrollActive && !isCompactReviewLayout()) {
      activeNode.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  function renderQueue(deps) {
    const {
      state,
      renderDetail,
      formatNumber: formatNumberFn,
      getSortRelevantScore: getSortRelevantScoreFn,
      getScoreColor,
      escapeHtml,
      pathLeaf,
      pathDirectory,
      selectFile,
      handleError,
    } = deps;

    void getSortRelevantScoreFn;

    const queueList = document.getElementById("queue-list");
    if (!queueList) {
      return;
    }

    const previousScroll = queueList.scrollTop;
    updateSelectionSummary(state);

    queueList.innerHTML = state.queue.length
      ? state.queue.map((item) => {
          const primaryScore = displayScore(item);
          const filename = escapeHtml(pathLeaf(item.path));
          return `
        <article class="queue-item ${state.activeId === item.id ? "active" : ""}" data-id="${item.id}">
          <label class="queue-select">
            <input class="queue-check" type="checkbox" data-select-id="${item.id}" ${isFileSelected(state, item.id) ? "checked" : ""} aria-label="Select ${filename}">
          </label>
          <button type="button" class="queue-open" data-open-id="${item.id}" aria-label="Open details for ${filename}">
            <img data-src="/api/media/preview?id=${item.id}" alt="Thumbnail of ${filename}" class="lazy-thumb">
            <div class="queue-copy">
              <div class="queue-kicker">
                <div class="badge-row">
                  ${item.delete_marked ? '<span class="badge badge-reject">reject</span>' : ""}
                  ${item.export_marked ? '<span class="badge badge-select">select</span>' : ""}
                </div>
              </div>
              <div class="queue-primary">
                <strong class="queue-file">${filename}</strong>
                <span class="queue-score ${getScoreColor(primaryScore)}">${formatNumberFn(primaryScore)}</span>
              </div>
              <div class="queue-path">${escapeHtml(pathDirectory(item.path))}</div>
            </div>
          </button>
        </article>
      `;
        }).join("")
      : `
      <div class="queue-empty-state">
        <div class="empty-icon">📁</div>
        <h3>No photos found</h3>
        <p class="muted">Either you haven't scanned this folder yet, or the current filters are hiding everything.</p>
        <button type="button" class="ghost" data-action="clear-filters">Clear Filters</button>
      </div>
    `;

    queueList.scrollTop = previousScroll;

    const clearFiltersProxy = queueList.querySelector("[data-action='clear-filters']");
    if (clearFiltersProxy) {
      clearFiltersProxy.addEventListener("click", () => {
        document.getElementById("clear-filters").click();
      });
    }

    if (_thumbObserver) {
      _thumbObserver.disconnect();
      _thumbObserver = null;
    }
    _thumbObserver = new IntersectionObserver((entries, observer) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          const img = entry.target;
          if (img.dataset.src) {
            img.src = img.dataset.src;
            img.removeAttribute("data-src");
          }
          observer.unobserve(img);
        }
      });
    }, { root: queueList, rootMargin: "200px" });
    queueList.querySelectorAll("img.lazy-thumb").forEach((img) => _thumbObserver.observe(img));

    queueList.querySelectorAll("[data-open-id]").forEach((button, buttonIndex) => {
      button.addEventListener("click", () => {
        state.lastSelectedIndex = buttonIndex;
        selectFile(Number(button.dataset.openId)).catch(handleError);
      });
    });

    let pendingSelectionGesture = null;
    queueList.querySelectorAll("[data-select-id]").forEach((checkbox, checkIndex) => {
      checkbox.addEventListener("click", (event) => {
        pendingSelectionGesture = {
          checked: checkbox.checked,
          index: checkIndex,
          shiftKey: event.shiftKey,
        };
      });

      checkbox.addEventListener("change", (event) => {
        const fileId = Number(event.target.dataset.selectId);
        const gesture = pendingSelectionGesture && pendingSelectionGesture.index === checkIndex
          ? pendingSelectionGesture
          : {
              checked: event.target.checked,
              index: checkIndex,
              shiftKey: false,
            };
        pendingSelectionGesture = null;

        if (gesture.shiftKey && state.lastSelectionAnchorIndex >= 0) {
          setSelectionRange(state, state.lastSelectionAnchorIndex, checkIndex, gesture.checked);
        } else {
          setFileSelection(state, fileId, gesture.checked);
        }
        state.lastSelectionAnchorIndex = checkIndex;
        updateSelectionState(deps);
      });
    });

    if (!state.queue.length) {
      state.activeId = null;
      state.detail = null;
      renderDetail();
    }
  }

  function renderDetail(deps) {
    const {
      state,
      modelDisplayNames,
      pathLeaf,
      escapeHtml,
      formatNumber: formatNumberFn,
      scoreCard,
      statusPill,
    } = deps;

    void modelDisplayNames;
    void escapeHtml;
    void scoreCard;
    void statusPill;

    if (!state.detail) {
      document.getElementById("detail-empty").classList.remove("hidden");
      document.getElementById("detail-content").classList.add("hidden");
      updateCompactReviewNavigation(state);
      return;
    }

    const detail = state.detail;
    const aiScore = displayScore(detail);
    const issueText = typeof detail.last_error === "string" ? detail.last_error.trim() : "";
    const issuesNote = document.getElementById("detail-issues-note");
    document.getElementById("detail-empty").classList.add("hidden");
    document.getElementById("detail-content").classList.remove("hidden");
    document.getElementById("detail-title").textContent = pathLeaf(detail.path);
    const queueIndex = state.queue.findIndex((item) => item.id === detail.id);
    const queueCount = state.totalFiles || state.queue.length;
    const reviewPositionNumber = queueIndex >= 0
      ? Math.min(queueCount, (state.page * state.pageSize) + queueIndex + 1)
      : -1;
    const reviewPositionText = reviewPositionNumber >= 0 && queueCount > 0
      ? `${reviewPositionNumber} of ${queueCount}`
      : (queueCount > 0 ? `${queueCount} loaded` : "");
    const reviewPosition = document.getElementById("review-position");
    if (reviewPosition) {
      reviewPosition.textContent = reviewPositionText;
    }
    updateCompactReviewNavigation(state, { queueIndex });
    document.getElementById("detail-path").textContent = detail.path;
    document.getElementById("detail-image").src = `/api/media/preview?id=${detail.id}`;
    document.getElementById("open-original").href = `/api/media/source?id=${detail.id}`;

    document.getElementById("detail-scoreline").textContent = `AI score ${formatNumberFn(aiScore)}`;
    if (issuesNote) {
      if (issueText) {
        issuesNote.textContent = issueText;
        issuesNote.classList.remove("hidden");
      } else {
        issuesNote.textContent = "";
        issuesNote.classList.add("hidden");
      }
    }
  }

  window.ShotSieveReview = {
    getSortRelevantScore,
    renderDetail,
    renderQueue,
    updateSelectionState,
  };
})();
