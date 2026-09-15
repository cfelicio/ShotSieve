(() => {
  const RETRY_CHUNK_SIZE = 500;

  function operationItems(result) {
    return Array.isArray(result?.items)
      ? result.items.filter((item) => item && typeof item === "object")
      : [];
  }

  function operationItemId(item) {
    const fileId = Number(item?.file_id || item?.id);
    return Number.isInteger(fileId) && fileId > 0 ? fileId : null;
  }

  function operationRequestIds(request) {
    const rawIds = request?.payload?.file_ids;
    return Array.isArray(rawIds)
      ? rawIds.map(Number).filter((fileId) => Number.isInteger(fileId) && fileId > 0)
      : [];
  }

  function operationActionLabel(result, request = null) {
    const rawAction = String(result?.action || "").toLowerCase();
    const action = rawAction === "export"
      ? String(request?.payload?.mode || "copy").toLowerCase()
      : (rawAction || String(request?.payload?.mode || "operation").toLowerCase());
    return action === "copy" ? "Copy" : action === "move" ? "Move" : action === "delete" ? "Delete" : "Operation";
  }

  function operationTone(result) {
    const outcome = String(result?.outcome || "").toLowerCase();
    if (!outcome) {
      if (Array.isArray(result?.failed) && result.failed.length) return "error";
      return Number(result?.copied || 0) || Number(result?.moved || 0) || Number(result?.deleted_count || 0)
        ? "success"
        : "warning";
    }
    if (outcome === "success") return result.warnings?.length ? "warning" : "success";
    if (outcome === "partial") return "warning";
    if (outcome === "noop") return "warning";
    return "error";
  }

  function operationDetailsText(result) {
    return JSON.stringify(result || {}, null, 2);
  }

  function mergeOperationResults(previous, next) {
    if (!previous) {
      return {
        ...next,
        items: operationItems(next).map((item) => ({ ...item })),
        warnings: Array.isArray(next?.warnings) ? [...next.warnings] : [],
      };
    }
    if (!next) {
      return previous;
    }

    const itemsById = new Map();
    const itemOrder = [];
    for (const item of [...operationItems(previous), ...operationItems(next)]) {
      const fileId = operationItemId(item);
      if (fileId === null) {
        continue;
      }
      if (!itemsById.has(fileId)) {
        itemOrder.push(fileId);
      }
      itemsById.set(fileId, { ...item, file_id: fileId, id: fileId });
    }
    const items = itemOrder.map((fileId) => itemsById.get(fileId));
    const warnings = [];
    const warningKeys = new Set();
    for (const warning of [...(previous.warnings || []), ...(next.warnings || [])]) {
      const key = JSON.stringify(warning);
      if (!warningKeys.has(key)) {
        warningKeys.add(key);
        warnings.push(warning);
      }
    }

    const merged = { ...previous };
    for (const [key, value] of Object.entries(next)) {
      if (value !== undefined) {
        merged[key] = value;
      }
    }
    Object.assign(merged, {
      items,
      warnings,
      deleted_ids: [...new Set([
        ...(previous.deleted_ids || []),
        ...(next.deleted_ids || []),
      ].map(Number).filter((fileId) => Number.isInteger(fileId) && fileId > 0))],
    });
    if (items.length) {
      merged.completed_count = items.filter((item) => item.outcome === "success").length;
      merged.failed_count = items.filter((item) => item.outcome === "failed").length;
      merged.partial_count = items.filter((item) => ["partial", "uncertain", "catalog_failed"].includes(item.outcome)).length;
      merged.unprocessed_count = items.filter((item) => item.outcome === "unprocessed").length;
      merged.failed = items.filter((item) => item.outcome !== "success");
      merged.safe_retry_ids = items
        .filter((item) => item.retry_safe)
        .map((item) => operationItemId(item))
        .filter((fileId) => fileId !== null);
      merged.copied = items.filter((item) => item.outcome === "success" && item.action === "copy").length;
      merged.moved = items.filter((item) => item.outcome === "success" && item.action === "move").length;
      merged.deleted_count = merged.deleted_ids.length;
    }
    const unresolved = items.some((item) => item.outcome !== "success");
    merged.cancelled = Boolean((next.cancelled || previous.cancelled) && unresolved);
    if (next.job_status === "unknown" || next.outcome === "unknown"
      || previous.job_status === "unknown" || previous.outcome === "unknown") {
      merged.job_status = "unknown";
      merged.outcome = "unknown";
    } else if (merged.cancelled) {
      merged.outcome = "cancelled";
    } else if (merged.partial_count || merged.unprocessed_count) {
      merged.outcome = "partial";
    } else if (merged.failed_count) {
      merged.outcome = merged.completed_count ? "partial" : "failed";
    } else {
      merged.outcome = items.length ? "success" : (next.outcome || previous.outcome || "noop");
    }
    return merged;
  }

  function appendRetryItems(result, fileIds, { action, outcome, error, retrySafe, stage }) {
    if (!fileIds.length) {
      return result;
    }
    const existing = new Set(operationItems(result).map(operationItemId).filter((fileId) => fileId !== null));
    const appended = fileIds
      .filter((fileId) => !existing.has(fileId))
      .map((fileId) => ({
        id: fileId,
        file_id: fileId,
        source: "",
        path: "",
        destination: null,
        action,
        requested_action: action,
        outcome,
        stage,
        error_text: String(error || "Retry was not started."),
        error: String(error || "Retry was not started."),
        retry_safe: retrySafe,
      }));
    return mergeOperationResults(result, {
      action,
      items: appended,
      outcome: outcome === "uncertain" ? "unknown" : "partial",
      job_status: outcome === "uncertain" ? "unknown" : undefined,
      fatal_error: outcome === "uncertain" ? String(error || "Job status is unknown.") : undefined,
    });
  }

  function retainOperationSelection(state, result, request = null) {
    if (result?.job_status === "unknown" || result?.outcome === "unknown") return;
    const items = operationItems(result);
    const operatedIds = new Set(operationRequestIds(request));
    items.forEach((item) => {
      const fileId = operationItemId(item);
      if (fileId !== null) operatedIds.add(fileId);
    });
    const preservedIds = [...state.selectedIds].filter((fileId) => !operatedIds.has(Number(fileId)));
    if (!items.length) {
      state.bulkSelection = null;
      state.selectedIds = request?.payload?.selection ? new Set() : new Set(preservedIds);
      state.lastSelectionAnchorIndex = -1;
      return;
    }

    const remainingIds = items
      .filter((item) => String(item.outcome || "") !== "success")
      .map(operationItemId)
      .filter((fileId) => fileId !== null);
    state.bulkSelection = null;
    state.selectedIds = new Set([...preservedIds, ...remainingIds]);
    state.lastSelectionAnchorIndex = -1;
  }

  function retryableIds(result) {
    return [...new Set((result?.safe_retry_ids || []).map(Number))]
      .filter((id) => Number.isInteger(id) && id > 0);
  }

  async function retrySafeOperation({
    state,
    withBusy,
    fetchSelectionRevision,
    runTrackedOperation,
    refreshWorkspace,
    presentResult,
    showToast,
    confirmRetry = (message) => window.confirm(message),
  }) {
    const result = state.latestOperationResult;
    const request = state.latestOperationRequest;
    const safeIds = retryableIds(result);
    if (!safeIds.length || !request) {
      showToast("There are no safely retryable files.", "error");
      return;
    }

    const payload = { ...(request.payload || {}) };
    const originalSelection = payload.selection;
    const pageSelection = payload.page_selection
      ? { ...payload.page_selection }
      : originalSelection
        ? { ...originalSelection }
        : null;
    if (!pageSelection) {
      throw new Error("The original review scope is unavailable. Refresh the results and select again.");
    }
    delete pageSelection.selection_revision;
    delete pageSelection.exclude_file_ids;
    delete payload.selection;
    delete payload.exclude_file_ids;
    payload.page_selection = pageSelection;
    payload.file_ids = safeIds;
    payload.count = safeIds.length;
    const mode = String(payload.mode || result.action || "").toLowerCase();
    if ((mode === "move" || mode === "delete") && !confirmRetry(`Retry ${mode} for ${safeIds.length} file(s)?`)) {
      return;
    }

    let aggregate = mergeOperationResults(null, result);
    let currentChunkIds = [];
    let currentChunkEnd = 0;
    let retryStopped = false;

    const stopBeforeChunk = (error) => {
      const message = `Retry could not verify the current selection: ${String(error?.message || error)}`;
      aggregate = appendRetryItems(
        aggregate,
        currentChunkIds,
        { action: mode, outcome: "unprocessed", error: message, retrySafe: true, stage: "not_started" },
      );
      aggregate = appendRetryItems(
        aggregate,
        safeIds.slice(currentChunkEnd),
        { action: mode, outcome: "unprocessed", error: message, retrySafe: true, stage: "not_started" },
      );
      aggregate.outcome = "partial";
      aggregate.fatal_error = message;
      state.latestOperationResult = aggregate;
      presentResult(aggregate, request);
      retryStopped = true;
    };

    const finishRetry = async () => {
      state.latestOperationResult = aggregate;
      presentResult(aggregate, {
        ...request,
        payload: { ...payload, file_ids: safeIds, count: safeIds.length },
      });
      await refreshWorkspace();
    };

    await withBusy(`Retrying ${safeIds.length} file(s)...`, async () => {
      for (let offset = 0; offset < safeIds.length; offset += RETRY_CHUNK_SIZE) {
        const chunkIds = safeIds.slice(offset, offset + RETRY_CHUNK_SIZE);
        currentChunkIds = chunkIds;
        currentChunkEnd = offset + chunkIds.length;
        const chunkPayload = { ...payload, file_ids: chunkIds, count: chunkIds.length };
        try {
          chunkPayload.selection_revision = await fetchSelectionRevision(pageSelection);
        } catch (error) {
          stopBeforeChunk(error);
          break;
        }
        if (!chunkPayload.selection_revision) {
          stopBeforeChunk(new Error("the review selection changed or is still loading"));
          break;
        }
        try {
          const next = await runTrackedOperation({
            startPath: request.startPath,
            payload: chunkPayload,
            fallbackLabel: request.fallbackLabel,
            failureMessage: request.failureMessage,
          });
          if (next) {
            aggregate = mergeOperationResults(aggregate, next);
          }
          if (next?.job_status === "failed" || ["cancelled", "unknown"].includes(String(next?.outcome || ""))) {
            aggregate = appendRetryItems(
              aggregate,
              safeIds.slice(currentChunkEnd),
              { action: mode, outcome: "unprocessed", error: next.fatal_error || next.job_error || "Retry stopped.", retrySafe: true, stage: "not_started" },
            );
            retryStopped = true;
            break;
          }
        } catch (error) {
          if (error?.name === "AbortError") {
            throw error;
          }
          const unknown = state.latestOperationResult?.job_status === "unknown";
          aggregate = unknown
            ? mergeOperationResults(aggregate, state.latestOperationResult)
            : appendRetryItems(
              aggregate,
              currentChunkIds,
              { action: mode, outcome: "uncertain", error: error.message || error, retrySafe: false, stage: "status_unknown" },
            );
          aggregate = appendRetryItems(
            aggregate,
            safeIds.slice(currentChunkEnd),
            { action: mode, outcome: "unprocessed", error: error.message || error, retrySafe: true, stage: "not_started" },
          );
          aggregate.job_status = "unknown";
          aggregate.outcome = "unknown";
          aggregate.fatal_error = String(error.message || error);
          state.latestOperationResult = aggregate;
          presentResult(aggregate, request);
          retryStopped = true;
          return;
        }
      }
      if (!retryStopped) {
        await finishRetry();
      } else if (state.recoveryJob?.kind !== "operation") {
        await finishRetry();
      }
    }, {
      operationType: "operation",
      onCancelled: async ({ confirmed, result: cancelledResult }) => {
        if (confirmed && cancelledResult) {
          aggregate = mergeOperationResults(aggregate, cancelledResult);
        } else {
          aggregate = appendRetryItems(
            aggregate,
            currentChunkIds,
            { action: mode, outcome: "uncertain", error: "Cancellation status is unknown.", retrySafe: false, stage: "status_unknown" },
          );
          aggregate.job_status = "unknown";
          aggregate.outcome = "unknown";
          aggregate.fatal_error = "Cancellation was requested, but the server has not confirmed a terminal state.";
        }
        aggregate = appendRetryItems(
          aggregate,
          safeIds.slice(currentChunkEnd),
          { action: mode, outcome: "unprocessed", error: confirmed ? "Retry was cancelled." : "Cancellation status is unknown.", retrySafe: true, stage: "not_started" },
        );
        state.latestOperationResult = aggregate;
        presentResult(aggregate, request);
        if (confirmed) {
          await refreshWorkspace();
        }
      },
    });
    return state.latestOperationResult;
  }

  window.ShotSieveWorkflowResults = {
    RETRY_CHUNK_SIZE,
    operationActionLabel,
    operationItems,
    operationItemId,
    operationRequestIds,
    operationTone,
    operationDetailsText,
    mergeOperationResults,
    appendRetryItems,
    retainOperationSelection,
    retryableIds,
    retrySafeOperation,
  };
})();
