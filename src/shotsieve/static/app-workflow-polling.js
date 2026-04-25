(() => {
  function percentFromCounts(processed, total) {
    const totalValue = Number(total || 0);
    if (totalValue <= 0) {
      return null;
    }
    const doneValue = Math.max(0, Math.min(totalValue, Number(processed || 0)));
    return (doneValue / totalValue) * 100;
  }

  function pipelineOverallPercent(percent, pipeline) {
    if (!pipeline || !Number.isFinite(Number(pipeline.totalSteps)) || Number(pipeline.totalSteps) <= 0) {
      return percent;
    }
    const totalSteps = Number(pipeline.totalSteps);
    const stepIndex = Math.max(1, Math.min(totalSteps, Number(pipeline.stepIndex || 1)));
    const stepBase = ((stepIndex - 1) / totalSteps) * 100;
    if (percent === null || percent === undefined || Number.isNaN(Number(percent))) {
      return stepBase;
    }
    const clamped = Math.max(0, Math.min(100, Number(percent)));
    return stepBase + (clamped / totalSteps);
  }

  function resolvePhase(phaseValue, phaseMap, fallbackKey) {
    const normalized = String(phaseValue || fallbackKey).toLowerCase();
    return phaseMap[normalized] || phaseMap[fallbackKey];
  }

  function createJobPollers({ state, api, busy, progress }) {
    const { fetchJson } = api;
    const { setBusyMessage, setBusyPhaseProgress, setBusyProgress } = busy;
    const {
      compareProgressMessage,
      compareProgressPercent,
      scanProgressMessage,
      scanProgressPercent,
      scoreProgressMessage,
      scoreProgressPercent,
    } = progress;

    async function fetchWithRetry(url, options = {}, maxRetries = 5) {
      let lastError = null;
      for (let attempt = 0; attempt <= maxRetries; attempt += 1) {
        try {
          return await fetchJson(url, options);
        } catch (error) {
          if (error?.name === "AbortError") {
            throw error;
          }
          const message = String(error?.message || "");
          const isTransient = message === "Failed to fetch"
            || message.includes("NetworkError")
            || message.includes("network")
            || (error instanceof TypeError);
          if (!isTransient || attempt >= maxRetries) {
            throw error;
          }
          lastError = error;
          const delay = Math.min(500 * Math.pow(2, attempt), 8000);
          await new Promise((resolve) => window.setTimeout(resolve, delay));
        }
      }
      throw lastError;
    }

    function createStatusFetcher(path) {
      return function fetchStatus(jobId) {
        return fetchWithRetry(`${path}?job_id=${encodeURIComponent(jobId)}`, {
          signal: state.abortController?.signal,
        });
      };
    }

    function createResultFetcher(path) {
      return function fetchResult(jobId) {
        return fetchWithRetry(`${path}?job_id=${encodeURIComponent(jobId)}`, {
          signal: state.abortController?.signal,
        });
      };
    }

    async function pollJob({ jobId, fetchStatus, fetchResult, onProgress, progressMessage, progressTotal, failureMessage }) {
      while (true) {
        const status = await fetchStatus(jobId);
        const progressPayload = status?.progress && typeof status.progress === "object" ? status.progress : null;
        const progressState = onProgress({ progress: progressPayload, status });

        if (progressState?.overallPercent !== null && progressState?.overallPercent !== undefined) {
          setBusyProgress(Math.min(99, progressState.overallPercent));
        }

        if (progressState?.phaseState) {
          setBusyPhaseProgress(progressState.phaseState);
        }

        setBusyMessage(
          progressMessage(progressPayload, Number(status?.elapsed_seconds || 0), progressTotal),
        );

        if (status?.status === "completed") {
          return await fetchResult(jobId);
        }

        if (status?.status === "failed") {
          throw new Error(status.error || failureMessage);
        }

        if (state.abortController?.signal?.aborted) {
          throw new DOMException("Aborted", "AbortError");
        }

        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
    }

    const fetchCompareJobStatus = createStatusFetcher("/api/compare-models/status");
    const fetchCompareJobResult = createResultFetcher("/api/compare-models/result");
    const fetchScoreJobStatus = createStatusFetcher("/api/score/status");
    const fetchScoreJobResult = createResultFetcher("/api/score/result");
    const fetchScanJobStatus = createStatusFetcher("/api/scan/status");
    const fetchScanJobResult = createResultFetcher("/api/scan/result");

    async function pollScoreJob(scoreJobId, { rowsTotal, pipeline = null }) {
      const phaseMap = {
        loading: { label: "Loading model", defaultPhaseIndex: 1, percent: () => 0 },
        generating_previews: {
          label: "Generating previews",
          defaultPhaseIndex: 2,
          percent: ({ progress }) => percentFromCounts(progress?.files_processed, progress?.files_total),
        },
        scoring: {
          label: "Model scoring",
          defaultPhaseIndex: 3,
          percent: ({ progress }) => {
            const scoringTotal = rowsTotal && rowsTotal > 0 ? rowsTotal : Number(progress?.files_total || 0);
            return percentFromCounts(progress?.files_processed, scoringTotal);
          },
        },
      };

      return pollJob({
        jobId: scoreJobId,
        fetchStatus: fetchScoreJobStatus,
        fetchResult: fetchScoreJobResult,
        progressMessage: scoreProgressMessage,
        progressTotal: rowsTotal,
        failureMessage: "Scoring pass failed.",
        onProgress: ({ progress, status }) => {
          const phaseValue = String(progress?.phase || "loading").toLowerCase();
          const phase = resolvePhase(phaseValue, phaseMap, "loading");
          const phasePercent = phase.percent({ progress });
          const operationPercent = scoreProgressPercent(progress);

          let overallPercent = operationPercent;
          let phaseIndex = phase.defaultPhaseIndex;
          let phaseCount = 3;
          if (pipeline) {
            const loadingStepIndex = Math.max(1, Number(pipeline.stepIndex || 1));
            const scoringStepIndex = Math.min(Number(pipeline.totalSteps || loadingStepIndex), loadingStepIndex + 1);
            phaseIndex = phaseValue === "scoring" ? scoringStepIndex : loadingStepIndex;
            phaseCount = pipeline.totalSteps;
            const activeStepPercent = phaseValue === "loading" ? 0 : (phasePercent ?? operationPercent);
            overallPercent = pipelineOverallPercent(activeStepPercent, {
              stepIndex: phaseIndex,
              totalSteps: pipeline.totalSteps,
            });
          }

          return {
            overallPercent,
            phaseState: {
              percent: phasePercent,
              phaseIndex,
              phaseCount,
              phaseLabel: phase.label,
            },
          };
        },
      });
    }

    async function pollCompareJob(compareJobId, { rowsTotal, pipeline = null }) {
      return pollJob({
        jobId: compareJobId,
        fetchStatus: fetchCompareJobStatus,
        fetchResult: fetchCompareJobResult,
        progressMessage: compareProgressMessage,
        progressTotal: rowsTotal,
        failureMessage: "Model comparison failed.",
        onProgress: ({ progress }) => {
          const phaseValue = String(progress?.phase || "loading").toLowerCase();
          const operationPercent = compareProgressPercent(progress);
          const modelCount = Math.max(1, Number(progress?.model_count || 1));
          const modelIndex = Math.max(1, Math.min(modelCount, Number(progress?.model_index || 1)));
          const modelName = progress?.model_name || "model";

          let phasePercent = null;
          if (phaseValue === "loading") {
            phasePercent = 0;
          } else if (phaseValue === "generating_previews") {
            phasePercent = percentFromCounts(progress?.files_processed, progress?.files_total);
          } else if (phaseValue === "scoring") {
            const scoringTotal = rowsTotal && rowsTotal > 0 ? rowsTotal : Number(progress?.files_total || 0);
            phasePercent = percentFromCounts(progress?.files_processed, scoringTotal);
          }

          let overallPercent = operationPercent;
          let phaseIndex = phaseValue === "scoring" ? 3 : phaseValue === "generating_previews" ? 2 : 1;
          let phaseCount = 3;
          if (pipeline) {
            phaseIndex = phaseValue === "scoring" ? 3 : 2;
            phaseCount = pipeline.totalSteps;
            const activeStepPercent = phaseValue === "loading" ? 0 : (phasePercent ?? operationPercent);
            overallPercent = pipelineOverallPercent(activeStepPercent, {
              stepIndex: phaseIndex,
              totalSteps: pipeline.totalSteps,
            });
          }

          const phaseLabel = phaseValue === "scoring"
            ? `Model scoring (${modelIndex}/${modelCount} ${modelName})`
            : phaseValue === "generating_previews"
              ? (pipeline ? "Loading models (generating previews)" : "Generating previews")
              : `Loading models (${modelIndex}/${modelCount} ${modelName})`;

          return {
            overallPercent,
            phaseState: {
              percent: phasePercent,
              phaseIndex,
              phaseCount,
              phaseLabel,
            },
          };
        },
      });
    }

    async function pollScanJob(scanJobId, { filesTotalRef, pipeline = null }) {
      return pollJob({
        jobId: scanJobId,
        fetchStatus: fetchScanJobStatus,
        fetchResult: fetchScanJobResult,
        progressMessage: scanProgressMessage,
        progressTotal: Number(filesTotalRef?.value || 0) || null,
        failureMessage: "Scan failed.",
        onProgress: ({ progress }) => {
          const filesTotal = Number(filesTotalRef?.value || 0) || null;
          const phaseValue = String(progress?.phase || "indexing").toLowerCase();
          const percent = scanProgressPercent(progress, filesTotal);
          const overallPercent = pipelineOverallPercent(percent, pipeline);
          const phaseLabel = phaseValue === "scanning"
            ? (pipeline ? "Scanning library" : "Scanning files")
            : phaseValue === "failed"
              ? "Scan failed"
              : "Indexing files";
          return {
            overallPercent,
            phaseState: {
              percent,
              phaseIndex: pipeline ? pipeline.stepIndex : (phaseValue === "scanning" || phaseValue === "failed" ? 2 : 1),
              phaseCount: pipeline ? pipeline.totalSteps : 2,
              phaseLabel,
            },
          };
        },
      });
    }

    return {
      fetchCompareJobResult,
      fetchCompareJobStatus,
      fetchScanJobResult,
      fetchScanJobStatus,
      fetchScoreJobResult,
      fetchScoreJobStatus,
      percentFromCounts,
      pipelineOverallPercent,
      pollCompareJob,
      pollJob,
      pollScanJob,
      pollScoreJob,
      resolvePhase,
    };
  }

  window.ShotSieveWorkflowPolling = {
    createJobPollers,
    percentFromCounts,
    pipelineOverallPercent,
    resolvePhase,
  };
})();
