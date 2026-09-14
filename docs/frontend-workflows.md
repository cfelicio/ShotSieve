# Frontend workflow ownership

The browser loads the workflow modules in this order:

1. `app-workflow-polling.js`
2. `app-workflow-compare.js`
3. `app-workflow-export.js`
4. `app-workflow-library.js`
5. `app-workflows.js`

`app-workflows.js` is the composition-only facade. It creates the module
instances, wires their dependencies, and exposes the historical
`window.ShotSieveWorkflows` surface. It does not contain feature workflow
implementations.

## Public facade

The facade combines these module-owned methods:

| Owner | Public methods |
|---|---|
| Library | `runTrackedOperation`, `checkTrackedJob`, `checkTrackedOperation`, `resetReviewFiltersForAnalyze`, `runScan`, `runScore`, `prepareSelectedModel`, `installAiSupport`, `analyzeLibrary`, `renderLibraryRoots`, `downloadDecisionCsv`, `installDecisionCsvEvents`, `clearCache`, `reviewMissingEntries`, `deleteSelectedFiles`, `navigateSelection`, `openOriginalFile`, `openBrowser`, `browseDirectory`, `chooseBrowserPath`, `handleError` |
| Export | `saveReview`, `reviewDecisionPayload`, `hasActiveSelection`, `clearActiveSelection`, `currentSelectionRevision`, `fetchReviewStateSelectionRevision`, `activeSelectionRequest`, `saveReviewDecision`, `nextReviewCandidateId`, `saveReviewDecisionWithOptions`, `runBatchReview`, `runBatchReviewDecision`, `fetchMarkedFileIds`, `summarizeExportResult`, `buildSelectedExportRequest`, `openExportDialog`, `installExportDialogEvents`, `installRejectedActionEvents`, `presentOperationResult`, `operationTone`, `mergeOperationResults`, `retrySafeOperation`, `fetchSelectionRevision`, `installOperationResultEvents` |
| Compare | `compareRowSortChoices`, `syncCompareSortControls`, `comparisonFailureText`, `comparisonFailureDetails`, `comparisonFailureSummaryText`, `renderComparisonWarnings`, `renderComparisonSummary`, `renderComparisonResults`, `renderComparisonModelOptions`, `runModelComparison` |
| Polling aliases | `fetchCompareJobStatus`, `fetchCompareJobResult`, `pollCompareJob` |

Library and export workflows have a mutual dependency for operation and
selection handling. The facade supplies shared bridge objects while composing
them so each module sees the completed peer at call time. Keep public names
stable when changing ownership; callers in `app.js` and `app-events.js` use the
facade rather than importing feature modules directly.

## Scan and score job lifecycle

`app-workflow-library.js` uses its private `runTrackedJob` seam for the shared
start, job identity, tracking, polling, successful cleanup, and unresolved-job
handling used by scan, score, and file-operation workflows. The busy controller
still owns server-side cancellation and recovery presentation: an aborted job
keeps its identity long enough to request cancellation, while a lost
non-abort status request remains available to **Check status**.

`runScan` and `runScore` retain their own estimate requests, start payloads,
poller options, progress phases, result messages, diagnostics, and workspace
refresh behavior. Changes to those workflow-specific contracts should stay in
the owning function; changes to tracked-job cleanup or recovery belong in the
shared seam and its lifecycle tests.
