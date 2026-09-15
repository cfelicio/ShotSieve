# Frontend workflow ownership

The browser loads the workflow modules in this order:

1. `app-workflow-polling.js`
2. `app-workflow-operation-results.js`
3. `app-workflow-compare.js`
4. `app-workflow-export-ui.js`
5. `app-workflow-export.js`
6. `app-workflow-library-operations.js`
7. `app-workflow-library-analysis.js`
8. `app-workflow-library-browser.js`
9. `app-workflow-library.js`
10. `app-workflows.js`

`app-workflows.js` is the composition-only facade. It creates the module
instances, wires their dependencies, and exposes the historical
`window.ShotSieveWorkflows` surface. It does not contain feature workflow
implementations.

## Public facade

The facade combines these module-owned methods:

| Owner | Public methods |
|---|---|
| Library façade (operations / analysis / browser) | `runTrackedOperation`, `checkTrackedJob`, `checkTrackedOperation`, `resetReviewFiltersForAnalyze`, `runScan`, `runScore`, `prepareSelectedModel`, `installAiSupport`, `analyzeLibrary`, `renderLibraryRoots`, `downloadDecisionCsv`, `installDecisionCsvEvents`, `clearCache`, `reviewMissingEntries`, `deleteSelectedFiles`, `navigateSelection`, `openOriginalFile`, `openBrowser`, `browseDirectory`, `chooseBrowserPath`, `handleError` |
| Export core | `saveReview`, `reviewDecisionPayload`, `hasActiveSelection`, `clearActiveSelection`, `currentSelectionRevision`, `fetchReviewStateSelectionRevision`, `activeSelectionRequest`, `saveReviewDecision`, `nextReviewCandidateId`, `saveReviewDecisionWithOptions`, `runBatchReview`, `runBatchReviewDecision`, `fetchMarkedFileIds`, `summarizeExportResult`, `presentOperationResult`, `operationTone`, `mergeOperationResults`, `retrySafeOperation`, `fetchSelectionRevision` |
| Export UI | `buildSelectedExportRequest`, `openExportDialog`, `installExportDialogEvents`, `installRejectedActionEvents`, `installOperationResultEvents` |
| Compare | `compareRowSortChoices`, `syncCompareSortControls`, `comparisonFailureText`, `comparisonFailureDetails`, `comparisonFailureSummaryText`, `renderComparisonWarnings`, `renderComparisonSummary`, `renderComparisonResults`, `renderComparisonModelOptions`, `runModelComparison` |
| Polling aliases | `fetchCompareJobStatus`, `fetchCompareJobResult`, `pollCompareJob` |

Library and export workflows have a mutual dependency for operation and
selection handling. The facade supplies shared bridge objects while composing
them so each module sees the completed peer at call time. Keep public names
stable when changing ownership; callers in `app.js` and `app-events.js` use the
facade rather than importing feature modules directly.

The library façade composes three domain factories: operation polling and
maintenance, scan/score/model analysis, and library/browser navigation. Export
result shape, retry aggregation, and retry-safety rules are in the injected
`ShotSieveWorkflowResults` utility; export and rejected-file dialog handlers
remain in `app-workflow-export-ui.js`. These modules communicate through the
same injected workflow bridges, so the historical `ShotSieveWorkflows` names
and script order remain stable.

## Scan and score job lifecycle

`app-workflow-library-operations.js` owns the shared `runTrackedJob` seam for
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
