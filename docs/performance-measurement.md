# Performance measurement

This guide explains how to measure ShotSieve responsiveness without uploading a photo library or changing normal catalog behavior.

## What to measure

Separate these potential sources of perceived slowness:

1. SQLite catalog and Review queries.
2. Filesystem discovery and directory traversal.
3. Preview decoding and generation.
4. Learned-IQA model startup and per-image scoring.

Do not tune indexes, change pagination, or split catalogs until measurements identify the dominant cost on the target machine.

## Do we need real photos?

**Not for the database baseline.** `tests/test_performance_baseline.py` creates 100,000 cached/scored rows in one active library plus 10,000 additional cached rows. It measures global and active-library Review behavior deterministically without reading image files, generating previews, or initializing an IQA model.

**Yes, for filesystem, preview, and model measurements.** Use a local representative sample to measure directory walking, image decoding, preview generation, and learned-IQA execution. A public or uploaded data source is not required and is usually less representative than the user's actual camera workflow.

Use a copy of photos rather than originals if possible. Benchmark output should contain only aggregate timings and counts—never file paths, filenames, EXIF values, or image content.

## Prepare a representative local sample

Create or choose a stable, non-production benchmark folder with these cohorts:

| Cohort | Suggested size | Why it matters |
| --- | ---: | --- |
| Warm active library | 100–500 photos | Measures the common small-current-folder workflow. |
| Mixed formats | At least several of each JPEG, RAW, HEIF/HEIC, TIFF, and PNG used in practice | Captures decoder and preview differences. |
| Changed/new files | 10–20 files | Verifies stale detection and incremental work. |
| Existing large catalog | 100,000+ cached/scored rows in one active library, real or synthetic | Captures deep active-library and global Review navigation. |
| Ignored-tree fixture | One large directory tree that will later be excluded | Establishes the traversal baseline before ignore rules are implemented. |

For repeatability:

- Do not edit, rename, move, or delete benchmark inputs between warm runs.
- Keep the database and preview directory on the same type of storage used in normal operation (for example, local SSD versus external/network storage).
- Record whether the benchmark uses an existing warm cache or a new data directory.
- Use a new dedicated `--data-dir` when a clean/cold catalog is required. Never point a clean-run experiment at a catalog you want to preserve.

## Run the synthetic database and navigation baseline

The opt-in test is skipped during normal test runs. Enable it explicitly.

PowerShell:

```powershell
$env:SHOTSIEVE_RUN_PERFORMANCE_BASELINE = "1"
python -m pytest tests/test_performance_baseline.py -s -q
Remove-Item Env:SHOTSIEVE_RUN_PERFORMANCE_BASELINE
```

POSIX shells:

```bash
SHOTSIEVE_RUN_PERFORMANCE_BASELINE=1 python -m pytest tests/test_performance_baseline.py -s -q
```

The test prints one JSON report containing the named machine, fixture size, setup timings, and a Review-navigation matrix. The matrix keeps list/count/revision timings separate from matching `EXPLAIN QUERY PLAN` details for global and active scopes. It samples early, middle, and deep pages for score, path, and date sorts, plus score, format, review-state, metadata, and path filters. It does not benchmark real image analysis or impose absolute timing thresholds.

The baseline's active-root measurements answer how a 100,000-row current library behaves, while the global measurements include the additional cached rows. Compare the same scope, sort, page, and filter on the same named machine. Treat a slow result as evidence to investigate; do not change pagination or add an index until the corresponding query plan identifies a clear bottleneck.

## Measure a local photo folder

`scripts/measure_performance.py` performs a safe, reproducible local measurement:

- two metadata-only scans (first and warm re-scan);
- disposable Review/score SQL queries against the scanned metadata, including global and active early/middle/deep navigation for score/path/date sorts and representative filters;
- separate Review list/count/revision timings and matching query plans;
- a small, format-stratified preview sample;
- an environment/capability report that states whether HEIF, RAW, and learned-IQA dependencies are available.

It never runs learned-IQA inference, and it does not include source paths or filenames in its JSON report. Use a disposable data directory and a local, ignored report location.

PowerShell example:

```powershell
python scripts/measure_performance.py C:\Photos\Sample `
  --data-dir .\.performance-data\sample `
  --output .\performance-reports\sample-performance.json
```

The repository ignores `.performance-data/` and `performance-reports/` so generated databases, previews, and measurement results are not committed.

## Capture application query timings

Set `SHOTSIEVE_PERFORMANCE_LOGGING=1` before launching ShotSieve. This enables debug-level JSON duration events for catalog overview, Review list/count/revision, and score-row fetch/count queries. It does not change scoring behavior or send data anywhere.

PowerShell:

```powershell
$env:SHOTSIEVE_PERFORMANCE_LOGGING = "1"
shotsieve-desktop --data-dir .\benchmark-data
Remove-Item Env:SHOTSIEVE_PERFORMANCE_LOGGING
```

Configure Python logging to show debug records when collecting these events. If debug logging is not configured, the environment variable remains harmless and no extra output is produced.

For each sample, record these workflows separately:

1. **Cold scan:** empty dedicated data directory; scan with preview generation enabled.
2. **Warm re-scan:** repeat without changing files; verify previews are reused and note duration.
3. **Incremental scan:** alter or add a small changed/new cohort; record only the expected files being regenerated.
4. **Cold score:** start with unscored cached rows; record model-load time separately from scoring progress.
5. **Warm score:** repeat with unchanged files and the same model; verify no unnecessary learned-IQA work occurs.
6. **Review switch:** open Review for a 100–500 photo active folder while the catalog contains 100,000+ cached/scored rows; record overview, first queue load, and navigation responsiveness.
7. **Deep Review navigation:** compare global and active scopes at early, middle, and deep offsets for score, path, and date sorts. Repeat count, selection-revision, and an early page for representative score, format, review-state, metadata, and path filters. Review renders a bounded page (60 photos by default) and lazy-loads queue thumbnails; investigate list/count/revision timing and the matching query plan before assuming the browser loaded the catalog.
8. **Compare:** if comparison is a supported workflow, record model loading and per-image execution separately.

For the current I11 baseline, use the synthetic 100,000-active-row fixture above or the real-photo utility's `review_navigation` report. Both reports include global and active scopes, early/middle/deep page offsets, score/path/date sorts, representative filters, and separate timings with matching query-plan records.

### Current Review query behavior

ShotSieve creates score-order indexes during normal database initialization for the default lowest-AI-score sort and the score-descending sort. These improve deep-page Review queries without changing the stable file-ID tie-breaker.

Single-photo Keep, Reject, and Reset actions update the visible Review page from the server response instead of fetching the queue again. Summary totals still refresh, so this is not a substitute for measuring page navigation or a bulk action.

## Record the environment with every run

Include the machine name, operating system, CPU/RAM/GPU/runtime, Python and SQLite versions, ShotSieve commit, selected model/device/resource profile/batch size, storage medium, source format counts/bytes, and cold/warm state. Do not compare timings from different machines or cache states as though they were equivalent.

## Measurement completion checklist

- [ ] Synthetic 100,000-active-row baseline output with global/active list, count, revision timings, and query plans.
- [ ] Cold and warm scan timings, including preview-generation counts.
- [ ] Incremental scan timing with changed/new files.
- [ ] Cold and warm scoring timings, with model startup separated from per-image scoring.
- [ ] Review first-page/count/revision timings for a small active library within a large catalog.
- [ ] If large-library Review is a reported concern, early-, middle-, and deep-page timings for score/path/date sorts in both relevant scopes.
- [ ] Format-specific observations for every format actually used by the workflow.
- [ ] A written finding identifying the largest measured bottleneck and whether it is SQL, filesystem traversal, preview/decode work, model initialization, or model inference.
- [ ] Query plans captured alongside every measured Review list/count/revision operation, especially any slow sort or filter.

Choose the next engineering change from the measured bottleneck: scope/UI behavior, database query work, scanner/preview work, or runtime/model work.

## Safety notes

- Use a dedicated benchmark data directory for cold runs; do not reset a real catalog just to reproduce a test.
- The synthetic benchmark seeds only SQLite metadata, so it cannot delete or modify source photos.
- Do not copy benchmark logs containing user paths into issues or public documentation.
- Keep destructive Review actions out of performance runs unless that specific workflow is being safely tested against disposable copies.
