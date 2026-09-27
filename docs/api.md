# Application interfaces

This page describes the interfaces used by the local review UI and by
integration tests. It is a local application protocol, not a supported remote
or public Internet API.

## Desktop CLI

The supported entry point is `shotsieve-desktop`. Its options are documented
in [configuration.md](configuration.md). The most useful diagnostics are:

```bash
shotsieve-desktop --help
shotsieve-desktop --check-runtime
shotsieve-desktop --no-browser --port 9001
```

The `--check-runtime` command exits after checking native Torch/TorchVision
operations and learned imports. It does not prepare model weights.

## HTTP basics

The default base URL is `http://127.0.0.1:8765`. JSON endpoints use the
standard GET/POST methods shown below. Media routes return the requested local
file bytes rather than JSON.

The server accepts only loopback clients with a loopback Host header matching
the server port. For POST requests, an `Origin` header, when present, must
also be an `http` or `https` loopback origin on that port. An absent Origin is
allowed. Do not treat `--host 0.0.0.0` as a supported way to expose the app
to a LAN.

## Read routes

| Route | Contract |
| --- | --- |
| `/api/overview` | Catalog summary; optional `root`. |
| `/api/options` | UI/runtime options; optional `resource_profile`. |
| `/api/analysis-diagnostics` | Scan/analysis diagnostics; optional `root` and `limit` (default 100, max 500). |
| `/api/files` and `/api/files/count` | Filtered catalog rows or only the count. Filters include root, review state, query, score, format, dimensions, megapixels, size, and metadata. `/api/files` defaults to 60 rows and caps `limit` at 500. |
| `/api/file?id=ID` | One catalog file detail. |
| `/api/review/file-ids` | IDs for a marked selection (`approved`, `rejected`, or `none`); requires `marked` and supports root/selection pagination. |
| `/api/review/decisions.csv` | CSV export of decisions; requires `root` and `decision=approved`, `rejected`, or `both`. |
| `/api/fs/roots` and `/api/fs/list` | Local filesystem browsing for existing roots/directories. |
| `/api/media/preview` and `/api/media/source` | Preview/source bytes for a positive catalog `id`; paths are safety-checked. |
| `/api/cache/missing/preview` | Preview of missing managed-cache entries; requires `root`. |
| `/api/*/status` and `/api/*/result` | Status or completed result for the job families below, using `job_id`. |

`/api/files` responses include `items`, `total`, and `selection_revision`.
Selection-bearing routes use that revision to detect a catalog change before a
bulk mutation.

## Write routes

### Review and file actions

`POST /api/review` updates one file. Its payload contains `file_id` and may
include a decision state plus delete/export mark changes.

`POST /api/review/batch` accepts either a page-level selection (`file_ids`,
page selection, and `selection_revision`) or a filter/scope selection plus the
revision. A stale revision is rejected instead of applying the operation to a
different catalog.

`POST /api/files/open` accepts `{ "file_id": ID }` and asks the local file
manager to reveal the source file. The response reports `opened`, `path`, and
the platform `method`.

File deletion and export each have a synchronous compatibility route and an
asynchronous route:

```text
POST /api/files/delete
POST /api/files/delete/start
POST /api/files/export
POST /api/files/export/start
```

Use the `/start` routes for UI-sized or larger selections. Their response
contains a `job_id` and `status: "running"`; the operation result retains
per-file outcomes and uncertainty information.

### Analysis and cache jobs

The asynchronous start routes are:

```text
POST /api/scan/start
POST /api/score/start
POST /api/compare-models/start
POST /api/models/prepare/start
POST /api/cache/clear/start
```

`POST /api/cache/missing/apply` applies reviewed, root-scoped missing-entry
cleanup. `/api/cache/clear` remains as a synchronous compatibility route.
Estimate routes are `/api/score-estimate` and `/api/compare-estimate`.

## Job lifecycle

Analysis, model-preparation, cache, delete, and export jobs return a process-
local job ID. Poll the matching status route:

```text
POST /api/score/start
GET  /api/score/status?job_id=JOB_ID
GET  /api/score/result?job_id=JOB_ID
POST /api/score/cancel?job_id=JOB_ID
```

The job status is `running`, `completed`, or `failed` and may include
`progress`, `summary`, `error`, and `elapsed_seconds`. A result request returns
the summary after completion, a conflict while the job is still running, and a
not-found response for an unknown or evicted process-local job. Cancel routes
return the job ID and whether cancellation was accepted; workers still report
their final operation state.

The same status/result/cancel pattern applies to `scan`, `compare-models`,
`operations`, and `models/prepare` with their corresponding route prefixes.

## Errors and compatibility

Malformed requests and unavailable optional learned-runtime operations normally
produce `400`. Loopback/Host/Origin failures produce `403`; request-body
timeouts produce `408`; database contention produces `409`; unexpected
server errors produce `500`. A missing catalog file, directory, media path, or
job is reported as not found.

The synchronous delete, export, and cache-clear routes remain for compatibility
with existing clients. New integrations should preserve the selection revision
contract and prefer the asynchronous routes for work that can take noticeable
time.

The stable JavaScript workflow facade is
`window.ShotSieveWorkflows`. Its names and the static script order are
documented in [frontend-workflows.md](frontend-workflows.md). Internal Python
helpers, route dependency adapters, and functions such as
`build_review_server` are implementation/test entry points, not a promised
general embedding API.
