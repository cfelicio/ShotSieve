# Architecture

ShotSieve is a local application made of a Python process, a local SQLite
catalog, managed preview files, optional model/runtime caches, and a bundled
browser UI. The normal entry point is the `shotsieve-desktop` console script.
It starts a standard-library HTTP server, serves the packaged static assets,
and handles the UI's local JSON and media requests.

## Data flow

1. The desktop entry point selects the data directory and model-cache roots,
   prepares any configured runtime sidecar, and starts the review server.
2. The browser UI calls the `/api` routes to enumerate files, scan roots,
   score files, prepare models, compare models, and update review state.
3. Catalog and review operations use SQLite. Image previews, runtime sidecars,
   logs, and model-preparation state are stored outside the database under the
   selected data directory.
4. Long-running analysis and file operations run in process-local worker
   registries. The UI polls their status and result routes.

The application is local-first. The default bind address is loopback, and
request validation also requires a loopback Host header. A `--host` value
changes where the process listens, but does not turn the HTTP protocol into a
remote or LAN service.

## Ownership map

| Area | Main implementation |
| --- | --- |
| CLI and startup | `src/shotsieve/desktop.py` |
| Runtime-pack bootstrap | `src/shotsieve/bootstrap.py`, `bootstrap_assets.py`, and `bootstrap_sidecar.py` |
| HTTP server and route composition | `src/shotsieve/web.py` and `web_routes.py` |
| Route families | `web_route_*.py` |
| Catalog, migrations, and review state | `catalog.py`, `database.py`, and review modules |
| Preview/cache ownership | `config.py`, `preview.py`, and cache modules |
| Learned-IQA catalog and runtime | `learned_iqa_catalog.py`, `learned_iqa_runtime.py`, and `model_assets.py` |
| Browser workflows | `src/shotsieve/static/*.js`; ownership and load order are in [frontend-workflows.md](frontend-workflows.md) |
| Release target identity | `src/shotsieve/release_targets.py` |

`shotsieve.__init__` intentionally exposes only the package version. The
desktop console script and local HTTP protocol are the application interfaces;
internal module functions and route helpers are not a general-purpose Python
library API.

## Durable and ephemeral state

The selected data directory contains the local database at `shotsieve.db`,
managed previews under `previews/`, runtime sidecars and installation logs
under `runtime/`, and model-preparation state at `model-preparation.json`.
Model weights and framework caches use the separate model-cache configuration
described in [configuration.md](configuration.md).

The catalog uses SQLite with WAL mode, foreign-key enforcement, and additive
migrations. File paths are normalized before they become catalog keys. Managed
previews carry a ShotSieve ownership marker; cleanup must remain root-scoped
and must not treat an unavailable root as an empty directory.

Job status, progress, cancellation, and retained result summaries are
process-local. A process exit or crash loses the job registry even when the
catalog and operation records remain on disk. Clients must treat an unknown
job ID as requiring a fresh operation or a manual inspection of the affected
files.

## Compatibility boundaries

The following are compatibility surfaces and should be changed only with
matching tests and documentation:

- CLI flags and the loopback default;
- HTTP route paths, request/response shapes, and selection revisions;
- persisted catalog and exported decision/operation data;
- the three learned-model IDs in `learned_iqa_catalog.py`;
- runtime target IDs, launcher names, archive names, and the torchless
  portable-archive boundary in `release_targets.py`;
- `window.ShotSieveWorkflows` and the script order documented in
  [frontend-workflows.md](frontend-workflows.md).

Bulk review, export, and delete operations use a selection revision so a
stale browser selection cannot silently mutate a changed catalog. Preserve
that check when changing selection or pagination code.

## Runtime and release boundary

Source installs and packaged runtime packs share the application but have
different dependency boundaries. Portable archives are intentionally
torchless and contain no model weights. A frozen launcher may install its
target-specific Torch sidecar under `data/runtime/` and model preparation
remains a separate first-use step. The ten target IDs and their artifact names
are authoritative in `release_targets.py`; do not infer them from launcher
display text.

There is no hosted deployment protocol, authentication layer, cloud worker,
Docker image, or supported system-service/LAN configuration in this
repository. Operational guidance is therefore limited to a local process,
local files, and the repository's native release workflows.
