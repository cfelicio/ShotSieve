# ShotSieve manual release steps

Updated 2026-09-14.

This is the remaining candidate, target, hardware, storage, and human-review
checklist. Bounded implementation work is complete. Remaining code investigations
are tracked in `implement2.md`; `implement.md` retains the completion history.
Q-ReAlign Mini is now integrated as the forward model.
DirectML retirement
implementation is complete, but its remaining release and hardware validation
gates are still open; Windows ML/ONNX is deferred, and native XPU/ROCm are
separate unvalidated target tracks. I06 and the candidate-specific gates remain
open. Do not mark a release candidate ready until the applicable items below
have recorded evidence.

## 0.4.0 review handoff - 2026-09-14

The implementation review found and fixed Q-ReAlign checkpoint pinning and the
missing Chromium installation in the tag-release test job. It also removed an
environment-dependent state-reset test fixture that caused browser setup to
time out when real model options were unavailable. W06-W19 are complete;
W01-W04 code paths are present, while their applicable acceptance evidence and
W05 remain open. The remaining work is validation, but a failed model or target
check can still require a code fix. The Q-ReAlign acquisition stall was resolved
by the exact-target runs recorded below; no product defect was reproduced.

Run the applicable gates in this order:

1. **R01 / W04:** Complete fresh online and new-process offline Q-ReAlign Mini
   CPU/CUDA smokes, plus the other advertised models and missing/corrupt-asset
   cases. Previous attempts stopped before downloading a complete checkpoint.
2. **R02 / I06:** Rebuild and validate the final candidate's shipped bundles.
   The Windows hashes below predate this review's loader fix and must be replaced
   with new evidence. Linux CPU/NVIDIA and macOS CPU/MPS still lack target evidence.
   Review the complete Q-ReAlign checkpoint, Qwen base-model, and PyIQA terms
   against `THIRD_PARTY_LICENSES.md`, and record the revision and notices shipped.
3. **R03 / W01:** Finish advertised-accelerator and Windows-without-CUDA fallback
   checks. Python 3.11-3.12 source-stack checks remain open; the packaged matrix
   uses Python 3.13. XPU/ROCm are separate source-only tracks and do not gate the
   six packaged targets unless those providers are advertised as validated.
4. **R04 and human review:** Complete the disposable-file storage cases and
   `docs/accessibility-checklist.md`, then record evidence for the exact commit/tag.
5. **Publication:** Review the candidate diff, notices, manifests and hashes,
   then follow the publication section. The tag workflow publishes automatically.

### Review validation

- Final full suite with installed Chromium: **662 passed, 1 skipped** in
  **220.40 seconds**. The skip is the opt-in performance baseline. The earlier
  state-reset setup timeout was reproduced, fixed by removing its local fixture
  override, and verified in this complete run; browser skips are not counted as
  passes.
- Focused runtime/model-assets/release coverage passed **99 tests**; the corrected
  state-reset subset passed **5 tests**. Ruff `--isolated --select F,E9`, Python
  compilation, `git diff --check`, and six-target release-matrix generation passed.
- Both existing exact Windows Python 3.13.14 CPU/NVIDIA environments passed a
  fresh `pip check` with no broken requirements.
- A real CPU Q-ReAlign smoke with an empty cache and network disabled correctly
  exited with a sanitized `missing_offline_assets` diagnostic and recovery action
  in about **4.2 seconds**. Its expected-failure report is
  `build/audit-reports/review-040-missing-qrealign.json`. This validates only the
  missing-pinned-checkpoint failure path; it is not successful inference or
  complete-cache offline reuse evidence.
- Exact Python 3.13.14 Windows CPU and NVIDIA target environments completed
  fresh Q-ReAlign Mini online and new-process offline smokes. All four reports
  use immutable revision `fe1f45a7574c9e9d908875af9f7e90cb946aa19f` and are
  recorded at `build/audit-reports/manual-040-w01-qrealign-cpu-online.json`,
  `build/audit-reports/manual-040-w01-qrealign-cpu-offline.json`,
  `build/audit-reports/manual-040-w01-qrealign-cuda-online.json`, and
  `build/audit-reports/manual-040-w01-qrealign-cuda-offline.json`. CPU used
  Torch `2.14.0+cpu` and scored normalized `14.4073` online/offline; CUDA
  used Torch `2.14.0+cu130`, scored normalized `14.2039` online/offline,
  and recorded about `2262.48 MB` peak allocated memory. Both exact
  environments passed `pip check`; the offline processes reused their complete
  caches with network access disabled.
- These results cover the working tree based on `671e533` plus this review's
  fixes. No bundle was rebuilt or published during this review; the broader
  release, storage, license, and human-review gates remain open.

### Focused 0.4.0 polish pass

Use the final rebuilt candidate for these checks alongside R01-R04 and the
accessibility checklist. Add a concrete defect to `implement2.md` if a check
requires a code change.

- [ ] **Upgrade continuity:** Open a disposable copy of an existing data
  directory. Confirm roots, review decisions, historical scores, and preferences
  remain usable. An old DirectML selection must give clear migration/fallback
  guidance. A preparation record from the unpinned Mini loader must require
  preparation again rather than appear current.
- [ ] **First-run and recovery wording:** With a fresh data directory, check
  empty-library and unavailable-model states. During model preparation, confirm
  the current phase is understandable; exercise an interrupted run and confirm
  restart/retry guidance does not leave a false ready state.
- [ ] **Operation completion:** After deleting or moving the last item on a
  Review page, confirm pagination and selection update correctly. For a partial
  failure, confirm the result explains what happened and only eligible files
  are offered for retry. Check results against disk and catalog state under R04.
- [ ] **Visible finishing details:** At the required sizes and zoom, review
  long filenames, empty results, errors, dialogs, focus, and primary actions.
  Confirm model names, runtime labels, and support wording are consistent across
  Library, Compare, Review, Settings, and the release notes.
- [ ] **Candidate consistency:** Confirm the actual rebuilt distribution's
  package version is 0.4.0 and its changelog, notices, manifest, hashes, and
  support claims describe that same candidate. Preserve the test/model evidence
  for the final commit/tag; choose the intended release channel before following
  the publication steps.

## Follow-up setup and evidence - 2026-09-13

The original `.venv` was preserved. Runtime upgrades were tested in isolated
workspace environments; release constraints and Install / Repair AI support now
select the updated stack. Existing installations are not silently upgraded.

To use the already tested CUDA environment from this checkout:

```powershell
build/audit-latest/Scripts/python.exe -m shotsieve.desktop --model-cache-dir "$PWD/build/audit-model-cache-latest"
```

Check the cache setting in Settings before preparing a model. The prepared
weights live under `build/audit-model-cache-latest`. The old DirectML cache at
`build/audit-model-cache` is historical and should not be used for new runs.
These are disposable build locations: move or copy a cache to a durable
location and pass it via `--model-cache-dir` before deleting build output.
Q-ReAlign Mini's published safetensors are about 2.2 GB; allow
additional tokenizer/configuration, temporary loading, and runtime memory.

For a normal source environment upgrade, use the appropriate target constraints
from `docs/building.md`. For a portable install, run Settings > Install / Repair
AI support, restart, and explicitly Prepare the selected model. Do not install
the retired DirectML Torch trio into a new environment.

Sanitized smoke reports are in `build/audit-reports/`. Preserve them with the
exact release commit/tag before cleaning build output. Current tests and the
model-validation outcomes are summarized in the follow-up validation below.

For repeatable browser checks in this checkout:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD/build/audit-browsers"
.venv/Scripts/python.exe -m playwright install chromium --only-shell
.venv/Scripts/python.exe -m pytest -q --basetemp build/manual-tests -o cache_dir=build/manual-pytest-cache
```

Use a fresh `--basetemp` directory when old test files are inaccessible. Local
browser skips must not be recorded as passes; CI treats launch problems as errors.

## Prior follow-up validation results - 2026-09-13

- Existing-environment full suite: **584 passed, 52 skipped**. Browser skips were
  due to a missing Chromium binary, subsequently installed in the workspace.
- Latest dependency stack, core suite: **585 passed, 1 skipped, 51 deselected**.
- Final model/readiness/diagnostic regressions after the OOM fix: **42 passed**.
- Real browser suite: **50 passed, 1 setup timeout**. The affected accessibility
  test passed on its isolated rerun (**1 passed**). Do not describe this as one
  uninterrupted all-green browser run.
- Focused Ruff `--isolated --select F,E9`, Python compilation and `git diff --check`
  passed. A bare Ruff invocation inherits unrelated extra rules in this workspace.
- **At the time of this prior snapshot,** Q-ReAlign Mini was integrated but had
  not yet been downloaded or scored in this checkout. The exact-target online
  and network-disabled validation is now recorded in the 0.4.0 review handoff
  above. Do not treat the upstream under-4-GB estimate as local validation.

At that point Q-ReAlign Mini still required a fresh online and new-process
offline smoke after code integration. DirectML is being retired rather than
ported: its available Torch 2.4.1 stack conflicts with Q-ReAlign's published
Torch >=2.6 requirement, so installing Python 3.12 or downloading the weights
does not establish support.

## Current automated release follow-up - 2026-09-14

This follow-up started from commit `ee8e9fc` and includes the current working-tree
frontend fix and the 0.4.0 version bump, using fresh ignored build directories. It improves the Windows
evidence but does not close the model, cross-platform, storage, or human-review
gates below.

- The fresh workspace full suite passed **601 tests with 54 skips** in 149.51
  seconds. Focused Ruff (`--isolated --select F,E9`), Python compilation,
  `git diff --check`, and six-target release-matrix generation also passed.
- Installed headless Chromium enabled the frontend checks: **51 browser tests
  passed** across accessibility, responsive layout, state reset, and workflow
  coverage. This caught and fixed a real operation-result wiring error that had
  prevented disk-delete completion from refreshing and clamping the review page;
  the isolated regression test and the complete browser subsets now pass. The
  two remaining warnings are from the workspace development Torch build not
  supporting this host's RTX 5060 Ti architecture, not from the exact release
  target environments.
- The current exact Python 3.13.14 `windows-cpu` environment passed `pip check`.
  The rebuilt archive is
  `build/agent-release-040-dist/ShotSieve-windows-cpu-x64.zip`, SHA-256
  `ACE6E3F63F0A0FDE41781C0577EDCF35850A24C7E7E163BE28A9354EB8D8CC6E`.
  Its staged bundle has 19,105 files, one `ShotSieve-CPU.exe`, no model-weight
  files, and no DirectML artifacts. The launcher accepted `--help`; the frozen
  server answered `/api/options`, scanned three disposable JPEGs (`files_seen=3`,
  `files_added=3`, `files_failed=0`), and completed a CPU TOPIQ score for three
  disposable JPEGs (`files_scored=3`, `learned_scored=3`, `files_failed=0`).
  The frozen bundle contains the corrected operation-result reconciliation call.
- The current exact Python 3.13.14 `windows-nvidia` environment also passed
  `pip check`. The rebuilt archive is
  `build/agent-release-040-dist/ShotSieve-windows-nvidia-x64.zip`, SHA-256
  `D96C43520120F0C3892946C9A1C7FDD89F984D1B97FD18D5AFCAAEC4C963A8F4`.
  Its staged bundle has 19,143 files, one `ShotSieve-NVIDIA.exe`, no
  model-weight files, and no DirectML artifacts. The launcher accepted `--help`;
  CUDA `2.14.0+cu130` on the RTX 5060 Ti answered `/api/options`, scanned three
  disposable JPEGs (`files_seen=3`, `files_added=3`, `files_failed=0`),
  and completed a CUDA TOPIQ score (`files_scored=3`, `learned_scored=3`,
  `files_failed=0`). The frozen bundle contains the corrected operation-result
  reconciliation call. This supersedes the earlier
  unclaimed-NVIDIA-score status for this current source build only.
- Earlier Q-ReAlign Mini CPU and CUDA online attempts reached only
  `preparing_model`. The records are
  `build/agent-qrealign-cpu-online-data/model-preparation.json` and
  `build/agent-qrealign-cuda-online-data/model-preparation.json`; both retain
  revision `fe1f45a7574c9e9d908875af9f7e90cb946aa19f`, zero validation images,
  no score, and no completed report. The attempts were stopped after the
  bounded preparation window; no offline run was started because neither cache
  contains a complete checkpoint. These remain historical diagnostic blockers,
  not passes; the exact-target completion is recorded in the review handoff
  above.

The new Windows bundle evidence is still only Windows CPU/NVIDIA evidence.
Linux CPU/NVIDIA, macOS CPU/MPS, Windows-without-CUDA fallback, R04 storage
cases, and the human visual/accessibility review remain open. XPU and ROCm
remain source-only and are not release passes.

## W05 model and release evidence - 2026-09-13

This session completed one exact Windows candidate-target gate against the
current source. W05 is not a release-ready or fully validated item.

The exact `windows-cpu` environment at
`build/w01-release-targets/windows-cpu/.venv` reports Python 3.13.14,
Torch 2.14.0+cpu, Torchvision 0.29.0+cpu, PyIQA 0.1.16, and `pip check` with
no broken requirements. The matching frozen bundle passed launcher `--help`,
contained 20,962 archive entries, exactly one expected launcher, and zero
`.safetensors`, `.ckpt`, `.pth`, or `.pt` files. Its archive also contains the
exact package `dist-info` records and dependency license trees. Preserve this
archive as:

`build/w05-dist/ShotSieve-windows-cpu-x64.zip`

with SHA-256
`1A8C2BAAA8AB54963AD9592B092194300F4A13BD69A6BC4506A95CA2270E79C7`.

The extracted CPU bundle used fresh isolated data at
`build/w05-cpu-bundle-data` and the external cache
`build/audit-model-cache-latest`. It scanned one disposable JPEG and completed
one CPU TOPIQ score: scan `files_seen=1`, `added=1`, `failed=0`, about
`0.0267s`; score `files_scored=1`, `learned_scored=1`, `skipped=0`, `failed=0`,
about `5.798s`. The SQLite score row recorded raw `0.42987313866615295`,
normalized/overall `42.987313866615295`, and
`learned:pyiqa:0.1.16:topiq_nr:cpu`. This is TOPIQ candidate evidence only;
it is not Q-ReAlign Mini evidence and is not a no-CUDA hardware pass because
the host has an NVIDIA RTX 5060 Ti.

The exact `windows-nvidia` environment also built and passed launcher help,
archive inventory, zero-weight, and single-launcher checks. Its archive is
`build/w05-dist/ShotSieve-windows-nvidia-x64.zip` with SHA-256
`3A448D7222BE923595D8E22791118695E571F206A9D5FE428F9B147D1D0E0900`.
The NVIDIA bundle was not scored: the sandboxed support install could not open
network sockets, while the escalated sidecar install reached Torch/Torchvision/
PyIQA installation but the packaged repair process exited before dependencies
stabilized; a restart still reported missing `sympy`. No NVIDIA pass is
claimed.

The fresh Q-ReAlign Mini CPU attempt reached only `preparing_model` after about
150 seconds. Its pending record is
`build/w05-qrealign-mini-cpu-data/model-preparation.json` and contains the
immutable revision `fe1f45a7574c9e9d908875af9f7e90cb946aa19f`, requested CPU
runtime, and resolved dependency metadata, but no downloaded weights, report,
score, peak memory, or processed image. The prior CUDA attempt likewise has
only a pending preparation record. These unavailable model checks are blockers,
not passes.

Continuation attempt: a fresh CPU run with network access enabled used
`build/w05-qrealign-mini-cpu-online2-cache` and
`build/w05-qrealign-mini-cpu-online2-data`. It obtained tokenizer and
configuration files, but no Q-ReAlign checkpoint or incomplete weight payload;
after about 150 seconds it had no active Hub connection and was stopped. The
preparation record remains `preparing_model` with zero images and no report or
score. No offline command was run because the complete-cache prerequisite was
not satisfied. This confirms an external model-acquisition blocker, not a
successful CPU or offline result.

Gate status after this session: I06 has exact Windows CPU/NVIDIA build evidence
only; R01 still lacks fresh model-candidate online/offline and corrupt-cache
evidence; R02 still lacks stable NVIDIA scoring and Linux/macOS bundles; R03
lacks Mini/XPU/ROCm/MPS provider evidence; R04 and the human visual/accessibility
review were not run. Continue with the fresh Q-ReAlign commands below on a
network-capable runner, then repeat each completed cache in a new process with
network access disabled.

## DirectML retirement implementation follow-up - 2026-09-13

The source implementation and regression coverage are complete. DirectML is no
longer a dependency extra, runtime probe/status, model runtime, UI target,
Windows release target, sidecar branch, or PyInstaller collection path. The
release matrix now emits six targets and Windows AMD requests report an explicit
CPU fallback until native ROCm support is validated. An old persisted DirectML
selection is migrated to CPU with a visible retirement message.

Automated evidence for this patch:

- Full suite: **632 passed, 1 skipped**; the skip is the opt-in performance
  baseline and is not counted as a pass.
- Focused retirement/runtime/release suite: **114 passed**.
- Frontend/static regression suite: **138 passed**.
- Focused Ruff `--isolated --select F,E9`, Python compilation, `git diff --check`,
  and release-matrix generation passed.
- Exact Python 3.13.14 `windows-cpu` and `windows-nvidia` release environments
  both pass `pip check`. Their PyInstaller builds pass archive/launcher checks,
  contain no model weights, and contain no DML target, executable, sidecar, or
  constraint artifact. `windows-cpu` TOPIQ and scanner/scoring API checks pass
  on CPU; `windows-nvidia` TOPIQ and scanner/scoring API checks pass on CUDA.
- The frozen CPU executable passed `/api/options`, `/api/scan/start`, and
  `/api/score/start` against one disposable JPEG. The options payload exposes
  only `auto`, `cpu`, `cuda`, `xpu`, and `mps` as runtime choices and contains
  no `directml` or `torch_directml` entry. A historical score row whose stored
  backend was `directml` remains readable.
- Stale ignored `windows-dml` build/release outputs were removed; the
  historical `build/audit-dml` audit directory was preserved.
- The host's NVIDIA RTX 5060 Ti was used for the CUDA pass. WSL is not
  installed, so Linux and macOS/MPS target validation was not available here.
- The host `pip check` is not clean: `hf-xet 1.5.0` and `pyarrow 24.0.0` are
  reported unsupported on this platform. The preserved `build/audit-latest`
  environment separately reports `signaler 0.1.0` requiring
  `pyarrow<25.0,>=24.0` while `25.0.1` is installed; neither environment is
  used as release evidence.

Still required before checking the retirement gate:

- [x] Run `pip check` in each exact Windows Python 3.13 release-target
  environment and build/smoke the two available Windows bundles. The CPU and
  NVIDIA environments both report `No broken requirements found`.
- [ ] Run the Windows CPU fallback scan/score on a host without CUDA and finish
  the human Settings review. The CPU bundle ran successfully on a host that
  also has CUDA hardware, so this is not counted as a no-CUDA pass.
- [ ] Build and repeat the retained Linux CPU/NVIDIA and macOS MPS target
  checks. No Linux or macOS runner is available in this workspace.
- [ ] Validate the required Python 3.11-3.12 release-target combinations.
- [x] Keep Windows ML/ONNX out of this release; XPU and ROCm remain separate
  unvalidated source-install tracks and are not release passes.

## Q-ReAlign Mini validation after integration

Use a fresh cache and disposable data directory for each target. Do not reuse a
cache from another model or runtime.

```powershell
$cudaPython = "build/w01-release-targets/windows-nvidia/.venv/Scripts/python.exe"
$cpuPython = "build/w01-release-targets/windows-cpu/.venv/Scripts/python.exe"
$cache = "$PWD/build/manual-040-w01-qrealign-cuda-cache"
$data = "$PWD/build/manual-040-w01-qrealign-cuda-data"

& $cudaPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "CUDA target requirements are not clean" }
& $cpuPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "CPU target requirements are not clean" }

# Online preparation and one-image validation.
& $cudaPython scripts/model_smoke.py --model qrealign-mini --device cuda `
  --cache-dir $cache --data-dir $data `
  --report-path build/audit-reports/manual-040-w01-qrealign-cuda-online.json
if ($LASTEXITCODE -ne 0) { throw "CUDA online smoke failed; inspect its report before offline validation" }

# New-process offline reuse of the complete cache.
& $cudaPython scripts/model_smoke.py --model qrealign-mini --device cuda --offline `
  --cache-dir $cache --data-dir $data `
  --report-path build/audit-reports/manual-040-w01-qrealign-cuda-offline.json
if ($LASTEXITCODE -ne 0) { throw "CUDA offline smoke failed" }

# Repeat in a fresh CPU cache/data directory when CPU support is being claimed.
$cpuCache = "$PWD/build/manual-040-w01-qrealign-cpu-cache"
$cpuData = "$PWD/build/manual-040-w01-qrealign-cpu-data"
& $cpuPython scripts/model_smoke.py --model qrealign-mini --device cpu `
  --cache-dir $cpuCache --data-dir $cpuData `
  --report-path build/audit-reports/manual-040-w01-qrealign-cpu-online.json
if ($LASTEXITCODE -ne 0) { throw "CPU online smoke failed; inspect its report before offline validation" }
& $cpuPython scripts/model_smoke.py --model qrealign-mini --device cpu --offline `
  --cache-dir $cpuCache --data-dir $cpuData `
  --report-path build/audit-reports/manual-040-w01-qrealign-cpu-offline.json
if ($LASTEXITCODE -ne 0) { throw "CPU offline smoke failed" }
```

These commands use the existing exact Windows target environments, not the
historical Python 3.14 audit environment. Recreate them with the release build
helper if absent. Choose new cache/data/report paths for each fresh candidate
attempt; only the paired offline run should reuse its online cache. Allow the
approximately 2.2 GB checkpoint acquisition to finish; an interrupted or stalled
preparation is not a successful smoke. While the online
run executes, record peak GPU memory (or process working set for CPU), elapsed
time, model revision, resolved versions, and the returned score range. Confirm
the offline process makes no network connection and that failure reports redact
local image paths and model artifacts. Do not add a DirectML model run to release
evidence; DirectML removal is covered by the retirement checklist below.

Historical validation attempt on 2026-09-13: the CUDA online command was
started with the `build/audit-latest` environment and a fresh cache. It
remained in `preparing_model` for about 150 seconds without creating the model
cache or a report, so it was stopped before inference. The resulting pending
`model-preparation.json` remains diagnostic context only, not a pass. The
later exact-target CPU/CUDA runs recorded above supersede this attempt and
completed both online inference and new-process offline reuse.

## DirectML retirement verification

Run these checks after the DirectML removal patch, using a clean environment and
disposable data. They replace the former Windows-DML release gate.

- [x] Confirm `pip check` passes for the modern non-DML stack and that
  `torch-directml`, the DirectML extra, and the DML constraint file are absent.
- [x] Confirm release planning and generated artifacts contain no `windows-dml`,
  `ShotSieve-DML.exe`, DML sidecar install, or DirectML-only PyInstaller files.
- [ ] On a Windows machine without CUDA, launch the CPU target and scan/score
  one disposable JPEG. Confirm Auto falls back to CPU with an honest runtime
  report and no attempt to import `torch_directml`.
- [x] Confirm Settings/API runtime choices contain only the retained targets and
  that Windows AMD/Intel hardware is not silently labeled as DirectML.
- [x] Open an existing data directory containing historical scores and confirm
  rows remain readable even though DirectML is no longer selectable for new runs.
- [x] Confirm `README.md` and `CHANGELOG.md` explain that unsupported Windows
  AMD/Intel hardware falls back to CPU until a native provider is validated.
- [x] Keep Windows ML/ONNX out of this release. XPU and ROCm are not release
  passes until their exact installation, provider packaging, license, offline
  behavior, and one-image score checks are documented.

## Native XPU and ROCm validation

These are optional target tracks after DirectML removal. They do not change the
planned CPU/CUDA/MPS release matrix until the exact hardware and package paths
have passed.

The Intel source-install implementation is documented in
[docs/intel-xpu.md](docs/intel-xpu.md). It pins the Torch 2.14.0 / Torchvision
0.29.0 XPU wheels from the official PyTorch XPU index, keeps the environment
outside the packaged release matrix, and uses `scripts/model_smoke.py` to
write sanitized one-image evidence. The code/docs portion is complete; the
hardware checks below remain open until they run on supported Intel hardware.

The AMD ROCm source-install implementation is documented in
[docs/amd-rocm.md](docs/amd-rocm.md). It uses the AMD-validated ROCm 7.2.1
PyTorch 2.9.1/Python 3.12 wheel family, keeps Linux as the primary path, and
keeps the optional Windows path limited to AMD's explicit compatibility
matrix. The logical `rocm` runtime maps to Torch's HIP-backed CUDA device API
but reports AMD/ROCm identity in the app and smoke report. The code/docs portion
is complete; no AMD hardware or model evidence is available in this workspace.

Automated evidence for the source implementation: the directly affected
runtime/model-smoke/release suite passed **54 tests**; the full suite passed
**636 tests, 1 skipped** (the opt-in performance baseline); focused Ruff,
Python compilation, `git diff --check`, and six-target release-matrix
generation passed. These are code/regression checks only and do not close the
AMD hardware or model gates below.

- [x] Keep Intel XPU source-only and out of the packaged release target matrix.
- [x] Document isolated Windows/Linux installation, exact Python/wheel
  recording, driver recording, native tensor verification, cache isolation,
  online/offline smoke commands, and CPU fallback handling.
- [x] Preserve explicit `xpu` runtime selection and `intel -> xpu` resolution;
  unavailable XPU requests fail with actionable guidance instead of silently
  using CPU. Auto mode may still fall back to CPU when XPU is unavailable.
- [ ] Do not mark the Intel track validated until a supported Windows host and
  supported Linux host each complete the checks below. Q-ReAlign Mini waits for
  W04 model integration.

- [ ] **Intel XPU:** on a supported Intel Arc/Core Ultra Windows host and a
  supported Linux host, install the official PyTorch XPU wheel and matching
  driver in an isolated environment using [docs/intel-xpu.md](docs/intel-xpu.md)
  and `scripts/source-constraints-xpu.txt`. Run TOPIQ and CLIPIQA on one
  disposable JPEG, then run Q-ReAlign Mini when the target advertises it. Repeat each
  model from a complete network-disabled cache. Record the Torch/XPU wheel,
  Python, driver, runtime, cache paths, raw/normalized score, elapsed time,
  and peak memory from the smoke report. A source-install pass does not
  authorize a packaged XPU target.
- [x] Keep AMD ROCm source-only and out of the packaged release target matrix.
- [x] Document the Linux-first AMD install, exact ROCm/PyTorch wheel pair,
  supported Windows limitation, HIP tensor check, cache isolation, and
  online/offline smoke commands in [docs/amd-rocm.md](docs/amd-rocm.md).
- [x] Preserve an explicit `rocm` runtime and `amd -> rocm` alias for a
  detected HIP build; unavailable AMD requests retain CPU fallback with an
  actionable ROCm diagnostic.
- [x] Extend smoke evidence with HIP/ROCm version, GPU name/architecture,
  runtime, driver field, score, elapsed time, and peak memory without adding a
  packaged target.
- [ ] **AMD ROCm/Linux:** on a GPU listed in AMD's current matrix, install the
  exact supported ROCm/PyTorch pair in a fresh Python 3.12 environment using
  [docs/amd-rocm.md](docs/amd-rocm.md) and
  `scripts/source-constraints-rocm.txt`. Run TOPIQ and CLIPIQA on one
  disposable JPEG, then Q-ReAlign Mini when the target advertises it. Repeat every
  model from a complete network-disabled cache. Record the ROCm version, GPU
  architecture, driver, Python, Torch build, cache paths, raw/normalized
  scores, elapsed time, and peak memory. Keep unsupported Radeon cards on CPU.
- [ ] **AMD ROCm/Windows (optional):** test only the exact GPU, Python 3.12,
  driver, Torch, and ROCm combination explicitly listed by AMD, using the
  Windows source constraints in `scripts/source-constraints-rocm-windows.txt`.
  Windows ROCm support is narrower than Linux and covers PyTorch rather than
  the full ROCm stack; do not infer support from a Linux run or generic
  CPU/CUDA wheels.
- [ ] For each successful track, update the runtime catalog, installer/build
  constraints, release target, user-facing hardware wording, and third-party
  notices together. If a provider cannot run Q-ReAlign Mini, record model-level
  fallback rather than claiming the accelerator supports all learned models.

## Historical evidence from the previous pass

- Fixed the narrow Settings layout overflow by allowing the flex column to shrink within the mobile grid track.
- Full offline suite from the previous pass: **584 passed, 52 skipped, 1 warning** in 145.51 seconds.
- Focused responsive/accessibility suite: **35 passed**.
- Focused local file-operation/scanner regression suite: **45 passed, 1 warning**.
- Ruff `F,E9`, Python compilation, `git diff --check`, and the release/build tests passed.
- Fresh isolated online and new-process offline CPU smokes passed for TOPIQ and CLIPIQA in that previous pass. Resolved versions for the CPU smokes were `pyiqa 0.1.16`, `timm 1.0.28`, `huggingface-hub 1.24.0`, `transformers 5.14.1`, `openai-clip 1.0.1`, `torch 2.13.0+cu126`, and `torchvision 0.28.0+cu126`; the local environment's `pip check` passed.
- A local Windows CPU bundle was built, its launcher started, and its bundled server scanned one generated JPEG and scored it with `topiq_nr` on CPU: `files_scored=1`, `learned_scored=1`, `files_failed=0`.
- The CPU archive contained 21,552 entries, one launcher, and zero `.safetensors`, `.ckpt`, `.pth`, or `.pt` files.

The local bundle rehearsal used Python 3.14 and a CUDA-enabled Torch install, not the exact Python 3.13 CPU release environment. It is useful evidence, but it does not close the release-target gates.

## R01 — release-candidate model validation

For the exact commit/tag intended for the test release:

1. Prepare fresh isolated caches for `topiq_nr`, `clipiqa`, and
   `qrealign-mini` using the selected target constraints. Run Q-ReAlign Mini on
   CPU where that target is claimed, and on CUDA where advertised; its initial
   batch size is one until throughput and memory measurements justify a change.
2. Run the online smoke, then a new network-disabled process using the complete cache. Capture the sanitized JSON report, exact resolved versions, cache paths, date, commit/tag, and `pip check` result.
3. Exercise a missing or corrupt offline asset for each selected model and confirm the diagnostic is actionable and does not expose credentials, model weights, or private image data.
4. Repeat for each retained target stack where the cache is claimed to be
   portable. Do not reuse the local rehearsal as evidence for another platform;
   there is no DirectML release target after retirement.

## R02 — actual shipped bundles

Build and test every target selected for the test release, using the exact release workflow/environment. After DirectML retirement, the planned matrix is:

`windows-cpu`, `windows-nvidia`, `linux-cpu`, `linux-nvidia`, `macos-cpu`, and
`macos-mps`.

The former `windows-dml` target must be absent rather than silently rebuilt with
the non-DML Torch stack. Add Windows XPU, AMD ROCm, or Windows ML/ONNX only after
the separate provider evidence is complete.

For each shipped target:

1. Confirm target Python/Torch constraints resolve, `pip check` passes, and the produced launcher responds to `--help`.
2. Start the extracted bundle with an isolated data directory and externally prepared model cache.
3. Scan and score one disposable JPEG. Confirm the requested and actual runtime in the result or diagnostic. Use Q-ReAlign Mini where the target advertises it.
4. Inspect the staged bundle and archive for model weights, exact package versions, notices, and asset/license terms. Preserve the archive hash and release-manifest evidence.

## R03 — advertised accelerators

The follow-up review created isolated environments under `build/`. The DirectML
environment is historical evidence only and is not a release target:

- `build/audit-dml`: Python 3.12.13, Torch 2.4.1, torchvision 0.19.1,
  torch-directml 0.2.5.dev240914. Basic GPU operations and TOPIQ/CLIPIQA
  one-image inference passed. Both models also passed in new network-disabled
  processes with the updated shared dependencies.
- `build/audit-latest`: Python 3.14.5, Torch 2.14.0+cu130 and torchvision
  0.29.0+cu130. The RTX 5060 Ti runs successfully with sm_120 kernels. TOPIQ
  and CLIPIQA passed CPU and CUDA inference, including offline cache reuse.
- The shared versions are timm 1.0.29, huggingface-hub 1.31.0,
  Transformers 5.17.0, Accelerate 1.15.0, PyIQA 0.1.16,
  openai-clip 1.0.1, SentencePiece 0.2.2 and Einops 0.8.2.
  `uv pip check` passed in both isolated environments.

These source-environment checks do not validate the exact Python version,
packaging or drivers of every shipped target. Repeat on Windows release Python,
Linux CUDA and macOS MPS. CUDA 13.0 also requires a compatible installed NVIDIA
driver; the local driver passed, but older-driver compatibility is not claimed.

No Q-ReAlign Mini download or inference has yet been completed in either
environment. Record it as unverified until the fresh-cache checks below pass.

Torch 2.6 is only Q-ReAlign's minimum supported loader version; do not downgrade
the current 2.14.0 retained stack to 2.6. Use the exact Torch/driver wheel pair
required by each XPU or ROCm target when those tracks are tested.

After the retirement patch, repeat the positive CPU/CUDA/MPS checks in the
planned matrix and use the DirectML retirement checklist above only to verify
that the old path is gone.

Q-ReAlign Mini is not currently validated on DirectML. Its published runtime
requires Torch >=2.6/Transformers >=5.2, while the available DirectML package is
tied to Torch 2.4.1. No Q-ReAlign DirectML port is planned; this incompatibility
is one reason the adapter is being retired. CUDA one-image, memory and offline
validation remain separate gates.

## R04 — local and network file operations

Using disposable photos only, validate the release's claimed storage matrix:

- Windows same-volume and cross-volume moves.
- Denied and locked files.
- UNC and/or mapped shares.
- Copy-success/delete-failure and disconnect/reconnect during operations.
- Non-ASCII and long paths.
- Cancellation and unavailable-root scans, preserving catalog decisions.
- Linux/macOS local paths and any available mounted-share cases.

For each case, verify bytes on disk, catalog paths, visible result wording, retry eligibility, and decision CSV root/path accuracy. Record untested combinations; do not claim general NAS certification from the automated suite.

## Human visual and accessibility review

Run `docs/accessibility-checklist.md` on a fresh small library before the test release. Check Chromium at approximately 390px, 768px, desktop width, and 125% zoom across Library, Compare, Review, and Settings. Confirm no distracting horizontal scroll, readable contrast, usable dialogs, clear selection/current-photo states, lightbox behavior, and visible primary actions. Pair this with keyboard/focus and any available assistive-technology checks; automated browser tests do not replace this review.

## Test-release publication

1. Review the exact diff, version, changelog, third-party notices, generated release manifest, and archive hashes.
2. Commit the release changes and create the annotated version tag using the documented helpers in `docs/building.md`.
3. After the tag workflow publishes, mark the GitHub release as a pre-release manually if that is the intended test-release channel; the current workflow does not set `-PreRelease` automatically.
4. Attach the recorded R01–R04 evidence to the release notes or build documentation, including explicit untested targets.
