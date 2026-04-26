# ShotSieve

ShotSieve is a local-first photo culling app for people who want AI help without handing their library to a cloud service. You point it at a folder, let it analyze the images on your machine, then make the final keep/reject decisions yourself in the desktop review workflow.

<img width="1844" height="1251" alt="image" src="https://github.com/user-attachments/assets/31cb0d90-6ec9-4e2e-88f9-9ecfbaecdca6" />


## What ShotSieve does

- Scores images with learned image-quality models on supported runtimes
- Lets you compare models on the same library before trusting one for a bigger culling pass
- Gives you a visual `Review` workflow for keep/reject decisions, filtering, and batch actions
- Stays local-first: the review UI runs on loopback and your library stays on your machine

## Who it is for

ShotSieve is aimed at photographers and hobbyists who:

- shoot large folders and need help narrowing them down
- want AI ranking to help find low quality photos (e.g. blurry, out of focus, over/under exposed)
- prefer a local workflow over uploading a library to a hosted service

## Which package should I download?

If you are choosing between the packaged builds, pick the one that matches your hardware:

| Package | Best for | What it means |
|---|---|---|
| `CPU` | Any machine, safest fallback | Runs entirely on the processor. Slowest, but the most compatible. |
| `NVIDIA / CUDA` | Windows or Linux machines with an NVIDIA GPU | Best choice when you have a supported NVIDIA card and want the fastest learned-IQA scoring. |
| `DML / DirectML` | Windows machines with non-NVIDIA GPUs, like AMD and Intel (can also work with NVIDIA) | Uses Microsoft's DirectML stack. Usually the right Windows accelerator option when CUDA is not available. |
| `Apple Silicon / MPS` | Recent Macs with Apple Silicon | Best choice on Apple Silicon when you want GPU acceleration without a separate CUDA stack. |

Practical rule of thumb:

- If you have an NVIDIA GPU, choose **CUDA**.
- If you are on Apple Silicon, choose **MPS**.
- If you are on Windows without NVIDIA but do have a modern GPU, try **DirectML**.
- If you just want the most reliable option or are unsure, choose **CPU**.

Intel XPU remains a source-install/runtime option today, but it is not one of the packaged runtime downloads listed above.

## Quick start with `shotsieve-desktop`

For source installs and editable installs, use the Python entry point when you want ShotSieve to manage the local DB location and open the review UI directly:

```bash
shotsieve-desktop
```

Downloaded bundles use platform- and runtime-specific launcher names instead of the `shotsieve-desktop` command:

| Platform | Runtime pack | Launcher |
|---|---|---|
| Windows | CPU | `ShotSieve-CPU.exe` |
| Windows | NVIDIA / CUDA | `ShotSieve-NVIDIA.exe` |
| Windows | DirectML | `ShotSieve-DML.exe` |
| Linux | CPU | `ShotSieve-CPU` |
| Linux | NVIDIA / CUDA | `ShotSieve-NVIDIA` |
| macOS | CPU | `ShotSieve-CPU` |
| macOS | Apple Silicon / MPS | `ShotSieve-MPS` |

So the quick rule is:

- if you installed ShotSieve from Python packaging, launch `shotsieve-desktop`
- if you downloaded a runtime pack, launch the bundled `ShotSieve-*` app inside that archive

Useful flags:

```bash
shotsieve-desktop --data-dir ./shot-data
shotsieve-desktop --host 127.0.0.1 --port 9001 --no-browser
```

Default data location for downloaded bundles:

- Downloaded Windows, Linux, and macOS bundles keep writable app state in a local `data/` folder next to the launcher
- Editable source checkouts (`pip install -e .` from this repository) now also default to a local `data/` folder at the project root
- The main database lives at `data/shotsieve.db`
- Generated previews live under `data/previews/`
- Runtime sidecars, repairs, and related pip logs live under `data/runtime/`

Installed packages outside a source checkout still fall back to an OS-level app data location (`%LOCALAPPDATA%\ShotSieve` on Windows, falling back to `%APPDATA%` if needed, otherwise `~/.shotsieve`). On any install style, `--data-dir` overrides the default.

## How you use it

The intended workflow is simple and visual:

1. Launch `shotsieve-desktop`
2. Choose a photo folder in `Library`
3. Run `Analyze` to scan the folder and score supported images
4. Start in `Review` and work through the images with keep/reject decisions
5. Use `Compare` if you want to test different models on the same library
6. Use `Settings` for runtime info, resource profile, and maintenance actions

If you are using a downloaded runtime pack, replace step 1 with the matching bundled launcher from the table above.

ShotSieve is meant to speed up your judgment, not replace it. The app helps surface likely throwaways; you still make the final call.

## Models and runtimes

The main in-app model choices are currently centered on:

- `topiq_nr` is the default model
- `clipiqa` is a fast secondary option for quick comparisons
- `qalign` is available only on supported accelerator-backed paths (`cuda`, `mps`)

Runtime names you may see in settings or developer docs:

- `cpu`: no GPU acceleration
- `cuda`: NVIDIA GPU acceleration
- `xpu`: Intel accelerator path for source installs where the local PyTorch runtime exposes it
- `directml`: Windows GPU acceleration through DirectML
- `mps`: Apple Silicon GPU acceleration

Runtime compatibility note: `qalign` is **not** available on `cpu` or `directml`; on those paths, `topiq_nr` and `clipiqa` are the practical in-app choices.

## Review UI

The review server binds to `127.0.0.1:8765` by default. The UI is organized around four tabs:

1. `Library` for root selection, scan/score actions, and analysis options
2. `Compare` for side-by-side learned-model benchmarking
3. `Review` for queue navigation, filtering, marking, and export/delete flows
4. `Settings` for runtime info, resource profile, and maintenance actions

Current keyboard shortcuts in review mode:

| Key | Action |
|---|---|
| `↓` / `→` | Next photo |
| `↑` / `←` | Previous photo |
| `S` | Keep current photo or selected photos |
| `R` | Reject current photo or selected photos |
| `Esc` | Close the lightbox or active overlay |

## For developers and builders

If you are installing from source, running tests, or building release archives, use the dedicated build guide:

- [docs/building.md](docs/building.md)

That guide covers:

- source installs and optional extras
- test and lint commands
- Playwright setup
- Windows runtime-pack build instructions

The browser-focused frontend checks are intentionally about **visual QA** and visual usability for the photo-review workflow, not broad accessibility conformance claims.

## Security and licensing

- [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md): third-party model and library licensing
- [LICENSE](LICENSE): GNU Affero General Public License v3.0 or later (AGPLv3+)

ShotSieve uses AI models and libraries that carry **non-commercial or stricter research-use restrictions**. No model weights are bundled; they are downloaded on first use from Hugging Face. In particular, Q-Align should be treated as research-use only unless its authors publish clearer licensing terms. See [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md) for the canonical breakdown.

## Project docs

- [README.md](README.md): user-facing overview, quick start, and package selection guide
- [CHANGELOG.md](CHANGELOG.md): release history, starting with the initial `0.1.0` release
- [docs/building.md](docs/building.md): source install, testing, linting, and release-build instructions
