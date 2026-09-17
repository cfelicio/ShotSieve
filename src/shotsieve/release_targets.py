from __future__ import annotations

from dataclasses import asdict, dataclass


# Runtime-pack IDs are part of manifests, cache paths, and build tooling. Keep
# old IDs readable so an upgraded bootstrap can consume manifests and sidecars
# produced before the runtime names were made explicit.
LEGACY_RELEASE_TARGET_ID_ALIASES = {
    "windows-nvidia": "windows-nvidia-cuda",
    "windows-cuda": "windows-nvidia-cuda",
    "windows-intel": "windows-intel-xpu",
    "windows-xpu": "windows-intel-xpu",
    "windows-amd": "windows-amd-rocm",
    "windows-rocm": "windows-amd-rocm",
    "linux-nvidia": "linux-nvidia-cuda",
    "linux-cuda": "linux-nvidia-cuda",
    "linux-intel": "linux-intel-xpu",
    "linux-xpu": "linux-intel-xpu",
    "linux-amd": "linux-amd-rocm",
    "linux-rocm": "linux-amd-rocm",
    "macos-mps": "macos-apple-mps",
}


def canonical_release_target_id(target_id: str) -> str:
    """Return the explicit runtime-pack ID for a requested target ID."""
    normalized = str(target_id).strip().casefold()
    return LEGACY_RELEASE_TARGET_ID_ALIASES.get(normalized, normalized)


def release_target_id_aliases(target_id: str) -> tuple[str, ...]:
    """Return the canonical ID followed by IDs accepted from older releases."""
    canonical = canonical_release_target_id(target_id)
    legacy_ids = tuple(
        alias
        for alias, replacement in LEGACY_RELEASE_TARGET_ID_ALIASES.items()
        if replacement == canonical
    )
    return (canonical, *legacy_ids)


@dataclass(frozen=True, slots=True)
class ReleaseTarget:
    id: str
    platform: str
    runtime: str
    runsOn: str
    pythonVersion: str
    extras: tuple[str, ...]
    torchVariant: str
    variantFolderName: str
    executableName: str
    archiveName: str
    buildProfile: str = "runtime-pack"
    specPath: str = "shotsieve.spec"
    constraintsFile: str = "scripts/release-constraints.txt"

    def to_json(self) -> dict[str, object]:
        payload = asdict(self)
        payload["extras"] = list(self.extras)
        return payload


def runtime_pack_release_targets() -> tuple[ReleaseTarget, ...]:
    return (
        ReleaseTarget(
            id="windows-cpu",
            platform="windows",
            runtime="cpu",
            runsOn="windows-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa", "windows-build"),
            torchVariant="cpu",
            variantFolderName="ShotSieve-windows-cpu",
            executableName="ShotSieve-CPU.exe",
            archiveName="ShotSieve-windows-cpu-x64.zip",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/release-constraints-torch.txt",
        ),
        ReleaseTarget(
            id="windows-nvidia-cuda",
            platform="windows",
            runtime="cuda",
            runsOn="windows-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa", "windows-build"),
            torchVariant="cuda",
            variantFolderName="ShotSieve-windows-nvidia-cuda",
            executableName="ShotSieve-NVIDIA-CUDA.exe",
            archiveName="ShotSieve-windows-nvidia-cuda-x64.zip",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/release-constraints-torch.txt",
        ),
        ReleaseTarget(
            id="windows-intel-xpu",
            platform="windows",
            runtime="xpu",
            runsOn="windows-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa", "windows-build"),
            torchVariant="xpu",
            variantFolderName="ShotSieve-windows-intel-xpu",
            executableName="ShotSieve-Intel-XPU.exe",
            archiveName="ShotSieve-windows-intel-xpu-x64.zip",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/source-constraints-xpu.txt",
        ),
        ReleaseTarget(
            id="windows-amd-rocm",
            platform="windows",
            runtime="rocm",
            runsOn="windows-latest",
            pythonVersion="3.12",
            extras=("format-loaders", "learned-iqa", "windows-build"),
            torchVariant="rocm",
            variantFolderName="ShotSieve-windows-amd-rocm",
            executableName="ShotSieve-AMD-ROCm.exe",
            archiveName="ShotSieve-windows-amd-rocm-x64.zip",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/source-constraints-rocm-windows.txt",
        ),
        ReleaseTarget(
            id="linux-cpu",
            platform="linux",
            runtime="cpu",
            runsOn="ubuntu-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="cpu",
            variantFolderName="ShotSieve-linux-cpu",
            executableName="ShotSieve-CPU",
            archiveName="ShotSieve-linux-cpu-x64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/release-constraints-torch.txt",
        ),
        ReleaseTarget(
            id="linux-intel-xpu",
            platform="linux",
            runtime="xpu",
            runsOn="ubuntu-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="xpu",
            variantFolderName="ShotSieve-linux-intel-xpu",
            executableName="ShotSieve-Intel-XPU",
            archiveName="ShotSieve-linux-intel-xpu-x64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/source-constraints-xpu.txt",
        ),
        ReleaseTarget(
            id="linux-amd-rocm",
            platform="linux",
            runtime="rocm",
            runsOn="ubuntu-latest",
            pythonVersion="3.12",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="rocm",
            variantFolderName="ShotSieve-linux-amd-rocm",
            executableName="ShotSieve-AMD-ROCm",
            archiveName="ShotSieve-linux-amd-rocm-x64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/source-constraints-rocm.txt",
        ),
        ReleaseTarget(
            id="linux-nvidia-cuda",
            platform="linux",
            runtime="cuda",
            runsOn="ubuntu-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="cuda",
            variantFolderName="ShotSieve-linux-nvidia-cuda",
            executableName="ShotSieve-NVIDIA-CUDA",
            archiveName="ShotSieve-linux-nvidia-cuda-x64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/release-constraints-torch.txt",
        ),
        ReleaseTarget(
            id="macos-cpu",
            platform="macos",
            runtime="cpu",
            runsOn="macos-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="default",
            variantFolderName="ShotSieve-macos-cpu",
            executableName="ShotSieve-CPU",
            archiveName="ShotSieve-macos-cpu-arm64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/release-constraints-torch.txt",
        ),
        ReleaseTarget(
            id="macos-apple-mps",
            platform="macos",
            runtime="mps",
            runsOn="macos-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="default",
            variantFolderName="ShotSieve-macos-apple-mps",
            executableName="ShotSieve-Apple-MPS",
            archiveName="ShotSieve-macos-apple-mps-arm64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
            constraintsFile="scripts/release-constraints-torch.txt",
        ),
    )


def tier1_release_targets() -> tuple[ReleaseTarget, ...]:
    return runtime_pack_release_targets()


def all_release_targets() -> tuple[ReleaseTarget, ...]:
    return runtime_pack_release_targets()


def tier1_release_matrix() -> list[dict[str, object]]:
    return [target.to_json() for target in runtime_pack_release_targets()]
