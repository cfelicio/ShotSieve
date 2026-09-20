from __future__ import annotations

from dataclasses import asdict, dataclass


def canonical_release_target_id(target_id: str) -> str:
    """Normalize a current runtime-pack ID at an input boundary."""
    return str(target_id).strip().casefold()


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
            id="windows-amd-rocm10-gfx1103",
            platform="windows",
            runtime="rocm",
            runsOn="windows-latest",
            pythonVersion="3.12",
            extras=("format-loaders", "learned-iqa", "windows-build"),
            torchVariant="rocm10-gfx1103",
            variantFolderName="ShotSieve-windows-amd-rocm10-gfx1103",
            executableName="ShotSieve-AMD-ROCm10-GFX1103.exe",
            archiveName="ShotSieve-windows-amd-rocm10-gfx1103-x64.zip",
            buildProfile="rocm10-candidate",
            specPath="shotsieve.spec",
            constraintsFile="scripts/source-constraints-rocm10-gfx1103.txt",
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
            id="linux-amd-rocm10-gfx1103",
            platform="linux",
            runtime="rocm",
            runsOn="ubuntu-latest",
            pythonVersion="3.12",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="rocm10-gfx1103",
            variantFolderName="ShotSieve-linux-amd-rocm10-gfx1103",
            executableName="ShotSieve-AMD-ROCm10-GFX1103",
            archiveName="ShotSieve-linux-amd-rocm10-gfx1103-x64.tar.gz",
            buildProfile="rocm10-candidate",
            specPath="shotsieve.spec",
            constraintsFile="scripts/source-constraints-rocm10-gfx1103.txt",
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
