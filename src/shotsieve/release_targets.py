from __future__ import annotations

from dataclasses import asdict, dataclass


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
        ),
        ReleaseTarget(
            id="windows-nvidia",
            platform="windows",
            runtime="cuda",
            runsOn="windows-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa", "windows-build"),
            torchVariant="cuda",
            variantFolderName="ShotSieve-windows-nvidia",
            executableName="ShotSieve-NVIDIA.exe",
            archiveName="ShotSieve-windows-nvidia-x64.zip",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
        ),
        ReleaseTarget(
            id="windows-dml",
            platform="windows",
            runtime="directml",
            runsOn="windows-latest",
            pythonVersion="3.12",
            extras=("format-loaders", "learned-iqa-directml", "windows-build"),
            torchVariant="directml",
            variantFolderName="ShotSieve-windows-dml",
            executableName="ShotSieve-DML.exe",
            archiveName="ShotSieve-windows-dml-x64.zip",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
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
        ),
        ReleaseTarget(
            id="linux-nvidia",
            platform="linux",
            runtime="cuda",
            runsOn="ubuntu-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="cuda",
            variantFolderName="ShotSieve-linux-nvidia",
            executableName="ShotSieve-NVIDIA",
            archiveName="ShotSieve-linux-nvidia-x64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
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
        ),
        ReleaseTarget(
            id="macos-mps",
            platform="macos",
            runtime="mps",
            runsOn="macos-latest",
            pythonVersion="3.13",
            extras=("format-loaders", "learned-iqa"),
            torchVariant="default",
            variantFolderName="ShotSieve-macos-mps",
            executableName="ShotSieve-MPS",
            archiveName="ShotSieve-macos-mps-arm64.tar.gz",
            buildProfile="runtime-pack",
            specPath="shotsieve.spec",
        ),
    )


def tier1_release_targets() -> tuple[ReleaseTarget, ...]:
    return runtime_pack_release_targets()


def all_release_targets() -> tuple[ReleaseTarget, ...]:
    return runtime_pack_release_targets()


def tier1_release_matrix() -> list[dict[str, object]]:
    return [target.to_json() for target in runtime_pack_release_targets()]