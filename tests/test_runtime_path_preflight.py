from __future__ import annotations

import errno
import contextlib
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from shotsieve import bootstrap_sidecar as sidecar
from shotsieve.release_targets import runtime_pack_release_targets


DEEP_PORTABLE_ROOT = Path(
    "C:/Users/Photographer/Downloads/windows-amd-rocm-preview-build/"
    "ShotSieve-windows-amd-rocm-x64"
)


def _unexpected_install(*args, **kwargs):
    pytest.fail("An unsafe runtime path must not start pip, a download, or staging")


@pytest.mark.parametrize("target", runtime_pack_release_targets(), ids=lambda target: target.id)
@pytest.mark.parametrize("installer", [sidecar.install_torch_sidecar, sidecar.install_learned_iqa_sidecar])
@pytest.mark.parametrize("repair", [False, True], ids=["first-install", "repair"])
def test_deep_portable_path_blocks_all_targets_before_install(
    target, installer, repair: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(sidecar, "_run_sidecar_install_subprocess", _unexpected_install)
    monkeypatch.setattr(sidecar, "_prepare_learned_iqa_staging", _unexpected_install)
    monkeypatch.setattr(sidecar, "_create_sidecar_staging_dir", _unexpected_install)
    messages: list[str] = []
    destination = sidecar.sidecar_site_packages_dir(DEEP_PORTABLE_ROOT / "data" / "runtime", target.id)

    assert installer(
        runtime=target.runtime, site_packages=destination,
        force_reinstall=repair, output_func=messages.append,
    ) is False
    message = "\n".join(messages)
    assert str(destination) in message
    assert "too deeply" in message
    assert "180 required" in message
    assert "Move the entire ShotSieve folder" in message
    assert "C:\\ShotSieve" in message
    assert "Then start ShotSieve again" in message
    assert "No runtime files were installed" in message
    assert "Traceback" not in message
    assert "registry" not in message


@pytest.mark.parametrize("target", [target for target in runtime_pack_release_targets() if target.platform == "windows"], ids=lambda target: target.id)
def test_c_shotsieve_has_headroom_and_probes_before_passing(
    target, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = sidecar.sidecar_site_packages_dir(Path("C:/ShotSieve/data/runtime"), target.id)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists", lambda self: self == destination or original_exists(self))
    original_staging = sidecar._create_sidecar_staging_dir
    probe_parents: list[Path] = []

    def mapped_probe(parent: Path) -> Path:
        # Keep the illustrative C:\ShotSieve path read-only; perform the probe
        # in pytest's sandbox instead, with the same suffix and filesystem API.
        probe_parents.append(parent)
        return original_staging(tmp_path)

    monkeypatch.setattr(sidecar, "_create_sidecar_staging_dir", mapped_probe)
    monkeypatch.setattr(sidecar, "_sidecar_install_lock", lambda path: contextlib.nullcontext())
    writes: list[Path] = []
    original_write = Path.write_bytes

    def record_write(self: Path, data: bytes) -> int:
        writes.append(self)
        return original_write(self, data)

    monkeypatch.setattr(Path, "write_bytes", record_write)
    assert sidecar.runtime_path_preflight(destination) is True
    assert probe_parents == [destination]
    assert len(writes) == 1
    assert len(str(writes[0])) - len(str(tmp_path)) == sidecar.RUNTIME_PATH_RESERVE
    assert list(tmp_path.iterdir()) == []


def test_real_short_path_passes_and_removes_new_destination_parents() -> None:
    # Avoid pytest's descriptive test directories consuming the Windows budget.
    with tempfile.TemporaryDirectory(prefix=".ss") as temporary:
        root = Path(temporary)
        destination = root / "r" / "s"
        if sys.platform == "win32" and len(str(destination.resolve())) + sidecar.RUNTIME_PATH_RESERVE > sidecar.WINDOWS_RUNTIME_PATH_LIMIT:
            pytest.skip("The system temporary directory has no short-path headroom")
        assert sidecar.runtime_path_preflight(destination) is True
        assert list(root.iterdir()) == []


@pytest.mark.parametrize("installer", [sidecar.install_torch_sidecar, sidecar.install_learned_iqa_sidecar])
def test_real_short_path_reaches_installer(installer, monkeypatch: pytest.MonkeyPatch) -> None:
    operations: list[str] = []

    def fake_install(*, operation, site_packages, **kwargs):
        operations.append(operation)
        package = site_packages / ("torch" if operation == "torch" else "pyiqa")
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("", encoding="utf-8")
        return True

    monkeypatch.setattr(sidecar, "_run_sidecar_install_subprocess", fake_install)
    with tempfile.TemporaryDirectory(prefix=".ss") as temporary:
        destination = Path(temporary) / "windows-cpu"
        if sys.platform == "win32" and len(str(destination.resolve())) + sidecar.RUNTIME_PATH_RESERVE > sidecar.WINDOWS_RUNTIME_PATH_LIMIT:
            pytest.skip("The system temporary directory has no short-path headroom")
        assert installer(runtime="cpu", site_packages=destination) is True
        assert len(operations) == 1
        assert list(Path(temporary).iterdir()) == [destination]


@pytest.mark.parametrize("system", ["linux", "darwin"])
def test_posix_allows_deep_paths_without_windows_limit(
    system: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", system)
    destination = tmp_path / ("nested-" * 15) / "runtime" / "site-packages" / "linux-amd-rocm"
    assert len(str(destination.resolve())) + sidecar.RUNTIME_PATH_RESERVE > sidecar.WINDOWS_RUNTIME_PATH_LIMIT
    assert sidecar.runtime_path_preflight(destination) is True
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error", [errno.ENAMETOOLONG, errno.EACCES])
def test_failed_probe_cleans_up_and_reports_the_cause(
    error: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    messages: list[str] = []
    destination = tmp_path / "data" / "runtime" / "site-packages" / "linux-cpu"

    def fail_write(self: Path, data: bytes) -> int:
        assert self.parent.is_dir()
        raise OSError(error, "injected filesystem failure")

    monkeypatch.setattr(Path, "write_bytes", fail_write)
    assert sidecar.runtime_path_preflight(destination, output_func=messages.append) is False
    assert list(tmp_path.iterdir()) == []
    assert str(destination) in messages[0]
    assert "Traceback" not in messages[0]
    assert ("too deeply" if error == errno.ENAMETOOLONG else "writable") in messages[0]


def test_probe_preserves_existing_runtime_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    installed = tmp_path / "torch" / "__init__.py"
    installed.parent.mkdir()
    installed.write_text("existing runtime", encoding="utf-8")
    assert sidecar.runtime_path_preflight(tmp_path) is True
    assert list(tmp_path.iterdir()) == [installed.parent]
    assert installed.read_text(encoding="utf-8") == "existing runtime"


def test_probe_is_created_and_removed_under_install_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    locked = False
    original_write = Path.write_bytes

    @contextlib.contextmanager
    def lock(destination: Path):
        nonlocal locked
        assert destination == tmp_path
        locked = True
        try:
            yield
        finally:
            assert list(tmp_path.iterdir()) == []
            locked = False

    def write(self: Path, data: bytes) -> int:
        assert locked
        return original_write(self, data)

    monkeypatch.setattr(sidecar, "_sidecar_install_lock", lock)
    monkeypatch.setattr(Path, "write_bytes", write)
    assert sidecar.runtime_path_preflight(tmp_path) is True
    assert locked is False


def test_bootstrap_forwards_preflight_message_to_output_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(sidecar, "runtime_bundle_contains_torch", lambda path: False)
    monkeypatch.setattr(sidecar, "torch_sidecar_is_valid", lambda *args, **kwargs: False)
    monkeypatch.setattr(sidecar, "_run_sidecar_install_subprocess", _unexpected_install)
    monkeypatch.setenv(sidecar.DEFAULT_TORCH_AUTO_INSTALL_ENV, "1")
    messages: list[str] = []
    assert sidecar.maybe_prepare_torch_runtime(
        SimpleNamespace(id="windows-amd-rocm", runtime="rocm"),
        install_dir=DEEP_PORTABLE_ROOT, runtime_root=DEEP_PORTABLE_ROOT / "data/runtime",
        output_func=messages.append,
    ) == {}
    assert any("too deeply" in message for message in messages)


@pytest.mark.parametrize("installer", [sidecar._install_torch_sidecar_with_embedded_pip, sidecar._install_learned_iqa_sidecar_with_embedded_pip])
def test_direct_embedded_installer_rejects_before_loading_pip(
    installer, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(sidecar, "_load_embedded_pip_main", _unexpected_install)
    assert installer(runtime="rocm", site_packages=DEEP_PORTABLE_ROOT / "data/runtime/site-packages/windows-amd-rocm") is False


def test_private_helper_command_rejects_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "resolve", lambda self: self)
    monkeypatch.setattr(sidecar, "_load_embedded_pip_main", _unexpected_install)
    assert sidecar.dispatch_sidecar_install_command([
        sidecar.SIDECAR_INSTALL_COMMAND, "torch", "rocm",
        str(DEEP_PORTABLE_ROOT / "data/runtime/site-packages/windows-amd-rocm"), "0",
    ]) == 1
    output = capsys.readouterr()
    assert "too deeply" in output.out
    assert output.err == ""


def test_preflight_checks_resolved_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    destination = Path("C:/ShotSieve/data/runtime/site-packages/windows-cpu")
    resolved = DEEP_PORTABLE_ROOT / "data/runtime/site-packages/windows-cpu"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "resolve", lambda self: resolved)
    messages: list[str] = []
    assert sidecar.runtime_path_preflight(destination, output_func=messages.append) is False
    assert str(resolved) in messages[0]


def test_excessive_posix_path_resolution_is_a_controlled_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")

    def fail_resolve(self: Path) -> Path:
        raise OSError(errno.ENAMETOOLONG, "path too long")

    monkeypatch.setattr(Path, "resolve", fail_resolve)
    messages: list[str] = []
    assert sidecar.runtime_path_preflight(Path("/deep/runtime"), output_func=messages.append) is False
    assert "too deeply" in messages[0]
    assert "C:\\" not in messages[0]
