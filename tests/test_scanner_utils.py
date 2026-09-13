from pathlib import Path

import pytest
from conftest import create_image

from shotsieve.scanner import (
    FileDiscoveryError,
    IgnoreMatcher,
    check_overlapping_roots,
    discover_files,
)


def test_ignore_matcher(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    
    rules = [
        "exports",
        "**/temp/**",
        "*.tmp",
    ]
    matcher = IgnoreMatcher(root, rules)

    assert matcher.should_ignore(root / "exports") is True
    assert matcher.should_ignore(root / "exports" / "sub") is True
    assert matcher.should_ignore(root / "vacation" / "temp") is True
    assert matcher.should_ignore(root / "vacation" / "temp" / "file.jpg") is True
    assert matcher.should_ignore(root / "vacation" / "file.tmp") is True
    
    assert matcher.should_ignore(root / "vacation") is False
    assert matcher.should_ignore(root / "vacation" / "sub") is False
    assert matcher.should_ignore(root) is False


def test_discover_files_pruning(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    
    good_dir = root / "good"
    good_dir.mkdir()
    ignored_dir = root / "temp"
    ignored_dir.mkdir()
    
    create_image(good_dir / "a.jpg")
    create_image(good_dir / "b.jpg")
    create_image(ignored_dir / "c.jpg")
    
    files = list(discover_files(
        root,
        recursive=True,
        extensions=(".jpg",),
        ignore_rules=("**/temp/**",),
    ))
    
    assert len(files) == 2
    paths = {f.name for f in files}
    assert "a.jpg" in paths
    assert "b.jpg" in paths
    assert "c.jpg" not in paths


def test_discover_files_reports_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileDiscoveryError, match="Unable to enumerate"):
        list(
            discover_files(
                tmp_path / "not-mounted",
                recursive=True,
                extensions=(".jpg",),
            )
        )


def test_discover_files_reports_nonrecursive_enumeration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "photos"
    root.mkdir()

    def denied_scandir(_path):
        raise PermissionError("access denied")

    monkeypatch.setattr("shotsieve.scanner.os.scandir", denied_scandir)

    with pytest.raises(FileDiscoveryError, match="photos"):
        list(
            discover_files(
                root,
                recursive=False,
                extensions=(".jpg",),
            )
        )


def test_discover_files_reports_recursive_child_enumeration_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "photos"
    child = root / "private"
    child.mkdir(parents=True)

    def failing_walk(_root, *, topdown, onerror):
        error = PermissionError("access denied")
        error.filename = str(child)
        onerror(error)
        yield str(root), [child.name], []

    monkeypatch.setattr("shotsieve.scanner.os.walk", failing_walk)

    with pytest.raises(FileDiscoveryError, match="private"):
        list(
            discover_files(
                root,
                recursive=True,
                extensions=(".jpg",),
            )
        )


def test_check_overlapping_roots(tmp_path: Path) -> None:
    root1 = tmp_path / "photos"
    root2 = tmp_path / "photos" / "vacation"
    root3 = tmp_path / "other"
    root4 = tmp_path / "photos" / "vacation" / "2026"

    roots = [root1, root2, root3, root4]
    overlaps = check_overlapping_roots(roots)

    assert len(overlaps) == 3
    paths = {(parent.resolve(), child.resolve()) for parent, child in overlaps}

    assert (root1.resolve(), root2.resolve()) in paths
    assert (root1.resolve(), root4.resolve()) in paths
    assert (root2.resolve(), root4.resolve()) in paths
