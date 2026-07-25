from pathlib import Path
from conftest import create_image
from shotsieve.scanner import IgnoreMatcher, discover_files, check_overlapping_roots


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


def test_check_overlapping_roots() -> None:
    roots = [
        Path("C:/photos"),
        Path("C:/photos/vacation"),
        Path("D:/other"),
        Path("C:/photos/vacation/2026"),
    ]
    overlaps = check_overlapping_roots(roots)
    
    assert len(overlaps) >= 2
    paths = {(str(parent), str(child)) for parent, child in overlaps}
    
    assert (str(Path("C:/photos")), str(Path("C:/photos/vacation"))) in paths
    assert (str(Path("C:/photos")), str(Path("C:/photos/vacation/2026"))) in paths
    assert (str(Path("C:/photos/vacation")), str(Path("C:/photos/vacation/2026"))) in paths
