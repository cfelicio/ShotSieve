import json
import time
import urllib.request
import urllib.error
from pathlib import Path
import pytest
from conftest import create_image
from shotsieve.scanner import IgnoreMatcher, discover_files, check_overlapping_roots, preflight_root


def test_ignore_matcher(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    
    # Rules: exact folder, subfolder glob, and pattern matches
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
    
    # Should not ignore non-matching items
    assert matcher.should_ignore(root / "vacation") is False
    assert matcher.should_ignore(root / "vacation" / "sub") is False
    assert matcher.should_ignore(root) is False


def test_discover_files_pruning(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    
    # Setup some directories
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
    
    # c.jpg should have been ignored
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
    
    # Path("C:/photos") is a parent of "C:/photos/vacation" and "C:/photos/vacation/2026"
    # Path("C:/photos/vacation") is a parent of "C:/photos/vacation/2026"
    assert len(overlaps) >= 2
    paths = {(str(parent), str(child)) for parent, child in overlaps}
    
    assert (str(Path("C:/photos")), str(Path("C:/photos/vacation"))) in paths
    assert (str(Path("C:/photos")), str(Path("C:/photos/vacation/2026"))) in paths
    assert (str(Path("C:/photos/vacation")), str(Path("C:/photos/vacation/2026"))) in paths


def test_preflight_root_statistics(tmp_path: Path) -> None:
    root = tmp_path / "photos"
    root.mkdir()
    
    good_dir = root / "good"
    good_dir.mkdir()
    ignored_dir = root / "exports"
    ignored_dir.mkdir()
    
    create_image(good_dir / "a.jpg")
    create_image(good_dir / "b.jpg")
    create_image(ignored_dir / "c.jpg")
    
    res = preflight_root(
        root,
        recursive=True,
        extensions=(".jpg",),
        ignore_rules=("exports",),
    )
    
    assert res["candidate_assets"] == 2
    assert res["ignored_directories"] == 1
    assert res["unreadable_directories"] == 0
    assert res["unreadable_files"] == 0


def test_preflight_api_lifecycle(test_server) -> None:
    base_url, db_path, tmp_path = test_server
    
    root = tmp_path / "photos"
    root.mkdir()
    good_dir = root / "good"
    good_dir.mkdir()
    create_image(good_dir / "a.jpg")
    
    payload = {
        "roots": [str(root.resolve())],
        "ignore_rules": ["exports"],
        "recursive": True,
        "extensions": "jpg",
    }
    
    # 1. Start Preflight Job
    req = urllib.request.Request(
        f"{base_url}/api/library/preflight/start",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        start_res = json.loads(resp.read().decode("utf-8"))
        
    job_id = start_res["job_id"]
    assert job_id is not None
    assert start_res["status"] == "running"
    
    # 2. Poll Status
    status = "running"
    result = None
    for _ in range(20):
        time.sleep(0.1)
        with urllib.request.urlopen(f"{base_url}/api/library/preflight/status?job_id={job_id}") as resp:
            status_res = json.loads(resp.read().decode("utf-8"))
            status = status_res["status"]
            if status in ("completed", "failed"):
                break
                
    assert status == "completed", f"Preflight job failed: {status_res.get('error')}"
    
    # 3. Get Result
    with urllib.request.urlopen(f"{base_url}/api/library/preflight/result?job_id={job_id}") as resp:
        result = json.loads(resp.read().decode("utf-8"))
        
    assert result["candidate_assets"] == 1
    assert result["ignored_directories"] == 0
    assert len(result["sources"]) == 1
    assert result["sources"][0]["root"] == str(root.resolve())


def test_preflight_api_uses_the_same_default_extensions_as_scan(test_server) -> None:
    base_url, _db_path, tmp_path = test_server
    root = tmp_path / "photos"
    root.mkdir()
    (root / "iphone.heic").write_bytes(b"heif")

    request = urllib.request.Request(
        f"{base_url}/api/library/preflight/start",
        data=json.dumps({
            "roots": [str(root)],
            "ignore_rules": [],
            "recursive": True,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    job_id = json.loads(urllib.request.urlopen(request).read().decode("utf-8"))["job_id"]

    deadline = time.time() + 2
    while time.time() < deadline:
        with urllib.request.urlopen(f"{base_url}/api/library/preflight/status?job_id={job_id}") as response:
            status = json.loads(response.read().decode("utf-8"))
        if status["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)

    assert status["status"] == "completed"
    with urllib.request.urlopen(f"{base_url}/api/library/preflight/result?job_id={job_id}") as response:
        result = json.loads(response.read().decode("utf-8"))
    assert result["candidate_assets"] == 1


def test_preflight_api_cancellation(test_server) -> None:
    base_url, db_path, tmp_path = test_server
    
    root = tmp_path / "photos"
    root.mkdir()
    
    payload = {
        "roots": [str(root.resolve())],
        "ignore_rules": [],
        "recursive": True,
        "extensions": "jpg",
    }
    
    # Start Preflight Job
    req = urllib.request.Request(
        f"{base_url}/api/library/preflight/start",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        start_res = json.loads(resp.read().decode("utf-8"))
        
    job_id = start_res["job_id"]
    
    # Cancel Preflight Job
    cancel_req = urllib.request.Request(
        f"{base_url}/api/library/preflight/cancel",
        data=json.dumps({"job_id": job_id}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(cancel_req) as resp:
        cancel_res = json.loads(resp.read().decode("utf-8"))
        
    assert cancel_res["cancelled"] is True
    
    # Check status becomes failed due to user cancellation
    status = "running"
    status_res = {}
    for _ in range(20):
        time.sleep(0.1)
        with urllib.request.urlopen(f"{base_url}/api/library/preflight/status?job_id={job_id}") as resp:
            status_res = json.loads(resp.read().decode("utf-8"))
            status = status_res["status"]
            if status == "failed":
                break
                
    assert status == "failed"
    assert "cancelled" in status_res.get("error", "").lower()


def test_preflight_api_overlapping_rejection(test_server) -> None:
    base_url, db_path, tmp_path = test_server
    
    root1 = tmp_path / "photos"
    root1.mkdir()
    root2 = root1 / "subfolder"
    root2.mkdir()
    
    payload = {
        "roots": [str(root1.resolve()), str(root2.resolve())],
        "ignore_rules": [],
        "recursive": True,
        "extensions": "jpg",
    }
    
    req = urllib.request.Request(
        f"{base_url}/api/library/preflight/start",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with pytest.raises(urllib.error.HTTPError) as exc_info:
        urllib.request.urlopen(req)
        
    assert exc_info.value.code == 400
    body = exc_info.value.read().decode("utf-8")
    assert "Overlapping folders detected" in body
