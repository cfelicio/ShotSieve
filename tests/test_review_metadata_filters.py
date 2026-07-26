import json
import sqlite3
import urllib.request

def test_metadata_filtering_and_sorting(test_server) -> None:
    base_url, db_path, tmp_path = test_server

    # 1. Establish database connection and insert mock metadata files
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    # Enable FOREIGN KEY support
    conn.execute("PRAGMA foreign_keys = ON")
    
    # Mock data configuration
    mock_files = [
        # format, width, height, size_bytes, path
        ("jpg", 6000, 4000, 5000000, "photos/jpeg_24mp.jpg"),  # 24 MP, 5 MB
        ("png", 800, 600, 200000, "photos/screenshot.png"),  # 0.48 MP, 200 KB
        ("tiff", 3000, 2000, 18000000, "photos/scan.tiff"),  # 6.0 MP, 18 MB
        ("heic", 4032, 3024, 1200000, "photos/iphone.heic"),  # 12.19 MP, 1.2 MB
        ("nef", 8256, 5504, 45000000, "photos/nikon_raw.nef"),  # 45.44 MP, 45 MB
        ("bmp", 1024, 768, 3000000, "photos/bitmap.bmp"),  # 0.78 MP, 3 MB
        (None, None, None, None, "photos/corrupted.xyz"),  # Unknown metadata
    ]
    
    # Insert files and corresponding score records (so they are visible in review browser)
    for idx, (fmt, w, h, size, p) in enumerate(mock_files, start=1):
        conn.execute(
            """
            INSERT INTO files(id, path, path_key, format, width, height, size_bytes, preview_status, scan_status)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'ready', 'new')
            """,
            (idx, p, p.lower(), fmt, w, h, size)
        )
        conn.execute(
            """
            INSERT INTO scores(file_id, overall_score, learned_score_normalized)
            VALUES (?, 50.0, 50.0)
            """
            , (idx,)
        )
        
    conn.commit()
    conn.close()

    def get_results(query_str: str) -> dict:
        with urllib.request.urlopen(f"{base_url}/api/files?{query_str}") as resp:
            return json.loads(resp.read().decode("utf-8"))

    # Verify basic format filtering
    # JPEG group matching (jpg, jpeg)
    res = get_results("formats=jpeg")
    assert res["total"] == 1
    assert res["items"][0]["path"] == "photos/jpeg_24mp.jpg"

    # Multi formats
    res = get_results("formats=raw,png")
    assert res["total"] == 2
    paths = {item["path"] for item in res["items"]}
    assert "photos/screenshot.png" in paths
    assert "photos/nikon_raw.nef" in paths

    # Other category (bmp is other, xyz is other but it has format=None so it is NULL, should match other)
    res = get_results("formats=other")
    assert res["total"] == 2
    paths = {item["path"] for item in res["items"]}
    assert "photos/bitmap.bmp" in paths
    assert "photos/corrupted.xyz" in paths

    # Metadata completeness filter
    # Unknown metadata
    res = get_results("metadata=unknown")
    assert res["total"] == 1
    assert res["items"][0]["path"] == "photos/corrupted.xyz"

    # Valid metadata
    res = get_results("metadata=valid")
    assert res["total"] == 6

    # Megapixels filtering (min_mp / max_mp)
    # min_mp = 10 (heic, jpg, raw)
    res = get_results("min_mp=10")
    assert res["total"] == 3
    paths = {item["path"] for item in res["items"]}
    assert "photos/jpeg_24mp.jpg" in paths
    assert "photos/iphone.heic" in paths
    assert "photos/nikon_raw.nef" in paths

    # max_mp = 5 (png, bmp)
    res = get_results("max_mp=5")
    assert res["total"] == 2
    paths = {item["path"] for item in res["items"]}
    assert "photos/screenshot.png" in paths
    assert "photos/bitmap.bmp" in paths

    # Width precision
    res = get_results("min_width=5000")
    assert res["total"] == 2
    paths = {item["path"] for item in res["items"]}
    assert "photos/jpeg_24mp.jpg" in paths
    assert "photos/nikon_raw.nef" in paths

    # File size precision
    # min_size = 10 MB (10,000,000 bytes) -> tiff (18MB), raw (45MB)
    res = get_results("min_size=10000000")
    assert res["total"] == 2
    paths = {item["path"] for item in res["items"]}
    assert "photos/scan.tiff" in paths
    assert "photos/nikon_raw.nef" in paths

    # max_size = 500 KB (500,000 bytes) -> png (200KB)
    res = get_results("max_size=500000")
    assert res["total"] == 1
    assert res["items"][0]["path"] == "photos/screenshot.png"

    # Sorting options
    # Size Descending
    res = get_results("sort=size_desc&metadata=valid")
    assert [item["path"] for item in res["items"]] == [
        "photos/nikon_raw.nef",
        "photos/scan.tiff",
        "photos/jpeg_24mp.jpg",
        "photos/bitmap.bmp",
        "photos/iphone.heic",
        "photos/screenshot.png",
    ]

    # Resolution Ascending
    res = get_results("sort=resolution_asc&metadata=valid")
    assert [item["path"] for item in res["items"]] == [
        "photos/screenshot.png",
        "photos/bitmap.bmp",
        "photos/scan.tiff",
        "photos/iphone.heic",
        "photos/jpeg_24mp.jpg",
        "photos/nikon_raw.nef",
    ]

    # Format A-Z
    res = get_results("sort=format&metadata=valid")
    assert [item["path"] for item in res["items"]] == [
        "photos/bitmap.bmp",
        "photos/iphone.heic",
        "photos/jpeg_24mp.jpg",
        "photos/nikon_raw.nef",
        "photos/screenshot.png",
        "photos/scan.tiff",
    ]
