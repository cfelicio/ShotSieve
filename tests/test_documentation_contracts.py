from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from shotsieve.learned_iqa_catalog import supported_learned_models  # noqa: E402
from shotsieve.release_targets import runtime_pack_release_targets  # noqa: E402


def read_project_file(relative_path: str) -> str:
    return (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")


def test_readme_links_the_maintained_developer_guides() -> None:
    readme = read_project_file("README.md")
    for guide in (
        "docs/architecture.md",
        "docs/configuration.md",
        "docs/api.md",
        "docs/troubleshooting.md",
        "docs/contributing.md",
    ):
        assert f"]({guide})" in readme
    assert "shotsieve-desktop --check-runtime" in readme


def test_building_uses_current_commands_and_target_ids() -> None:
    building = read_project_file("docs/building.md")
    assert 'python -m pip install -e ".[lint]"' in building
    assert "python -m build --wheel --outdir dist" in building
    for target in runtime_pack_release_targets():
        assert target.id in building


def test_public_docs_track_catalog_and_release_names() -> None:
    readme = read_project_file("README.md")
    building = read_project_file("docs/building.md")
    configuration = read_project_file("docs/configuration.md")
    for target in runtime_pack_release_targets():
        assert target.executableName in readme
        assert target.archiveName in building or target.archiveName in readme
    for model_name in supported_learned_models():
        assert model_name in readme
        assert model_name in configuration


def test_configuration_and_api_docs_track_runtime_contracts() -> None:
    configuration = read_project_file("docs/configuration.md")
    api = read_project_file("docs/api.md")
    for env_name in (
        "HF_TOKEN",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "TORCH_HOME",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_TORCH",
        "SHOTSIEVE_BOOTSTRAP_AUTO_INSTALL_LEARNED_IQA",
        "SHOTSIEVE_BOOTSTRAP_MANIFEST_URL",
    ):
        assert env_name in configuration
    for route in (
        "/api/options",
        "/api/files",
        "/api/review/batch",
        "/api/scan/start",
        "/api/score/status",
        "/api/score/result",
        "/api/models/prepare/start",
    ):
        assert route in api
    assert "selection_revision" in api
    assert "127.0.0.1:8765" in api
