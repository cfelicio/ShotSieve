from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "preview-build.yml"


def test_preview_workflow_builds_branch_artifacts_without_publishing_a_release() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "name: preview-build" in workflow
    assert "      - 0.5.x" in workflow
    assert "  workflow_dispatch:" in workflow
    assert workflow.count("if: github.ref == 'refs/heads/0.5.x'") == 3
    assert "contents: read" in workflow
    assert "tier1_release_matrix" in workflow
    assert "python -m pytest -q" in workflow
    assert "python scripts/build_portable_bundle.py --target" in workflow
    assert "uses: actions/upload-artifact@v6" in workflow
    assert "retention-days: 7" in workflow
    assert "softprops/action-gh-release" not in workflow
    assert "publish-release" not in workflow
    assert "contents: write" not in workflow


def test_build_guide_explains_preview_artifact_scope() -> None:
    guide = (PROJECT_ROOT / "docs" / "building.md").read_text(encoding="utf-8")

    assert "Pushes to `0.5.x` run the `preview-build` workflow" in guide
    assert "does not create or update a" in guide
    assert "anyone with repository read access can download them" in guide
