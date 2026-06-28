from __future__ import annotations

import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_public_package_metadata_is_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    project = pyproject["project"]

    assert project["name"] == "av-toolbox"
    assert project["license"] == "MIT"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"]
    assert project["maintainers"]
    assert "audio-visual" in project["keywords"]
    assert "Programming Language :: Python :: 3.12" in project["classifiers"]

    urls = project["urls"]
    assert urls["Repository"] == "https://github.com/yanpeng0520/av-toolbox"
    assert urls["Documentation"].endswith("/tree/main/docs")
    assert urls["Issues"].endswith("/issues")


def test_license_and_manifest_protect_release_archives() -> None:
    assert (ROOT / "LICENSE").exists()

    manifest = (ROOT / "MANIFEST.in").read_text()
    for path in (
        ".codex",
        "omni-video-pipeline",
        "outputs",
        "testing",
        "data_samples",
        "multicam_samples",
        "models",
        "data_segments",
    ):
        assert f"prune {path}" in manifest

    for pattern in ("*.ckpt", "*.pt", "*.pth", "*.onnx", "*.safetensors", "*.h5"):
        assert f"global-exclude {pattern}" in manifest

    assert "include data_segments/Clever_Cat_Outsmarts_Warrior_square.mp4" in manifest
