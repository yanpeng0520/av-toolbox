from __future__ import annotations

import pytest

from av_toolbox.core.cache import ModelCache
from av_toolbox.core.hardware import HardwareConfig
from av_toolbox.video import yolo_utils


def test_auto_yolo_cuda_falls_back_to_cpu_when_nms_is_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HardwareConfig, "resolved_device", lambda self: "cuda")
    monkeypatch.setattr(yolo_utils, "_torchvision_cuda_nms_available", lambda: (False, "no kernel image"))

    assert yolo_utils.resolve_yolo_device(HardwareConfig()) == "cpu"


def test_explicit_yolo_cuda_reports_nms_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HardwareConfig, "resolved_device", lambda self: "cuda")
    monkeypatch.setattr(yolo_utils, "_torchvision_cuda_nms_available", lambda: (False, "no kernel image"))

    with pytest.raises(RuntimeError, match="YOLO CUDA inference is not usable"):
        yolo_utils.resolve_yolo_device(HardwareConfig(device="cuda"))


def test_yolo_cuda_keeps_cuda_when_nms_is_usable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(HardwareConfig, "resolved_device", lambda self: "cuda")
    monkeypatch.setattr(yolo_utils, "_torchvision_cuda_nms_available", lambda: (True, None))

    assert yolo_utils.resolve_yolo_device(HardwareConfig()) == "cuda"


def test_bare_yolo_weight_resolves_inside_model_cache(tmp_path) -> None:
    cache = ModelCache(tmp_path / "cache")

    resolved = yolo_utils.resolve_yolo_model_path("yolov8n.pt", cache)

    assert resolved == str(tmp_path / "cache" / "weights" / "yolov8n.pt")
    assert (tmp_path / "cache" / "weights").is_dir()


def test_custom_yolo_path_and_url_are_preserved(tmp_path) -> None:
    cache = ModelCache(tmp_path / "cache")

    assert yolo_utils.resolve_yolo_model_path("models/custom.pt", cache) == "models/custom.pt"
    assert yolo_utils.resolve_yolo_model_path("https://example.test/yolov8n.pt", cache) == "https://example.test/yolov8n.pt"
