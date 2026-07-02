"""Shared Ultralytics/YOLO device helpers."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from av_toolbox.core.cache import ModelCache
from av_toolbox.core.hardware import HardwareConfig


_CUDA_NMS_ERROR_HINTS = (
    "no kernel image is available",
    "torchvision::nms",
    "cuda error",
    "not compiled with cuda",
)


def resolve_yolo_device(hardware: HardwareConfig) -> str:
    """Return a YOLO inference device, falling back to CPU for unsafe auto-CUDA.

    Ultralytics uses torchvision NMS during warmup/inference. Some systems expose
    CUDA through PyTorch while torchvision lacks kernels for the installed GPU
    architecture. In auto mode, prefer a working CUDA NMS path; otherwise keep the
    demo/CLI running on CPU. Explicit ``--device cuda`` still reports the setup
    problem instead of silently changing user intent.
    """
    requested = str(hardware.device or "auto")
    candidate = hardware.resolved_device()
    if not str(candidate).startswith("cuda"):
        return "cpu"
    ok, error = _torchvision_cuda_nms_available()
    if ok:
        return str(candidate)
    if requested == "auto":
        return "cpu"
    raise RuntimeError(
        "YOLO CUDA inference is not usable because torchvision NMS failed on "
        f"{candidate}: {error}. Use --device cpu, or install PyTorch/torchvision "
        "CUDA wheels built for this GPU."
    )


def resolve_yolo_model_path(model_name: str, cache: ModelCache) -> str:
    """Resolve bare YOLO .pt names into the shared weights cache.

    Ultralytics downloads missing bare weights into the current working
    directory. Passing an explicit cache path keeps generated model files out of
    the repository root while preserving custom paths and URLs.
    """
    raw = str(model_name).strip()
    if raw.startswith(("http://", "https://")):
        return raw

    candidate = Path(raw).expanduser()
    if candidate.is_absolute() or candidate.parent != Path(".") or candidate.suffix != ".pt":
        return str(candidate)

    cache.ensure()
    return str(cache.weights_dir / candidate.name)


def is_cuda_kernel_compatibility_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(hint in text for hint in _CUDA_NMS_ERROR_HINTS)


@lru_cache(maxsize=1)
def _torchvision_cuda_nms_available() -> tuple[bool, str | None]:
    try:
        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover - depends on optional deps
        return False, f"{type(exc).__name__}: {exc}"
    try:
        boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]], device="cuda")
        scores = torch.tensor([1.0], device="cuda")
        torchvision.ops.nms(boxes, scores, 0.5)
    except Exception as exc:  # pragma: no cover - hardware dependent
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def yolo_predict_with_auto_cpu_retry(model: Any, frame: Any, kwargs: dict[str, Any]) -> Any:
    """Run YOLO prediction and retry once on CPU if auto-CUDA still fails."""
    try:
        return model.predict(frame, **kwargs)[0]
    except Exception as exc:
        if kwargs.get("device") != "cpu" and is_cuda_kernel_compatibility_error(exc):
            retry_kwargs = dict(kwargs)
            retry_kwargs["device"] = "cpu"
            return model.predict(frame, **retry_kwargs)[0]
        raise
