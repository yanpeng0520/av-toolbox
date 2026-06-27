"""DenseAV decoding and buffer preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def coerce_video_frames(
    frames: Any,
    frame_format: str = "rgb",
) -> Any:
    """Convert buffered frames to THWC RGB uint8 on CPU."""
    np, torch = _numpy_torch()

    if isinstance(frames, torch.Tensor):
        tensor = frames.detach().cpu()
    elif isinstance(frames, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(frames))
    elif isinstance(frames, (list, tuple)):
        if len(frames) == 0:
            return torch.zeros(0, 0, 0, 3, dtype=torch.uint8)
        if isinstance(frames[0], torch.Tensor):
            tensor = torch.stack([f.detach().cpu() for f in frames])
        else:
            tensor = torch.from_numpy(np.ascontiguousarray(np.stack(frames)))
    else:
        raise TypeError("frames must be a tensor, ndarray, or list of frames")

    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("frames must have shape THWC, TCHW, or HWC")

    if tensor.shape[-1] in (1, 3, 4):
        thwc = tensor
    elif tensor.shape[1] in (1, 3, 4):
        thwc = tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError("frames must include 1, 3, or 4 image channels")

    if thwc.shape[-1] == 4:
        thwc = thwc[..., :3]
    elif thwc.shape[-1] == 1:
        thwc = thwc.repeat_interleave(3, dim=-1)

    frame_format = frame_format.lower()
    if frame_format == "bgr":
        thwc = thwc[..., [2, 1, 0]]
    elif frame_format != "rgb":
        raise ValueError("frame_format must be 'rgb' or 'bgr'")

    if torch.is_floating_point(thwc):
        if thwc.numel() and float(thwc.max()) <= 1.0:
            thwc = thwc * 255.0
        thwc = thwc.clamp(0, 255).to(torch.uint8)
    elif thwc.dtype != torch.uint8:
        thwc = thwc.to(torch.int64).clamp(0, 255).to(torch.uint8)

    return thwc.contiguous()


def coerce_audio_samples(audio: Any) -> Any:
    """Convert buffered audio to CN float32 on CPU."""
    np, torch = _numpy_torch()

    if isinstance(audio, torch.Tensor):
        tensor = audio.detach().cpu()
    elif isinstance(audio, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(audio))
    else:
        raise TypeError("audio must be a tensor or ndarray")

    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 2:
        if tensor.shape[0] > 8 and tensor.shape[1] <= 8:
            tensor = tensor.transpose(0, 1)
    else:
        raise ValueError("audio must have shape N, CN, or NC")

    if torch.is_floating_point(tensor):
        return tensor.to(torch.float32).contiguous()
    if tensor.dtype == torch.uint8:
        return tensor.to(torch.float32).sub(128.0).div(128.0).contiguous()

    dtype_info = torch.iinfo(tensor.dtype)
    scale = float(max(abs(dtype_info.min), dtype_info.max))
    return tensor.to(torch.float32).div(scale).contiguous()


def trim_av_to_duration(
    frames: Any,
    audio: Any,
    *,
    video_fps: float,
    audio_sample_rate: int,
    max_seconds: float | None = None,
) -> tuple[Any, Any]:
    """Trim frames and audio to a shared maximum duration."""
    if video_fps <= 0:
        raise ValueError("video_fps must be greater than zero")
    if audio_sample_rate <= 0:
        raise ValueError("audio_sample_rate must be greater than zero")
    if max_seconds is None:
        return frames, audio

    max_frames = max(1, int(float(max_seconds) * video_fps))
    max_samples = max(1, int(float(max_seconds) * audio_sample_rate))
    return frames[:max_frames], audio[:, :max_samples]


def downsample_video_frames(
    frames: Any,
    *,
    source_fps: float,
    target_fps: float | None = None,
) -> tuple[Any, float]:
    """Downsample a THWC frame tensor and return the effective FPS."""
    if source_fps <= 0:
        raise ValueError("source_fps must be greater than zero")
    if target_fps is None:
        return frames, float(source_fps)

    target_fps = float(target_fps)
    if target_fps <= 0:
        raise ValueError("target_fps must be greater than zero")
    if target_fps >= source_fps:
        return frames, float(source_fps)

    torch = _torch()
    duration = float(frames.shape[0]) / float(source_fps)
    sample_times = torch.arange(0.0, duration, 1.0 / target_fps)
    indices = torch.floor(sample_times * source_fps).to(torch.long)
    indices = torch.clamp(indices, min=0, max=max(int(frames.shape[0]) - 1, 0))
    if indices.numel() == 0:
        indices = torch.tensor([0], dtype=torch.long)
    indices = torch.unique_consecutive(indices)
    sampled = frames.index_select(0, indices).contiguous()
    return sampled, target_fps


def read_video_pyav(
    path: str | Path,
    *,
    max_seconds: float | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Read video frames and audio with PyAV.

    Returns frames as THWC RGB uint8, audio as CN float32, and a metadata dict
    containing ``video_fps`` and ``audio_fps``.
    """
    torch = _torch()
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "av.denseav requires PyAV. Install with: pip install -e '.[denseav]'"
        ) from exc

    container = av.open(str(path))
    try:
        if not container.streams.video:
            raise ValueError(f"No video stream decoded from {path}")
        video_stream = container.streams.video[0]
        video_fps = float(video_stream.average_rate or 0.0)
        if video_fps <= 0:
            raise ValueError(f"Cannot determine video FPS for {path}")
        max_video_frames = (
            max(1, int(float(max_seconds) * video_fps))
            if max_seconds is not None
            else None
        )

        frames = []
        for idx, frame in enumerate(container.decode(video=0)):
            if max_video_frames is not None and idx >= max_video_frames:
                break
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))

        frames_tensor = (
            torch.stack(frames)
            if frames
            else torch.zeros(0, 0, 0, 3, dtype=torch.uint8)
        )

        container.seek(0)
        audio_stream = container.streams.audio[0] if container.streams.audio else None
        if audio_stream:
            audio_fps = int(audio_stream.sample_rate)
            max_audio_samples = (
                max(1, int(float(max_seconds) * audio_fps))
                if max_seconds is not None
                else None
            )
            samples = []
            total_samples = 0
            for frame in container.decode(audio=0):
                tensor = _audio_frame_to_tensor(frame)
                samples.append(tensor)
                total_samples += tensor.shape[1]
                if max_audio_samples is not None and total_samples >= max_audio_samples:
                    break
            audio_tensor = torch.cat(samples, dim=1) if samples else torch.zeros(1, 0)
            if max_audio_samples is not None:
                audio_tensor = audio_tensor[:, :max_audio_samples]
        else:
            audio_fps = 16000
            audio_tensor = torch.zeros(1, 0)
    finally:
        container.close()

    return frames_tensor, audio_tensor, {"video_fps": video_fps, "audio_fps": audio_fps}


def load_video_for_denseav(
    path: str | Path,
    *,
    load_size: int = 224,
    sr: int = 16000,
    patch_size: int = 8,
    max_seconds: float | None = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load and preprocess a video using DenseAV's standard tensor contract."""
    torch = _torch()
    try:
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "av.denseav requires torch. Install with: pip install -e '.[denseav]'"
        ) from exc

    frames, audio, info = read_video_pyav(path, max_seconds=max_seconds)
    if frames.numel() == 0:
        raise ValueError(f"No video frames decoded from {path}")
    if audio.numel() == 0:
        raise ValueError(f"No audio decoded from {path} (DenseAV requires audio)")

    frames = frames.permute(0, 3, 1, 2).contiguous().float().div_(255.0)
    _, _, h, w = frames.shape
    if h <= w:
        new_h = load_size
        new_w = int(round(w * load_size / h))
    else:
        new_w = load_size
        new_h = int(round(h * load_size / w))
    frames = F.interpolate(
        frames,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    _, _, h, w = frames.shape
    frames = frames[:, :, : patch_size * (h // patch_size), : patch_size * (w // patch_size)]

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    frames = (frames - mean) / std

    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    src_sr = int(info["audio_fps"])
    if src_sr != sr:
        try:
            from torchaudio.functional import resample
        except ImportError as exc:
            raise ImportError(
                "av.denseav audio resampling requires torchaudio. "
                "Install with: pip install -e '.[denseav]'"
            ) from exc
        audio = resample(audio, orig_freq=src_sr, new_freq=sr)

    return frames, audio, info


def _audio_frame_to_tensor(frame: Any) -> Any:
    torch = _torch()
    audio = torch.from_numpy(frame.to_ndarray())
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)

    if torch.is_floating_point(audio):
        return audio.float()
    if audio.dtype == torch.uint8:
        return audio.float().sub(128.0).div(128.0)

    dtype_info = torch.iinfo(audio.dtype)
    scale = float(max(abs(dtype_info.min), dtype_info.max))
    return audio.float().div(scale)


def _numpy_torch() -> tuple[Any, Any]:
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        raise ImportError(
            "av.denseav buffer helpers require NumPy and torch. "
            "Install with: pip install -e '.[denseav]'"
        ) from exc
    return np, torch


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "av.denseav requires torch. Install with: pip install -e '.[denseav]'"
        ) from exc
    return torch
