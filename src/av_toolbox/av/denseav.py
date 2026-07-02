"""DenseAV audio-visual correspondence tool."""

from __future__ import annotations

import csv
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import yaml

from av_toolbox.av.denseav_utils import (
    coerce_audio_samples,
    coerce_video_frames,
    downsample_video_frames,
    read_video_pyav,
    trim_av_to_duration,
)
from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.cache import ModelCache
from av_toolbox.core.outputs import make_artifact_paths
from av_toolbox.core.result import AVResult

MODEL_CHECKPOINTS = {
    "sound_and_language": "denseav_2head.ckpt",
    "sound": "denseav_sound.ckpt",
}


class DenseAVTool(BaseTool):
    """Run DenseAV spatial audio-visual correspondence on a video segment."""

    name = "av.denseav"
    category = "av"
    description = "DenseAV audio-visual correspondence and attention overlays."

    def __init__(self) -> None:
        self._model = None
        self._model_source: Path | None = None
        self._model_device: str | None = None
        self._torch = None
        self._transforms = None
        self._resample = None
        self._denseav_shared: dict[str, Any] = {}
        self._denseav_plotting: dict[str, Any] = {}

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        model_name: str = "sound_and_language",
        checkpoint: str | Path | None = None,
        sample_fps: float | None = 5.0,
        load_size: int | None = 224,
        plot_size: int | None = 720,
        audio_sample_rate: int = 16000,
        max_seconds: float | None = None,
        expected_sha256: str | None = None,
        offline: bool = False,
        include_sim_matrix: bool = False,
        export_json: bool = True,
        export_csv: bool = True,
        export_report: bool = True,
        export_overlay: bool = True,
        **_: Any,
    ) -> AVResult:
        if input_path is None:
            raise ValueError("av.denseav requires input_path")
        model_name = model_name or "sound_and_language"
        load_size = load_size or 224
        plot_size = plot_size or 720

        model_source, model_source_kind = self._resolve_model_source(
            context,
            model_name=model_name,
            checkpoint=checkpoint,
            expected_sha256=expected_sha256,
            offline=offline,
        )

        frames, audio, info = read_video_pyav(input_path, max_seconds=max_seconds)
        result_payload = self._analyze_arrays(
            frames=frames,
            audio=audio,
            video_fps=float(info["video_fps"]),
            audio_sample_rate=int(info["audio_fps"]),
            input_path=input_path,
            context=context,
            model_name=model_name,
            model_source=model_source,
            model_source_kind=model_source_kind,
            sample_fps=sample_fps,
            load_size=load_size,
            plot_size=plot_size,
            target_audio_sample_rate=audio_sample_rate,
            max_seconds=max_seconds,
            include_sim_matrix=include_sim_matrix,
            export_json=export_json,
            export_csv=export_csv,
            export_report=export_report,
            export_overlay=export_overlay,
        )
        return result_payload

    def _resolve_model_source(
        self,
        context: ToolRunContext,
        *,
        model_name: str,
        checkpoint: str | Path | None,
        expected_sha256: str | None,
        offline: bool,
    ) -> tuple[Path, str]:
        if checkpoint is not None:
            checkpoint_path = Path(checkpoint).expanduser()
            candidates = [checkpoint_path]
            if not checkpoint_path.is_absolute():
                candidates.extend([
                    Path.cwd() / checkpoint_path,
                    context.cache.weights_dir / checkpoint_path,
                ])

            for candidate in candidates:
                if candidate.exists():
                    resolved = candidate.resolve()
                    if expected_sha256:
                        actual = ModelCache.sha256(resolved)
                        if actual != expected_sha256:
                            raise ValueError(
                                f"Checksum mismatch for {resolved}: "
                                f"expected {expected_sha256}, got {actual}"
                            )
                    return resolved, "explicit_checkpoint"

            searched = ", ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                "DenseAV checkpoint was not found. Searched: "
                f"{searched}. Pass --checkpoint /path/to/file.ckpt or place "
                f"weights under {context.cache.weights_dir}."
            )

        checkpoint_name = MODEL_CHECKPOINTS.get(model_name)
        if checkpoint_name is None:
            options = ", ".join(sorted(MODEL_CHECKPOINTS))
            raise ValueError(
                f"Unknown DenseAV model_name {model_name!r}. Expected one of: {options}"
            )

        try:
            cached = context.cache.resolve_weight(
                checkpoint_name,
                expected_sha256=expected_sha256,
                offline=offline,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{exc}. DenseAV weights are not committed to av-toolbox; "
                f"copy {checkpoint_name} into {context.cache.weights_dir} or pass "
                "--checkpoint /path/to/file.ckpt."
            ) from exc
        return cached.path.resolve(), "av_toolbox_cache"

    def _analyze_arrays(
        self,
        *,
        frames: Any,
        audio: Any,
        video_fps: float,
        audio_sample_rate: int,
        input_path: Path,
        context: ToolRunContext,
        model_name: str,
        model_source: Path,
        model_source_kind: str,
        sample_fps: float | None,
        load_size: int,
        plot_size: int,
        target_audio_sample_rate: int,
        max_seconds: float | None,
        include_sim_matrix: bool,
        export_json: bool,
        export_csv: bool,
        export_report: bool,
        export_overlay: bool,
    ) -> AVResult:
        original_frames = coerce_video_frames(frames, frame_format="rgb")
        audio_tensor = coerce_audio_samples(audio)
        source_video_fps = float(video_fps)
        original_frames, audio_tensor = trim_av_to_duration(
            original_frames,
            audio_tensor,
            video_fps=source_video_fps,
            audio_sample_rate=int(audio_sample_rate),
            max_seconds=max_seconds,
        )
        original_frames, effective_video_fps = downsample_video_frames(
            original_frames,
            source_fps=source_video_fps,
            target_fps=sample_fps,
        )
        if len(original_frames) == 0:
            raise ValueError("No frames provided for DenseAV analysis")
        if audio_tensor.numel() == 0:
            raise ValueError("No audio provided for DenseAV analysis")

        device = context.hardware.resolved_device()
        self._load_model(model_source, device)
        audio_tensor = self._prepare_audio_for_denseav(
            audio_tensor,
            audio_sample_rate,
            target_audio_sample_rate,
        )

        torch = self._torch
        transforms = self._transforms
        assert torch is not None
        assert transforms is not None

        from PIL import Image

        crop_to_divisor = self._denseav_shared["crop_to_divisor"]
        norm = self._denseav_shared["norm"]

        img_transform = transforms.Compose([
            transforms.Resize(load_size, Image.BILINEAR),
            lambda x: crop_to_divisor(x, 8),
            lambda x: x.to(torch.float32) / 255,
            norm,
        ])
        plotting_img_transform = transforms.Compose([
            transforms.Resize(plot_size, Image.BILINEAR),
            lambda x: crop_to_divisor(x, 8),
            lambda x: x.to(torch.float32) / 255,
        ])

        dev = next(self._model.parameters()).device
        frames_for_model = torch.cat(
            [
                img_transform(frame.permute(2, 0, 1)).unsqueeze(0)
                for frame in original_frames
            ],
            dim=0,
        )
        frames_to_plot = plotting_img_transform(original_frames.permute(0, 3, 1, 2))

        batch_size = context.hardware.batch_size or 2
        amp_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if context.hardware.fp16 and dev.type == "cuda"
            else nullcontext()
        )

        with torch.no_grad(), amp_context:
            audio_feats = self._model.forward_audio({"audio": audio_tensor.to(dev)})
            audio_feats = {key: value.cpu() for key, value in audio_feats.items()}

            image_feats = self._model.forward_image(
                {"frames": frames_for_model.unsqueeze(0).to(dev)},
                max_batch_size=batch_size,
            )
            image_feats = {key: value.cpu() for key, value in image_feats.items()}

            if (
                getattr(self._model.sim_agg, "use_cls", False)
                and "audio_cls" in audio_feats
            ):
                image_batch = image_feats["image_feats"].shape[0]
                if audio_feats["audio_cls"].shape[0] == 1 and image_batch > 1:
                    audio_feats["audio_cls"] = (
                        audio_feats["audio_cls"].expand(image_batch, -1).contiguous()
                    )

            sim_by_head = self._model.sim_agg.get_pairwise_sims(
                {**image_feats, **audio_feats},
                raw=False,
                agg_sim=False,
                agg_heads=False,
            ).mean(dim=-2).cpu()
            sim_by_head = self._denseav_shared["blur_dim"](
                sim_by_head,
                window=3,
                dim=-1,
            )

        artifacts = make_artifact_paths(
            input_path=input_path,
            output_dir=context.output_dir,
            tool_name=self.name,
        )
        config = {
            "tool_name": self.name,
            "model_name": model_name,
            "model_source": str(model_source),
            "model_source_kind": model_source_kind,
            "sample_fps": sample_fps,
            "effective_video_fps": effective_video_fps,
            "source_video_fps": source_video_fps,
            "load_size": load_size,
            "plot_size": plot_size,
            "source_audio_sample_rate": audio_sample_rate,
            "audio_sample_rate": target_audio_sample_rate,
            "max_seconds": max_seconds,
            "batch_size": batch_size,
            "fp16": bool(context.hardware.fp16 and dev.type == "cuda"),
            "device": str(dev),
        }
        artifacts.config_path.write_text(yaml.safe_dump(config, sort_keys=True))

        overlay_path = None
        overlay_warning = None
        attention_videos: list[str] = []
        if export_overlay:
            if int(sim_by_head.shape[0]) < 2:
                overlay_warning = (
                    "DenseAV plotting requires at least two sampled frames; "
                    "increase --max-seconds or --sample-fps to render overlays."
                )
            else:
                overlay_path = artifacts.overlay_path
                self._denseav_plotting["plot_attention_video"](
                    sim_by_head,
                    frames_to_plot,
                    audio_tensor,
                    effective_video_fps,
                    target_audio_sample_rate,
                    str(overlay_path),
                )
                attention_videos.append(str(overlay_path))

                num_heads = int(getattr(self._model.sim_agg, "num_heads", 1))
                if num_heads >= 2:
                    two_head_path = artifacts.prefix.with_name(
                        artifacts.prefix.name + "_2head_attention.mp4"
                    )
                    self._denseav_plotting["plot_2head_attention_video"](
                        sim_by_head,
                        frames_to_plot,
                        audio_tensor,
                        effective_video_fps,
                        target_audio_sample_rate,
                        str(two_head_path),
                    )
                    attention_videos.append(str(two_head_path))

        sim_matrix = self._squeeze_sim_by_head(sim_by_head)
        summary = _summarize_similarity(sim_by_head)
        summary.update({
            "model_name": model_name,
            "model_source_kind": model_source_kind,
            "source_video_fps": round(source_video_fps, 6),
            "denseav_video_fps": round(float(effective_video_fps), 6),
            "num_video_frames": int(original_frames.shape[0]),
            "source_audio_sample_rate": int(audio_sample_rate),
            "audio_sample_rate": int(target_audio_sample_rate),
            "attention_video_count": len(attention_videos),
            "overlay_warning": overlay_warning,
        })

        timeline_payload: dict[str, Any] = {
            "tool_name": self.name,
            "input_path": str(input_path),
            "summary": summary,
            "attention_videos": attention_videos,
            "overlay_warning": overlay_warning,
            "similarity_shape": [int(dim) for dim in sim_by_head.shape],
            "per_head": summary["per_head"],
        }
        if include_sim_matrix:
            timeline_payload["sim_matrix"] = sim_matrix.tolist()

        if export_json:
            artifacts.timeline_json.write_text(json.dumps(timeline_payload, indent=2))
        if export_csv:
            _write_csv(artifacts.csv_path, summary["per_head"])
        if export_report:
            artifacts.report_html.write_text(_html_report(timeline_payload))

        artifacts.log_path.write_text(
            "\n".join([
                f"tool={self.name}",
                f"input={input_path}",
                f"model_name={model_name}",
                f"model_source={model_source}",
                f"frames={int(original_frames.shape[0])}",
                f"denseav_video_fps={effective_video_fps:.6f}",
                f"mean_similarity={summary['mean_similarity']:.6f}",
                f"attention_videos={len(attention_videos)}",
                f"overlay_warning={overlay_warning}",
            ])
            + "\n"
        )

        return AVResult(
            tool_name=self.name,
            input_path=input_path,
            output_dir=context.output_dir,
            overlay_path=overlay_path,
            timeline_json=artifacts.timeline_json if export_json else None,
            csv_path=artifacts.csv_path if export_csv else None,
            report_html=artifacts.report_html if export_report else None,
            config_path=artifacts.config_path,
            log_path=artifacts.log_path,
            metadata={
                "summary": summary,
                "attention_videos": attention_videos,
                "overlay_warning": overlay_warning,
                "model_source": str(model_source),
                "model_source_kind": model_source_kind,
            },
        )

    def _load_model(self, model_source: Path, device: str) -> None:
        if (
            self._model is not None
            and self._model_source == model_source
            and self._model_device == device
        ):
            return

        torch, transforms, resample, shared, plotting, aligner = _denseav_imports()
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(f"Requested DenseAV device {device!r}, but CUDA is unavailable")

        self._torch = torch
        self._transforms = transforms
        self._resample = resample
        self._denseav_shared = shared
        self._denseav_plotting = plotting

        model = aligner.load_from_checkpoint(
            str(model_source),
            loss_leak=0.0,
            use_cached_embs=False,
            strict=True,
        )
        if hasattr(model, "set_full_train"):
            model.set_full_train(True)
        model.eval()
        model = model.to(device)

        self._model = model
        self._model_source = model_source
        self._model_device = device

    def _prepare_audio_for_denseav(
        self,
        audio: Any,
        audio_sample_rate: int,
        target_audio_sample_rate: int,
    ) -> Any:
        """Downmix to mono first, then resample to DenseAV's sample rate."""
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio_sample_rate != target_audio_sample_rate:
            if self._resample is None:
                _, _, resample, _, _, _ = _denseav_imports()
                self._resample = resample
            audio = self._resample(audio, audio_sample_rate, target_audio_sample_rate)
        return audio

    @staticmethod
    def _squeeze_sim_by_head(sim_by_head: Any) -> Any:
        """Drop a single head dimension while preserving real multi-head output."""
        if sim_by_head.ndim >= 1 and sim_by_head.shape[0] == 1:
            return sim_by_head.squeeze(0)
        return sim_by_head


def _denseav_imports() -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], Any]:
    try:
        import torch
        import torchvision.transforms as transforms
        from torchaudio.functional import resample
    except ImportError as exc:
        raise ImportError(
            "av.denseav requires torch, torchvision, and torchaudio. "
            "Install with: pip install -e '.[denseav]'"
        ) from exc

    try:
        from denseav.plotting import plot_2head_attention_video, plot_attention_video
        from denseav.shared import blur_dim, crop_to_divisor, norm
        from denseav.train import LitAVAligner
    except ImportError as exc:
        raise ImportError(
            "av.denseav requires the DenseAV package. "
            "Install with: pip install \"git+https://github.com/mhamilton723/DenseAV.git\""
        ) from exc

    return (
        torch,
        transforms,
        resample,
        {
            "blur_dim": blur_dim,
            "crop_to_divisor": crop_to_divisor,
            "norm": norm,
        },
        {
            "plot_attention_video": plot_attention_video,
            "plot_2head_attention_video": plot_2head_attention_video,
        },
        LitAVAligner,
    )


def _summarize_similarity(sim_by_head: Any) -> dict[str, Any]:
    sim = sim_by_head.detach().float().cpu()
    if sim.ndim >= 2:
        head_tensors = [sim[:, head_idx, ...] for head_idx in range(int(sim.shape[1]))]
    else:
        head_tensors = [sim]

    per_head = []
    for idx, head in enumerate(head_tensors):
        per_head.append({
            "head": int(idx),
            "mean": round(float(head.mean()), 6),
            "min": round(float(head.min()), 6),
            "max": round(float(head.max()), 6),
            "std": round(float(head.std(unbiased=False)), 6),
        })
    return {
        "mean_similarity": round(float(sim.mean()), 6),
        "min_similarity": round(float(sim.min()), 6),
        "max_similarity": round(float(sim.max()), 6),
        "std_similarity": round(float(sim.std(unbiased=False)), 6),
        "num_similarity_frames": int(sim.shape[0]) if sim.ndim >= 1 else 0,
        "num_heads": len(head_tensors),
        "similarity_shape": [int(dim) for dim in sim.shape],
        "per_head": per_head,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["head", "mean", "min", "max", "std"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _html_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    rows = "\n".join(
        "<tr>"
        f"<td>{row['head']}</td>"
        f"<td>{row['mean']:.6f}</td>"
        f"<td>{row['min']:.6f}</td>"
        f"<td>{row['max']:.6f}</td>"
        f"<td>{row['std']:.6f}</td>"
        "</tr>"
        for row in summary["per_head"]
    )
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>av.denseav report</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 2rem; line-height: 1.45; }}
    table {{ border-collapse: collapse; margin-top: 1rem; }}
    th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>av.denseav</h1>
  <p>Mean similarity: {summary['mean_similarity']:.6f}</p>
  <p>Frames: {summary['num_video_frames']} at {summary['denseav_video_fps']:.3f} fps</p>
  <table>
    <thead><tr><th>Head</th><th>Mean</th><th>Min</th><th>Max</th><th>Std</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</body>
</html>
"""
