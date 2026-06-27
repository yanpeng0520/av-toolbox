from __future__ import annotations

import os
from pathlib import Path

import pytest

import av_toolbox
from av_toolbox.av.denseav import DenseAVTool
from av_toolbox.cli import build_parser
from av_toolbox.av.denseav_utils import (
    coerce_audio_samples,
    coerce_video_frames,
    downsample_video_frames,
    trim_av_to_duration,
)
from av_toolbox.core.base_tool import ToolRunContext
from av_toolbox.core.cache import ModelCache, default_cache_dir
from av_toolbox.core.hardware import HardwareConfig


def test_denseav_registered() -> None:
    assert "av.denseav" in [tool["name"] for tool in av_toolbox.list_tools()]


def test_denseav_defaults_to_720px_plot_size_and_cli_override() -> None:
    params = DenseAVTool._run.__kwdefaults__
    assert params["load_size"] == 224
    assert params["plot_size"] == 720

    args = build_parser().parse_args([
        "av",
        "denseav",
        "clip.mp4",
        "--plot-size",
        "1080",
        "--load-size",
        "256",
    ])
    assert args.plot_size == 1080
    assert args.load_size == 256


def test_denseav_default_cache_resolution(tmp_path) -> None:
    cache = ModelCache(tmp_path / "cache")
    cache.ensure()
    checkpoint = cache.weights_dir / "denseav_2head.ckpt"
    checkpoint.write_bytes(b"not a real checkpoint")
    context = ToolRunContext(
        output_dir=tmp_path,
        hardware=HardwareConfig(device="cpu"),
        cache=cache,
        workspace=tmp_path / "workspace",
    )

    source, source_kind = DenseAVTool()._resolve_model_source(
        context,
        model_name="sound_and_language",
        checkpoint=None,
        expected_sha256=None,
        offline=True,
    )

    assert source == checkpoint.resolve()
    assert source_kind == "av_toolbox_cache"


def test_denseav_explicit_checkpoint_resolution(tmp_path) -> None:
    checkpoint = tmp_path / "legacy_denseav.ckpt"
    checkpoint.write_bytes(b"legacy checkpoint")
    context = ToolRunContext(
        output_dir=tmp_path,
        hardware=HardwareConfig(device="cpu"),
        cache=ModelCache(tmp_path / "cache"),
        workspace=tmp_path / "workspace",
    )

    source, source_kind = DenseAVTool()._resolve_model_source(
        context,
        model_name="sound_and_language",
        checkpoint=checkpoint,
        expected_sha256=None,
        offline=True,
    )

    assert source == checkpoint.resolve()
    assert source_kind == "explicit_checkpoint"


def test_denseav_missing_checkpoint_message_uses_cache_dir(tmp_path) -> None:
    root = Path(__file__).resolve().parents[1]
    demo = root / "data_segments" / "Clever_Cat_Outsmarts_Warrior_square.mp4"

    with pytest.raises(FileNotFoundError, match="denseav_2head.ckpt"):
        av_toolbox.run_tool(
            "av.denseav",
            input_path=demo,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            model_name="sound_and_language",
            max_seconds=0.2,
            device="cpu",
            export_json=False,
            export_csv=False,
            export_report=False,
            export_overlay=False,
        )


def test_denseav_buffer_helpers_accept_common_shapes() -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")

    bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    bgr[..., 0] = 10
    bgr[..., 1] = 20
    bgr[..., 2] = 30
    frames = coerce_video_frames([bgr, bgr], frame_format="bgr")

    assert tuple(frames.shape) == (2, 4, 5, 3)
    assert frames.dtype == torch.uint8
    assert frames[0, 0, 0].tolist() == [30, 20, 10]

    nc = np.zeros((160, 2), dtype=np.float32)
    cn = np.zeros((2, 160), dtype=np.float32)
    assert tuple(coerce_audio_samples(nc).shape) == (2, 160)
    assert tuple(coerce_audio_samples(cn).shape) == (2, 160)


def test_denseav_trim_downsample_and_audio_prep() -> None:
    torch = pytest.importorskip("torch")

    frames = torch.zeros(20, 4, 5, 3, dtype=torch.uint8)
    audio = torch.stack([torch.zeros(1000), torch.ones(1000)])

    clipped_frames, clipped_audio = trim_av_to_duration(
        frames,
        audio,
        video_fps=10.0,
        audio_sample_rate=1000,
        max_seconds=0.5,
    )
    sampled, effective_fps = downsample_video_frames(
        clipped_frames,
        source_fps=10.0,
        target_fps=5.0,
    )

    tool = DenseAVTool()
    tool._resample = lambda samples, orig_freq, new_freq: samples
    prepared = tool._prepare_audio_for_denseav(
        clipped_audio,
        audio_sample_rate=1000,
        target_audio_sample_rate=1000,
    )

    assert clipped_frames.shape[0] == 5
    assert clipped_audio.shape[1] == 500
    assert sampled.shape[0] == 3
    assert effective_fps == 5.0
    assert prepared.shape == (1, 500)
    assert torch.allclose(prepared, torch.full((1, 500), 0.5))


def test_denseav_downsample_supports_non_integer_frame_rate_ratio() -> None:
    torch = pytest.importorskip("torch")

    frames = torch.zeros(60, 4, 5, 3, dtype=torch.uint8)
    sampled, effective_fps = downsample_video_frames(
        frames,
        source_fps=30.0,
        target_fps=25.0,
    )

    assert sampled.shape[0] == 50
    assert effective_fps == 25.0


def test_denseav_squeeze_sim_by_head_drops_single_head_only() -> None:
    torch = pytest.importorskip("torch")

    single = torch.zeros(1, 25, 28, 28)
    multi = torch.zeros(2, 25, 28, 28)

    assert DenseAVTool._squeeze_sim_by_head(single).shape == (25, 28, 28)
    assert DenseAVTool._squeeze_sim_by_head(multi).shape == (2, 25, 28, 28)


@pytest.mark.skipif(
    os.environ.get("AV_TOOLBOX_RUN_DENSEAV_SMOKE") != "1",
    reason="set AV_TOOLBOX_RUN_DENSEAV_SMOKE=1 to run DenseAV inference",
)
def test_denseav_clever_cat_opt_in_smoke(tmp_path) -> None:
    pytest.importorskip("denseav")
    pytest.importorskip("av")
    pytest.importorskip("torch")
    checkpoint = default_cache_dir() / "weights" / "denseav_2head.ckpt"
    if not checkpoint.exists():
        pytest.skip(f"DenseAV checkpoint missing: {checkpoint}")

    root = Path(__file__).resolve().parents[1]
    demo = root / "data_segments" / "Clever_Cat_Outsmarts_Warrior_square.mp4"
    result = av_toolbox.run_tool(
        "av.denseav",
        input_path=demo,
        output_dir=tmp_path / "out",
        model_name="sound_and_language",
        max_seconds=0.5,
        sample_fps=1.0,
        device="cpu",
        export_overlay=False,
    )

    assert result.timeline_json is not None
    assert Path(result.timeline_json).exists()
