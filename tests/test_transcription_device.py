from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from av_toolbox.audio import transcription
from av_toolbox.core.base_tool import ToolRunContext
from av_toolbox.core.cache import ModelCache
from av_toolbox.core.hardware import HardwareConfig


def test_auto_cuda_falls_back_to_cpu_when_ctranslate2_has_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription, "_ctranslate2_cuda_device_count", lambda: 0)

    device = transcription._resolve_faster_whisper_device(
        candidate_device="cuda",
        requested_device="auto",
    )

    assert device == "cpu"


def test_explicit_cuda_reports_ctranslate2_support_gap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription, "_ctranslate2_cuda_device_count", lambda: 0)

    with pytest.raises(ValueError, match="CTranslate2"):
        transcription._resolve_faster_whisper_device(
            candidate_device="cuda",
            requested_device="cuda",
        )


def test_ctranslate2_cuda_support_keeps_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(transcription, "_ctranslate2_cuda_device_count", lambda: 1)

    device = transcription._resolve_faster_whisper_device(
        candidate_device="cuda",
        requested_device="auto",
    )

    assert device == "cuda"


def test_pyav_audio_namespace_guard_loads_audio_attribute(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_av = types.ModuleType("av")
    fake_audio = types.ModuleType("av.audio")
    imported: list[str] = []

    def fake_import_module(name: str) -> types.ModuleType:
        imported.append(name)
        if name == "av.audio":
            return fake_audio
        return types.ModuleType(name)

    monkeypatch.setitem(sys.modules, "av", fake_av)
    monkeypatch.setattr(transcription.importlib, "import_module", fake_import_module)

    transcription._ensure_pyav_audio_namespace()

    assert fake_av.audio is fake_audio
    assert imported == ["av.audio", "av.audio.fifo", "av.audio.resampler"]


def test_transcription_passes_decoded_waveform_to_whisper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"placeholder")
    waveform = np.zeros(1600, dtype=np.float32)
    seen: dict[str, object] = {}

    class FakeSegment:
        start = 0.0
        end = 0.5
        text = " hello"
        avg_logprob = -0.1
        no_speech_prob = 0.01

    class FakeInfo:
        language = "en"
        language_probability = 1.0
        duration = 0.1

    class FakeWhisperModel:
        def __init__(self, *args: object, **kwargs: object) -> None:
            seen["init_kwargs"] = kwargs

        def transcribe(self, audio: object, **kwargs: object) -> tuple[list[FakeSegment], FakeInfo]:
            seen["audio"] = audio
            seen["transcribe_kwargs"] = kwargs
            return [FakeSegment()], FakeInfo()

    monkeypatch.setattr(transcription, "_import_faster_whisper", lambda: FakeWhisperModel)
    monkeypatch.setattr(transcription, "_ensure_pyav_audio_namespace", lambda: None)
    monkeypatch.setattr(transcription, "_decode_audio_for_whisper", lambda path, max_seconds=None: waveform)

    context = ToolRunContext(
        output_dir=tmp_path / "out",
        hardware=HardwareConfig(device="cpu"),
        cache=ModelCache(tmp_path / "cache"),
        workspace=tmp_path / "workspace",
    )
    context.output_dir.mkdir()

    transcription.TranscriptionTool()._run(input_path=input_path, context=context)

    assert seen["audio"] is waveform
    assert seen["init_kwargs"] == {
        "device": "cpu",
        "compute_type": "int8",
        "download_root": str(context.cache.weights_dir),
    }


def test_pyav_resampler_class_falls_back_to_submodule(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_av = types.ModuleType("av")
    fake_audio = types.ModuleType("av.audio")
    fake_resampler_module = types.ModuleType("av.audio.resampler")

    class FakeAudioResampler:
        pass

    fake_resampler_module.AudioResampler = FakeAudioResampler

    def fake_import_module(name: str) -> types.ModuleType:
        if name == "av.audio":
            return fake_audio
        if name == "av.audio.resampler":
            return fake_resampler_module
        raise ImportError(name)

    monkeypatch.setattr(transcription.importlib, "import_module", fake_import_module)

    assert transcription._pyav_audio_resampler_class(fake_av) is FakeAudioResampler
    assert fake_av.audio is fake_audio


def test_import_pyav_ignores_project_av_shadow(monkeypatch: pytest.MonkeyPatch) -> None:
    package_dir = Path(transcription.__file__).resolve().parents[1]
    previous_av = sys.modules.pop("av", None)
    monkeypatch.syspath_prepend(str(package_dir))
    try:
        pyav = transcription._import_pyav()

        assert not transcription._is_project_av_shadow(pyav)
        assert hasattr(pyav, "AudioResampler")
    finally:
        sys.modules.pop("av", None)
        if previous_av is not None:
            sys.modules["av"] = previous_av
