from __future__ import annotations

from pathlib import Path

import pytest

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.cache import DEFAULT_CACHE_ENV, ModelCache, default_cache_dir
from av_toolbox.core.result import AVResult


class CacheEchoTool(BaseTool):
    name = "test.cache_echo"
    category = "test"

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        **_: object,
    ) -> AVResult:
        return AVResult(
            tool_name=self.name,
            input_path=input_path,
            output_dir=context.output_dir,
            metadata={
                "workspace_seen": str(context.workspace),
                "weights_seen": str(context.cache.weights_dir),
            },
        )


def test_default_cache_dir_honors_env_override(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "env-cache"
    monkeypatch.setenv(DEFAULT_CACHE_ENV, str(cache_root))

    assert default_cache_dir() == cache_root

    cache = ModelCache()
    assert cache.cache_dir == cache_root
    assert cache.weights_dir == cache_root / "weights"


def test_model_cache_resolves_weight_and_verifies_checksum(tmp_path) -> None:
    cache = ModelCache(tmp_path / "cache")
    cache.ensure()
    weight_path = cache.weights_dir / "toy-model.bin"
    weight_path.write_bytes(b"toy model bytes")
    expected_sha256 = ModelCache.sha256(weight_path)

    resolved = cache.resolve_weight("toy-model.bin", expected_sha256=expected_sha256)

    assert resolved.name == "toy-model.bin"
    assert resolved.path == weight_path
    assert resolved.sha256 == expected_sha256

    with pytest.raises(ValueError, match="Checksum mismatch"):
        cache.resolve_weight("toy-model.bin", expected_sha256="0" * 64)


def test_model_cache_offline_missing_weight_mentions_cache_path(tmp_path) -> None:
    cache = ModelCache(tmp_path / "cache")

    with pytest.raises(FileNotFoundError, match="offline model weight not found"):
        cache.resolve_weight("missing.bin", offline=True)


def test_base_tool_run_includes_cache_metadata(tmp_path) -> None:
    cache_root = tmp_path / "cache"

    result = CacheEchoTool().run(
        input_path=tmp_path / "input.mp4",
        output_dir=tmp_path / "out",
        cache_dir=cache_root,
        device="cpu",
    )

    assert result.metadata["cache"] == {
        "cache_dir": str(cache_root),
        "weights_dir": str(cache_root / "weights"),
    }
    assert result.metadata["hardware"]["device"] == "cpu"
    assert result.metadata["weights_seen"] == str(cache_root / "weights")
