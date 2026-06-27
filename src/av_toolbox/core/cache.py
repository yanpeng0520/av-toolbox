"""Model cache helpers for av-toolbox."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_CACHE_ENV = "AV_TOOLBOX_CACHE_DIR"


def default_cache_dir() -> Path:
    return Path(os.environ.get(DEFAULT_CACHE_ENV, "~/.cache/av_toolbox")).expanduser()


@dataclass(frozen=True, slots=True)
class CachedModel:
    """Resolved model artifact metadata."""

    name: str
    path: Path
    sha256: str | None = None


class ModelCache:
    """Resolve model artifacts without putting weights in the repository."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()
        self.weights_dir = self.cache_dir / "weights"

    def ensure(self) -> None:
        self.weights_dir.mkdir(parents=True, exist_ok=True)

    def resolve_weight(
        self,
        name: str,
        *,
        expected_sha256: str | None = None,
        offline: bool = False,
    ) -> CachedModel:
        """Resolve an existing cached weight.

        Downloading is intentionally deferred to tool-specific resolvers in the
        next milestone; this method enforces the shared cache location and
        checksum verification for already-present artifacts.
        """
        self.ensure()
        path = self.weights_dir / name
        if not path.exists():
            mode = "offline " if offline else ""
            raise FileNotFoundError(f"{mode}model weight not found in cache: {path}")
        actual = self.sha256(path) if expected_sha256 else None
        if expected_sha256 and actual != expected_sha256:
            raise ValueError(
                f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual}"
            )
        return CachedModel(name=name, path=path, sha256=actual)

    @staticmethod
    def sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

