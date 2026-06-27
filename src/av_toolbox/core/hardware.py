"""Hardware configuration shared by model-backed tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HardwareConfig:
    """Explicit hardware controls for inference tools."""

    device: str = "auto"
    batch_size: int | None = None
    fp16: bool = False

    @classmethod
    def from_values(
        cls,
        *,
        device: str | None = None,
        batch_size: int | None = None,
        fp16: bool | None = None,
        hardware: "HardwareConfig | None" = None,
    ) -> "HardwareConfig":
        if hardware is not None:
            base = hardware
        else:
            base = cls()
        return cls(
            device=device if device is not None else base.device,
            batch_size=batch_size if batch_size is not None else base.batch_size,
            fp16=fp16 if fp16 is not None else base.fp16,
        )

    def resolved_device(self) -> str:
        """Resolve ``auto`` to CUDA when available, otherwise CPU."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "resolved_device": self.resolved_device(),
            "batch_size": self.batch_size,
            "fp16": self.fp16,
        }

