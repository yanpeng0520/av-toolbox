from __future__ import annotations

from pathlib import Path

from av_toolbox.core.base_tool import BaseTool, ToolRunContext
from av_toolbox.core.registry import ToolRegistry
from av_toolbox.core.result import AVResult


class EchoTool(BaseTool):
    name = "test.echo"
    category = "test"
    description = "Echo tool used by registry tests."

    def _run(
        self,
        *,
        input_path: Path | None,
        context: ToolRunContext,
        **kwargs,
    ) -> AVResult:
        return AVResult(
            tool_name=self.name,
            input_path=input_path,
            output_dir=context.output_dir,
            metadata={"received": kwargs},
        )


def test_registry_registers_and_runs_tool(tmp_path) -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())

    assert registry.list_names() == ["test.echo"]
    result = registry.run(
        "test.echo",
        input_path=tmp_path / "input.mp4",
        output_dir=tmp_path / "out",
        device="cpu",
        batch_size=4,
        fp16=False,
        value=123,
    )

    assert result.tool_name == "test.echo"
    assert result.metadata["received"] == {"value": 123}
    assert result.metadata["hardware"]["resolved_device"] == "cpu"
    assert result.metadata["workspace"]["workspace_cleaned"] is True

