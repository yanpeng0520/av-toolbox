from __future__ import annotations


def test_public_api_imports() -> None:
    import av_toolbox

    assert av_toolbox.AVResult
    assert av_toolbox.BaseTool
    assert av_toolbox.HardwareConfig
    assert av_toolbox.ToolRegistry
    assert isinstance(av_toolbox.list_tools(), list)

