from __future__ import annotations


def test_public_api_imports() -> None:
    import av_toolbox

    assert av_toolbox.AVResult
    assert av_toolbox.BaseTool
    assert av_toolbox.HardwareConfig
    assert av_toolbox.ToolRegistry
    assert isinstance(av_toolbox.list_tools(), list)

def test_base_public_api_import_does_not_require_numpy(tmp_path) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = """
import importlib.abc
import sys

class BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'numpy' or fullname.startswith('numpy.'):
            raise ModuleNotFoundError('blocked numpy import')
        return None

sys.meta_path.insert(0, BlockNumpy())
import av_toolbox
print(len(av_toolbox.list_tools()))
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert int(result.stdout.strip()) > 0

