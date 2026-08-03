from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from langchain_mcp_adapters.interceptors import MCPToolCallResult

# Type-form usages guarded by the pyright run below. If MCPToolCallResult
# loses its TypeAlias annotation, every one of these emits
# reportInvalidTypeForm and the test fails.
_annotated: MCPToolCallResult | None = None


def _accepts(value: MCPToolCallResult) -> MCPToolCallResult:
    return value


def _accepts_optional(value: MCPToolCallResult | None) -> MCPToolCallResult | None:
    return value


def _accepts_container(
    values: list[MCPToolCallResult],
) -> tuple[MCPToolCallResult, ...]:
    return tuple(values)


@pytest.mark.skipif(
    shutil.which("pyright") is None,
    reason="pyright not installed; skipping static type-form regression check",
)
def test_pyright_accepts_mcptoolcallresult_type_forms() -> None:
    pyright = shutil.which("pyright")
    assert pyright is not None  # narrowed by the ``skipif`` above

    result = subprocess.run(  # noqa: S603
        [pyright, "--outputjson", __file__],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, (
        "pyright reported errors on the MCPToolCallResult type-form usages:\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
