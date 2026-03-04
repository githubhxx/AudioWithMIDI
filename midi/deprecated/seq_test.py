"""Deprecated placeholder test module.

Historically this file contained ad-hoc executable code and invalid imports,
which caused pytest collection failures.
"""

import pytest


@pytest.mark.skip(reason="Deprecated experimental script; retained as placeholder.")
def test_deprecated_placeholder() -> None:
    """Keep pytest collection stable for deprecated module."""
    assert True
