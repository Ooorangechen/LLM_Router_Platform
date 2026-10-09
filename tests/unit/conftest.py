"""Fixtures shared by unit tests."""

import time
from collections import deque
from unittest.mock import Mock

import pytest

from main import LLMRouterPlatform


@pytest.fixture
def make_platform():
    """Build an LLMRouterPlatform without __init__ (no config.yaml read, no log files).

    Sets the runtime attributes __init__ would; keep this in sync with
    LLMRouterPlatform.__init__ in main.py.
    """
    def _make(config=None, services=None):
        platform = object.__new__(LLMRouterPlatform)
        platform.config = config if config is not None else {"api": {"cors_origins": ["*"]}}
        platform.services = services if services is not None else {}
        platform.logger = Mock()
        platform._prom_server = None
        platform._start_time = time.time()
        platform._traffic_window = deque(maxlen=100_000)
        return platform
    return _make
