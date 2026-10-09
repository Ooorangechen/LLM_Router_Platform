"""Fixtures for acceptance tests that run the real service process."""

import pytest

from tests.acceptance.server import start_server


@pytest.fixture(scope="module")
def server_factory(project_root, tmp_path_factory):
    """start(overrides=..., port=..., env=...) -> Server; every server is stopped at module end."""
    workdir = tmp_path_factory.mktemp("servers")
    servers = []

    def start(**kwargs):
        server = start_server(project_root, workdir, **kwargs)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()
