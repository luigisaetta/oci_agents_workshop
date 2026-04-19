"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: Unit tests for OpenAI-compatible OCI client builders.
"""

from __future__ import annotations

# pylint: disable=too-few-public-methods

from common import oci_openai_clients


def test_build_oci_openai_client_uses_user_principal(monkeypatch) -> None:
    """It should build OpenAI client with OCI user_principal auth."""
    captured: dict[str, object] = {}

    class _FakeAuth:
        def __init__(self, profile_name: str) -> None:
            self.profile_name = profile_name

    class _FakeHttpxClient:
        def __init__(self, *, auth=None) -> None:
            captured["httpx_auth"] = auth

    class _FakeOpenAI:
        def __init__(
            self, *, base_url: str, api_key: str, project: str, http_client=None
        ) -> None:
            captured["base_url"] = base_url
            captured["api_key"] = api_key
            captured["project"] = project
            captured["http_client"] = http_client

    monkeypatch.setattr(oci_openai_clients, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(
        oci_openai_clients,
        "httpx",
        type("_FakeHttpxModule", (), {"Client": _FakeHttpxClient}),
    )
    monkeypatch.setattr(oci_openai_clients, "OciUserPrincipalAuth", _FakeAuth)

    client = oci_openai_clients.build_oci_openai_client(
        base_url="https://example.test/v1",
        project_id="proj_123",
        auth_profile="DEFAULT",
    )

    assert isinstance(client, _FakeOpenAI)
    assert captured["base_url"] == "https://example.test/v1"
    assert captured["api_key"] == "unused"
    assert captured["project"] == "proj_123"
    assert isinstance(captured["httpx_auth"], _FakeAuth)
