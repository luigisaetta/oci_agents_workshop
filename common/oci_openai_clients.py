"""
Author: L. Saetta
Date last modified: 2026-04-19
License: MIT
Description: OpenAI-compatible OCI client builders for user_principal authentication.
"""

from __future__ import annotations

from typing import Any

import httpx
from oci_genai_auth import OciUserPrincipalAuth
from openai import OpenAI


def build_oci_openai_client(
    *,
    base_url: str,
    project_id: str,
    auth_profile: str = "DEFAULT",
) -> Any:
    """Build an OpenAI-compatible OCI client using user_principal auth.

    Args:
        base_url: OCI OpenAI-compatible API base URL.
        project_id: OCI project identifier.
        auth_profile: OCI config profile name (for example ``DEFAULT``).

    Returns:
        Any: OpenAI-compatible client.
    """
    auth = OciUserPrincipalAuth(profile_name=auth_profile)
    return OpenAI(
        base_url=base_url,
        # OCI signer auth is used; API key is not used here.
        api_key="unused",
        project=project_id,
        http_client=httpx.Client(auth=auth),
    )
