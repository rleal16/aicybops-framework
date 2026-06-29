"""Resolve the AICybOps service base URL from environment variables."""

from __future__ import annotations

import os

DEFAULT_AICYBOPS_SERVICE_HOST = "nexus-wp7.lis.ipn.pt"
DEFAULT_AICYBOPS_SERVICE_PORT = "443"
DEFAULT_AICYBOPS_SERVICE_PROTOCOL = "https"

DEFAULT_AICYBOPS_SERVICE_URL = f"{DEFAULT_AICYBOPS_SERVICE_PROTOCOL}://{DEFAULT_AICYBOPS_SERVICE_HOST}"


def resolve_aicybops_service_url() -> str:
    """Return service base URL with no trailing slash."""
    explicit = os.getenv("AICYBOPS_SERVICE_URL", "").strip()
    if explicit:
        return explicit.rstrip("/")
    host = os.getenv("AICYBOPS_SERVICE_HOST", DEFAULT_AICYBOPS_SERVICE_HOST)
    port = os.getenv("AICYBOPS_SERVICE_PORT", DEFAULT_AICYBOPS_SERVICE_PORT)
    protocol = os.getenv("AICYBOPS_SERVICE_PROTOCOL", DEFAULT_AICYBOPS_SERVICE_PROTOCOL)
    if protocol == "https" and port == "443":
        return f"https://{host}".rstrip("/")
    if protocol == "http" and port == "80":
        return f"http://{host}".rstrip("/")
    return f"{protocol}://{host}:{port}".rstrip("/")
