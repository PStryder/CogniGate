"""
Authentication for CogniGate REST API.

Simple API key authentication for protecting admin/config endpoints.
Uses environment-based keys since CogniGate is a stateless worker.
"""

import logging
import secrets
from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from .config import Settings


logger = logging.getLogger(__name__)

# API key prefix for CogniGate
API_KEY_PREFIX = "cg_"


def verify_api_key(
    settings: Settings,
    authorization: Optional[str] = Header(None),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
) -> bool:
    """
    Verify API key for protected endpoints.

    Checks Authorization: Bearer or X-API-Key header against
    the configured COGNIGATE_API_KEY environment variable.

    Security: Fails closed - if api_key is not configured and we're not
    in explicit insecure dev mode, all requests are rejected.
    """
    # Check if auth is required
    if not settings.require_auth:
        if settings.allow_insecure_dev:
            return True
        # If require_auth is False but not explicitly insecure, still require it
        # to prevent accidental exposure
        logger.warning("require_auth=False but allow_insecure_dev=False - requiring auth anyway")

    # Extract API key from headers
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    elif x_api_key:
        api_key = x_api_key

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization. Use Authorization: Bearer <key> or X-API-Key header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate against configured API key
    if not settings.api_key:
        logger.error(
            "SECURITY VIOLATION: api_key not configured. "
            "Set COGNIGATE_API_KEY or enable COGNIGATE_ALLOW_INSECURE_DEV=true (dev only)."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server misconfigured: authentication not properly initialized",
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


def generate_api_key() -> str:
    """Generate a new API key with cg_ prefix.

    Utility function for generating keys - the key should be stored
    in COGNIGATE_API_KEY environment variable.
    """
    return f"{API_KEY_PREFIX}{secrets.token_urlsafe(32)}"


class AuthDependency:
    """FastAPI dependency for API key authentication."""

    def __init__(self, settings: Settings):
        self.settings = settings

    async def __call__(
        self,
        authorization: Optional[str] = Header(None),
        x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
    ) -> bool:
        return verify_api_key(self.settings, authorization, x_api_key)
