from __future__ import annotations

import secrets
from dataclasses import dataclass

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str


def build_api_key_dependency(expected_api_key: str | None, principal: Principal):
    async def require_api_key(x_api_key: str | None = Header(default=None)) -> Principal:
        if not expected_api_key:
            return principal
        if x_api_key is None or not secrets.compare_digest(x_api_key, expected_api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key.",
            )
        return principal

    return require_api_key
