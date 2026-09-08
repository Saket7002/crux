"""Session authentication. Requests carry a bearer token."""

from fastapi import Header, HTTPException


async def current_user(authorization: str = Header()) -> dict:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    return {"id": authorization.removeprefix("Bearer ")}
