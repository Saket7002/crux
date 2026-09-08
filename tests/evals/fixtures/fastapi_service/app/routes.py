"""API routes."""

from app.auth import current_user
from fastapi import APIRouter, Depends

router = APIRouter(prefix="/api/v2")


@router.get("/items")
async def list_items(user=Depends(current_user)) -> list[dict]:
    return []


@router.post("/items")
async def create_item(payload: dict, user=Depends(current_user)) -> dict:
    return payload
