from fastapi import APIRouter

from app.api.handler.search.search_by_title import search_by_title_handle


router = APIRouter(prefix="/search", tags=["search"])


@router.get("")
async def search_by_title(title: str = None):
    return await search_by_title_handle(title)
