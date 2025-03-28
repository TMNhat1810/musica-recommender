from fastapi import APIRouter, UploadFile

from app.api.handler.query_search import query_Search_handle
from app.api.handler.vector import upload_vector_handle

router = APIRouter(prefix="/vector", tags=["vector"])


@router.get("/query")
async def read_item(id: str = None, k: int = 5):
    return query_Search_handle(id, k)


@router.post("/upload")
async def upload_file(id: str, title: str, media: UploadFile):
    return await upload_vector_handle(id, title, media)
