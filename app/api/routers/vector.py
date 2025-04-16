from fastapi import APIRouter, File, Form, UploadFile

from app.api.handler.vector import (
    delete_vector_handle,
    query_Search_handle,
    upload_vector_handle,
)

router = APIRouter(prefix="/vector", tags=["vector"])


@router.get("")
async def read_item(id: str = None, k: int = 5):
    return query_Search_handle(id, k)


@router.delete("")
async def delete_vector(id: str = None):
    return delete_vector_handle(id)


@router.post("/upload")
async def upload_file(
    id: str = Form(...), title: str = Form(...), media: UploadFile = File(...)
):
    return await upload_vector_handle(id, title, media)
