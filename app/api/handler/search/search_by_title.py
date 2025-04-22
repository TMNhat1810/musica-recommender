from app.services.qdrant.service import QdrantService
from app.utils.embedding.text import extract_text_embedding


async def search_by_title_handle(title: str):
    vector = extract_text_embedding(title)
    return QdrantService.search_by_title(vector)
