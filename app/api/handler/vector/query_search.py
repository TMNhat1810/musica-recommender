from app.common.constants import FEATURE_COLLECTION_NAME
from app.services.qdrant import QdrantService


def query_Search_handle(id: str, k: int):

    return QdrantService.query_search(
        collection_name=FEATURE_COLLECTION_NAME, id=id, limit=k
    )
