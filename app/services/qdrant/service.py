from app.common.constants import FEATURE_COLLECTION_NAME
from app.services.qdrant import qdrant


class QdrantService:
    def __init__(self):
        pass

    @staticmethod
    def retrieve(collection_name=FEATURE_COLLECTION_NAME, id=None):
        data = qdrant.retrieve(collection_name, [id], with_vectors=True)
        if not data:
            return None
        return data

    @staticmethod
    def query_search(collection_name=FEATURE_COLLECTION_NAME, id=None, k=5):
        records = QdrantService.retrieve(collection_name, id)
        if not records:
            return None
        video_vector = records[0].vector["video_vector"]

        search_results = qdrant.search(
            collection_name=collection_name,
            query_vector=("video_vector", video_vector),
            limit=k + 1,
        )

        return {"response": search_results}
