from app.common.constants import FEATURE_COLLECTION_NAME
from app.services.qdrant import qdrant
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue


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

    @staticmethod
    def upload_audio_vector(id, title_vector, feature_vector):
        qdrant.upsert(
            collection_name="audio_features",
            points=[
                PointStruct(
                    id=id,
                    vector={
                        "title_vector": title_vector,
                        "audio_vector": feature_vector,
                    },
                    payload={"id": id},
                )
            ],
        )

    @staticmethod
    def upload_video_vector(id, title_vector, feature_vectors):
        qdrant.upsert(
            collection_name="video_features",
            points=[
                PointStruct(
                    id=id,
                    vector={
                        "title_vector": title_vector,
                        "video_vector": feature_vectors[0],
                        "audio_vector": feature_vectors[1],
                    },
                    payload={"id": id},
                )
            ],
        )

    @staticmethod
    def delete_audio_vector(id: str):
        result, _ = qdrant.scroll(
            collection_name="audio_features",
            limit=1,
            with_payload=False,
            with_vectors=False,
            offset=None,
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=id))]
            ),
        )

        if result:
            qdrant.delete(collection_name="audio_features", points_selector=[id])
            return True
        else:
            return False

    @staticmethod
    def delete_video_vector(id: str):
        result, _ = qdrant.scroll(
            collection_name="video_features",
            limit=1,
            with_payload=False,
            with_vectors=False,
            offset=None,
            scroll_filter=Filter(
                must=[FieldCondition(key="id", match=MatchValue(value=id))]
            ),
        )

        if result:
            qdrant.delete(collection_name="video_features", points_selector=[id])
            return True
        else:
            return False
