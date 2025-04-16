from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


qdrant = QdrantClient("localhost", port=6333)


if not qdrant.collection_exists("video_features"):
    qdrant.create_collection(
        collection_name="video_features",
        vectors_config={
            "title_vector": VectorParams(size=384, distance=Distance.COSINE),
            "video_vector": VectorParams(size=512, distance=Distance.COSINE),
            "audio_vector": VectorParams(size=20000, distance=Distance.COSINE),
        },
    )
    print(f"Collection video_features created.")
else:
    print(f"Collection video_features already exists.")


if not qdrant.collection_exists("audio_features"):
    qdrant.create_collection(
        collection_name="audio_features",
        vectors_config={
            "title_vector": VectorParams(size=384, distance=Distance.COSINE),
            "audio_vector": VectorParams(size=20000, distance=Distance.COSINE),
        },
    )
    print(f"Collection audio_features created.")
else:
    print(f"Collection audio_features already exists.")
