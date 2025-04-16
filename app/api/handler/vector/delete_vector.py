from app.services.qdrant.service import QdrantService


def delete_vector_handle(id: str):
    success = (
        QdrantService.delete_audio_vector(id)
        if not QdrantService.delete_video_vector(id)
        else False
    )

    return {"success": success}
