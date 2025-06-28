from fastapi import HTTPException, UploadFile
from app.configs.allowed_mimetype import AUDIO_MIMETYPES, VIDEO_MIMETYPES
from app.services.qdrant.service import QdrantService
from app.utils.content_moderator import flag_content
from app.utils.embedding import (
    extract_audiofile_embedding,
    extract_text_embedding,
    extract_videofile_embedding,
)


async def upload_vector_handle(id: str, title: str, media: UploadFile):
    if media.content_type in VIDEO_MIMETYPES:
        feature_vectors = await extract_videofile_embedding(media)
    elif media.content_type in AUDIO_MIMETYPES:
        feature_vectors = await extract_audiofile_embedding(media)
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload a video or audio file.",
        )

    title_vector = extract_text_embedding(title)

    if media.content_type in VIDEO_MIMETYPES:
        QdrantService.upload_video_vector(id, title_vector, feature_vectors)
    else:
        QdrantService.upload_audio_vector(id, title_vector, feature_vectors)

    if media.content_type in VIDEO_MIMETYPES:
        pred_label, pred_class, probs = flag_content(id)

    return {
        "success": True,
        "data": {"label": pred_label, "class": pred_class, "problity": probs},
    }
