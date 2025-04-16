from io import BytesIO
import os
from click import File
from fastapi import UploadFile
import tempfile
from app.utils.embedding.audio import extract_audio_embedding
from app.utils.embedding.clip import extract_video_embedding
from app.utils.embedding.text import extract_text_embedding


async def extract_audiofile_embedding(file: UploadFile = File(...)):
    buffer = BytesIO(await file.read())
    return extract_audio_embedding(buffer)


async def extract_videofile_embedding(file: UploadFile = File(...)):
    video_bytes = await file.read()
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_file.write(video_bytes)
    tmp_file.flush()

    vectors = extract_video_embedding(tmp_file.name), extract_audio_embedding(
        tmp_file.name
    )

    tmp_file.close()
    os.remove(tmp_file.name)

    return vectors
