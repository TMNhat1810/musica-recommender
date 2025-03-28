from io import BytesIO
import os
from click import File
import cv2
from fastapi import HTTPException, UploadFile
import librosa
import torch
from app.models import clip_model, sentence_transformer
from app.configs import device
import numpy as np
from PIL import Image
from torchvision import transforms
import tempfile


def extract_text_embedding(text):
    """Generate text vector using Sentence Transformers"""
    return sentence_transformer.encode(text).tolist()


def extract_video_embedding(file, num_frames=5):
    """Extracts frames from a video and computes an average embedding"""

    cap = cv2.VideoCapture(file)

    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in np.linspace(0, frame_count - 1, num_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        preprocess = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor()]
        )
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = (
                clip_model.encode_image(image_tensor).cpu().numpy().flatten()
            )

        frames.append(image_features)

    cap.release()

    if frames:
        return np.mean(frames, axis=0).tolist()
    else:
        return [0] * 512


def extract_audio_embedding(audio):
    """Extracts audio from video and computes an MFCC-based embedding"""
    y, sr = librosa.load(audio, sr=16000, duration=10)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    audio_vector = np.mean(mfcc, axis=1)

    return audio_vector.tolist()


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
