import cv2
import librosa
import torch
from app.models import clip_model, sentence_transformer
from app.configs import device
import numpy as np
from PIL import Image
from torchvision import transforms


def extract_text_embedding(text):
    """Generate text vector using Sentence Transformers"""
    return sentence_transformer.encode(text).tolist()


def extract_video_embedding(video_path, num_frames=5):
    """Extracts frames from a video and computes an average embedding"""

    cap = cv2.VideoCapture(video_path)
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


def extract_audio_embedding(video_path):
    """Extracts audio from video and computes an MFCC-based embedding"""
    y, sr = librosa.load(video_path, sr=16000, duration=10)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128)
    audio_vector = np.mean(mfcc, axis=1)

    return audio_vector.tolist()
