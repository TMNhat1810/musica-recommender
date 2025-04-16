import cv2
from PIL import Image
import torch
from torchvision import transforms
from app.configs import device
from app.models import clip_model
import numpy as np


def extract_video_embedding(file, num_frames=5):
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
