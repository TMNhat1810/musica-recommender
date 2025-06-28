import os
import joblib
import torch
from app.configs import device
from app.models.content_moderator import model
from app.services.qdrant.service import QdrantService

label_encoder = joblib.load(os.path.join("app", "models", "_classes.pkl"))


def flag_content(id: str):
    point = QdrantService.retrieve(collection_name="video_features", id=id)[0]
    vectors = point.vector

    video_vec = (
        torch.tensor(vectors["video_vector"], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    audio_vec = (
        torch.tensor(vectors["audio_vector"], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )
    title_vec = (
        torch.tensor(vectors["title_vector"], dtype=torch.float32)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        probabilities = model(video_vec, audio_vec, title_vec)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label, predicted_class, probabilities
