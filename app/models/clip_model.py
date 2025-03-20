import clip
from app.configs import device


clip_model, preprocess = clip.load("ViT-B/32", device=device)
