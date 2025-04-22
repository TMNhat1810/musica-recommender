from sentence_transformers import SentenceTransformer

from app.configs import device


sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2", device=device).to(device)
