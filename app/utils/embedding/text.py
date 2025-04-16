from app.models import sentence_transformer


def extract_text_embedding(text):
    return sentence_transformer.encode(text).tolist()
