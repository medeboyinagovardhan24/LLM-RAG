from sentence_transformers import SentenceTransformer
# Switch from all-MiniLM-L6-v2 to multilingual model
multilingual_model = SentenceTransformer("intfloat/multilingual-e5-base")

def embed_multilingual(text):
    """Embed text in multiple languages"""
    return multilingual_model.encode(text, convert_to_numpy=True)
