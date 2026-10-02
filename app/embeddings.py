from functools import cache
from app import config
@cache
def model():
    from fastembed import TextEmbedding
    return TextEmbedding(config.EMBEDDING_MODEL)

def embed(texts):
    return [vector.tolist() for vector in model().embed(texts)]