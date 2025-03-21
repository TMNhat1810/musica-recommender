from app.services.qdrant import QdrantService


def query_Search_handle(id: str, k: int):

    return QdrantService.query_search(id=int(id), k=k)
