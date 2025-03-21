from fastapi import FastAPI
from app.api.handler import root_handle, query_Search_handle


app = FastAPI()


@app.get("/")
def read_root():
    return root_handle()


@app.get("/vector/query")
def read_item(id: str = None, k: int = 5):
    return query_Search_handle(id, k)
