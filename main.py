from fastapi import FastAPI
from app.api.handler import root_handle
from app.api.routers import vector, search


app = FastAPI()


@app.get("/")
def read_root():
    return root_handle()


app.include_router(vector.router)
app.include_router(search.router)


print("Swagger Docs: http://localhost:8000/docs")
