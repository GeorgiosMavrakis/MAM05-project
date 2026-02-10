from fastapi import FastAPI
import uvicorn
import database, embedder
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

@app.on_event("startup")
def startup():

    if database.collection.count() == 0:
        embedder.embedding()
    else:
        print("DB loaded.")


@app.get("/ask")
def ask(q: str):

    return {
        "question": q,
        "answer": ask_rag(q)
    }

if __name__ == "__main__":

    uvicorn.run(
        "rag_local:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)