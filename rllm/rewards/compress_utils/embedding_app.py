# server.py
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer
import uvicorn
from pydantic import BaseModel
import argparse

app = FastAPI()

class EmbeddingRequest(BaseModel):
    text: str

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    parser.add_argument("--ip", type=str, help="ip to run the server on")
    return parser.parse_args()

device = "cuda:0"
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device=device
)

@app.post("/embedding")
async def get_embedding(request: EmbeddingRequest):
    embedding = model.encode(request.text, normalize_embeddings=True)
    return {"embedding": embedding.tolist()}

if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(
        app, 
        host=args.ip, 
        port=args.port,
        workers=1,
    )