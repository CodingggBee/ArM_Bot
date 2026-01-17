import traceback
from typing import List
from pydantic import BaseModel

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

# INTERNAL imports
from api.core.database import add_documents_to_store, similarity_search
from api.utils.parser import parse_file


class Message(BaseModel):
    role: str
    content: str


class QueryRequest(BaseModel):
    query: str
    chat_history: List[Message] = []


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class _Chunk:
    def __init__(self, text: str):
        self.page_content = text


def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    if not text:
        return []
    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        chunks.append(text[start:start+chunk_size])
        start += step
    return chunks


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = await parse_file(file.filename, content)

        parts = _chunk_text(text, chunk_size=1000, overlap=200)
        chunks = [_Chunk(p) for p in parts]

        add_documents_to_store(chunks, source_name=file.filename)
        return {"status": "ok", "file": file.filename}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}


@app.post("/api/query")
async def query(request: QueryRequest):
    try:
        result = similarity_search(request.query)
        return {"response": result}
    except Exception as e:
        return {"response": f"Internal Error: {str(e)}"}
