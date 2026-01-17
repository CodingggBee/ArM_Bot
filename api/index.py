from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
import traceback
from pydantic import BaseModel
from typing import List

# INTERNAL IMPORTS - Notice we only import get_response
from api.core.agent import get_response
from api.core.database import add_documents_to_store
from api.utils.parser import parse_file


class _SimpleDoc:
    def __init__(self, text: str):
        self.page_content = text

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Message]


@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/query")
async def query(request: QueryRequest):
    try:
        # We call get_response because that is what is in your agent.py
        answer = get_response(request.query, request.chat_history)
        return {"response": answer}
    except Exception as e:
        return {"response": f"Internal Error: {str(e)}"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        # parse_file is async and returns the extracted text
        text = await parse_file(file.filename, content)

        # Split into chunks and wrap into objects that have `page_content`
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        parts = []
        if hasattr(splitter, "split_text"):
            parts = splitter.split_text(text) if text else []
        else:
            # Fallback: don't split
            parts = [text] if text else []

        chunks = [_SimpleDoc(p) for p in parts]

        # Add to vector store (in-memory list in this simple example)
        add_documents_to_store(chunks, source_name=file.filename)

        return {"status": "ok", "file": file.filename}
    except Exception as e:
        traceback.print_exc()
        return {"status": "error", "detail": str(e)}