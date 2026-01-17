from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
import traceback

# Internal Imports
from api.core.agent import get_response
from api.core.database import add_documents_to_store
from api.utils.parsers import parse_file

from pydantic import BaseModel
from typing import List

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

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = await parse_file(file.filename, content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.create_documents([text])
        add_documents_to_store(chunks, file.filename)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/query")
async def query(request: QueryRequest):
    try:
        # Call the manual RAG chain
        answer = get_response(request.query, request.chat_history)
        return {"response": answer}
    except Exception as e:
        # This will catch the error and show it in your frontend!
        error_details = traceback.format_exc()
        print(error_details)
        return {"response": f"Internal Error: {str(e)}"}