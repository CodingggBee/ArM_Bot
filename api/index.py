from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter
from api.core.agent import get_response
from api.core.database import add_documents_to_store
from api.utils.parser import parse_file
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

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    text = await parse_file(file.filename, content)
    chunks = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).create_documents([text])
    add_documents_to_store(chunks, file.filename)
    return {"status": "success"}

@app.post("/api/query")
async def query(request: QueryRequest):
    try:
        # Use the function you actually defined in agent.py
        answer = get_response(request.query, request.chat_history)
        return {"response": answer}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}