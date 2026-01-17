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

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Message]

@app.get("/")
def root():
    return {"message": "ArM Bot API is running!"}
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