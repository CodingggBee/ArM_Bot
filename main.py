from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Absolute imports based on the new structure
from api.core.agent import agent_executor, memory
from api.core.database import add_documents_to_store
from api.utils.parsers import parse_file

from pydantic import BaseModel
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    query: str
    chat_history: List[Message]

@app.get("/api/health")
def health():
    return {"status": "alive"}

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = await parse_file(file.filename, content)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = text_splitter.create_documents([text])
        add_documents_to_store(chunks, file.filename)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
async def query(request: QueryRequest):
    try:
        memory.chat_memory.clear()
        for m in request.chat_history[-6:]:
            if m.role == "user": memory.chat_memory.add_user_message(m.content)
            else: memory.chat_memory.add_ai_message(m.content)
        
        res = agent_executor.invoke({"input": request.query})
        return {"response": res.get("output", "Error processing request.")}
    except Exception as e:
        return {"response": f"Error: {str(e)}"}