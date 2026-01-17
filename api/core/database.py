import os
import requests
import math

KNOWLEDGE_BASE = []

class LiteHuggingFaceEmbeddings:
    def __init__(self):
        self.api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_TOKEN')}"}

    def embed_documents(self, texts):
        try:
            response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts}, timeout=10)
            return response.json()
        except Exception:
            return [[0.0] * 384 for _ in texts]

    def embed_query(self, text):
        res = self.embed_documents([text])
        return res[0] if isinstance(res, list) and res else [0.0] * 384

embedding_model = LiteHuggingFaceEmbeddings()

def add_documents_to_store(chunks, source_name):
    texts = [c.page_content for c in chunks]
    vectors = embedding_model.embed_documents(texts)
    if not isinstance(vectors, list): return
    
    for i, vec in enumerate(vectors):
        if isinstance(vec, list):
            KNOWLEDGE_BASE.append({"vector": vec, "text": texts[i], "source": source_name})

def dot_product(v1, v2):
    return sum(x * y for x, y in zip(v1, v2))

def magnitude(v):
    return math.sqrt(sum(x * x for x in v))

def similarity_search(query):
    if not KNOWLEDGE_BASE: return "No documents found."
    
    query_vec = embedding_model.embed_query(query)
    query_mag = magnitude(query_vec)
    
    scored_results = []
    for item in KNOWLEDGE_BASE:
        sim = dot_product(item["vector"], query_vec) / (magnitude(item["vector"]) * query_mag + 1e-9)
        scored_results.append((sim, item))
    
    # Sort by similarity
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    results = []
    for score, item in scored_results[:3]:
        results.append(f"Source: {item['source']}\nContent: {item['text']}")
    
    return "\n\n".join(results)