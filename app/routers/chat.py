from fastapi import APIRouter
from pydantic import BaseModel
from app.db.milvus_client import insert_documents
from app.models.schemas import ChatRequest, ChatResponse
from typing import List

from app.utils.embedder import LocalEmbedder


router = APIRouter(tags=["chat"])
embedder = LocalEmbedder()

class DocumentsIn(BaseModel):
    texts: list[str]

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    fake_answer = f"Received message: {request.message}"
    return ChatResponse(answer=fake_answer)

@router.post("/add_documents")
def add_documents(docs: DocumentsIn):
    """
        Birden fazla dökümanın API üzerinden kaydedilmesi
    """
    embeddings = [embedder.embed_text(text) for text in docs.texts]
    insert_documents(docs.texts, embeddings)
    return {"message": "Dökümanlar başarıyla kaydedildi."}