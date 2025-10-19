from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import requests
import time

# setup_milvus.py
from app.db.milvus_client import connect_to_milvus, create_collections, insert_documents
from app.utils.embedder import LocalEmbedder
from pymilvus import utility, Collection

# 1. Milvus'a bağlan
connect_to_milvus()

# 2. Koleksiyon varsa sil
if "documents" in utility.list_collections():
    utility.drop_collection("documents")
    print("Eski 'documents' silindi.")

# 3. Yeni koleksiyon oluştur
create_collections()  # create_collections() dim'i embedder ile uyumlu olsun

# 4. Örnek metinler ve embedding
embedder = LocalEmbedder()
texts = [
    "Milvus vektör tabanlı bir veritabanıdır.",
    "RAG uygulamaları embedding kullanır.",
    "Sentence Transformers, metinleri sayısal vektörlere dönüştürür."
]
embeddings = embedder.embed(texts)

# 5. Milvus'a insert
insert_documents(texts, embeddings.tolist())
print("Veriler insert edildi.")

# 6. Flush ve load
coll = Collection("documents")
coll.flush()
coll.create_index(
    field_name="embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}}
)
coll.load()
print("Koleksiyon hazır, toplam kayıt sayısı:", coll.num_entities)


# 1️⃣ Milvus bağlantısı
connections.connect("default", host="localhost", port="19530")
print("✅ Connected to Milvus")

# 2️⃣ Collection kontrol / oluştur
collection_name = "documents"

try:
    collection = Collection(collection_name)
    print("📦 Collection bulundu:", collection_name)
except:
    print("🆕 Yeni collection oluşturuluyor:", collection_name)
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "document embeddings")
    collection = Collection(collection_name, schema)

# 3️⃣ Embedding modeli
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# 4️⃣ Doküman ekleme fonksiyonu
def add_document(doc_id, text):
    embedding = embed_model.encode(text).tolist()
    collection.insert([[doc_id], [embedding], [text]])
    collection.flush()
    print(f"✅ Doküman eklendi: {doc_id} -> {text[:40]}...")

# 5️⃣ Arama fonksiyonu
def query_documents(question, top_k=3):
    q_emb = embed_model.encode(question).tolist()
    collection.load()  # 🔥 Arama öncesi şart
    print("🔍 Collection loaded for search")
    
    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    
    docs = []
    for hits in results:
        for hit in hits:
            docs.append(hit.entity.get("text"))
    return docs

# 6️⃣ Ollama API (Llama3)
def ask_llama3(context, question):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    url = "http://localhost:11434/api/generate"  # ✅ doğru endpoint
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False,   # True olursa parça parça gelir
        "options": {"temperature": 0.7}
    }

    response = requests.post(url, json=data)
    if response.status_code != 200:
        raise Exception(f"Ollama request failed: {response.text}")
    
    result = response.json()
    return result.get("response", "").strip()

# 7️⃣ Test dokümanlar (tek seferlik)
if not collection.num_entities:
    add_document(1, "Python bir programlama dilidir.")
    add_document(2, "Milvus bir vektör veritabanıdır.")
    add_document(3, "Ollama, yerel dil modellerini çalıştırmak için kullanılır.")
    time.sleep(1)

# 8️⃣ Test sorgusu
question = "Milvus nedir?"
docs = query_documents(question)
context = "\n".join(docs)
answer = ask_llama3(context, question)

print("\n📄 Dokümanlar:", docs)
print("\n🤖 Cevap:", answer)
