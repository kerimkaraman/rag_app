# rag_ollama.py
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import requests
import time

# 1. Milvus bağlantısı
connections.connect("default", host="localhost", port="19530")
print("Milvus'a başarıyla bağlanıldı.")

# 2. Koleksiyon kontrolü veya oluşturulması
collection_name = "documents"
try:
    collection = Collection(collection_name)
    print(f"'{collection_name}' koleksiyonu bulundu.")
except Exception as e:
    print(f"Koleksiyon bulunamadı, yeni koleksiyon oluşturuluyor... Hata: {e}")
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="RAG Doküman Koleksiyonu")
    collection = Collection(collection_name, schema)
    print(f"'{collection_name}' koleksiyonu başarıyla oluşturuldu.")

# 3. Embedding modeli yükleniyor
print("Embedding modeli yükleniyor...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Model başarıyla yüklendi.")

# 4. Doküman ekleme fonksiyonu
def add_document(text: str):
    embedding = embed_model.encode(text).tolist()
    collection.insert([[embedding], [text]])
    print(f"Doküman eklendi: '{text}'")

# Flush ve index işlemleri
print("Dokümanlar Milvus'a kaydediliyor (flush işlemi yapılıyor)...")
collection.flush()

if len(collection.indexes) == 0:
    print("Embedding alanı için index oluşturuluyor...")
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128}
    }
    collection.create_index("embedding", index_params)
    print("Index oluşturuldu.")
else:
    print("Index zaten mevcut.")

# Sorgulama fonksiyonu
def query_documents(question, top_k=3):
    q_emb = embed_model.encode(question).tolist()
    print("Koleksiyon belleğe yükleniyor...")
    collection.load()
    print("Arama işlemi başlatıldı...")
    results = collection.search(
        data=[q_emb],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text"]
    )
    docs = [hit.entity.get("text") for hits in results for hit in hits]
    collection.release()
    return docs

# Ollama API ile Llama3'e istek gönderme
def ask_llama3(context, question):
    prompt = f"Aşağıdaki bağlamı kullanarak soruya cevap ver.\n\nBağlam:\n{context}\n\nSoru: {question}\nCevap:"
    url = "http://localhost:11434/api/generate"
    data = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json().get("response", "Hata: 'response' anahtarı bulunamadı.")
    except requests.exceptions.RequestException as e:
        print(f"Ollama API'sine bağlanırken bir hata oluştu: {e}")
        return "Ollama API'sine bağlantı sağlanamadı."

# Örnek doküman ekleme
if collection.num_entities == 0:
    print("Koleksiyon boş, örnek dokümanlar yükleniyor...")
    add_document("Python, Guido van Rossum tarafından oluşturulmuş, genel amaçlı bir programlama dilidir.")
    add_document("Milvus, yapay zeka uygulamaları için özel olarak tasarlanmış açık kaynaklı bir vektör veritabanıdır.")
    add_document("RAG, büyük dil modellerinin harici bir bilgi tabanından veri alarak daha doğru ve güncel cevaplar üretmesini sağlayan bir tekniktir.")
    collection.flush()
else:
    print("Koleksiyonda halihazırda veri mevcut, yeni doküman eklenmedi.")

# Test
deger = "Milvus nedir?"
print(f"\n--- Soru: {deger} ---")
docs = query_documents(deger)
context = "\n- ".join(docs)
print(f"Bulunan ilgili dokümanlar:\n- {context}")
print("\nLlama3'e soru gönderiliyor...")
answer = ask_llama3(context, deger)
print("\n--- Cevap ---")
print(answer)