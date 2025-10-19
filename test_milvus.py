# setup_milvus.py
from app.db.milvus_client import connect_to_milvus, create_collections, insert_documents
from app.utils.embedder import LocalEmbedder
from pymilvus import utility, Collection

# 1️⃣ Milvus'a bağlan
connect_to_milvus()

# 2️⃣ Koleksiyon varsa sil
if "documents" in utility.list_collections():
    utility.drop_collection("documents")
    print("Eski 'documents' silindi.")

# 3️⃣ Yeni koleksiyon oluştur
create_collections()  # create_collections() dim'i embedder ile uyumlu olsun

# 4️⃣ Örnek metinler ve embedding
embedder = LocalEmbedder()
texts = [
    "Milvus vektör tabanlı bir veritabanıdır.",
    "RAG uygulamaları embedding kullanır.",
    "Sentence Transformers, metinleri sayısal vektörlere dönüştürür."
]
embeddings = embedder.embed(texts)

# 5️⃣ Milvus'a insert
insert_documents(texts, embeddings.tolist())
print("Veriler insert edildi.")

# 6️⃣ Flush ve load
coll = Collection("documents")
coll.flush()
coll.create_index(
    field_name="embedding",
    index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 1024}}
)
coll.load()
print("Koleksiyon hazır, toplam kayıt sayısı:", coll.num_entities)
