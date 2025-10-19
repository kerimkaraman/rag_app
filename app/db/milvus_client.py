from pymilvus import connections, utility, FieldSchema, DataType, Collection, CollectionSchema

def connect_to_milvus():
    """Milvus'a bağlanır ve bağlantı durumunu döner."""
    connections.connect(alias="default", host="localhost", port="19530")
    print("✅ Connected to Milvus")

def check_collections():
    """Milvus'ta var olan koleksiyonları listeler."""
    print("📦 Existing collections:", utility.list_collections())

def create_collections():
    """
    'documents' adlı bir koleksiyon oluşturur.
    - id: her dokümanın benzersiz numarası
    - content: dokümanın metni
    - embedding: dokümanın vektör embedding'i (float list)
    """
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)  # örnek dim
    ]

    schema = CollectionSchema(fields, description="RAG Documents Collection")
    Collection(name="documents", schema=schema)
    print("✅ 'documents' koleksiyonu oluşturuldu")

def insert_document(content: str, embedding: list[float]):
    """
    TEK döküman ekleme
    """
    collection = Collection("documents")
    collection.insert([ [content], [embedding] ])  # TEK doküman için bu doğru
    print("✅ Tek döküman kaydedildi")

def insert_documents(contents: list[str], embeddings: list[list[float]]):
    """
    Birden fazla döküman ekleme
    """
    collection = Collection("documents")
    
    # İç içe liste hatasını düzeltmek için:
    # Her alan kendi listesinde olmalı ama listeleri tekrar sarmamıza gerek yok
    collection.insert([contents, embeddings])
    
    print(f"✅ {len(contents)} döküman kaydedildi")
