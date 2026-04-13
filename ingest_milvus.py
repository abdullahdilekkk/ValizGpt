import json
import os
import time
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer

import socket

# 1. AYARLAR
JSON_PATH = "hotels_big_db.json"
COLLECTION_NAME = "oteller_milvus_v3"

def get_milvus_host():
    """Bağlantı adresini ortama göre dinamik belirler."""
    env_host = os.environ.get("MILVUS_HOST")
    if env_host: return env_host
    try:
        socket.gethostbyname("host.docker.internal")
        return "host.docker.internal"
    except (socket.gaierror, socket.error):
        return "localhost"

MILVUS_HOST = get_milvus_host()

def main():
    print(f"🚀 Milvus Veri Yükleme (V3) Başlatılıyor... (Host: {MILVUS_HOST})")
    
    # Embedding modelini yükle
    print("⏳ Vektör modeli yükleniyor...")
    model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

    # Milvus Bağlantısı
    print(f"⏳ Milvus'a bağlanılıyor ({MILVUS_HOST})...")
    try:
        connections.connect("default", host=MILVUS_HOST, port="19530")
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}. Lütfen Milvus'un çalıştığından emin olun.")
        return

    # Koleksiyon Temizliği (Varsa sil)
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
        print(f"🧹 Eski koleksiyon '{COLLECTION_NAME}' silindi.")

    # 2. ŞEMA TANIMI
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
        FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="city", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="district", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="region", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="vibe", dtype=DataType.VARCHAR, max_length=100),
        FieldSchema(name="segment", dtype=DataType.VARCHAR, max_length=100), # 🚨 YENİ EKLENDİ
        FieldSchema(name="stars", dtype=DataType.INT64),
        FieldSchema(name="min_price", dtype=DataType.INT64),
        FieldSchema(name="concept", dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name="suitable_months", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="rooms", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, description="ValizGPT Otel Veritabanı V3")
    collection = Collection(COLLECTION_NAME, schema)
    print(f"✅ Yeni şema oluşturuldu: {COLLECTION_NAME}")

    # 3. VERİ HAZIRLAMA
    print(f"⏳ '{JSON_PATH}' okunuyor...")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    ids, embeddings, names, cities, districts, regions, vibes, segments = [], [], [], [], [], [], [], []
    stars, min_prices, concepts, months, documents, rooms_data = [], [], [], [], [], []

    print(f"🧠 {len(data)} otel için embedding üretiliyor...")
    
    for idx, otel in enumerate(data):
        # En ucuz oda fiyatını bul
        rooms_list = otel.get("rooms", [])
        prices = [r.get("price", 0) for r in rooms_list if r.get("price")]
        min_p = min(prices) if prices else 0

        # RAG Metni (Embedding için)
        tags_str = ", ".join(otel.get("rag_data", {}).get("search_tags", []))
        text_blob = f"{otel['info']['name']} {otel['location']['city']} {otel['location']['district']} {tags_str} {otel['rag_data']['description']}"
        
        ids.append(int(otel.get("id", idx)))
        names.append(otel["info"].get("name", ""))
        cities.append(otel["location"].get("city", ""))
        districts.append(otel["location"].get("district", ""))
        regions.append(otel["location"].get("region", ""))
        vibes.append(otel["info"].get("vibe", ""))
        segments.append(otel["info"].get("segment", "")) 
        stars.append(int(otel["info"].get("stars", 0)))
        min_prices.append(int(min_p))
        concepts.append(otel["details"].get("concept", ""))
        months.append(otel.get("suitable_months", ""))
        documents.append(otel["rag_data"].get("description", "")[:1900])
        # 🚨 ODALARI JSON STRING OLARAK SAKLA
        rooms_data.append(json.dumps(rooms_list, ensure_ascii=False))

        if (idx + 1) % 500 == 0:
            print(f"   -> {idx + 1} otel işlendi.")

    # 4. TOPLU EMBEDDING ÜRETİMİ
    print(f"🧠 {len(documents)} otel için vektörler üretiliyor (bu biraz sürebilir)...")
    all_embeddings = model.encode(documents, batch_size=64, show_progress_bar=True).tolist()

    # 5. INSERT
    print("💾 Milvus'a yazılıyor...")
    collection.insert([
        ids, all_embeddings, names, cities, districts, regions, vibes, segments,
        stars, min_prices, concepts, months, documents, rooms_data
    ])
    collection.flush()
    print("✅ Veriler yüklendi.")

    # 5. INDEX OLUŞTURMA
    print("⚡ Index oluşturuluyor (HNSW)...")
    index_params = {
        "index_type": "HNSW",
        "metric_type": "L2",
        "params": {"M": 8, "efConstruction": 64}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print("🎉 İŞLEM TAMAM! Yeni mimari kullanıma hazır.")

if __name__ == "__main__":
    main()
