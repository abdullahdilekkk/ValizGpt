import os
import json
import re
import unicodedata
import zeyrek
from typing import Optional, List, Dict, Any

from pymilvus import connections, Collection, utility
from sentence_transformers import SentenceTransformer
from langchain_core.tools import tool

import datetime


# ─────────────────────────────────────────────────────────────────────────────
# 1. MILVUS VE MODEL BAĞLANTISI 
# ─────────────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("⏳ [Tools] Vektör modeli yükleniyor...")
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

print("⏳ [Tools] Zeyrek NLP motoru yükleniyor...")
analyzer = zeyrek.MorphAnalyzer()

print("⏳ [Tools] Milvus'a bağlanılıyor...")
milvus_host = os.environ.get("MILVUS_HOST", "host.docker.internal")
connections.connect("default", host=milvus_host, port="19530")

COLLECTION_NAME = "oteller_milvus_v3"

if utility.has_collection(COLLECTION_NAME):
    collection = Collection(COLLECTION_NAME)
    collection.load()
else:
    raise Exception(f"HATA: '{COLLECTION_NAME}' tablosu bulunamadı! Lütfen önce verileri Milvus'a yükleyin.")

# ─────────────────────────────────────────────────────────────────────────────
# 2. YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────────────────────
def build_db_scope() -> Dict[str, Any]:
    try:
        res = collection.query(expr="min_price >= 0", output_fields=["city", "district", "region", "vibe", "segment", "min_price"], limit=16384)
        cities, districts, regions, vibes, segments = set(), set(), set(), set(), set()
        prices = []
        for m in res:
            if m.get("city"): cities.add(m["city"].strip())
            if m.get("district"): districts.add(m["district"].strip())
            if m.get("region"): regions.add(m["region"].strip())
            if m.get("vibe"): vibes.add(m["vibe"].strip())
            if m.get("segment"): segments.add(m["segment"].strip())
            p = m.get("min_price", 0)
            if p > 0: prices.append(p)
            
        return {
            "cities": cities, "districts": districts, "regions": regions,
            "vibes": vibes, "segments": segments,
            "min_price": min(prices) if prices else 0, "max_price": max(prices) if prices else 0,
            "hotel_count": len(res),
        }
    except Exception:
        return {"cities": set(), "districts": set(), "regions": set(), "vibes": set(), "segments": set(), "min_price": 0, "max_price": 0, "hotel_count": 0}

DB_SCOPE: Dict[str, Any] = build_db_scope()

def tr_normalize(text: str) -> str:
    """Türkçe karakterleri normalize eder ve benzerlik hatalarını önler."""
    return "".join(c for c in unicodedata.normalize('NFC', text))

def tr_slugify(text: str) -> str:
    """Türkçe karakterleri İngilizce karşılıklarına çevirir (ğ -> g, ı -> i vb.)."""
    chars = str.maketrans("ğüşıöçĞÜŞİÖÇ", "gusioçGUSIOC")
    return text.translate(chars)


def pro_normalize(text: str, is_month: bool = False) -> Dict[str, str]:
    """Zeyrek NLP destekli profesyonel normalizasyon (Lemma + NFC + Slug).
    Göreli zamanları, 'X ay sonra' matematiğini ve tam tarihleri gerçek aylara çevirir."""
    t_raw = text.strip().lower()
    
    if is_month:
        # 1. Sistemin şu anki zamanını al
        now = datetime.datetime.now()
        current_month_idx = now.month
        months_tr_list = ["", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
        
        current_month = months_tr_list[current_month_idx]
        next_month = months_tr_list[current_month_idx + 1] if current_month_idx < 12 else "Ocak"
        
        # 2. Göreli Zaman Sözlüğü
        relative_map = {
            "yarın": current_month, "yarin": current_month, "bugün": current_month, "şimdi": current_month,
            "bu hafta": current_month, "haftaya": current_month, "önümüzdeki hafta": current_month,
            "gelecek ay": next_month, "önümüzdeki ay": next_month,
            "bu yaz": "Temmuz", "yazın": "Temmuz", "bu kış": "Ocak", "kışın": "Ocak",
            "baharda": "Mayıs", "ilkbaharda": "Mayıs", "sonbaharda": "Eylül"
        }
        
        if t_raw in relative_map:
            t_raw = relative_map[t_raw].lower()
        else:
            # 🚨 YENİ: "X ay sonra" Dinamik Zaman Hesaplayıcısı (Regex + Modulo)
            match = re.search(r"(\d+)\s*ay\s*sonra", t_raw)
            if match:
                months_to_add = int(match.group(1))
                # Mod 12 mantığı: 36 ay da dese, 400 ay da dese döngüyü hesaplar
                target_month_idx = (current_month_idx - 1 + months_to_add) % 12
                t_raw = months_tr_list[target_month_idx + 1].lower()
            else:
                # 3. Cümle içindeki ("10 nisan", "nisanın ilk haftası") ay ismini agresif şekilde çekip çıkarma
                tr_months_map = {
                    "ocak": "Ocak", "şubat": "Şubat", "subat": "Şubat", "mart": "Mart",
                    "nisan": "Nisan", "mayıs": "Mayıs", "mayis": "Mayıs", "haziran": "Haziran",
                    "temmuz": "Temmuz", "ağustos": "Ağustos", "agustos": "Ağustos",
                    "eylül": "Eylül", "eylul": "Eylül", "ekim": "Ekim", "kasım": "Kasım",
                    "kasim": "Kasım", "aralık": "Aralık", "aralik": "Aralık",
                    "january": "Ocak", "february": "Şubat", "march": "Mart", "april": "Nisan",
                    "may": "Mayıs", "june": "Haziran", "july": "Temmuz", "august": "Ağustos",
                    "september": "Eylül", "october": "Ekim", "november": "Kasım", "december": "Aralık"
                }
                
                found_month = ""
                for key, val in tr_months_map.items():
                    if key in t_raw:
                        found_month = val
                        break
                
                if found_month:
                    t_raw = found_month.lower()

    # 4. Zeyrek ile kök bul (Lemmatization)
    try:
        lemmas = analyzer.lemmatize(t_raw)
        if lemmas and len(lemmas[0]) > 1 and lemmas[0][1]:
            root = lemmas[0][1][0] 
        else:
            root = t_raw
    except Exception:
        root = t_raw
        
    # 5. NFC ve Slug formlarını üret
    norm = tr_normalize(root)
    slug = tr_slugify(root)
    
    if is_month:
        norm = norm.capitalize()
        slug = slug.capitalize()
        
    return {"norm": norm, "slug": slug}


def resolve_region_to_cities(region: str) -> List[str]:
    if not region: return []
    r_lower = region.strip().lower()

    for loc in DB_SCOPE["cities"]:
        if r_lower in loc.lower() or loc.lower() in r_lower: return [loc]

    for loc in DB_SCOPE["districts"]:
        if r_lower in loc.lower() or loc.lower() in r_lower:
            try:
                res = collection.query(expr=f'district == "{loc}"', output_fields=["city"], limit=1)
                return [res[0].get("city")] if res else []
            except Exception: return []

    try:
        query_emb = embedder.encode([region]).tolist()
        res = collection.search(data=query_emb, anns_field="embedding", param={"metric_type": "L2"}, limit=30, output_fields=["city"])
        cities = {str(hit.entity.get("city", "")).strip() for hit in (res[0] if res else []) if hit.entity.get("city")}
        return list(cities) if cities else [region]
    except Exception: return [region]

# ─────────────────────────────────────────────────────────────────────────────
import math

def calculate_real_total_price(rooms_json_str: str, pax: int, duration: int) -> tuple[float, str, float, int]:
    """Oda tiplerine, kapasitelere ve kişi sayısına göre GERÇEK toplam tatil tutarını ve ODA TİPİNİ hesaplar."""
    if not rooms_json_str or rooms_json_str == "[]":
        return float('inf'), "", 0.0, 0
        
    try:
        rooms = json.loads(rooms_json_str)
    except Exception:
        return float('inf'), "", 0.0, 0
        
    best_total_price = float('inf')
    best_room_name = ""
    best_nightly_price = 0.0
    best_room_count = 0
    
    for room in rooms:
        rtype = room.get("type", "").lower()
        original_rname = room.get("type", "Standart Oda") # Arayüzde düzgün görünsün diye orijinal adını alıyoruz
        rprice = room.get("price", float('inf'))
        
        # Kapasite Algoritması
        if "villa" in rtype:
            capacity = 7
        elif "aile" in rtype or "family" in rtype or "2+1" in rtype:
            capacity = 3
        elif "superior" in rtype:
            capacity = 2
        else:
            capacity = 1 # Standart Oda
            
        # Gerekli oda sayısını bul
        required_rooms = math.ceil(pax / capacity)
        
        # Toplam Tatil Fiyatı
        total_price = required_rooms * rprice * (duration if duration > 0 else 1)
        
        # En ucuz kombinasyonu yakala
        if total_price < best_total_price:
            best_total_price = total_price
            best_room_name = original_rname
            best_nightly_price = rprice
            best_room_count = required_rooms
            
    return best_total_price, best_room_name, best_nightly_price, best_room_count

# ─────────────────────────────────────────────────────────────────────────────
# 3. YARDIMCI SÖZLÜKLER (Query Expansion)
# ─────────────────────────────────────────────────────────────────────────────
# Arama motorunu şaşırtabilecek istisnai kelimeler ve genişletmeleri
DOMAIN_SYNONYMS = {
    # ❄️ Kış & Kayak (Homograph / Kano Karışıklığını Önleme)
    "kayak": "kayak kar kış sporları dağ uludağ palandöken erciyes sarıkamış kartepe ski snowboard telesiyej şömine",
    "kar": "kayak kar kış sporları dağ uludağ palandöken erciyes sarıkamış kartepe ski snowboard telesiyej şömine",
    "kış": "kayak kar kış sporları dağ uludağ palandöken erciyes sarıkamış kartepe ski snowboard telesiyej şömine dağ evi",
    
    # 🌊 Deniz & Sahil
    "deniz": "deniz kum güneş sahil plaj kumsal koy kıyı beach tatil köyü resort yüzme şezlong iskele",
    "plaj": "deniz kum güneş sahil plaj kumsal koy kıyı beach tatil köyü resort yüzme şezlong iskele",
    
    # ⛵ Yat & Su Sporları (İngilizce 'yacht' ve su sporu ile ayrıştırma)
    "yat": "yat tekne mavi tur gulet yelkenli kano deniz açıkları koy gezisi marina",
    "tekne": "yat tekne mavi tur gulet yelkenli kano deniz açıkları koy gezisi marina",
    
    # 🌲 Doğa & Kamp
    "doğa": "doğa orman kamp trekking yürüyüş yeşil yeşillik göl şelale yayla karadeniz bungalov oksijen temiz hava",
    "orman": "doğa orman kamp trekking yürüyüş yeşil yeşillik göl şelale yayla karadeniz bungalov oksijen temiz hava",
    "kamp": "kamp çadır karavan glamping doğa ateş bungalov izci",
    
    # ♨️ Termal & Spa & Sağlık
    "termal": "termal kaplıca sıcak su spa ılıca hamam kese masaj şifalı su afyon pamukkale sağlık kür",
    "spa": "termal kaplıca sıcak su spa ılıca hamam kese masaj şifalı su rahatlama bakım sauna",
    
    # 🧘 Dinlenme & Huzur
    "dinlenme": "dinlenme huzur sakin kafa dinleme sessiz inziva yoga detoks retreat yavaş yetişkin oteli",
    "huzur": "dinlenme huzur sakin kafa dinleme sessiz inziva yoga detoks retreat yavaş yetişkin oteli",
    
    # 🪩 Eğlence & Gece Hayatı
    "eğlence": "eğlence gece hayatı parti kulüp dj bar pub festival konser hareketli dans canlı müzik",
    "parti": "eğlence gece hayatı parti kulüp dj bar pub festival konser hareketli dans canlı müzik",
    
    # ❤️ Romantik & Balayı
    "romantik": "romantik balayı çift şömine jakuzi gün batımı şarap sevgili otantik lüks özel havuz",
    "balayı": "romantik balayı çift şömine jakuzi gün batımı şarap sevgili otantik lüks özel havuz",
    
    # 🏛️ Kültür & Tarih
    "kültür": "kültür tarih antik kent müze ören yeri mimari arkeoloji otantik butik taş ev mardin kapadokya",
    "tarih": "kültür tarih antik kent müze ören yeri mimari arkeoloji otantik butik taş ev mardin kapadokya",
    
    # 🍽️ Gastronomi
    "gastronomi": "gastronomi gurme yemek lezzet yöresel mutfak şarap tadımı meze restoran lezzet turu şef",
    
    # 👨👩👧👦 Aile & Çocuk
    "aile": "aile çocuk dostu aquapark su kaydırağı mini kulüp animasyon güvenli aile odası bebek bakıcısı",
    "çocuk": "aile çocuk dostu aquapark su kaydırağı mini kulüp animasyon güvenli aile odası bebek bakıcısı"
}

# 🚀 GENİŞLETİLEBİLİR İŞ KURALLARI MOTORU (Business Rules Engine)
# Vektör modelinin (Yapay Zekanın) halüsinasyon görmesini engelleyecek kesin sınırlar.
# 🚀 GENİŞLETİLEBİLİR KESİN EŞLEŞME MOTORU (Exact Match Engine)
# Veritabanındaki (Milvus) 56 spesifik vibe etiketi ile kullanıcının doğal dilini eşleştirir.
DOMAIN_CONSTRAINTS = {
    "deniz_ve_su": {
        "triggers": ["deniz", "plaj", "kum", "sahil", "yüz", "kıyı", "koy", "tekne", "mavi tur"],
        "banned_regions": ["iç anadolu", "güneydoğu", "doğu anadolu"],
        "required_keywords": ["deniz", "plaj", "yaz tatili", "mavi yolculuk", "su sporları"]
    },
    "kis_ve_kayak": {
        "triggers": ["kayak", "kar", "kış", "snowboard", "telesiyej"],
        "banned_regions": ["güneydoğu", "ege", "akdeniz"], 
        "required_keywords": ["kayak", "kar", "snowboard", "kış tatili", "sömestr tatili"]
    },
    "gastronomi": {
        "triggers": ["yemek", "gurme", "lezzet", "kebap", "gastronomi", "tadım", "şarap", "meze"],
        "banned_regions": [],
        "required_keywords": ["gastronomi", "gurme turu", "yemek tadımı", "yöresel lezzet", "kebap turu"]
    },
    "tarih_ve_kultur": {
        "triggers": ["tarih", "kültür", "antik", "müze", "eski", "nostalji", "ören", "mimari"],
        "banned_regions": [],
        "required_keywords": ["tarih", "kültür", "antik kent", "müze turu", "nostalji"]
    },
    "doga_ve_kamp": {
        "triggers": ["doğa", "kamp", "orman", "yeşil", "yayla", "çadır", "trekking", "şelale", "göl"],
        "banned_regions": [],
        "required_keywords": ["doğa", "kamp", "trekking", "eko turizm", "yeşillik", "yayla"]
    },
    "saglik_ve_wellness": {
        "triggers": ["sağlık", "spa", "termal", "şifa", "sıcak su", "kaplıca", "masaj", "detoks", "wellness"],
        "banned_regions": [],
        "required_keywords": ["sağlık", "spa", "termal", "şifa", "sıcak su", "detoks", "wellness"]
    },
    "eglence_ve_gece_hayati": {
        "triggers": ["eğlence", "parti", "gece", "konser", "festival", "dj", "kulüp", "bar", "dans"],
        "banned_regions": [],
        "required_keywords": ["eğlence", "parti", "gece hayatı", "konser", "festival", "dj performansı", "eğlence kulübü"]
    },
    "spor_ve_aksiyon": {
        "triggers": ["spor", "koşu", "maraton", "bisiklet", "rafting", "triatlon", "ironman"],
        "banned_regions": [],
        "required_keywords": ["spor", "koşu", "maraton", "bisiklet turu", "rafting", "triatlon", "ironman"]
    },
    "zihin_ve_meditasyon": {
        "triggers": ["meditasyon", "yoga", "mindfulness", "inziva", "ruh", "kafa dinleme", "retreat"],
        "banned_regions": [],
        "required_keywords": ["meditasyon", "yoga kampı", "mindfulness", "retreat"]
    },
    "huzur_ve_dinlenme": {
        "triggers": ["huzur", "dinlenme", "sakin", "sessiz", "kaçamak", "hafta sonu", "yavaş"],
        "banned_regions": [],
        "required_keywords": ["huzur", "dinlenme", "kaçamak", "hafta sonu"]
    },
    "is_seyahati": {
        "triggers": ["iş", "toplantı", "kongre", "seminer", "çalışma"],
        "banned_regions": [],
        "required_keywords": ["iş"]
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. ARAÇLAR (TOOLS)
# ─────────────────────────────────────────────────────────────────────────────

@tool
def search_regions(travel_vibe: str, month: str = "", companions: str = "", max_budget: int = 0, duration: int = 0, target_region: str = "") -> str:
    """Kullanıcının tercihine, aya, GECE SAYISINA, BÜTÇESİNE ve isteğe bağlı COĞRAFİ BÖLGESİNE göre Milvus'tan uygun bölgeleri bulur."""
    if not travel_vibe or travel_vibe.strip() == "": return "Eksik bilgi: Tatil türü."
    if not companions or companions.strip() == "": return "Eksik bilgi: Kiminle gidileceği."
    if not month or month.strip() == "": return "Eksik bilgi: Zaman/Ay."

    nightly_budget = 0
    if max_budget > 0 and duration > 0: nightly_budget = int(max_budget / duration)
    elif max_budget > 0 and duration == 0: nightly_budget = max_budget

    month_res = pro_normalize(month, is_month=True)
    norm_month, slug_month = month_res["norm"], month_res["slug"]
    
    raw_vibe = travel_vibe.lower().strip()
    expanded_parts = [DOMAIN_SYNONYMS.get(word, word) for word in raw_vibe.split()]
    expanded_vibe = " ".join(expanded_parts)
    query = f"{expanded_vibe} {companions}".strip()

    month_variants = list(set([norm_month.lower(), norm_month.capitalize(), slug_month.lower(), slug_month.capitalize()]))
    month_clauses = [f'suitable_months like "%{m}%"' for m in month_variants]
    month_filter = "(" + " or ".join(month_clauses) + ")"

    expr_parts = [month_filter]
    if nightly_budget > 0: expr_parts.append(f"min_price <= {nightly_budget}")
    
    if target_region and target_region.strip() != "":
        clean_region = target_region.strip().replace('"', '\\"').title()
        # Hem 'region' hem de 'city' içinde arama yapıyoruz ki kaçak olmasın
        expr_parts.append(f'(region like "%{clean_region}%" or city like "%{clean_region}%")')
    final_expr = " and ".join([f"({e})" for e in expr_parts])
    
    try:
        query_emb = embedder.encode([query]).tolist()
        
        # 800 Limitli Geniş Arama
        results = collection.search(
            data=query_emb, anns_field="embedding", param={"metric_type": "L2", "params": {"nprobe": 10}}, 
            limit=800, expr=final_expr, output_fields=["city", "region", "min_price", "name", "vibe", "concept"]
        )
        hits = results[0] if results else []
        fallback_msg = ""
        
        if not hits and nightly_budget > 0:
            res_fb = collection.search(
                data=query_emb, anns_field="embedding", param={"metric_type": "L2"}, 
                limit=800, expr=month_filter, output_fields=["city", "region", "min_price", "name", "vibe", "concept"]
            )
            hits = res_fb[0] if res_fb else []
            if hits: fallback_msg = f"SİSTEM UYARISI: Bütçenizi aşan ancak en iyi seçenekler."

        if not hits: return json.dumps({"type": "error", "message": "Bulunamadı."}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"type": "error", "message": f"Arama hatası: {e}"}, ensure_ascii=False)

    # =====================================================================
    # 🧠 KESİN EŞLEŞME (EXACT MATCH) VE TEMİZLEME MOTORU
    # =====================================================================
    
    # 1. Matematiksel Dolgu Kalkanı
    dynamic_threshold = float('inf')
    if hits and hasattr(hits[0], "distance"):
        top_k = min(10, len(hits))
        avg_dist = sum(h.distance for h in hits[:top_k]) / top_k
        dynamic_threshold = avg_dist * 1.60 

    # 2. Aktif Kuralları Belirle
    active_constraints = []
    for constraint_name, rules in DOMAIN_CONSTRAINTS.items():
        if any(t in raw_vibe for t in rules["triggers"]):
            active_constraints.append(rules)

    # 3. Şehirleri Tekilleştir ve Kurallara Göre Süz
    city_data: Dict[str, Dict] = {}
    seen_hotel_names = set()
    
    for hit in hits:
        if hasattr(hit, "distance") and hit.distance > dynamic_threshold:
            continue
            
        city = hit.entity.get("city", "")
        name = hit.entity.get("name", "")
        region = str(hit.entity.get("region", "")).lower()
        vibe_concept = (str(hit.entity.get("vibe", "")) + " " + str(hit.entity.get("concept", ""))).lower()
        
        if not city or not name: continue
        
        # B) NOKTA ATIŞI FİLTRELEME (ESNETİLMİŞ OR MANTIĞI)
        is_valid = False
        is_banned = False

        if not active_constraints:
            is_valid = True 
        else:
            for rules in active_constraints:
                if any(b_reg in region for b_reg in rules.get("banned_regions", [])):
                    is_banned = True
                    break
                    
                req_words = rules.get("required_keywords", [])
                if req_words and any(rw in vibe_concept for rw in req_words):
                    is_valid = True 
                    
        if is_banned or not is_valid:
            continue 
            
        # C) Tekilleştirme ve Geçerli Otel Sayımı
        if name in seen_hotel_names: continue
        seen_hotel_names.add(name)
        
        if city not in city_data:
            city_data[city] = {"count": 0, "min_price": float("inf"), "max_price": 0, "region": region.title()}
            
        city_data[city]["count"] += 1
        price = hit.entity.get("min_price", 0)
        if price > 0:
            city_data[city]["min_price"] = min(city_data[city]["min_price"], price)
            city_data[city]["max_price"] = max(city_data[city]["max_price"], price)

    # 4. Şehirleri Kaliteye Göre Sırala ve 2-2-2 Seçimi Yap
    relevant_cities = sorted(city_data.items(), key=lambda x: x[1]["count"], reverse=True)
    
    quality_cities = [c for c in relevant_cities if c[1]["count"] >= 3]
    if len(quality_cities) < 3: 
        quality_cities = relevant_cities[:3]

    top_8_cities = sorted(quality_cities[:8], key=lambda x: x[1]["min_price"])

    if len(top_8_cities) <= 6:
        top_cities = top_8_cities
    else:
        ucuzlar = top_8_cities[:2]
        pahalilar = top_8_cities[-2:]
        kalanlar = top_8_cities[2:-2]
        orta_index = len(kalanlar) // 2
        ortalar = kalanlar[max(0, orta_index-1) : orta_index+1] if kalanlar else []
        top_cities = sorted(ucuzlar + ortalar + pahalilar, key=lambda x: x[1]["min_price"])

    regions = [{"name": c, "region": d["region"], "hotel_count": d["count"], "price_range": f"{d['min_price']:,.0f} – {d['max_price']:,.0f} TL", "min_price": d["min_price"], "max_price": d["max_price"]} for c, d in top_cities]
    
    if not regions:
        return json.dumps({"type": "error", "message": "İstediğiniz karma konseptlere tam uyan bir bölge bulamadım. Kriterleri biraz esnetmemi ister misiniz?"}, ensure_ascii=False)

    msg = fallback_msg if fallback_msg else f"{norm_month} ayı için harika bölgeler bulundu."
    return json.dumps({"type": "region_cards", "regions": regions, "message": msg}, ensure_ascii=False)


@tool
def search_hotels(cities: str, travel_vibe: str, month: str = "", duration: int = 1, max_budget: int = 0, companions: str = "", pax: int = 2) -> str:
    """🚨 DİKKAT: Kullanıcı bölge seçtikten sonra çalışır.
    pax parametresi: Kullanıcının 'sevgilimle', 'ailemle (4 kişi)' vb. beyanlarından çıkarılan NET KİŞİ SAYISI (int).
    """
    if not cities or cities.strip() in ["", "Türkiye", "Turkey", "Bilinmiyor", "Akdeniz", "Ege"]:
        return "SİSTEM UYARISI: HATA! Şehir parametresi eksik. Önce 'search_regions' kullan."
    
    month_res = pro_normalize(month, is_month=True)
    norm_month, slug_month = month_res["norm"], month_res["slug"]
    
    raw_vibe = travel_vibe.lower().strip()
    expanded_parts = [DOMAIN_SYNONYMS.get(word, word) for word in raw_vibe.split()]
    expanded_vibe = " ".join(expanded_parts)
    query = f"{expanded_vibe} {companions}".strip()
    
    city_list = [c.strip() for c in cities.split(",") if c.strip()]
    resolved_cities = list(set([res for c in city_list for res in resolve_region_to_cities(c)]))
    city_array_str = "[" + ", ".join([f"'{c}'" for c in resolved_cities]) + "]"
    city_expr = f"(city in {city_array_str} or region in {city_array_str} or district in {city_array_str})"
    
    month_variants = list(set([norm_month.lower(), norm_month.capitalize(), slug_month.lower(), slug_month.capitalize()]))
    month_clauses = [f'suitable_months like "%{m}%"' for m in month_variants]
    month_filter = "(" + " or ".join(month_clauses) + ")"

    final_expr = f"({city_expr}) and ({month_filter})"
    print(f"🔍 Saf Vektör Araması (Oteller): sorgu='{query}'", flush=True)

    try:
        query_emb = embedder.encode([query]).tolist()

        res = collection.search(
            data=query_emb, anns_field="embedding", param={"metric_type": "L2", "params": {"nprobe": 10}}, 
            limit=50, expr=final_expr, output_fields=["name", "city", "district", "region", "min_price", "stars", "segment", "concept", "vibe", "document", "rooms"]
        )
        hits = res[0] if res and len(res[0]) > 0 else []

        dynamic_threshold = float('inf')
        if hits and hasattr(hits[0], "distance"):
            dynamic_threshold = hits[0].distance * 1.35 + 1.0

        if not hits:
            return json.dumps({"type": "error", "message": f"SİSTEM UYARISI: '{cities}' bölgesinde {norm_month} ayında bu konsepte uygun otel bulunamadı."}, ensure_ascii=False)

    except Exception as e:
        return json.dumps({"type": "error", "message": f"Arama hatası: {e}"}, ensure_ascii=False)

    valid_hotels, seen_names = [], set()
    fallback_msg = ""
    
    for hit in hits:
        name = hit.entity.get("name", "")
        if name in seen_names: continue
        
        if hasattr(hit, "distance") and hit.distance > dynamic_threshold:
            continue
            
        rooms_json_str = hit.entity.get("rooms", "[]")
        
        calc_price, calc_room, nightly_p, r_count = calculate_real_total_price(rooms_json_str, pax, duration)
        
        if max_budget == 0 or calc_price <= max_budget:
            seen_names.add(name)
            valid_hotels.append({
                "name": name, "city": hit.entity.get("city", ""), "district": hit.entity.get("district", ""),
                "region": hit.entity.get("region", ""), 
                "price": calc_price, 
                "room_type": calc_room,
                "nightly_price": nightly_p, 
                "room_count": r_count,       
                "duration": duration,       
                "stars": hit.entity.get("stars", 0), "segment": hit.entity.get("segment", ""),
                "concept": hit.entity.get("concept", ""), "vibe": hit.entity.get("vibe", ""),
                "description": (hit.entity.get("document", "") or "")[:200].replace("\n", " "),
            })

    # 🚨 DÜRÜST VE MÜZAKERECİ FALLBACK
    if not valid_hotels and max_budget > 0:
        print("⚠️ Bütçe yetersiz, bütçesiz havuzdan en uygunlar çekiliyor...", flush=True)
        fallback_msg = f"SİSTEM UYARISI: {pax} kişilik grup ve {duration} gece için {max_budget} TL bütçe yetersiz. Sistemdeki en ucuz seçenekler getirildi. KULLANICIYA ŞUNU SÖYLE: 'Bu kadar kalabalık bir grup için bu bütçeyle yer bulmak bizi biraz zorluyor. Göz atmanız için en uygun fiyatlı seçenekleri getirdim ama daha rahat konseptler için bütçenizi biraz artırmak ister misiniz?' diyerek topu kullanıcıya at ve etkileşimi sürdür. Otel detaylarını metin olarak madde madde YAZMA."
        
        for hit in hits: 
            if len(valid_hotels) >= 6: break
            name = hit.entity.get("name", "")
            if name in seen_names: continue
            
            if hasattr(hit, "distance") and hit.distance > dynamic_threshold:
                continue

            rooms_json_str = hit.entity.get("rooms", "[]")
            calc_price, calc_room, nightly_p, r_count = calculate_real_total_price(rooms_json_str, pax, duration)
            if calc_price == float('inf') or calc_price == 0: continue
            
            seen_names.add(name)
            valid_hotels.append({
                "name": name, "city": hit.entity.get("city", ""), "district": hit.entity.get("district", ""),
                "price": calc_price, 
                "room_type": calc_room,
                "nightly_price": nightly_p, 
                "room_count": r_count,       
                "duration": duration,       
                "stars": hit.entity.get("stars", 0), "concept": hit.entity.get("concept", ""),
                "description": (hit.entity.get("document", "") or "")[:200].replace("\n", " "),
            })

    valid_hotels = sorted(valid_hotels, key=lambda x: x["price"]) 
    
    if len(valid_hotels) <= 6:
        selected_hotels = valid_hotels
    else:
        ucuzlar = valid_hotels[:2]
        pahalilar = valid_hotels[-2:]
        kalanlar = valid_hotels[2:-2]
        orta_index = len(kalanlar) // 2
        ortalar = kalanlar[max(0, orta_index-1) : orta_index+1] if kalanlar else []
        
        selected_hotels = ucuzlar + ortalar + pahalilar

    selected_hotels = sorted(selected_hotels, key=lambda x: x["price"])

    msg = fallback_msg if fallback_msg else f"{pax} kişi için {duration} gecelik toplam bütçenize uygun oteller bulundu."
    return json.dumps({"type": "hotel_cards", "hotels": selected_hotels, "message": msg}, ensure_ascii=False)


@tool
def get_hotel_detail(hotel_name: str, city: str, district: str, pax: int = 2, duration: int = 1) -> str:
    """Milvus'tan otel detayını getirir. İsim benzerliklerini önlemek için mutlaka hotel_name, city ve district parametrelerinin üçü de LLM tarafından sağlanmalıdır."""
    print(f"🔍 Tool: get_hotel_detail(name='{hotel_name}', city='{city}', district='{district}', pax={pax}, duration={duration})", flush=True)
    try:
        clean_name = hotel_name.replace('"', '\\"')
        clean_city = city.replace('"', '\\"') if city else ""
        clean_district = district.replace('"', '\\"') if district else ""
        
        expr_parts = [f'name == "{clean_name}"']
        if clean_city:
            expr_parts.append(f'city == "{clean_city}"')
        if clean_district:
            expr_parts.append(f'district == "{clean_district}"')
            
        final_expr = " and ".join(expr_parts)
        
        res = collection.query(expr=final_expr, output_fields=["name", "city", "district", "region", "min_price", "stars", "segment", "document", "rooms"], limit=1)
        
        if not res: 
            return json.dumps({"type": "hotel_detail", "hotel": None, "message": "Bulunamadı."}, ensure_ascii=False)
        
        meta = res[0]
        rooms_json_str = meta.get("rooms", "[]")
        
        calc_price, calc_room, nightly_p, r_count = calculate_real_total_price(rooms_json_str, pax, duration)
        
        final_price = calc_price if calc_price != float('inf') and calc_price != 0 else meta.get("min_price", 0)

        hotel_info = {
            "name": meta.get("name", ""), 
            "city": meta.get("city", ""), 
            "district": meta.get("district", ""), 
            "region": meta.get("region", ""), 
            "price": final_price, 
            "room_type": calc_room,
            "room_count": r_count,
            "nightly_price": nightly_p,
            "duration": duration,
            "stars": meta.get("stars", 0), 
            "segment": meta.get("segment", ""), 
            "description": meta.get("document", "")
        }
        return json.dumps({"type": "hotel_detail", "hotel": hotel_info, "message": "Detaylar getirildi."}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"type": "hotel_detail", "hotel": None, "message": f"Hata: {str(e)}"}, ensure_ascii=False)