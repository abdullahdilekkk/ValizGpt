import json

def get_suitable_months(vibe, city, region):
    vibe = vibe.lower() if vibe else ""
    city = city.strip().title() if city else ""  # Örn: "antalya" -> "Antalya"
    region = region.strip() if region else ""
    
    tum_aylar = ["Ocak", "Şubat", "Mart", "Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım", "Aralık"]
    kis_aylari = ["Aralık", "Ocak", "Şubat", "Mart"]
    yaz_aylari_genis = ["Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim"]
    yaz_aylari_dar = ["Haziran", "Temmuz", "Ağustos"]
    
    # 🌍 TÜRKİYE COĞRAFYA VERİTABANI (Sıkı Kurallar)
    akdeniz_kiyi = {"Antalya", "Mersin", "Adana", "Hatay"}
    ege_kiyi = {"İzmir", "Aydın", "Muğla"}
    kuzey_kiyi = {"İstanbul", "Kocaeli", "Yalova", "Bursa", "Balıkesir", "Çanakkale", "Tekirdağ", "Kırklareli", "Edirne", "Sakarya", "Düzce", "Zonguldak", "Bartın", "Kastamonu", "Sinop", "Samsun", "Ordu", "Giresun", "Trabzon", "Rize", "Artvin"}
    
    # Web verileriyle doğrulanmış Kar Kayağı merkezleri olan iller
    kayak_illeri = {"Erzurum", "Bursa", "Kayseri", "Kars", "Bolu", "Isparta", "Antalya", "Denizli", "İzmir", "Kocaeli", "Kastamonu", "Erzincan", "Burdur", "Bingöl", "Hakkari", "Sivas"}
    
    # 1. DENİZ / YÜZME / MAVİ YOLCULUK
    if any(k in vibe for k in ["deniz", "plaj", "yüzme", "kum", "su sporu", "mavi yolculuk", "tekne", "koy"]):
        if city in akdeniz_kiyi:
            return ["Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım"]
        elif city in ege_kiyi:
            return yaz_aylari_genis
        elif city in kuzey_kiyi:
            return yaz_aylari_dar
        else:
            # ⛔ KATI KURAL: İç Anadolu'da "deniz" diyen otele boş liste dön! Milvus'ta asla çıkmayacak.
            return []

    # 2. KAYAK / KIŞ SPORU
    if any(k in vibe for k in ["kayak", "kış", "kar", "snowboard"]):
        # Eğer şehirde gerçekten Kar Kayağı merkezi varsa (Antalya Saklıkent, Bursa Uludağ vb.)
        if city in kayak_illeri:
            return kis_aylari
        # Şehirde kar yok ama denize kıyısı varsa (Muğla, Aydın vb.) bu "Su Kayağı"dır
        elif city in akdeniz_kiyi or city in ege_kiyi or city in kuzey_kiyi:
            if city in akdeniz_kiyi or city in ege_kiyi:
                return yaz_aylari_genis
            else:
                return yaz_aylari_dar
        else:
            # ⛔ KATI KURAL: Ne kar var ne deniz, boş liste dön.
            return []

    # 3. DOĞA / YAYLA / KAMP / RAFTİNG
    if any(k in vibe for k in ["doğa", "yayla", "kamp", "orman", "rafting", "macera", "trekking", "çadır"]):
        if region == "Karadeniz":
            return ["Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim"]
        else:
            return ["Nisan", "Mayıs", "Haziran", "Temmuz", "Ağustos", "Eylül", "Ekim", "Kasım"]

    # 4. TERMAL / SPA
    if any(k in vibe for k in ["termal", "spa", "kaplıca", "sağlık", "detoks"]):
        return ["Eylül", "Ekim", "Kasım", "Aralık", "Ocak", "Şubat", "Mart", "Nisan", "Mayıs"]

    # 5. DİĞERLERİ (Tarih, Kültür, İş, Gece Hayatı vb.)
    return tum_aylar

def enrich_json():
    input_file = "hotels_big_db.json"
    print(f"⏳ '{input_file}' okunuyor ve Türkiye coğrafyasına göre filtreleniyor...")
    
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            hotels = json.load(f)
    except Exception as e:
        print(f"❌ Dosya okuma hatası: {e}")
        return
        
    for h in hotels:
        vibe = h.get("info", {}).get("vibe", "")
        city = h.get("location", {}).get("city", "")
        region = h.get("location", {}).get("region", "")
        
        months_list = get_suitable_months(vibe, city, region)
        h["suitable_months"] = ", ".join(months_list)

    try:
        with open(input_file, "w", encoding="utf-8") as f:
            json.dump(hotels, f, ensure_ascii=False, indent=2)
        print("🎉 İşlem tamam! Hatalı coğrafi veriler (Örn: İç Anadolu'da deniz) tamamen temizlendi.")
    except Exception as e:
        print(f"❌ Dosya yazma hatası: {e}")

if __name__ == "__main__":
    enrich_json()