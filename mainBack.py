"""
ValizGPT — Tool Calling Agent Mimarisi v3.0
=============================================
LangGraph + Ollama Tool Calling ile doğal sohbet tabanlı tatil asistanı.

Mimari:
  - Tek bir Agent Node (LLM + bound tools)
  - 3 Tool: search_regions, search_hotels, get_hotel_detail
  - LLM kendi karar veriyor: konuşmak mı, tool çağırmak mı
  - Mesaj başına 1 LLM çağrısı (+ tool loop)
  - Frontend'e structured card data (bölge kartları, otel kartları)
"""

from __future__ import annotations

import json
import re
import uuid
import os
import sqlite3
import asyncio
from typing import Optional, List, Dict, Any, AsyncIterator

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import httpx

# LangChain + LangGraph
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import (
    HumanMessage, AIMessage, ToolMessage, SystemMessage,
    BaseMessage
)
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Session memory
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    import sqlite3
    
    # Kilitlenmeleri (database is locked / deadlock) kökten çözmek için:
    # 1. timeout=15 ile beklemeleri tolere ediyoruz
    # 2. PRAGMA journal_mode=WAL ile okuma ve yazma işlemlerinin aynı anda yapılabilmesini sağlıyoruz
    _conn = sqlite3.connect("./sessions_v3.db", check_same_thread=False, timeout=15.0)
    _conn.execute("PRAGMA journal_mode=WAL;")
    
    memory = SqliteSaver(_conn)
    MEMORY_AVAILABLE = True
    print("✅ Session memory (SQLite - WAL Mode) aktif ve kilitlenmelere karşı korumalı.")
except Exception as e:
    try:
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        MEMORY_AVAILABLE = True
        print(f"⚠️  SQLite hatası ({e}), InMemorySaver kullanılıyor.")
    except Exception as e2:
        memory = None
        MEMORY_AVAILABLE = False
        print(f"⚠️  Session memory kurulamadı: {e2}")


# Zeyrek Türkçe NLP
try:
    from nlp_utils import normalize_text, get_root, extract_roots_from_text
    ZEYREK_AVAILABLE = True
    print("✅ Zeyrek NLP modülü yüklendi.")
except ImportError as e:
    ZEYREK_AVAILABLE = False
    print(f"⚠️  Zeyrek yüklenemedi: {e}. Kök analizi devre dışı.")
    def normalize_text(t): return t.lower().strip() if t else ""
    def get_root(w): return w.lower().strip() if w else ""
    def extract_roots_from_text(t): return set()


# ─────────────────────────────────────────────────────────────────────────────
# 0. AYARLAR & CHROMADB
# ─────────────────────────────────────────────────────────────────────────────

ollama_host = os.environ.get("OLLAMA_HOST", "http://host.docker.internal:11434")
LLM_MODEL = "gpt-oss:120b-cloud"

print(f"🔗 Ollama host: {ollama_host}")
print(f"🤖 Model: {LLM_MODEL}")

# Yeni oluşturduğumuz dosyadan Araçları ve DB verisini içeri aktarıyoruz
from agent_tools import search_regions, search_hotels, get_hotel_detail, DB_SCOPE, collection

tools_list = [search_regions, search_hotels, get_hotel_detail]

# ChatOllama — LangChain entegrasyonu ile tool calling
print(f"⏳ ChatOllama başlatılıyor ({LLM_MODEL})...")
llm = ChatOllama(
    model=LLM_MODEL,
    base_url=ollama_host,
    timeout=300,
    temperature=0.1
)
llm_with_tools = llm.bind_tools(tools_list)
print("✅ ChatOllama + Tools hazır.")

# ── System Prompt — VERİ BAZLI, kural değil ──────────────────────────────────

def build_system_prompt() -> str:
    """DB'deki gerçek verileri system prompt'a enjekte eder."""
    cities = sorted(DB_SCOPE.get("cities", set()))
    vibes = sorted(DB_SCOPE.get("vibes", set()))
    min_p = DB_SCOPE.get("min_price", 0)
    max_p = DB_SCOPE.get("max_price", 0)
    count = DB_SCOPE.get("hotel_count", 0)

    cities_str = ", ".join(cities) if cities else "bilgi yok"
    vibes_str = ", ".join(vibes) if vibes else "çeşitli"

    return f"""Sen ValizGPT'sin. İnsanların hayallerindeki tatili bulmalarına yardım eden, son derece enerjik, samimi ve espirili bir yapay zeka seyahat danışmanısın. Karşındaki insanla robot gibi değil, sanki yıllardır tanıdığın bir dostunla tatil planı yapıyormuş gibi sıcak, akıcı ve doğal konuş.

Mevcut Veriler:
- Şehirler: {cities_str}
- Türler: {vibes_str}
- Fiyatlar: {min_p:,} – {max_p:,} TL
- Toplam Otel: {count}

Akış Kuralları:
1. Kullanıcının tatil türünü (vibe), kiminle gideceğini, ne zaman gideceğini, kaç gece konaklayacağını VE bütçesini öğrenmen gerekiyor.
2. KESİN KURAL (CRITICAL) - BİLGİLERİ TEK TEK TOPLA: Eksik birden fazla bilgi olsa bile, tek mesajda SADECE 1 EKSİK BİLGİYİ SOR. Aynı mesajda asla iki farklı soru sorma. Mesajında sadece ve sadece 1 tane soru işareti (?) bulunabilir. "ve", "peki" gibi bağlaçlarla soruları birleştirmek KESİNLİKLE YASAKTIR.
   - DOĞRU ÖRNEK: "Hangi ay gitmeyi planlıyorsunuz?" (DUR. Başka bir şey sorma.)
   - DOĞRU ÖRNEK: "Bütçeniz ne kadar?" (DUR. Başka bir şey sorma.)
   - YASAKLI ÖRNEK: "Ne zaman gitmek istersiniz ve bütçeniz ne kadar?" (YASAK: İki soru birleşmiş)
   - YASAKLI ÖRNEK: "Kaç gece kalacaksınız? Peki kiminle gidiyorsunuz?" (YASAK: İki soru var)
3. "Hangi ay gideceksiniz?" yerine "Ne zamanlar gitmeyi düşünüyorsunuz?" gibi daha doğal ve sohbet havasında sorular sor. 
    ZAMAN VE TAKVİM KURALI: Kullanıcının zaman bildiren tüm ifadelerini ("yarın", "haftaya", "bu yaz", "kışın", "yakında", "ocak" vb.) HİÇ DEĞİŞTİRMEDEN doğrudan araçlara 'month' parametresi olarak gönder. Sistem arka planda bu kelimeleri gerçek takvime çevirecektir. Kullanıcı "yarın" dediyse zaman bilgisini alınmış kabul et ve KESİNLİKLE tekrar ay/zaman sorma.
4. PROFESYONEL BÜTÇE/SÜRE/PAX YÖNETİMİ: 
   - 'pax' (Kişi Sayısı) Hesaplama: Kullanıcının 'sevgilimle' (2), 'ailemle 4 kişi' (4), 'tek başıma' (1) gibi ifadelerinden net bir 'pax' rakamı çıkar.
5. Bilgiler tamamlanınca 'search_regions' çağır. Kullanıcı bölge seçince (ve bütçe/pax belli olunca) 'search_hotels' çağır.
6. FİYAT ÇIPALAMA (UX): 'search_hotels' artık 2 Ucuz, 2 Orta ve 2 Lüks otel getiriyor.Bütçe uygunsa "bütçenize tam uyan harika bir karma hazırladım" de.EĞER araç sana "Bütçe yetersiz" uyarısı verirse dürüst ol: "Bu kalabalık için bu bütçe biraz zorlayıcı oldu ama bulabildiğim en uygunları şunlar, daha rahat seçenekler için bütçeyi biraz artırmak ister misiniz?" diyerek topu kullanıcıya at.
7. GENERATIVE UI KURALI : Araçlardan dönen sonuçları, otel isimlerini veya fiyatları ASLA metin içinde madde madde yazma! Sistem bunları kart olarak gösteriyor. Kısa bir sunuş yap (gerekirse bütçeyi sor) ve SUS.
8. COĞRAFİ KESİNLİK KURALI: Eğer kullanıcı "Akdeniz", "Karadeniz", "Ege" gibi spesifik bir bölge veya "Antalya", "Muğla" gibi spesifik bir şehir ismi verirse, bu kelimeleri 'travel_vibe' parametresine DAHİL ETME. Bu lokasyon isimlerini doğrudan araçların 'target_region' (veya 'cities') parametresine gönder.

Yanıtların kısa (1-2 cümle) ama her zaman sıcak ve heyecanlı olsun. Veritabanı dışındaki bölgeleri önerme."""

SYSTEM_PROMPT = build_system_prompt()
print(f"✅ System prompt hazır ({len(SYSTEM_PROMPT)} karakter, veri enjekte edildi)")


# ── LangGraph Nodes ──────────────────────────────────────────────────────────

def agent_node(state: MessagesState) -> Dict:
    """Agent: LLM'i tools ile çağır. Gerekirse tool çağrısı yap, yoksa konuş."""
    messages = state["messages"]

    # 🧠 ENDÜSTRİ STANDARDI: Seçici Bağlam Sıkıştırma (Selective Context Compression)
    # Human (Kullanıcı) ve AI (Bot) mesajlarını ASLA silmiyoruz. Böylece "hap bilgiler" (vibe, kişi, bütçe) asla unutulmaz.
    # Sadece LLM'in Token sınırını patlatan eski "ToolMessage" (Devasa JSON) verilerini kırpıyoruz.
    
    compressed_messages = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            # Sadece son 4 mesaj içindeki güncel Tool verilerini tam tut
            is_recent = i >= len(messages) - 4
            if is_recent:
                compressed_messages.append(msg)
            else:
                # Eski tool verilerini LLM beynini şişirmemesi için ufalt
                short_content = msg.content[:100] + "... [BELLEK TASARRUFU: ESKİ JSON VERİSİ GİZLENDİ]"
                compressed_messages.append(ToolMessage(content=short_content, name=msg.name, tool_call_id=msg.tool_call_id))
        else:
            # Kullanıcının verdiği tüm hap bilgiler bu mesajların içinde sonsuza kadar yaşar.
            compressed_messages.append(msg)

    # System prompt'u her çağrıda başa ekle
    full_messages = [SystemMessage(content=SYSTEM_PROMPT)] + compressed_messages

    print(f"🤖 Agent çağrılıyor... (Hafızadaki Sohbet: {len(compressed_messages)} mesaj)", flush=True)
    try:
        response = llm_with_tools.invoke(full_messages)
    except Exception as e:
        print(f"❌ Agent LLM hatası: {e}", flush=True)
        response = AIMessage(content="Bir sorun oluştu, tekrar dener misin? 🙏")
    
    # Thinking tag temizliği (model thinking capability'ye sahip)
    if hasattr(response, "content") and response.content:
        response.content = re.sub(
            r"<think>.*?</think>", "", response.content, flags=re.DOTALL
        ).strip()
        # 🚨 KESİN ÇÖZÜM KALKANI: BİRDEN FAZLA SORUYU FİZİKSEL OLARAK KES 🚨
        # Eğer model bir tool çağırmıyorsa (sadece kullanıcıyla sohbet ediyorsa)
        if not hasattr(response, "tool_calls") or not response.tool_calls:
            # Eğer cümlede 1'den fazla soru işareti varsa
            if response.content.count('?') > 1:
                # Metni soru işaretlerinden böl
                parts = response.content.split('?')
                # Sadece ilk soruyu al ve sonuna soru işaretini geri koy. Gerisini çöpe at!
                response.content = parts[0].strip() + '?'
    # Tool çağrısı log
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        print(f"🛠️  Agent tool çağırdı: {tool_names}", flush=True)
    else:
        preview = (response.content or "")[:100]
        print(f"💬 Agent yanıt: {preview}...", flush=True)

    return {"messages": [response]}


# ToolNode — tool çağrılarını otomatik yönetir
tool_node = ToolNode(tools_list)


# ── Graph ────────────────────────────────────────────────────────────────────

workflow = StateGraph(MessagesState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", tools_condition)
workflow.add_edge("tools", "agent")

# Compile
if MEMORY_AVAILABLE and memory:
    app_graph = workflow.compile(checkpointer=memory)
    print("✅ LangGraph (tool calling + session memory) derlendi.")
else:
    app_graph = workflow.compile()
    print("✅ LangGraph (tool calling, memory yok) derlendi.")


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARD EXTRACTION — Tool sonuçlarından kart verisini çıkar
# ─────────────────────────────────────────────────────────────────────────────

def extract_cards_from_messages(messages: list) -> Optional[Dict]:
    """
    Mesaj geçmişinden SADECE EN SON TURDAKİ tool sonucundaki kart verisini çıkarır.
    Eski kartların hortlamasını engeller.
    """
    for msg in reversed(messages):
        # 🛡️ GÜVENLİK DUVARI: Eğer kullanıcı mesajına (HumanMessage) denk gelirsek,
        # demek ki bu turda hiçbir tool çalışmamış demektir. Taramayı durdur!
        if isinstance(msg, HumanMessage):
            break 
            
        if isinstance(msg, ToolMessage):
            try:
                data = json.loads(msg.content)
                if data.get("type") in ("region_cards", "hotel_cards", "hotel_detail"):
                    return data
            except (json.JSONDecodeError, TypeError):
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. FASTAPI
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="ValizGPT API", version="3.0")

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "3.0", "model": LLM_MODEL}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request/Response Modelleri ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    history: List[Dict[str, str]] = Field(default_factory=list)  # Frontend backup

class ChatResponse(BaseModel):
    reply: str
    cards: Optional[Dict] = None
    session_id: str


# ── Graph çalıştırıcı ────────────────────────────────────────────────────────

def _run_graph(user_message: str, session_id: str, history: List[Dict[str, str]] = None) -> Dict:
    """Graph'ı çalıştır. Checkpoint varsa onu kullan, yoksa frontend history'yi kullan."""
    config = {"configurable": {"thread_id": session_id}}

    # Checkpoint'te bu session için mesaj var mı kontrol et
    has_checkpoint = False
    try:
        checkpoint_state = app_graph.get_state(config)
        existing_msgs = checkpoint_state.values.get("messages", [])
        has_checkpoint = len(existing_msgs) > 0
        if has_checkpoint:
            print(f"📚 Checkpoint'te {len(existing_msgs)} mesaj mevcut.", flush=True)
    except Exception as e:
        print(f"⚠️  Checkpoint okunamadı: {e}", flush=True)

    if has_checkpoint:
        # Checkpoint var → sadece yeni mesajı ekle
        result = app_graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
    elif history and len(history) > 0:
        # Checkpoint yok ama frontend history var → tam geçmişi gönder
        print(f"📋 Frontend history kullanılıyor ({len(history)} mesaj)", flush=True)
        messages = []
        for h in history:
            if h.get("role") == "user":
                messages.append(HumanMessage(content=h["content"]))
            elif h.get("role") == "assistant":
                messages.append(AIMessage(content=h["content"]))
        messages.append(HumanMessage(content=user_message))
        result = app_graph.invoke({"messages": messages}, config=config)
    else:
        # İlk mesaj
        result = app_graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config=config,
        )
    return result


# ── Normal (JSON) endpoint ────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    print(f"\n{'='*60}")
    print(f"📩 Kullanıcı: {request.message}")
    print(f"🔑 Session: {session_id}")
    print(f"{'='*60}")

    try:
        result = await asyncio.to_thread(_run_graph, request.message, session_id, request.history)
    except Exception as e:
        print(f"❌ Graph hatası: {e}", flush=True)
        return ChatResponse(
            reply="Bir sorun oluştu. Tekrar dener misin? ",
            cards=None,
            session_id=session_id,
        )

    # Son AI mesajını bul
    reply = ""
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            reply = msg.content
            break

    if not reply:
        reply = "Seni anlayamadım, tekrar söyler misin? "

    # Kartları çıkar
    cards = extract_cards_from_messages(result["messages"])

    print(f"📤 Yanıt: {reply[:100]}...")
    if cards:
        print(f"🃏 Kart tipi: {cards.get('type')}")

    return ChatResponse(reply=reply, cards=cards, session_id=session_id)


# ── Streaming (SSE) endpoint ──────────────────────────────────────────────────

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """Server-Sent Events ile streaming yanıt."""
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncIterator[str]:
        # Graph'ı arka planda çalıştır
        graph_task = asyncio.create_task(
            asyncio.to_thread(_run_graph, request.message, session_id, request.history)
        )

        # Beklerken ping at
        while not graph_task.done():
            yield f"data: {json.dumps({'type': 'ping'})}\n\n"
            await asyncio.sleep(3)

        try:
            result = graph_task.result()
        except Exception as e:
            print(f"❌ Stream graph hatası: {e}", flush=True)
            yield f"data: {json.dumps({'type': 'text', 'content': 'Bir sorun oluştu. Tekrar dener misin? 🙏'})}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'session_id': session_id, 'cards': None})}\n\n"
            return

        # Son AI mesajını bul
        reply = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                reply = msg.content
                break

        if not reply:
            reply = "Seni anlayamadım, tekrar söyler misin? 🤔"

        # Kartları çıkar
        cards = extract_cards_from_messages(result["messages"])

        # Kelime kelime stream et
        words = reply.split(" ")
        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
            await asyncio.sleep(0.03)

        # State + cards gönder
        done_payload = {
            "type": "done",
            "session_id": session_id,
            "cards": cards,
        }
        yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ── Session sıfırlama ────────────────────────────────────────────────────────

@app.post("/api/reset")
async def reset_session(request: ChatRequest):
    """Frontend'den session sıfırlama isteği."""
    return {"status": "ok", "message": "Yeni oturum başlatabilirsiniz.", "session_id": str(uuid.uuid4())}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)