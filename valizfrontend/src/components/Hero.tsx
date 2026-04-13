import { useState, useRef, useEffect, useCallback } from 'react';
import type { FormEvent } from 'react';
import bgImage from '../assets/bg.png';
import RegionCard from './RegionCard';
import HotelCard from './HotelCard';

// ─── Tipler ───────────────────────────────────────────────────────────────────

interface CardData {
  type: 'region_cards' | 'hotel_cards' | 'hotel_detail';
  regions?: Array<{
    name: string;
    region?: string;
    hotel_count: number;
    price_range: string;
    min_price?: number;
    max_price?: number;
  }>;
  hotels?: Array<{
    name: string;
    city: string;
    district?: string;
    region?: string;
    price: number;
    stars?: number;
    segment?: string;
    concept?: string;
    vibe?: string;
    description?: string;
  }>;
  hotel?: Record<string, unknown>;
  message?: string;
}

interface Message {
  id: number;
  text: string;
  sender: 'user' | 'bot';
  isStreaming?: boolean;
  cards?: CardData | null;
  cardsLocked?: boolean; // Kart seçimi yapıldıktan sonra kilitlenir
}

const API_BASE = 'http://localhost:9005';
const STREAM_URL = `${API_BASE}/api/chat/stream`;
const JSON_URL = `${API_BASE}/api/chat`;

// Sayfa her yenilendiğinde yeni bir ID üretir (bellek sorununu çözer)
function generateSessionId(): string {
  const id = crypto.randomUUID();
  localStorage.setItem('valiz_session_id', id);
  return id;
}

// ─── Bileşen ──────────────────────────────────────────────────────────────────

export default function Hero() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(generateSessionId);
  const [selectedRegions, setSelectedRegions] = useState<string[]>([]);
  const [budget, setBudget] = useState(30000); // 💰 Bütçe Slider state
  // LLM mesaj geçmişi — backend'e backup olarak gönderilir
  const chatHistoryRef = useRef<Array<{ role: string; content: string }>>([]);

  // ▶ Senkron ID üretici
  const msgCounter = useRef(0);
  const nextId = useCallback(() => { msgCounter.current += 1; return msgCounter.current; }, []);

  // ▶ Double-submit kilidi
  const isSubmitting = useRef(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  // ── Bot mesajını ID ile güncelle ────────────────────────────────────────────
  const patchBot = useCallback((id: number, patch: Partial<Message>) => {
    setMessages(prev => prev.map(m => m.id === id ? { ...m, ...patch } : m));
  }, []);

  // ── Mesaj gönderici (hem UI mesajı hem API) ─────────────────────────────────
  const sendMessage = useCallback(async (text: string) => {
    if (!text.trim() || isSubmitting.current) return;
    isSubmitting.current = true;

    const sidSnap = sessionId;
    const userId = nextId();
    const botId = nextId();

    setMessages(prev => [
      ...prev,
      { id: userId, text, sender: 'user' },
      { id: botId, text: '', sender: 'bot', isStreaming: true },
    ]);
    setInputValue('');
    setIsLoading(true);
    setSelectedRegions([]);

    // History snapshot (yeni user mesajı EKLENMEMİŞ hali — backend kendisi ekler)
    const historySnap = [...chatHistoryRef.current];

    // SSE Streaming dene
    let streamOk = false;
    try {
      const res = await fetch(STREAM_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, session_id: sidSnap, history: historySnap }),
      });
      if (res.ok && res.body) {
        const reader = res.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        let full = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop() ?? '';

          for (const line of lines) {
            if (!line.startsWith('data: ')) continue;
            try {
              const p = JSON.parse(line.slice(6));
              if (p.type === 'text') {
                full += p.content;
                patchBot(botId, { text: full });
                bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
              } else if (p.type === 'done') {
                streamOk = true;
                patchBot(botId, {
                  isStreaming: false,
                  cards: p.cards || null,
                });
                if (p.session_id) {
                  setSessionId(p.session_id);
                  localStorage.setItem('valiz_session_id', p.session_id);
                }
                // History güncelle
                chatHistoryRef.current = [
                  ...historySnap,
                  { role: 'user', content: text },
                  { role: 'assistant', content: full },
                ];
              }
            } catch { /* JSON parse error, skip */ }
          }
        }
      }
    } catch { /* stream failed */ }

    // JSON fallback
    if (!streamOk) {
      try {
        const res = await fetch(JSON_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
          body: JSON.stringify({ message: text, session_id: sidSnap, history: historySnap }),
        });
        if (res.ok) {
          const data = await res.json();
          patchBot(botId, {
            text: (data.reply ?? 'Bağlantı hatası.').replace(/\*\*/g, ''),
            isStreaming: false,
            cards: data.cards || null,
          });
          if (data.session_id) {
            setSessionId(data.session_id);
            localStorage.setItem('valiz_session_id', data.session_id);
          }
          // History güncelle
          chatHistoryRef.current = [
            ...historySnap,
            { role: 'user', content: text },
            { role: 'assistant', content: data.reply || '' },
          ];
        } else {
          patchBot(botId, { text: 'Sunucuyla bağlantı kurulamadı.', isStreaming: false });
        }
      } catch {
        patchBot(botId, { text: 'Sunucuya ulaşılamıyor. 9000 portu açık mı?', isStreaming: false });
      }
    }

    isSubmitting.current = false;
    setIsLoading(false);
  }, [sessionId, nextId, patchBot]);

  // ── Sohbeti Sıfırla ─────────────────────────────────────────────────────────
  const resetChat = useCallback(() => {
    setMessages([]);
    setInputValue('');
    setIsLoading(false);
    const newSid = crypto.randomUUID();
    setSessionId(newSid);
    localStorage.setItem('valiz_session_id', newSid);
    chatHistoryRef.current = [];
    setSelectedRegions([]);
    msgCounter.current = 0;
    console.log("🧹 Sohbet sıfırlandı. Yeni session:", newSid);
  }, []);

  // ── Form submit ───────────────────────────────────────────────────────────────
  const handleSend = async (e: FormEvent) => {
    e.preventDefault();
    await sendMessage(inputValue.trim());
  };

  // ── Bölge seçim toggle ────────────────────────────────────────────────────────
  const toggleRegion = useCallback((name: string) => {
    setSelectedRegions(prev =>
      prev.includes(name) ? prev.filter(r => r !== name) : [...prev, name]
    );
  }, []);

  // ── Bölge seçimi onayla → mesaj olarak gönder ─────────────────────────────────
  const confirmRegions = useCallback(() => {
    if (selectedRegions.length === 0) return;

    // Kartı kilitle (tekrar seçim yapılmasın)
    setMessages(prev =>
      prev.map(m => m.cards?.type === 'region_cards' ? { ...m, cardsLocked: true } : m)
    );

    const text = selectedRegions.length === 1
      ? `${selectedRegions[0]} bölgesini seçiyorum`
      : `${selectedRegions.join(' ve ')} bölgelerini seçiyorum`;

    sendMessage(text);
  }, [selectedRegions, sendMessage]);

  // ── Otel seçimi → mesaj olarak gönder ──────────────────────────────────────────
  const selectHotel = useCallback((hotelName: string) => {
    // Kartı kilitle
    setMessages(prev =>
      prev.map(m => m.cards?.type === 'hotel_cards' ? { ...m, cardsLocked: true } : m)
    );
    sendMessage(`${hotelName} hakkında daha fazla bilgi istiyorum`);
  }, [sendMessage]);

  // ── Generative UI: Bot bütçe mi soruyor? ──────────────────────────────────────
  const lastMessage = messages[messages.length - 1];
  const isAskingBudget =
    lastMessage?.sender === 'bot' &&
    !lastMessage.isStreaming &&
    !lastMessage.cards && // 🚀 KARTLAR VARSA ARTIK SORU DEĞİL SONUÇTUR
    (lastMessage.text.toLowerCase().includes('bütç') ||
      lastMessage.text.toLowerCase().includes('fiyat') ||
      lastMessage.text.toLowerCase().includes('ne kadar')) &&
    lastMessage.text.includes('?'); // 🚀 KATİ KURAL: SORU İŞARETİ OLMAK ZORUNDA

  const handleBudgetSubmit = () => {
    sendMessage(`Maksimum bütçem ${budget.toLocaleString('tr-TR')} TL civarı.`);
  };

  // ─── JSX ─────────────────────────────────────────────────────────────────────

  return (
    <section>
      {/* Navbar'ın sağında görünecek sabit buton (State bozulmaması için Hero içinde tutuldu) */}
      <button
        onClick={resetChat}
        className='fixed top-3 right-6 z-[60] text-xs bg-gray-800 hover:bg-gray-700 text-gray-300 px-4 py-2 rounded-full transition-all border border-gray-700 flex items-center gap-2 shadow-xl'
      >
        <span className='hidden sm:inline'>YENİ SOHBET</span>
      </button>

      <div className='fixed inset-0 -z-10'
        style={{ backgroundImage: `url(${bgImage})`, backgroundSize: 'cover', backgroundPosition: 'top center' }}
      />

      <div className='absolute top-[52%] left-1/2 -translate-x-1/2 -translate-y-1/2 z-10
                      w-[95%] max-w-6xl h-[80vh] bg-black/70 backdrop-blur-md
                      rounded-lg shadow-2xl flex flex-col'>

        {/* Mesaj listesi */}
        <div className='flex-grow overflow-y-auto p-4 space-y-3
                        [&::-webkit-scrollbar]:w-1.5
                        [&::-webkit-scrollbar-thumb]:rounded-full
                        [&::-webkit-scrollbar-thumb]:bg-amber-500'>
          {messages.map(m => (
            <div key={m.id}>
              {/* Mesaj baloncuğu */}
              <div className={`flex ${m.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-sm lg:max-w-xl px-4 py-2 rounded-lg shadow text-sm
                                ${m.sender === 'user' ? 'bg-amber-500 text-white' : 'bg-gray-700 text-gray-200'}`}>
                  <pre className='whitespace-pre-wrap font-sans'>
                    {m.text}
                    {m.isStreaming && (
                      <span className='inline-block w-[7px] h-[14px] bg-amber-400 ml-0.5 align-middle animate-pulse rounded-sm' />
                    )}
                  </pre>
                </div>
              </div>

              {/* Kartlar — mesajın altında */}
              {m.cards && !m.isStreaming && (
                <div className={`flex ${m.sender === 'user' ? 'justify-end' : 'justify-start'} mt-2`}>
                  {m.cards.type === 'region_cards' && m.cards.regions && m.cards.regions.length > 0 && (
                    <RegionCard
                      regions={m.cards.regions}
                      selectedRegions={selectedRegions}
                      onToggle={toggleRegion}
                      onConfirm={confirmRegions}
                      disabled={m.cardsLocked || isLoading}
                    />
                  )}
                  {m.cards.type === 'hotel_cards' && m.cards.hotels && m.cards.hotels.length > 0 && (
                    <HotelCard
                      hotels={m.cards.hotels}
                      onSelect={selectHotel}
                      disabled={m.cardsLocked || isLoading}
                    />
                  )}
                  {/* 💎 Premium Otel Detay Kartı */}
                  {m.cards.type === 'hotel_detail' && m.cards.hotel && (
                    <div className='w-full max-w-2xl bg-gray-900/90 border border-amber-500/30 rounded-xl p-5 shadow-2xl backdrop-blur-xl'>
                      <div className='flex justify-between items-start mb-4'>
                        <div>
                          <h3 className='text-xl font-bold text-white mb-1'>{(m.cards.hotel as any).name}</h3>
                          <div className='flex items-center gap-2 text-xs text-amber-500/80'>
                            {'★'.repeat(Number((m.cards.hotel as any).stars || 0))}
                            <span className='text-gray-400'>|</span>
                            <span>{(m.cards.hotel as any).city}</span>
                          </div>
                        </div>
                        <div className='text-right'>
                          <span className='text-sm text-gray-500 block underline'>Toplam Tutar</span>
                          <span className='text-xl font-black text-amber-400'>
                            {Number((m.cards.hotel as any).price || 0).toLocaleString('tr-TR')} TL
                          </span>
                          {/* 🚨 Fiyat Dökümü (YENI) */}
                          {(m.cards.hotel as any).nightly_price > 0 && (
                            <span className='text-[10px] text-gray-500 block leading-tight mt-1'>
                              {(m.cards.hotel as any).room_count} Oda x {(m.cards.hotel as any).duration} Gece<br />
                              ({(m.cards.hotel as any).room_type})
                            </span>
                          )}
                        </div>
                      </div>

                      <div className='bg-black/40 rounded-lg p-4 text-sm text-gray-300 leading-relaxed border border-gray-800'>
                        {((m.cards.hotel as any).description as string || '').split('. ').map((sentence, sIdx) => {
                          if (sentence.includes('📍 Konum Detayı:')) {
                            return <div key={sIdx} className='mt-3 pt-3 border-t border-gray-800 text-amber-200'>
                              <strong>📍 {sentence.replace('📍 Konum Detayı:', '').trim()}</strong>
                            </div>
                          }
                          if (sentence.includes('✨ Sunulan İmkanlar:')) {
                            return <div key={sIdx} className='mt-3 bg-amber-500/10 p-3 rounded-md border border-amber-500/20'>
                              <div className='text-amber-500 font-bold mb-1 flex items-center gap-2'>
                                <span>✨</span> Sunulan İmkanlar
                              </div>
                              <div className='text-gray-200 text-xs italic'>
                                {sentence.replace('✨ Sunulan İmkanlar:', '').trim()}
                              </div>
                            </div>
                          }
                          return <span key={sIdx}>{sentence}. </span>
                        })}
                      </div>

                      <button
                        onClick={() => sendMessage(`Rezervasyon adımlarına geçmek istiyorum: ${(m.cards?.hotel as any)?.name}`)}
                        className='mt-5 w-full py-3 bg-amber-500 hover:bg-amber-600 text-white font-bold rounded-lg transition-all shadow-lg shadow-amber-500/30'
                      >
                        Rezervasyon İçin Devam Et
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        {/* 🧠 AKILLI BİLEŞEN: Bütçe Slider (Sadece bot bütçe sorduğunda görünür) */}
        {isAskingBudget && (
          <div className='p-4 bg-gray-900/90 border-t border-gray-700 flex flex-col gap-3 flex-shrink-0 animate-fade-in'>
            <div className='flex justify-between items-center'>
              <span className='text-gray-300 text-sm'>Maksimum Bütçenizi Belirleyin:</span>
              <span className='text-amber-400 font-bold text-lg'>{budget.toLocaleString('tr-TR')} TL</span>
            </div>

            <input
              type='range'
              min='5000'
              max='150000'
              step='5000'
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className='w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-500'
            />

            <div className='flex justify-between text-xs text-gray-500 mb-1'>
              <span>5.000 TL</span>
              <span>150.000+ TL</span>
            </div>

            <button
              onClick={handleBudgetSubmit}
              disabled={isLoading}
              className='w-full py-2.5 bg-amber-500 text-white rounded-lg font-semibold hover:bg-amber-600 transition-colors text-sm shadow-lg shadow-amber-500/20 disabled:opacity-50'
            >
              {budget.toLocaleString('tr-TR')} TL İle Onayla ve Otelleri Gör
            </button>
          </div>
        )}

        {/* Input */}
        <form onSubmit={handleSend} className='p-4 flex-shrink-0'>
          <div className='flex rounded-lg bg-black/50'>
            <input
              type='text' value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              placeholder={isLoading ? 'Yanıt bekleniyor...' : 'Mesajınızı yazın...'}
              disabled={isLoading}
              className='flex-grow px-4 py-3 bg-transparent text-white text-sm
                         focus:outline-none rounded-l-lg disabled:opacity-50'
            />
            <button type='submit' disabled={isLoading}
              className='bg-amber-500 text-white px-6 py-3 rounded-r-lg font-semibold
                         hover:bg-amber-600 disabled:bg-gray-600 disabled:cursor-not-allowed
                         transition-colors'>
              {isLoading ? '...' : 'Gönder'}
            </button>
          </div>
        </form>
      </div>
    </section>
  );
}
