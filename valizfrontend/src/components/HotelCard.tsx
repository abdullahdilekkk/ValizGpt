import type { FC } from 'react';

interface Hotel {
  name: string;
  city: string;
  district?: string;
  region?: string;
  price: number;
  stars?: number;
  segment?: string;
  concept?: string;
  vibe?: string;
  room_type?: string;
  nightly_price?: number;
  room_count?: number;
  duration?: number;
  description?: string;
}

interface HotelCardProps {
  hotels: Hotel[];
  onSelect: (hotelName: string) => void;
  disabled?: boolean;
}

function renderStars(count: number) {
  return '★'.repeat(Math.min(count, 5)) + '☆'.repeat(Math.max(0, 5 - count));
}

const HotelCard: FC<HotelCardProps> = ({ hotels, onSelect, disabled = false }) => {
  return (
    <div className='w-full max-w-lg'>
      <div className='grid gap-2.5'>
        {hotels.map((h, idx) => (
          <button
            key={h.name}
            onClick={() => !disabled && onSelect(h.name)}
            disabled={disabled}
            className={`
              text-left px-4 py-3 rounded-lg border border-gray-600
              bg-gray-800/60 hover:border-amber-500/60 hover:bg-gray-700/60
              transition-all duration-200 group
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {/* Header */}
            <div className='flex items-start justify-between mb-1'>
              <div>
                <span className='text-amber-400 text-xs font-bold mr-1.5'>{idx + 1}.</span>
                <span className='font-semibold text-sm text-white group-hover:text-amber-300 transition-colors'>
                  {h.name}
                </span>
              </div>
              {h.stars != null && h.stars > 0 && (
                <span className='text-amber-400 text-xs tracking-wider ml-2 flex-shrink-0'>
                  {renderStars(h.stars)}
                </span>
              )}
            </div>

            {/* Location */}
            <div className='text-xs text-gray-400 mb-1.5'>
              📍 {h.district ? `${h.district}, ` : ''}{h.city}
              {h.region ? ` · ${h.region}` : ''}
            </div>

            {/* Tags */}
            <div className='flex flex-wrap gap-1.5 mb-2'>
              {h.concept && (
                <span className='text-[10px] px-2 py-0.5 rounded-full bg-blue-500/20 text-blue-300 border border-blue-500/30'>
                  {h.concept}
                </span>
              )}
              {h.vibe && (
                <span className='text-[10px] px-2 py-0.5 rounded-full bg-purple-500/20 text-purple-300 border border-purple-500/30'>
                  {h.vibe}
                </span>
              )}
              {h.segment && (
                <span className='text-[10px] px-2 py-0.5 rounded-full bg-green-500/20 text-green-300 border border-green-500/30'>
                  {h.segment}
                </span>
              )}
              {h.room_type && (
                <span className='text-[10px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30 font-bold'>
                  🏨 {h.room_type}
                </span>
              )}
            </div>

            {/* Description */}
            {h.description && (
              <p className='text-xs text-gray-400 mb-2 line-clamp-2'>{h.description}</p>
            )}

            {/* Price */}
            <div className='flex items-center justify-between'>
              <div className='text-right'>
                <span className='text-amber-400 font-bold text-sm block'>
                  {h.price > 0 ? `${h.price.toLocaleString('tr-TR')} TL (Toplam Tutar)` : 'Fiyat bilgisi yok'}
                </span>
                {h.price > 0 && h.room_count && h.duration && h.nightly_price && (
                  <span className='text-[9px] text-gray-500 block leading-tight'>
                    {h.room_count} Oda x {h.duration} Gece (Oda başı: {h.nightly_price.toLocaleString('tr-TR')} TL)
                  </span>
                )}
              </div>
              <span className='text-[10px] text-gray-500 group-hover:text-amber-400/60 transition-colors'>
                Detay için tıkla →
              </span>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

export default HotelCard;
