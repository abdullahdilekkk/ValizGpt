import type { FC } from 'react';

interface Region {
  name: string;
  region?: string;
  hotel_count: number;
  price_range: string;
  min_price?: number;
  max_price?: number;
}

interface RegionCardProps {
  regions: Region[];
  selectedRegions: string[];
  onToggle: (regionName: string) => void;
  onConfirm: () => void;
  disabled?: boolean;
}

const RegionCard: FC<RegionCardProps> = ({
  regions,
  selectedRegions,
  onToggle,
  onConfirm,
  disabled = false,
}) => {
  return (
    <div className='w-full max-w-lg'>
      <div className='grid gap-2'>
        {regions.map((r) => {
          const isSelected = selectedRegions.includes(r.name);
          return (
            <button
              key={r.name}
              onClick={() => !disabled && onToggle(r.name)}
              disabled={disabled}
              className={`
                flex items-center justify-between px-4 py-3 rounded-lg
                border transition-all duration-200 text-left
                ${isSelected
                  ? 'border-amber-500 bg-amber-500/20 text-white shadow-lg shadow-amber-500/10'
                  : 'border-gray-600 bg-gray-800/60 text-gray-300 hover:border-gray-400 hover:bg-gray-700/60'}
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              <div className='flex-1'>
                <div className='flex items-center gap-2'>
                  {/* Checkbox */}
                  <div className={`
                    w-5 h-5 rounded border-2 flex items-center justify-center transition-colors
                    ${isSelected ? 'border-amber-500 bg-amber-500' : 'border-gray-500'}
                  `}>
                    {isSelected && (
                      <svg className='w-3 h-3 text-white' fill='none' viewBox='0 0 24 24' stroke='currentColor' strokeWidth={3}>
                        <path strokeLinecap='round' strokeLinejoin='round' d='M5 13l4 4L19 7' />
                      </svg>
                    )}
                  </div>
                  <div>
                    <span className='font-semibold text-sm'>{r.name}</span>
                    {r.region && <span className='text-xs text-gray-400 ml-1.5'>({r.region})</span>}
                  </div>
                </div>
              </div>
              <div className='text-right ml-4'>
                <div className='text-xs text-amber-400'>{r.hotel_count} otel</div>
                <div className='text-xs text-gray-400'>{r.price_range}</div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Devam Et butonu */}
      {selectedRegions.length > 0 && !disabled && (
        <button
          onClick={onConfirm}
          className='mt-3 w-full py-2.5 rounded-lg bg-amber-500 text-white font-semibold
                     hover:bg-amber-600 transition-colors shadow-lg shadow-amber-500/20
                     text-sm'
        >
          {selectedRegions.length === 1
            ? `${selectedRegions[0]} ile devam et`
            : `${selectedRegions.length} bölge seçildi — devam et`}
        </button>
      )}
    </div>
  );
};

export default RegionCard;
