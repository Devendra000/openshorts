import { useState } from 'react';
import { Scissors, Wand2 } from 'lucide-react';
import SegmentedControl from '../ui/SegmentedControl';
import { findChunks, placeChunks, PLACE_MODES } from '../../lib/autoChunks';
import { PEAKS_PER_SECOND } from './audioEngine';
import { fmtTime } from '../../lib/audioPlacements';

// 1) cut the audio into chunks at its pauses, 2) place them across the video.
export default function AutoChunker({ energy, videoDuration, marks, chunks, onChunks, onPlace }) {
    const [count, setCount] = useState(0); // 0 = one chunk per pause
    const [sensitivity, setSensitivity] = useState(0.5);
    const [mode, setMode] = useState('spread');
    const [found, setFound] = useState(null);

    const make = () => {
        const res = findChunks(energy, PEAKS_PER_SECOND, { count, sensitivity });
        setFound(res.found);
        onChunks(res.chunks);
    };
    const needMarks = mode === 'marks' && !marks.length;
    const place = () => {
        if (!chunks.length || needMarks) return;
        onPlace(placeChunks(chunks, videoDuration, mode, marks));
    };

    return (
        <div className="rounded-input border border-rule2 p-3 sm:p-4 space-y-3">
            <p className="eyebrow">Auto · chunk maker</p>

            <div className="flex flex-wrap items-end gap-3">
                <label className="text-xs text-muted">
                    <span className="block mb-1">chunks</span>
                    <div className="flex items-center gap-1">
                        <button type="button" className="btn-ghost py-1 px-2 text-xs" onClick={() => setCount((c) => Math.max(0, c - 1))}>−</button>
                        <span className="w-12 text-center font-mono text-ink">{count || 'auto'}</span>
                        <button type="button" className="btn-ghost py-1 px-2 text-xs" onClick={() => setCount((c) => Math.min(40, c + 1))}>+</button>
                    </div>
                </label>
                <label className="text-xs text-muted flex-1 min-w-[160px]">
                    <span className="block mb-1">pause sensitivity</span>
                    <input type="range" min="0" max="1" step="0.05" value={sensitivity}
                        onChange={(e) => setSensitivity(parseFloat(e.target.value))} className="w-full accent-[var(--color-accent)]" />
                </label>
                <button type="button" className="btn-primary py-1.5 px-3 text-xs" disabled={!energy} onClick={make}>
                    <Scissors size={12} /> make chunks
                </button>
            </div>

            {chunks.length > 0 && (
                <p className="text-xs text-muted font-mono leading-relaxed">
                    {chunks.length} chunk{chunks.length > 1 ? 's' : ''}
                    {found !== null && count > 0 && found < count - 1 && ` (only ${found} clear pauses; the rest cut at the quietest moment)`}:{' '}
                    {chunks.map((c, i) => `${i + 1}) ${fmtTime(c.aStart)}–${fmtTime(c.aEnd)}`).join('  ')}
                </p>
            )}

            <div>
                <span className="block text-xs text-muted mb-1">place on the video</span>
                <SegmentedControl size="sm" columns={3} options={PLACE_MODES} value={mode} onChange={setMode} />
                <p className="text-xs text-muted mt-1.5">
                    {mode === 'spread' && `The video cut into ${chunks.length || 'N'} equal parts, one chunk at the start of each.`}
                    {mode === 'rhythm' && 'Your audio\'s own timing, stretched so the last chunk ends with the video.'}
                    {mode === 'marks' && (marks.length
                        ? `Chunk 1 at your first mark, chunk 2 at the second… (${marks.length} mark${marks.length > 1 ? 's' : ''}).`
                        : 'Drop marks on the timeline first: move the playhead and press "mark" (or M).')}
                </p>
            </div>

            <button type="button" className="btn-primary w-full py-2 text-sm" disabled={!chunks.length || needMarks || !videoDuration} onClick={place}>
                <Wand2 size={14} /> place {chunks.length || ''} chunks on the video
            </button>
            {chunks.length > 0 && (
                <p className="text-xs text-muted">Replaces what is on the timeline. Then drag any chunk to fine-tune.</p>
            )}
        </div>
    );
}
