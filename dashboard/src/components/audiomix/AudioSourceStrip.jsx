import { useRef, useState, useEffect } from 'react';
import { Play, Square, Plus } from 'lucide-react';
import Waveform from './Waveform';
import { fmtTime } from '../../lib/audioPlacements';

// The whole uploaded track as a waveform. Drag across it to select a chunk,
// click to move the audio playhead; chunks already placed are marked.
export default function AudioSourceStrip({ peaks, duration, audioRef, pieces, proposed = [], selection, onSelection, onAdd, videoTime }) {
    const box = useRef(null);
    const drag = useRef(null);
    const [audioTime, setAudioTime] = useState(0);
    const [playingSel, setPlayingSel] = useState(false);

    // Follow the audio element (and stop at the end of a played selection).
    useEffect(() => {
        let raf = 0;
        const tick = () => {
            const a = audioRef.current;
            if (a) {
                setAudioTime(a.currentTime);
                if (playingSel && selection && a.currentTime >= selection.a1) {
                    a.pause();
                    setPlayingSel(false);
                }
            }
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [audioRef, playingSel, selection]);

    const timeAt = (clientX) => {
        const r = box.current.getBoundingClientRect();
        return Math.min(duration, Math.max(0, ((clientX - r.left) / r.width) * duration));
    };

    const onPointerDown = (e) => {
        if (!duration) return;
        e.currentTarget.setPointerCapture(e.pointerId);
        const t = timeAt(e.clientX);
        drag.current = { start: t, moved: false };
    };
    const onPointerMove = (e) => {
        if (!drag.current) return;
        const t = timeAt(e.clientX);
        if (Math.abs(t - drag.current.start) * (box.current.clientWidth / duration) > 3) drag.current.moved = true;
        if (drag.current.moved) {
            onSelection({ a0: Math.min(t, drag.current.start), a1: Math.max(t, drag.current.start) });
        }
    };
    const onPointerUp = (e) => {
        if (!drag.current) return;
        if (!drag.current.moved && audioRef.current) {
            audioRef.current.currentTime = timeAt(e.clientX);
            onSelection(null);
        }
        drag.current = null;
    };

    const playSelection = () => {
        const a = audioRef.current;
        if (!a || !selection) return;
        if (playingSel) { a.pause(); setPlayingSel(false); return; }
        a.currentTime = selection.a0;
        a.play();
        setPlayingSel(true);
    };

    const pct = (t) => `${(t / (duration || 1)) * 100}%`;
    const selLen = selection ? selection.a1 - selection.a0 : 0;

    return (
        <div>
            <div className="flex items-baseline justify-between mb-2 gap-2 flex-wrap">
                <p className="eyebrow">Audio · drag to select a chunk</p>
                <span className="text-xs text-muted font-mono">
                    {selection ? `${fmtTime(selection.a0)}–${fmtTime(selection.a1)} (${selLen.toFixed(1)}s)` : `${fmtTime(audioTime)} / ${fmtTime(duration)}`}
                </span>
            </div>
            <div
                ref={box}
                className="relative h-16 rounded-input bg-paper border border-rule2 overflow-hidden cursor-crosshair select-none touch-none"
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
            >
                <Waveform peaks={peaks} from={0} to={duration} className="text-muted opacity-70" />
                {pieces.map((p) => (
                    <div key={p.id} className="absolute bottom-0 h-1 bg-brass/70 pointer-events-none"
                        style={{ left: pct(p.aStart), width: pct(p.aEnd - p.aStart) }} />
                ))}
                {proposed.map((c, i) => (
                    <div key={`${c.aStart}-${i}`} className="absolute inset-y-0 border border-brass/80 bg-brass/10 pointer-events-none"
                        style={{ left: pct(c.aStart), width: pct(c.aEnd - c.aStart) }}>
                        <span className="absolute top-0.5 left-1 text-[9px] font-mono text-ink">{i + 1}</span>
                    </div>
                ))}
                {selection && (
                    <div className="absolute inset-y-0 bg-brass/25 border-x-2 border-brass pointer-events-none"
                        style={{ left: pct(selection.a0), width: pct(selLen) }} />
                )}
                <div className="absolute inset-y-0 w-px bg-ink pointer-events-none" style={{ left: pct(audioTime) }} />
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
                <button type="button" className="btn-ghost py-1.5 px-3 text-xs" disabled={!selection} onClick={playSelection}>
                    {playingSel ? <Square size={12} /> : <Play size={12} />} {playingSel ? 'stop' : 'play chunk'}
                </button>
                <button type="button" className="btn-primary py-1.5 px-3 text-xs" disabled={!selection || selLen < 0.05}
                    onClick={() => onAdd(selection.a0, selection.a1)}>
                    <Plus size={12} /> add chunk at {fmtTime(videoTime)}
                </button>
                <button type="button" className="btn-ghost py-1.5 px-3 text-xs" disabled={!duration}
                    onClick={() => onAdd(0, duration)}>
                    <Plus size={12} /> add whole audio at {fmtTime(videoTime)}
                </button>
            </div>
        </div>
    );
}
