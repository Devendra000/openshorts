import { useEffect, useRef, useState } from 'react';
import { ZoomIn, ZoomOut, Scissors, Trash2, Flag } from 'lucide-react';
import Waveform from './Waveform';
import { fmtTime } from '../../lib/audioPlacements';

const MIN_PIECE = 0.1;    // seconds
const SNAP_PX = 8;        // magnet distance to playhead, 0, video end, other blocks
const TICK_STEPS = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];

const round2 = (t) => Math.round(t * 100) / 100;

// The video's timeline with the placed audio chunks as blocks: drag a block
// to move it, drag its edges to trim, click the ruler to move the playhead.
// Keys while focused: S split at playhead, Delete remove, arrows nudge,
// M drops a mark (where the chunk maker can place chunks).
export default function Timeline({ pieces, onChange, selectedId, onSelect, videoDuration, audioDuration, playhead, onSeek, peaks, marks = [], onMarksChange }) {
    const scroller = useRef(null);
    const [width, setWidth] = useState(800);
    const [zoom, setZoom] = useState(1);
    const [frozenTotal, setFrozenTotal] = useState(null);
    const drag = useRef(null);
    const seeking = useRef(false);

    useEffect(() => {
        const el = scroller.current;
        if (!el) return undefined;
        const ro = new ResizeObserver(() => setWidth(el.clientWidth));
        ro.observe(el);
        setWidth(el.clientWidth);
        return () => ro.disconnect();
    }, []);

    const lastEnd = pieces.reduce((m, p) => Math.max(m, p.vStart + p.aEnd - p.aStart), 0);
    const total = frozenTotal ?? Math.max(videoDuration || 0, lastEnd, 5) * 1.08;
    const inner = width * zoom;
    const pps = inner / total;
    const step = TICK_STEPS.find((s) => s * pps >= 64) || 600;
    const ticks = [];
    for (let t = 0; t <= total; t += step) ticks.push(t);

    const selected = pieces.find((p) => p.id === selectedId);
    const canSplit = selected && playhead > selected.vStart + MIN_PIECE
        && playhead < selected.vStart + (selected.aEnd - selected.aStart) - MIN_PIECE;

    const update = (id, patch) => onChange(pieces.map((p) => (p.id === id ? { ...p, ...patch } : p)));
    const remove = (id) => { onChange(pieces.filter((p) => p.id !== id)); onSelect(null); };
    const split = () => {
        if (!canSplit) return;
        const cut = round2(selected.aStart + (playhead - selected.vStart));
        const right = { id: `${selected.id}b${Date.now()}`, aStart: cut, aEnd: selected.aEnd, vStart: round2(playhead) };
        onChange(pieces.flatMap((p) => (p.id === selected.id ? [{ ...p, aEnd: cut }, right] : [p])));
        onSelect(right.id);
    };

    // Magnet: the nearest edge within SNAP_PX, else the time itself.
    const snap = (t, ownId) => {
        const edges = [0, playhead, videoDuration || 0, ...marks];
        pieces.forEach((p) => {
            if (p.id !== ownId) edges.push(p.vStart, p.vStart + p.aEnd - p.aStart);
        });
        let best = t;
        let bestPx = SNAP_PX;
        edges.forEach((e) => {
            const d = Math.abs(e - t) * pps;
            if (d < bestPx) { bestPx = d; best = e; }
        });
        return best;
    };

    const startDrag = (e, piece, mode) => {
        e.stopPropagation();
        e.currentTarget.setPointerCapture(e.pointerId);
        onSelect(piece.id);
        setFrozenTotal(total);
        drag.current = { mode, x: e.clientX, orig: piece, pps };
    };
    const moveDrag = (e) => {
        const d = drag.current;
        if (!d) return;
        const ds = (e.clientX - d.x) / d.pps;
        const o = d.orig;
        const len = o.aEnd - o.aStart;
        if (d.mode === 'move') {
            let start = Math.max(0, o.vStart + ds);
            const snappedStart = snap(start, o.id);
            const snappedEnd = snap(start + len, o.id);
            if (snappedStart !== start) start = snappedStart;
            else if (snappedEnd !== start + len) start = snappedEnd - len;
            update(o.id, { vStart: round2(Math.max(0, start)) });
        } else if (d.mode === 'left') {
            // Trim the head: the audio and its place move together.
            let delta = Math.min(len - MIN_PIECE, Math.max(-o.aStart, -o.vStart, ds));
            const snapped = snap(o.vStart + delta, o.id);
            if (snapped !== o.vStart + delta) {
                delta = Math.min(len - MIN_PIECE, Math.max(-o.aStart, -o.vStart, snapped - o.vStart));
            }
            update(o.id, { aStart: round2(o.aStart + delta), vStart: round2(o.vStart + delta) });
        } else {
            let end = o.vStart + Math.min(audioDuration - o.aStart, Math.max(MIN_PIECE, len + ds));
            end = snap(end, o.id);
            const newLen = Math.min(audioDuration - o.aStart, Math.max(MIN_PIECE, end - o.vStart));
            update(o.id, { aEnd: round2(o.aStart + newLen) });
        }
    };
    const endDrag = () => {
        drag.current = null;
        setFrozenTotal(null);
    };

    const timeAt = (clientX) => {
        const r = scroller.current.getBoundingClientRect();
        return Math.max(0, (clientX - r.left + scroller.current.scrollLeft) / pps);
    };
    const seekDown = (e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        seeking.current = true;
        onSelect(null);
        onSeek(Math.min(videoDuration || 0, timeAt(e.clientX)));
    };
    const seekMove = (e) => {
        if (seeking.current) onSeek(Math.min(videoDuration || 0, timeAt(e.clientX)));
    };

    const addMark = () => {
        const t = round2(playhead);
        if (!marks.some((m) => Math.abs(m - t) < 0.05)) onMarksChange([...marks, t].sort((a, b) => a - b));
    };

    const onKeyDown = (e) => {
        if (e.key === 'm' || e.key === 'M') { e.preventDefault(); addMark(); return; }
        if (!selected) return;
        if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); remove(selected.id); }
        else if (e.key === 's' || e.key === 'S') { e.preventDefault(); split(); }
        else if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
            e.preventDefault();
            const by = (e.shiftKey ? 1 : 0.1) * (e.key === 'ArrowLeft' ? -1 : 1);
            update(selected.id, { vStart: round2(Math.max(0, selected.vStart + by)) });
        }
    };

    return (
        <div>
            <div className="flex items-center justify-between mb-2 gap-2 flex-wrap">
                <p className="eyebrow">Video timeline · drag chunks to move, edges to trim</p>
                <div className="flex items-center gap-1.5">
                    <button type="button" className="btn-ghost py-1 px-2 text-xs" disabled={!videoDuration} onClick={addMark} title="Mark the playhead (M)">
                        <Flag size={12} /> mark
                    </button>
                    {marks.length > 0 && (
                        <button type="button" className="btn-ghost py-1 px-2 text-xs" onClick={() => onMarksChange([])} title="Remove all marks">
                            clear {marks.length}
                        </button>
                    )}
                    <button type="button" className="btn-ghost py-1 px-2 text-xs" disabled={!canSplit} onClick={split} title="Split at playhead (S)">
                        <Scissors size={12} /> split
                    </button>
                    <button type="button" className="btn-ghost py-1 px-2 text-xs" disabled={!selected} onClick={() => selected && remove(selected.id)} title="Remove (Delete)">
                        <Trash2 size={12} />
                    </button>
                    <button type="button" className="btn-ghost py-1 px-2 text-xs" disabled={zoom <= 1} onClick={() => setZoom((z) => Math.max(1, z / 1.5))} title="Zoom out">
                        <ZoomOut size={12} />
                    </button>
                    <button type="button" className="btn-ghost py-1 px-2 text-xs" disabled={zoom >= 30} onClick={() => setZoom((z) => Math.min(30, z * 1.5))} title="Zoom in">
                        <ZoomIn size={12} />
                    </button>
                </div>
            </div>

            <div
                ref={scroller}
                tabIndex={0}
                onKeyDown={onKeyDown}
                className="relative overflow-x-auto custom-scrollbar rounded-input border border-rule2 bg-paper outline-none focus:border-brass select-none"
            >
                <div className="relative" style={{ width: inner }}>
                    {/* Ruler */}
                    <div className="relative h-6 border-b border-rule cursor-pointer touch-none"
                        onPointerDown={seekDown} onPointerMove={seekMove} onPointerUp={() => { seeking.current = false; }}>
                        {ticks.map((t) => (
                            <div key={t} className="absolute top-0 h-full border-l border-rule2 pl-1 text-[10px] text-muted font-mono"
                                style={{ left: t * pps }}>
                                {fmtTime(t)}
                            </div>
                        ))}
                    </div>

                    {/* Video track */}
                    <div className="relative h-8 touch-none" onPointerDown={seekDown} onPointerMove={seekMove}
                        onPointerUp={() => { seeking.current = false; }}>
                        <div className="absolute inset-y-1 left-0 rounded bg-paper3 border border-rule2 px-2 flex items-center text-[10px] text-muted font-mono overflow-hidden whitespace-nowrap"
                            style={{ width: Math.max(4, (videoDuration || 0) * pps) }}>
                            video · {fmtTime(videoDuration)}
                        </div>
                    </div>

                    {/* Audio chunks track */}
                    <div className="relative h-16 touch-none" onPointerDown={seekDown} onPointerMove={seekMove}
                        onPointerUp={() => { seeking.current = false; }}>
                        {pieces.length === 0 && (
                            <div className="absolute inset-0 flex items-center px-3 text-xs text-muted pointer-events-none">
                                Select a chunk in the audio above and press "add chunk", or paste a placement map below.
                            </div>
                        )}
                        {pieces.map((p) => {
                            const len = p.aEnd - p.aStart;
                            const isSel = p.id === selectedId;
                            return (
                                <div
                                    key={p.id}
                                    className={`absolute inset-y-1 rounded border overflow-hidden cursor-grab active:cursor-grabbing ${isSel ? 'border-brass bg-brass/25 z-10' : 'border-brass/40 bg-brass/10'}`}
                                    style={{ left: p.vStart * pps, width: Math.max(6, len * pps) }}
                                    onPointerDown={(e) => startDrag(e, p, 'move')}
                                    onPointerMove={moveDrag}
                                    onPointerUp={endDrag}
                                    title={`audio ${fmtTime(p.aStart)}–${fmtTime(p.aEnd)} at video ${fmtTime(p.vStart)}`}
                                >
                                    <div className="absolute inset-0 pointer-events-none">
                                        <Waveform peaks={peaks} from={p.aStart} to={p.aEnd} className="text-brass opacity-60" />
                                    </div>
                                    <span className="absolute top-0.5 left-2 text-[10px] text-ink font-mono pointer-events-none whitespace-nowrap">
                                        {fmtTime(p.vStart)}
                                    </span>
                                    <div className="absolute inset-y-0 left-0 w-2 cursor-ew-resize hover:bg-brass/50"
                                        onPointerDown={(e) => startDrag(e, p, 'left')} onPointerMove={moveDrag} onPointerUp={endDrag} />
                                    <div className="absolute inset-y-0 right-0 w-2 cursor-ew-resize hover:bg-brass/50"
                                        onPointerDown={(e) => startDrag(e, p, 'right')} onPointerMove={moveDrag} onPointerUp={endDrag} />
                                </div>
                            );
                        })}
                    </div>

                    {/* Past the video's end: cut in the render */}
                    {videoDuration > 0 && (
                        <div className="absolute top-6 bottom-0 right-0 bg-ink/35 pointer-events-none"
                            style={{ left: videoDuration * pps }} title="Past the video's end: cut" />
                    )}

                    {/* Marks: click a flag to remove it */}
                    {marks.map((m, i) => (
                        <div key={m} className="absolute top-0 bottom-0 z-10 pointer-events-none" style={{ left: m * pps }}>
                            <div className="absolute top-6 bottom-0 border-l border-dashed border-brass/70" />
                            <button
                                type="button"
                                className="pointer-events-auto absolute top-0 -translate-x-1/2 px-1 h-4 rounded-sm bg-brass text-[9px] leading-4 font-mono text-paper"
                                onPointerDown={(e) => e.stopPropagation()}
                                onClick={() => onMarksChange(marks.filter((x) => x !== m))}
                                title={`Mark ${i + 1} at ${fmtTime(m)}: click to remove`}
                            >
                                {i + 1}
                            </button>
                        </div>
                    ))}

                    {/* Playhead */}
                    <div className="absolute top-0 bottom-0 w-px bg-ink pointer-events-none z-20" style={{ left: playhead * pps }}>
                        <div className="w-2 h-2 -ml-[3.5px] rotate-45 bg-ink" />
                    </div>
                </div>
            </div>
            <p className="text-xs text-muted mt-1.5">
                Click the timeline to move the playhead. With the timeline focused: M marks the playhead, S splits the selected chunk at it, Delete removes the chunk, ← → nudge (Shift: 1 s).
            </p>
        </div>
    );
}
