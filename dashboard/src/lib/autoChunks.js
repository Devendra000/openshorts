// Chunk maker and auto placement for the Audio Mix editor. Pure functions
// over a loudness curve (RMS per frame, normalised 0..1), so they run on the
// decoded audio in the browser with no server round-trip.

const round2 = (t) => Math.round(t * 100) / 100;

// Runs of quiet frames inside the speech: [{ start, end }] in seconds.
function quietGaps(energy, perSec, threshold, minGap) {
    let first = 0;
    let last = energy.length - 1;
    while (first < energy.length && energy[first] < threshold) first++;
    while (last > first && energy[last] < threshold) last--;
    const gaps = [];
    let runStart = -1;
    for (let i = first; i <= last; i++) {
        if (energy[i] < threshold) {
            if (runStart < 0) runStart = i;
        } else if (runStart >= 0) {
            if ((i - runStart) / perSec >= minGap) gaps.push({ start: runStart / perSec, end: i / perSec });
            runStart = -1;
        }
    }
    return { gaps, speechStart: first / perSec, speechEnd: (last + 1) / perSec };
}

// The quietest frame between a and b seconds (a fallback cut point).
function quietestAt(energy, perSec, a, b) {
    let best = Math.round((a + b) / 2 * perSec);
    let bestV = Infinity;
    for (let i = Math.ceil(a * perSec); i < Math.floor(b * perSec); i++) {
        if (energy[i] < bestV) { bestV = energy[i]; best = i; }
    }
    return best / perSec;
}

/**
 * Cut speech into chunks at its pauses.
 *  count: wanted number of chunks (0 = one per pause).
 *  sensitivity 0..1: higher hears quieter sounds as speech (fewer pauses).
 * Returns { chunks: [{ aStart, aEnd }], found } — `found` is how many real
 * pauses were used; when there are not enough, the longest chunks are cut at
 * their quietest moment to reach `count`.
 */
export function findChunks(energy, perSec, { count = 0, sensitivity = 0.5, minGap = 0.18, pad = 0.06 } = {}) {
    if (!energy || !energy.length) return { chunks: [], found: 0 };
    const threshold = 0.02 + (1 - sensitivity) * 0.16;
    let { gaps, speechStart, speechEnd } = quietGaps(energy, perSec, threshold, minGap);
    if (count > 1 && gaps.length < count - 1) {
        // Shorter breaths count as pauses too before cutting mid-chunk.
        gaps = quietGaps(energy, perSec, threshold, 0.06).gaps;
    }
    let chosen = count > 0
        ? [...gaps].sort((a, b) => (b.end - b.start) - (a.end - a.start)).slice(0, Math.max(0, count - 1))
        : gaps;
    chosen = chosen.sort((a, b) => a.start - b.start);
    const found = chosen.length;

    let chunks = [];
    let from = speechStart;
    for (const g of chosen) {
        chunks.push({ aStart: from, aEnd: g.start });
        from = g.end;
    }
    chunks.push({ aStart: from, aEnd: speechEnd });

    // Not enough pauses: cut the longest chunk at its quietest point.
    while (count > 0 && chunks.length < count) {
        let k = 0;
        chunks.forEach((c, i) => { if (c.aEnd - c.aStart > chunks[k].aEnd - chunks[k].aStart) k = i; });
        const c = chunks[k];
        if (c.aEnd - c.aStart < 0.6) break;
        const cut = quietestAt(energy, perSec, c.aStart + 0.25, c.aEnd - 0.25);
        chunks.splice(k, 1, { aStart: c.aStart, aEnd: cut }, { aStart: cut, aEnd: c.aEnd });
    }

    const total = energy.length / perSec;
    // A little air around each chunk, never past the middle of the pause.
    chunks = chunks.map((c, i) => {
        const prev = chunks[i - 1];
        const next = chunks[i + 1];
        return {
            aStart: round2(Math.max(0, c.aStart - pad, prev ? (prev.aEnd + c.aStart) / 2 : 0)),
            aEnd: round2(Math.min(total, c.aEnd + pad, next ? (c.aEnd + next.aStart) / 2 : total)),
        };
    }).filter((c) => c.aEnd - c.aStart >= 0.05);
    return { chunks, found };
}

export const PLACE_MODES = [
    { value: 'spread', label: 'spread evenly' },
    { value: 'rhythm', label: 'keep rhythm, stretched' },
    { value: 'marks', label: 'at my marks' },
];

const GAP = 0.1;

/**
 * Where each chunk starts in the video: [{ aStart, aEnd, vStart }].
 *  spread: the video cut into equal slots, one chunk at the start of each.
 *  rhythm: the chunks' own timing stretched so the last one ends with the video.
 *  marks:  chunk i at mark i (sorted); chunks beyond the marks share the
 *          time after the last one.
 * Chunks never overlap: one that would is pushed after the previous.
 */
export function placeChunks(chunks, videoDuration, mode = 'spread', marks = []) {
    const n = chunks.length;
    if (!n) return [];
    const dur = Math.max(videoDuration || 0, 0);
    let targets;
    if (mode === 'marks' && marks.length) {
        // Chunk i at mark i; any left over share the time after the last mark.
        const m = [...marks].sort((a, b) => a - b).slice(0, n);
        const lastMark = m[m.length - 1];
        const rest = n - m.length;
        const slot = rest > 0 ? Math.max(0, dur - lastMark) / (rest + 1) : 0;
        targets = chunks.map((_, i) => (i < m.length ? m[i] : lastMark + (i - m.length + 1) * slot));
    } else if (mode === 'rhythm') {
        const first = chunks[0].aStart;
        const last = chunks[n - 1];
        const span = last.aStart - first;
        const room = Math.max(0, dur - (last.aEnd - last.aStart));
        const f = span > 0 ? Math.max(1, room / span) : 1;
        targets = chunks.map((c) => (c.aStart - first) * f);
    } else {
        const slot = dur / n;
        targets = chunks.map((_, i) => i * slot);
    }
    let free = 0;
    return chunks.map((c, i) => {
        const vStart = round2(Math.max(targets[i] ?? free, free));
        free = vStart + (c.aEnd - c.aStart) + GAP;
        return { aStart: c.aStart, aEnd: c.aEnd, vStart };
    });
}
