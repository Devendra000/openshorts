// Placing parts of an audio file in a video, one line each:
//   0:00-0:05 -> 0:05-0:10   audio 0:00–0:05 plays at video 0:05–0:10
//   0:12-0:20 -> 0:15        video end optional
// Mirrors translate.parse_audio_placements on the backend: only the times on
// a line count (lines without one, like a pasted table's headings, are
// skipped), and a shorter video range cuts the piece (never stretched).

export const TIME = '(?:\\d+:)?\\d{1,2}:\\d{2}(?:\\.\\d+)?';
const TIME_TOKEN = new RegExp(TIME, 'g');

export const toSeconds = (stamp) => stamp.split(':').reduce((t, p) => t * 60 + parseFloat(p), 0);

export const fmtTime = (sec) => {
    const s = Math.max(0, sec || 0);
    const whole = Math.floor(s);
    const tenth = Math.floor((s - whole) * 10);
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}${tenth ? `.${tenth}` : ''}`;
};

export const PLACEMENT_PLACEHOLDER = `Optional. Leave empty to play the audio from the start.
Audio part -> where it plays in the video:

0:00-0:05 -> 0:05-0:10
0:12-0:20 -> 0:15`;

// { pieces: [{ aStart, aEnd, vStart }] } or { error } naming the line.
export function parsePlacements(text) {
    const pieces = [];
    const lines = (text || '').split('\n');
    for (let i = 0; i < lines.length; i++) {
        if (!lines[i].trim()) continue;
        const t = (lines[i].match(TIME_TOKEN) || []).map(toSeconds);
        if (!t.length) continue; // a heading or a note in a pasted table
        if (t.length !== 3 && t.length !== 4) return { error: `line ${i + 1}: write it as 0:00-0:05 -> 0:05-0:10` };
        if (t[1] <= t[0]) return { error: `line ${i + 1}: the audio range ends before it starts` };
        if (t.length === 4 && t[3] <= t[2]) return { error: `line ${i + 1}: the video range ends before it starts` };
        const aEnd = t.length === 4 ? Math.min(t[1], t[0] + t[3] - t[2]) : t[1];
        pieces.push({ aStart: t[0], aEnd, vStart: t[2] });
    }
    if (pieces.length > 50) return { error: 'at most 50 placement lines' };
    return { pieces };
}

export const describePlacements = ({ pieces, error }, emptyText) => error
    || (pieces.length
        ? pieces.map((p) => `audio ${fmtTime(p.aStart)}–${fmtTime(p.aEnd)} at video ${fmtTime(p.vStart)}`).join(' · ')
        : emptyText);

// Precise m:ss(.cc) for the text box (fmtTime rounds to a tenth for display).
export const fmtPrecise = (sec) => {
    const s = Math.max(0, Math.round((sec || 0) * 100) / 100);
    const m = Math.floor(s / 60);
    const rest = (s - m * 60).toFixed(2).replace(/\.?0+$/, '');
    const [whole, frac] = rest.split('.');
    return `${m}:${whole.padStart(2, '0')}${frac ? `.${frac}` : ''}`;
};

export const serializePlacements = (pieces) => [...pieces]
    .sort((a, b) => a.vStart - b.vStart)
    .map((p) => `${fmtPrecise(p.aStart)}-${fmtPrecise(p.aEnd)} -> ${fmtPrecise(p.vStart)}`)
    .join('\n');
