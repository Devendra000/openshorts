// Re-time edited caption text against the words it was edited from.
//
// The caption editor used to spread the edited words evenly over the whole
// clip, so fixing one misheard word ("zombie" -> "off me") threw every word
// after it off the audio. Instead the new text is aligned with the original
// words (a word-level diff): unchanged words keep their own timestamps, a
// replaced run takes the time span of the words it replaces, and inserted
// words share the gap (or the previous word) they were typed into.

// Past this many cells the diff table gets heavy; a clip never has that many
// words, but a pasted essay should not freeze the tab.
const MAX_CELLS = 4_000_000;

const norm = (w) => w.toLowerCase().replace(/[^\p{L}\p{N}']/gu, '');

// Word-level diff: [['eq'|'sub', origStart, origEnd, newStart, newEnd], ...],
// every original and every new word covered exactly once, in order.
function diff(orig, next) {
    const n = orig.length;
    const m = next.length;
    const a = orig.map(norm);
    const b = next.map(norm);
    // lcs[i][j] = LCS length of a[i..] and b[j..], flattened.
    const w = m + 1;
    const lcs = new Uint32Array((n + 1) * w);
    for (let i = n - 1; i >= 0; i--) {
        for (let j = m - 1; j >= 0; j--) {
            lcs[i * w + j] = a[i] === b[j] && a[i] !== ''
                ? lcs[(i + 1) * w + j + 1] + 1
                : Math.max(lcs[(i + 1) * w + j], lcs[i * w + j + 1]);
        }
    }
    const ops = [];
    let i = 0;
    let j = 0;
    let runI = 0;
    let runJ = 0;
    const flush = () => {
        if (i > runI || j > runJ) ops.push(['sub', runI, i, runJ, j]);
    };
    while (i < n || j < m) {
        if (i < n && j < m && a[i] === b[j] && a[i] !== '') {
            flush();
            ops.push(['eq', i, i + 1, j, j + 1]);
            i += 1;
            j += 1;
            runI = i;
            runJ = j;
        } else if (j < m && (i >= n || lcs[i * w + j + 1] >= lcs[(i + 1) * w + j])) {
            j += 1;
        } else {
            i += 1;
        }
    }
    flush();
    return ops;
}

// Lay `words` over [startMs, endMs], each getting time in proportion to its
// length (a long word takes longer to say than "a").
function spread(words, startMs, endMs) {
    const weights = words.map((t) => Math.max(1, norm(t).length));
    const total = weights.reduce((s, x) => s + x, 0);
    const span = Math.max(0, endMs - startMs);
    let at = startMs;
    return words.map((text, k) => {
        const len = (span * weights[k]) / total;
        const word = { text, startMs: Math.round(at), endMs: Math.round(at + len) };
        at += len;
        return word;
    });
}

// Minimum time an inserted word gets before it borrows from its neighbour.
const MIN_INSERT_MS = 120;

/**
 * @param {{text: string, startMs: number, endMs: number}[]} original
 * @param {string} newText
 * @returns {{text: string, startMs: number, endMs: number}[]}
 */
export function retimeWords(original, newText) {
    const next = newText.split(/\s+/).filter((t) => t.length > 0);
    if (!next.length || !original.length) return [];
    const orig = original.map((c) => c.text);

    if (orig.length * next.length > MAX_CELLS) {
        // Too big to align: keep the old even spread over the clip.
        return spread(next, original[0].startMs, original[original.length - 1].endMs);
    }

    const out = [];
    for (const [op, i0, i1, j0, j1] of diff(orig, next)) {
        if (op === 'eq') {
            out.push({ ...original[i0], text: next[j0] });
            continue;
        }
        const added = next.slice(j0, j1);
        if (!added.length) continue; // pure deletion: the words just go
        if (i1 > i0) {
            out.push(...spread(added, original[i0].startMs, original[i1 - 1].endMs));
            continue;
        }
        // Pure insertion: the silence between the neighbours, or, when that is
        // too short, a share of the previous word's own time.
        const need = MIN_INSERT_MS * added.length;
        const prev = out[out.length - 1];
        const after = original[i0]; // undefined when typed after the last word
        if (!prev) {
            // Before the first word: a short slot just ahead of it.
            const end = after.startMs;
            out.push(...spread(added, Math.max(0, end - need), end));
            continue;
        }
        // After the last word there is no gap to fill: share its time.
        const gapEnd = after ? after.startMs : prev.endMs;
        if (gapEnd - prev.endMs >= need) {
            out.push(...spread(added, prev.endMs, gapEnd));
        } else {
            out.pop();
            out.push(...spread([prev.text, ...added], prev.startMs, Math.max(prev.endMs, gapEnd)));
        }
    }
    return out;
}
