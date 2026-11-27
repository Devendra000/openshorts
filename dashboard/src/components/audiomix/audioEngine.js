// Browser-side audio for the Audio Mix editor: decoding the picked file,
// waveform peaks, and a preview that plays the placed pieces in sync with
// the video element (Web Audio), before anything is sent to the server.
import { useEffect, useRef, useCallback } from 'react';

let shared = null;
export const audioContext = () => {
    if (!shared) shared = new (window.AudioContext || window.webkitAudioContext)();
    return shared;
};

export const PEAKS_PER_SECOND = 100;

export async function decodeAudioFile(file) {
    const data = await file.arrayBuffer();
    return audioContext().decodeAudioData(data);
}

// Max |sample| per 1/PEAKS_PER_SECOND s over all channels, normalised to 0..1.
export function computePeaks(buffer) {
    const per = Math.max(1, Math.floor(buffer.sampleRate / PEAKS_PER_SECOND));
    const n = Math.ceil(buffer.length / per);
    const peaks = new Float32Array(n);
    for (let c = 0; c < buffer.numberOfChannels; c++) {
        const data = buffer.getChannelData(c);
        for (let i = 0; i < n; i++) {
            let m = peaks[i];
            const end = Math.min(data.length, (i + 1) * per);
            for (let j = i * per; j < end; j++) {
                const v = data[j] < 0 ? -data[j] : data[j];
                if (v > m) m = v;
            }
            peaks[i] = m;
        }
    }
    let max = 0;
    for (let i = 0; i < n; i++) if (peaks[i] > max) max = peaks[i];
    if (max > 0) for (let i = 0; i < n; i++) peaks[i] /= max;
    return peaks;
}

// Loudness (RMS) per 1/PEAKS_PER_SECOND s, channels averaged, normalised to
// 0..1: what the chunk maker reads to find pauses.
export function computeEnergy(buffer) {
    const per = Math.max(1, Math.floor(buffer.sampleRate / PEAKS_PER_SECOND));
    const n = Math.ceil(buffer.length / per);
    const energy = new Float32Array(n);
    const channels = [];
    for (let c = 0; c < buffer.numberOfChannels; c++) channels.push(buffer.getChannelData(c));
    let max = 0;
    for (let i = 0; i < n; i++) {
        let sum = 0;
        let count = 0;
        const end = Math.min(buffer.length, (i + 1) * per);
        for (const data of channels) {
            for (let j = i * per; j < end; j++) { sum += data[j] * data[j]; count++; }
        }
        const v = Math.sqrt(sum / Math.max(1, count));
        energy[i] = v;
        if (v > max) max = v;
    }
    if (max > 0) for (let i = 0; i < n; i++) energy[i] /= max;
    return energy;
}

const VIDEO_LEVEL = { duck: 0.15, mix: 1, replace: 0 };

// Plays `pieces` from `buffer` whenever the video plays, from wherever it is.
export function useMixPreview(videoRef, videoKey, buffer, pieces, originalAudio) {
    const nodes = useRef([]);
    const state = useRef({ buffer, pieces });
    useEffect(() => { state.current = { buffer, pieces }; }, [buffer, pieces]);

    const stop = useCallback(() => {
        nodes.current.forEach((n) => { try { n.stop(); } catch { /* already stopped */ } });
        nodes.current = [];
    }, []);

    const schedule = useCallback(() => {
        stop();
        const video = videoRef.current;
        const { buffer: buf, pieces: list } = state.current;
        if (!video || !buf || video.paused) return;
        const ctx = audioContext();
        if (ctx.state === 'suspended') ctx.resume();
        const t0 = video.currentTime;
        const now = ctx.currentTime + 0.02;
        for (const p of list) {
            const len = p.aEnd - p.aStart;
            if (len <= 0 || p.vStart + len <= t0) continue;
            const into = Math.max(0, t0 - p.vStart);
            const src = ctx.createBufferSource();
            src.buffer = buf;
            src.connect(ctx.destination);
            src.start(now + Math.max(0, p.vStart - t0), p.aStart + into, len - into);
            nodes.current.push(src);
        }
    }, [videoRef, stop]);

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return undefined;
        const onSeeked = () => { if (!video.paused) schedule(); };
        video.addEventListener('playing', schedule);
        video.addEventListener('seeked', onSeeked);
        ['pause', 'seeking', 'ended', 'waiting'].forEach((e) => video.addEventListener(e, stop));
        return () => {
            stop();
            video.removeEventListener('playing', schedule);
            video.removeEventListener('seeked', onSeeked);
            ['pause', 'seeking', 'ended', 'waiting'].forEach((e) => video.removeEventListener(e, stop));
        };
    }, [videoRef, videoKey, schedule, stop]);

    // Edits while playing are heard at once.
    useEffect(() => {
        const video = videoRef.current;
        if (video && !video.paused) schedule();
    }, [videoRef, buffer, pieces, schedule]);

    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;
        const level = VIDEO_LEVEL[originalAudio] ?? 1;
        video.muted = level === 0;
        video.volume = level || 1;
    }, [videoRef, videoKey, originalAudio]);
}
