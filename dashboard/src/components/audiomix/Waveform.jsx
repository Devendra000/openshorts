import { useEffect, useRef } from 'react';
import { PEAKS_PER_SECOND } from './audioEngine';

// Peaks between `from` and `to` seconds, drawn in the canvas's CSS color.
export default function Waveform({ peaks, from = 0, to, className = '' }) {
    const ref = useRef(null);
    useEffect(() => {
        const canvas = ref.current;
        if (!canvas) return undefined;
        const draw = () => {
            const dpr = window.devicePixelRatio || 1;
            const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
            const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
            canvas.width = w;
            canvas.height = h;
            const g = canvas.getContext('2d');
            g.clearRect(0, 0, w, h);
            if (!peaks || !(to > from)) return;
            g.fillStyle = getComputedStyle(canvas).color;
            const first = from * PEAKS_PER_SECOND;
            const span = (to - from) * PEAKS_PER_SECOND;
            for (let x = 0; x < w; x++) {
                const i0 = Math.floor(first + (x / w) * span);
                const i1 = Math.max(i0 + 1, Math.floor(first + ((x + 1) / w) * span));
                let m = 0;
                for (let i = Math.max(0, i0); i < Math.min(peaks.length, i1); i++) if (peaks[i] > m) m = peaks[i];
                const bar = Math.max(dpr, m * h * 0.9);
                g.fillRect(x, (h - bar) / 2, 1, bar);
            }
        };
        draw();
        const ro = new ResizeObserver(draw);
        ro.observe(canvas);
        return () => ro.disconnect();
    }, [peaks, from, to]);
    return <canvas ref={ref} className={`block w-full h-full ${className}`} />;
}
