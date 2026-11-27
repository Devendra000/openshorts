import { useState, useRef, useEffect } from 'react';
import { Music, Film, Loader2, Download, AlertCircle, Wand2 } from 'lucide-react';
import { apiFetch } from '../lib/api';
import SegmentedControl from './ui/SegmentedControl';
import { parsePlacements, describePlacements, serializePlacements, PLACEMENT_PLACEHOLDER } from '../lib/audioPlacements';
import AudioSourceStrip from './audiomix/AudioSourceStrip';
import Timeline from './audiomix/Timeline';
import AutoChunker from './audiomix/AutoChunker';
import { decodeAudioFile, computePeaks, computeEnergy, useMixPreview } from './audiomix/audioEngine';

const MAX_VIDEO_BYTES = 1024 * 1024 * 1024;
const MAX_AUDIO_BYTES = 50 * 1024 * 1024;
const VIDEO_ACCEPT = '.mp4,.mov,.m4v,.webm,.mkv,video/*';
const AUDIO_ACCEPT = '.mp3,.wav,.m4a,.aac,.ogg,.oga,.opus,.flac,.webm,audio/*';

let nextId = 1;
const withIds = (list) => list.map((p) => ({ id: `p${nextId++}`, aStart: p.aStart, aEnd: p.aEnd, vStart: p.vStart }));

// Object URL for a picked file, revoked when the file changes or on unmount.
function useObjectUrl(file) {
    const [url, setUrl] = useState(null);
    useEffect(() => {
        if (!file) { setUrl(null); return undefined; }
        const u = URL.createObjectURL(file);
        setUrl(u);
        return () => URL.revokeObjectURL(u);
    }, [file]);
    return url;
}

function FilePick({ icon, label, accept, file, maxBytes, onPick, disabled }) {
    const tooBig = file && file.size > maxBytes;
    return (
        <div>
            <label className="eyebrow flex items-center gap-2 mb-2">{icon} {label}</label>
            <input
                type="file"
                accept={accept}
                disabled={disabled}
                onChange={(e) => onPick(e.target.files?.[0] || null)}
                className="input-field cursor-pointer file:mr-3 file:border-0 file:bg-paper3 file:text-ink file:rounded-input file:px-3 file:py-1"
            />
            {tooBig && (
                <p className="text-xs text-red-400 mt-1.5">Larger than {Math.round(maxBytes / (1024 * 1024))} MB.</p>
            )}
        </div>
    );
}

export default function AudioMixTab() {
    const [videoFile, setVideoFile] = useState(null);
    const [audioFile, setAudioFile] = useState(null);
    const [originalAudio, setOriginalAudio] = useState('duck');
    const [mixing, setMixing] = useState(false);
    const [error, setError] = useState('');
    const [result, setResult] = useState(null); // { url, name }

    // The placed chunks are the truth; the text box mirrors them both ways.
    const [pieces, setPieces] = useState([]);
    const [text, setText] = useState('');
    const [selectedId, setSelectedId] = useState(null);
    const [selection, setSelection] = useState(null); // { a0, a1 } in the audio

    const [buffer, setBuffer] = useState(null);
    const [peaks, setPeaks] = useState(null);
    const [energy, setEnergy] = useState(null);
    const [proposed, setProposed] = useState([]); // chunk maker output, not placed yet
    const [marks, setMarks] = useState([]);
    const [decodeError, setDecodeError] = useState('');
    const [videoDuration, setVideoDuration] = useState(0);
    const [playhead, setPlayhead] = useState(0);

    const videoRef = useRef(null);
    const audioRef = useRef(null);
    const videoUrl = useObjectUrl(videoFile);
    const audioUrl = useObjectUrl(audioFile);

    useEffect(() => () => { if (result?.url) URL.revokeObjectURL(result.url); }, [result]);

    // Decode the audio in the browser: waveform + preview.
    useEffect(() => {
        setBuffer(null);
        setPeaks(null);
        setEnergy(null);
        setProposed([]);
        setDecodeError('');
        setSelection(null);
        if (!audioFile) return undefined;
        let cancelled = false;
        decodeAudioFile(audioFile)
            .then((buf) => { if (!cancelled) { setBuffer(buf); setPeaks(computePeaks(buf)); setEnergy(computeEnergy(buf)); } })
            .catch(() => { if (!cancelled) setDecodeError('This browser cannot read that audio for the editor. Type the placement below; the mix itself still works.'); });
        return () => { cancelled = true; };
    }, [audioFile]);

    // Playhead follows the video.
    useEffect(() => {
        let raf = 0;
        const tick = () => {
            const v = videoRef.current;
            if (v) setPlayhead(v.currentTime);
            raf = requestAnimationFrame(tick);
        };
        raf = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(raf);
    }, [videoUrl]);

    useMixPreview(videoRef, videoUrl, buffer, pieces, originalAudio);

    const editPieces = (next) => {
        setPieces(next);
        setText(serializePlacements(next));
    };
    const editText = (value) => {
        setText(value);
        const parsed = parsePlacements(value);
        if (!parsed.error) {
            setPieces(withIds(parsed.pieces));
            setSelectedId(null);
        }
    };
    const addPiece = (a0, a1) => {
        const [piece] = withIds([{ aStart: Math.round(a0 * 100) / 100, aEnd: Math.round(a1 * 100) / 100, vStart: Math.round(playhead * 100) / 100 }]);
        editPieces([...pieces, piece]);
        setSelectedId(piece.id);
        setSelection(null);
    };
    const seek = (t) => {
        if (videoRef.current) videoRef.current.currentTime = t;
        setPlayhead(t);
    };

    const textState = parsePlacements(text);
    const audioDuration = buffer?.duration || 0;
    const ready = videoFile && audioFile && !mixing && !textState.error
        && videoFile.size <= MAX_VIDEO_BYTES && audioFile.size <= MAX_AUDIO_BYTES;

    const handleMix = async () => {
        if (!ready) return;
        setMixing(true);
        setError('');
        setResult(null);
        try {
            const form = new FormData();
            form.append('video', videoFile);
            form.append('audio', audioFile);
            form.append('original_audio', originalAudio);
            if (pieces.length) form.append('placements', serializePlacements(pieces));
            const res = await apiFetch('/api/audio-mix', { method: 'POST', body: form });
            if (!res.ok) {
                const raw = await res.text();
                let detail = raw;
                try { detail = JSON.parse(raw).detail || raw; } catch { /* not json */ }
                throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
            }
            const blob = await res.blob();
            const cd = res.headers.get('Content-Disposition') || '';
            const name = (cd.match(/filename="?([^";]+)"?/) || [])[1] || 'video_mixed.mp4';
            setResult({ url: URL.createObjectURL(blob), name });
        } catch (e) {
            setError(e.message || 'Could not mix those files');
        } finally {
            setMixing(false);
        }
    };

    return (
        <div className="h-full overflow-y-auto custom-scrollbar">
            <div className="max-w-6xl mx-auto p-4 sm:p-6 lg:p-8">
                <div className="mb-6">
                    <p className="eyebrow mb-2">07 · AUDIO MIX</p>
                    <h1 className="font-display lowercase text-2xl text-ink">audio mix</h1>
                    <p className="text-sm text-muted mt-1">
                        Your video + your audio. Cut the audio into chunks, place and move them on the video's timeline, preview, then mix. Nothing is generated or kept.
                    </p>
                </div>

                <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
                    <div className="card p-4 sm:p-5">
                        {videoUrl ? (
                            <div className="rounded-card overflow-hidden bg-black">
                                <video
                                    ref={videoRef}
                                    src={videoUrl}
                                    controls
                                    playsInline
                                    onLoadedMetadata={(e) => setVideoDuration(e.currentTarget.duration || 0)}
                                    className="w-full max-h-[420px] object-contain"
                                />
                            </div>
                        ) : (
                            <div className="aspect-video rounded-card bg-paper3 flex items-center justify-center text-sm text-muted">
                                <Film size={16} className="mr-2" /> pick a video to start
                            </div>
                        )}
                        <p className="text-xs text-muted mt-2">
                            Press play to preview the chunks over the video, with its own sound set as on the right.
                        </p>
                    </div>

                    <div className="card p-4 sm:p-5 space-y-4">
                        <FilePick icon={<Film size={12} />} label="Video" accept={VIDEO_ACCEPT} file={videoFile}
                            maxBytes={MAX_VIDEO_BYTES} onPick={setVideoFile} disabled={mixing} />
                        <FilePick icon={<Music size={12} />} label="Audio" accept={AUDIO_ACCEPT} file={audioFile}
                            maxBytes={MAX_AUDIO_BYTES} onPick={setAudioFile} disabled={mixing} />

                        <div>
                            <label className="eyebrow block mb-2">Video's own audio</label>
                            <SegmentedControl
                                size="sm"
                                columns={3}
                                options={[
                                    { value: 'duck', label: 'quiet under' },
                                    { value: 'mix', label: 'keep full' },
                                    { value: 'replace', label: 'replace' },
                                ]}
                                value={originalAudio}
                                onChange={(v) => !mixing && setOriginalAudio(v)}
                            />
                        </div>

                        <button type="button" className="btn-primary w-full" disabled={!ready} onClick={handleMix}>
                            {mixing ? <><Loader2 size={16} className="animate-spin" /> uploading &amp; mixing…</> : <><Wand2 size={16} /> mix &amp; download</>}
                        </button>
                        {!pieces.length && videoFile && audioFile && (
                            <p className="text-xs text-muted">No chunks placed: the whole audio plays from the start.</p>
                        )}

                        {error && (
                            <div className="flex items-start gap-2">
                                <span className="badge-warn shrink-0"><AlertCircle size={12} /> error</span>
                                <p className="text-sm text-muted">{error}</p>
                            </div>
                        )}

                        {result && (
                            <div className="space-y-3">
                                <div className="rounded-card overflow-hidden bg-black">
                                    <video src={result.url} controls playsInline className="w-full max-h-[300px] object-contain" />
                                </div>
                                <a href={result.url} download={result.name} className="btn-primary w-full">
                                    <Download size={16} /> download {result.name}
                                </a>
                            </div>
                        )}
                    </div>
                </div>

                <div className="card p-4 sm:p-5 mt-5 space-y-5">
                    {audioUrl && <audio ref={audioRef} src={audioUrl} className="hidden" />}
                    {peaks ? (
                        <AudioSourceStrip
                            peaks={peaks}
                            duration={audioDuration}
                            audioRef={audioRef}
                            pieces={pieces}
                            proposed={proposed}
                            selection={selection}
                            onSelection={setSelection}
                            onAdd={addPiece}
                            videoTime={playhead}
                        />
                    ) : (
                        <p className="text-sm text-muted">
                            {decodeError || (audioFile ? 'Reading the audio…' : 'Pick an audio file to see its waveform and cut it into chunks.')}
                        </p>
                    )}

                    {energy && (
                        <AutoChunker
                            energy={energy}
                            videoDuration={videoDuration}
                            marks={marks}
                            chunks={proposed}
                            onChunks={setProposed}
                            onPlace={(placed) => {
                                editPieces(withIds(placed));
                                setSelectedId(null);
                                setProposed([]);
                            }}
                        />
                    )}

                    {videoUrl && (
                        <Timeline
                            pieces={pieces}
                            onChange={editPieces}
                            selectedId={selectedId}
                            onSelect={setSelectedId}
                            videoDuration={videoDuration}
                            audioDuration={audioDuration || Infinity}
                            playhead={playhead}
                            onSeek={seek}
                            peaks={peaks}
                            marks={marks}
                            onMarksChange={setMarks}
                        />
                    )}

                    <details open={!peaks || undefined}>
                        <summary className="eyebrow cursor-pointer">Placement as text · paste a map here</summary>
                        <textarea
                            value={text}
                            onChange={(e) => editText(e.target.value)}
                            rows={6}
                            placeholder={PLACEMENT_PLACEHOLDER}
                            className="input-field resize-y min-h-[120px] font-mono text-xs mt-2"
                            disabled={mixing}
                        />
                        <p className={`text-xs mt-1.5 ${textState.error ? 'text-red-400' : 'text-muted'}`}>
                            {describePlacements(textState, 'Empty: the whole audio plays from the start of the video, cut at its end.')}
                        </p>
                    </details>
                </div>
            </div>
        </div>
    );
}
