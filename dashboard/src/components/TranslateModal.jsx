import React, { useState, useEffect } from 'react';
import { Loader2, Languages, AlertCircle, Mic, Upload } from 'lucide-react';
import Modal from './ui/Modal';
import SegmentedControl from './ui/SegmentedControl';
import { apiFetch } from '../lib/api';
import { TIME, toSeconds, parsePlacements, describePlacements, PLACEMENT_PLACEHOLDER } from '../lib/audioPlacements';

const MAX_SCRIPT_CHARS = 5000;
const MAX_AUDIO_BYTES = 50 * 1024 * 1024;
const AUDIO_ACCEPT = '.mp3,.wav,.m4a,.aac,.ogg,.oga,.opus,.flac,.webm,audio/*';
// ElevenLabs premade voices, used until the account's own list arrives.
const DEFAULT_VOICES = [
    { voice_id: '21m00Tcm4TlvDq8ikWAM', name: 'Rachel' },
    { voice_id: '29vD33N1CtxCmqQRPOHJ', name: 'Drew' },
    { voice_id: 'EXAVITQu4vr4xnSDxMaL', name: 'Bella' },
    { voice_id: 'ErXwobaYiN019PkySvjV', name: 'Antoni' },
];
// Roughly 15 characters a second of natural speech (audio tags not spoken).
const estimateSeconds = (text) => Math.round((text || '').replace(/\[[^\]\n]{1,40}\]/g, '').trim().length / 15);
// Same shape the backend reads (translate.parse_timed_script): a line that
// starts with "0:04" or "0:04–0:10" is placed at that time in the clip.
const TIMED_LINE = new RegExp(`^\\s*[\\[(]?\\s*(${TIME})(?:\\s*(?:-|\u2013|\u2014|to)\\s*(${TIME}))?`);
const timedLines = (text) => (text || '').split('\n').map((l) => l.match(TIMED_LINE)).filter(Boolean);
const SCRIPT_PLACEHOLDER = `Type what the voice should say, or time each line:

0:00–0:04  [curious] What's hiding inside this pack?
0:04–0:10  [explaining] See every item and its odds first.
0:10–0:17  [excited] Now... watch what came out!`;
const TTS_MODELS = [
    { value: 'eleven_v3', label: 'v3 · expressive, acts on [tags]' },
    { value: 'eleven_multilingual_v2', label: 'multilingual v2 · tags removed' },
];

const LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "hi": "Hindi",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "ar": "Arabic",
    "ru": "Russian",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "id": "Indonesian",
    "fil": "Filipino",
    "ms": "Malay",
    "vi": "Vietnamese",
    "th": "Thai",
    "uk": "Ukrainian",
    "el": "Greek",
    "cs": "Czech",
    "fi": "Finnish",
    "ro": "Romanian",
    "da": "Danish",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sk": "Slovak",
    "ta": "Tamil",
    "en": "English",
};

export default function TranslateModal({ isOpen, onClose, onTranslate, onVoiceover, onAudioUpload, isProcessing, videoUrl, hasApiKey, elevenLabsKey, clipSeconds = 0 }) {
    const [mode, setMode] = useState('script');
    const [targetLanguage, setTargetLanguage] = useState('es');
    const [script, setScript] = useState('');
    const [voices, setVoices] = useState(DEFAULT_VOICES);
    const [voiceId, setVoiceId] = useState(DEFAULT_VOICES[0].voice_id);
    const [originalAudio, setOriginalAudio] = useState('duck');
    const [model, setModel] = useState('eleven_v3');
    const [audioFile, setAudioFile] = useState(null);
    const [placements, setPlacements] = useState('');

    useEffect(() => {
        if (!isOpen || !elevenLabsKey) return;
        let cancelled = false;
        apiFetch('/api/voiceover/voices', { headers: { 'X-ElevenLabs-Key': elevenLabsKey } })
            .then((res) => (res.ok ? res.json() : { voices: [] }))
            .then((data) => {
                if (cancelled || !data.voices?.length) return;
                setVoices(data.voices);
                setVoiceId((cur) => (data.voices.some((v) => v.voice_id === cur) ? cur : data.voices[0].voice_id));
            })
            .catch(() => {});
        return () => { cancelled = true; };
    }, [isOpen, elevenLabsKey]);

    if (!isOpen) return null;

    const isScript = mode === 'script';
    const isUpload = mode === 'upload';
    const scriptReady = script.trim().length > 0 && script.length <= MAX_SCRIPT_CHARS;
    const audioTooBig = !!audioFile && audioFile.size > MAX_AUDIO_BYTES;
    const placement = parsePlacements(placements);
    const uploadReady = !!audioFile && !audioTooBig && !placement.error;
    const ready = isScript ? scriptReady && hasApiKey : isUpload ? uploadReady : hasApiKey;
    const spoken = estimateSeconds(script);
    const timed = timedLines(script);
    const timedEnd = timed.reduce((m, t) => Math.max(m, toSeconds(t[2] || t[1])), 0);

    const handleSubmit = () => {
        if (isScript) {
            onVoiceover({ script: script.trim(), voiceId, originalAudio, model });
        } else if (isUpload) {
            onAudioUpload({ file: audioFile, originalAudio, placements: placements.trim() });
        } else {
            onTranslate({ targetLanguage });
        }
    };

    return (
        <Modal
            isOpen={isOpen}
            onClose={isProcessing ? undefined : onClose}
            eyebrow="DUB"
            title="dub voice"
            size="md"
            footer={
                <div className="flex gap-3">
                    <button
                        onClick={onClose}
                        disabled={isProcessing}
                        className="btn-ghost flex-1"
                    >
                        Cancel
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={isProcessing || !ready}
                        className="btn-primary flex-1"
                    >
                        {isProcessing ? (
                            <>
                                <Loader2 size={16} className="animate-spin" />
                                {isScript ? 'Generating...' : isUpload ? 'Mixing...' : 'Dubbing...'}
                            </>
                        ) : isUpload ? (
                            <>
                                <Upload size={16} />
                                Mix Audio
                            </>
                        ) : isScript ? (
                            <>
                                <Mic size={16} />
                                Add Voiceover
                            </>
                        ) : (
                            <>
                                <Languages size={16} />
                                Dub Voice
                            </>
                        )}
                    </button>
                </div>
            }
        >
            <div className="flex items-center gap-3 mb-5">
                <div className="w-10 h-10 rounded-input bg-paper3 flex items-center justify-center shrink-0">
                    <Languages size={18} className="text-brass" />
                </div>
                <p className="text-xs text-muted">AI voice by ElevenLabs</p>
            </div>

            <div className="mb-5">
                <SegmentedControl
                    options={[
                        { value: 'script', label: 'from script', icon: Mic },
                        { value: 'upload', label: 'upload audio', icon: Upload },
                        { value: 'translate', label: 'translate', icon: Languages },
                    ]}
                    value={mode}
                    onChange={(v) => !isProcessing && setMode(v)}
                />
            </div>

            {!hasApiKey && !isUpload && (
                <div className="mb-4 flex items-start gap-2">
                    <span className="badge-warn shrink-0"><AlertCircle size={12} /> key missing</span>
                    <p className="text-sm text-muted">Configure ElevenLabs API Key in Settings first.</p>
                </div>
            )}

            {/* Preview */}
            <div className="mb-5 rounded-card overflow-hidden bg-black aspect-video">
                <video
                    src={videoUrl}
                    className="w-full h-full object-contain"
                    muted
                    playsInline
                />
            </div>

            {isScript ? (
                <>
                    <div className="mb-4">
                        <div className="flex items-baseline justify-between mb-2">
                            <label className="eyebrow">Script</label>
                            <span className={`text-xs ${script.length > MAX_SCRIPT_CHARS ? 'text-red-400' : 'text-muted'}`}>
                                {script.length}/{MAX_SCRIPT_CHARS}
                                {timed.length > 0
                                    ? ` · ${timed.length} timed line${timed.length > 1 ? 's' : ''} to ${Math.round(timedEnd)}s`
                                    : spoken > 0 && ` · ~${spoken}s spoken`}
                                {clipSeconds > 0 && ` · clip ${Math.round(clipSeconds)}s`}
                            </span>
                        </div>
                        <textarea
                            value={script}
                            onChange={(e) => setScript(e.target.value)}
                            rows={7}
                            placeholder={SCRIPT_PLACEHOLDER}
                            className="input-field resize-y min-h-[150px] font-mono text-xs"
                            disabled={isProcessing}
                        />
                        <p className="text-xs text-muted mt-1.5">
                            {timed.length > 0
                                ? 'Each line starts at its time, with silence between. A line too long for its slot is sped up a little or pushed later.'
                                : clipSeconds > 0 && spoken > clipSeconds * 1.15
                                    ? 'Longer than the clip: the voice is sped up slightly and the last frame is held until it finishes.'
                                    : 'Start lines with 0:04–0:10 to place them in the clip. [excited], [whispers]… set the tone with v3.'}
                        </p>
                    </div>

                    <div className="mb-4">
                        <label className="eyebrow block mb-2">Model</label>
                        <select
                            value={model}
                            onChange={(e) => setModel(e.target.value)}
                            className="input-field appearance-none cursor-pointer"
                            disabled={isProcessing}
                        >
                            {TTS_MODELS.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
                        </select>
                    </div>

                    <div className="mb-4">
                        <label className="eyebrow block mb-2">Voice</label>
                        <select
                            value={voiceId}
                            onChange={(e) => setVoiceId(e.target.value)}
                            className="input-field appearance-none cursor-pointer"
                            disabled={isProcessing}
                        >
                            {voices.map((v) => (
                                <option key={v.voice_id} value={v.voice_id}>
                                    {v.name}{v.labels?.gender ? ` · ${v.labels.gender}` : ''}{v.labels?.accent ? ` · ${v.labels.accent}` : ''}
                                </option>
                            ))}
                        </select>
                    </div>

                </>
            ) : isUpload ? (
                <div className="mb-4">
                    <label className="eyebrow block mb-2">Audio file</label>
                    <input
                        type="file"
                        accept={AUDIO_ACCEPT}
                        onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
                        disabled={isProcessing}
                        className="input-field cursor-pointer file:mr-3 file:border-0 file:bg-paper3 file:text-ink file:rounded-input file:px-3 file:py-1"
                    />
                    <p className={`text-xs mt-1.5 ${audioTooBig ? 'text-red-400' : 'text-muted'}`}>
                        {audioTooBig
                            ? 'That file is larger than 50 MB.'
                            : `mp3, wav, m4a, ogg, flac… up to 50 MB. Cut at the clip's end${clipSeconds > 0 ? ` (${Math.round(clipSeconds)}s)` : ''}.`}
                    </p>

                    <label className="eyebrow block mt-4 mb-2">Placement</label>
                    <textarea
                        value={placements}
                        onChange={(e) => setPlacements(e.target.value)}
                        rows={5}
                        placeholder={PLACEMENT_PLACEHOLDER}
                        className="input-field resize-y min-h-[110px] font-mono text-xs"
                        disabled={isProcessing}
                    />
                    <p className={`text-xs mt-1.5 ${placement.error ? 'text-red-400' : 'text-muted'}`}>
                        {describePlacements(placement, 'Empty: the whole audio plays from the start of the video.')}
                    </p>
                </div>
            ) : (
                <>
                {/* Language Selection */}
                <div className="mb-5">
                    <label className="eyebrow block mb-2">
                        Target Language
                    </label>
                    <select
                        value={targetLanguage}
                        onChange={(e) => setTargetLanguage(e.target.value)}
                        className="input-field appearance-none cursor-pointer"
                        disabled={isProcessing}
                    >
                        {Object.entries(LANGUAGES).sort((a, b) => a[1].localeCompare(b[1])).map(([code, name]) => (
                            <option key={code} value={code}>
                                {name}
                            </option>
                        ))}
                    </select>
                </div>

                {/* Info */}
                <p className="text-xs text-muted leading-relaxed mb-2">
                    The audio will be dubbed with AI-generated voice in the selected language, matching the original speaker's characteristics.
                </p>
                </>
            )}

            {(isScript || isUpload) && (
                <div className="mb-5">
                    <label className="eyebrow block mb-2">Original audio</label>
                    <SegmentedControl
                        size="sm"
                        columns={3}
                        options={[
                            { value: 'duck', label: 'quiet under' },
                            { value: 'mix', label: 'keep full' },
                            { value: 'replace', label: 'replace' },
                        ]}
                        value={originalAudio}
                        onChange={(v) => !isProcessing && setOriginalAudio(v)}
                    />
                </div>
            )}

            {/* AI Act art. 50: we mark the file, the person publishing it is the
                one who owes the audience the disclosure. Saying so here is the
                only place they will read it. */}
            {!isUpload && (
                <p className="text-xs text-muted leading-relaxed mb-2">
                    The new file is tagged as AI-generated content. When you publish it, disclose that the voice is synthetic.
                </p>
            )}

            {/* Processing State */}
            {isProcessing && (
                <div className="mt-4 p-3 bg-paper3 rounded-input">
                    <div className="flex items-center gap-3">
                        <Loader2 size={18} className="text-brass animate-spin" />
                        <div>
                            <p className="text-sm text-ink font-medium lowercase">{isScript ? 'Generating voiceover...' : isUpload ? 'Mixing audio...' : 'Dubbing audio...'}</p>
                            <p className="text-xs text-muted lowercase">This may take a few minutes</p>
                        </div>
                    </div>
                </div>
            )}
        </Modal>
    );
}
