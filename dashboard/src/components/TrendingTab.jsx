import React, { useState, useEffect, useCallback } from 'react';
import {
  Flame,
  Search,
  Loader2,
  ExternalLink,
  Scissors,
  Copy,
  Check,
  RefreshCw,
  Globe,
  X,
  AlertCircle,
} from 'lucide-react';
import { apiJson } from '../lib/api';

const REGIONS = [
  { code: 'US', label: 'United States' },
  { code: 'GB', label: 'United Kingdom' },
  { code: 'IN', label: 'India' },
  { code: 'CA', label: 'Canada' },
  { code: 'AU', label: 'Australia' },
  { code: 'NP', label: 'Nepal' },
  { code: 'DE', label: 'Germany' },
  { code: 'FR', label: 'France' },
  { code: 'JP', label: 'Japan' },
  { code: 'KR', label: 'South Korea' },
  { code: 'BR', label: 'Brazil' },
  { code: 'MX', label: 'Mexico' },
  { code: 'ES', label: 'Spain' },
  { code: 'ID', label: 'Indonesia' },
  { code: 'PH', label: 'Philippines' },
];

const CATEGORIES = [
  { id: 'all', label: 'All Trending' },
  { id: 'music', label: 'Music' },
  { id: 'gaming', label: 'Gaming' },
  { id: 'news', label: 'News' },
  { id: 'movies', label: 'Movies' },
  { id: 'shorts', label: 'Shorts' },
];

const SORTS = [
  { id: 'trending', label: 'Trending' },
  { id: 'views', label: 'Most Viewed' },
  { id: 'date', label: 'Newest' },
];

const QUICK_TOPICS = [
  'AI & Tech',
  'Podcasts',
  'Creator Economy',
  'Gaming',
  'Finance',
  'Fitness',
  'Coding',
];

export default function TrendingTab({ onClipUrl }) {
  const [queryInput, setQueryInput] = useState('');
  const [activeQuery, setActiveQuery] = useState('');
  const [region, setRegion] = useState('US');
  const [category, setCategory] = useState('all');
  const [sort, setSort] = useState('trending');
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [meta, setMeta] = useState({ source: '', cached: false });
  const [copiedId, setCopiedId] = useState(null);

  const loadTrending = useCallback(async (opts = {}) => {
    const q = opts.q !== undefined ? opts.q : activeQuery;
    const reg = opts.region || region;
    const cat = opts.category || category;
    const srt = opts.sort || sort;

    setLoading(true);
    setError('');
    try {
      const params = new URLSearchParams({
        q,
        region: reg,
        category: cat,
        sort: srt,
        limit: '24',
      });
      const data = await apiJson(`/api/youtube/trending?${params.toString()}`);
      setItems(Array.isArray(data?.items) ? data.items : []);
      setMeta({ source: data?.source || '', cached: Boolean(data?.cached) });
    } catch (err) {
      setError(err?.message || 'Could not load trending YouTube videos right now.');
    } finally {
      setLoading(false);
    }
  }, [activeQuery, region, category, sort]);

  useEffect(() => {
    loadTrending();
  }, [loadTrending]);

  const handleSearchSubmit = (e) => {
    e.preventDefault();
    const trimmed = queryInput.trim();
    setActiveQuery(trimmed);
  };

  const handleTopicClick = (topic) => {
    setQueryInput(topic);
    setActiveQuery(topic);
  };

  const handleClearSearch = () => {
    setQueryInput('');
    setActiveQuery('');
  };

  const handleCopy = async (video) => {
    try {
      await navigator.clipboard.writeText(video.url);
      setCopiedId(video.video_id);
      setTimeout(() => setCopiedId((prev) => (prev === video.video_id ? null : prev)), 1800);
    } catch {
      // ignore clipboard errors
    }
  };

  return (
    <div className="h-full overflow-y-auto custom-scrollbar p-4 sm:p-8 max-w-6xl mx-auto animate-fade">
      <div className="flex flex-wrap items-end justify-between gap-4 mb-6">
        <div>
          <p className="eyebrow mb-1.5 flex items-center gap-1.5">
            <Flame size={12} className="text-brass" /> YOUTUBE RADAR
          </p>
          <h1 className="font-display lowercase text-2xl sm:text-3xl text-ink">
            Trending on YouTube right now
          </h1>
          <p className="text-muted text-sm mt-1 lowercase">
            Discover what&apos;s blowing up right now or search any topic, then send a video straight to the Clip Generator.
          </p>
        </div>

        <button
          type="button"
          onClick={() => loadTrending()}
          disabled={loading}
          className="btn-ghost px-3 py-2 text-xs shrink-0"
          title="Refresh results"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin text-brass' : ''} />
          <span>{loading ? 'fetching…' : 'refresh'}</span>
        </button>
      </div>

      {/* Search & Filters */}
      <div className="card p-4 sm:p-5 mb-6 space-y-4">
        <form onSubmit={handleSearchSubmit} className="flex flex-col sm:flex-row gap-2.5">
          <div className="relative flex-1">
            <Search size={16} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
            <input
              type="text"
              value={queryInput}
              onChange={(e) => setQueryInput(e.target.value)}
              placeholder="Search trending topics, creators, podcasts, niches..."
              className="input-field pl-10 pr-9"
            />
            {queryInput && (
              <button
                type="button"
                onClick={handleClearSearch}
                aria-label="Clear search"
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted hover:text-ink p-1"
              >
                <X size={14} />
              </button>
            )}
          </div>

          <div className="flex gap-2">
            <div className="relative">
              <Globe size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
              <select
                value={region}
                onChange={(e) => setRegion(e.target.value)}
                aria-label="Region"
                className="input-field pl-9 pr-7 py-2.5 text-xs bg-paper2 cursor-pointer w-44"
              >
                {REGIONS.map((r) => (
                  <option key={r.code} value={r.code}>
                    {r.code} · {r.label}
                  </option>
                ))}
              </select>
            </div>

            <select
              value={sort}
              onChange={(e) => setSort(e.target.value)}
              aria-label="Sort by"
              className="input-field px-3 py-2.5 text-xs bg-paper2 cursor-pointer"
            >
              {SORTS.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>

            <button type="submit" disabled={loading} className="btn-primary px-4 py-2.5 text-xs shrink-0">
              search
            </button>
          </div>
        </form>

        {/* Category Pills + Quick Topics */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-rule">
          <div className="flex flex-wrap gap-1.5">
            {CATEGORIES.map((cat) => {
              const selected = !activeQuery && category === cat.id;
              return (
                <button
                  key={cat.id}
                  type="button"
                  onClick={() => {
                    setActiveQuery('');
                    setQueryInput('');
                    setCategory(cat.id);
                  }}
                  className={`px-3 py-1.5 rounded-input text-xs lowercase transition-colors ${
                    selected
                      ? 'bg-brass text-brassink font-medium'
                      : 'bg-paper3 text-ink2 hover:text-ink'
                  }`}
                >
                  {cat.label}
                </button>
              );
            })}
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            <span className="readout mr-1">topics:</span>
            {QUICK_TOPICS.map((topic) => (
              <button
                key={topic}
                type="button"
                onClick={() => handleTopicClick(topic)}
                className={`px-2.5 py-1 rounded-input text-xs transition-colors border ${
                  activeQuery.toLowerCase() === topic.toLowerCase()
                    ? 'border-brass text-brass bg-brass/10'
                    : 'border-rule text-muted hover:text-ink2 hover:border-rule2'
                }`}
              >
                {topic}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Status bar */}
      {!loading && !error && (
        <div className="flex items-center justify-between text-xs text-muted mb-4">
          <span className="readout">
            {activeQuery
              ? `results for "${activeQuery}" · ${items.length} videos · ${region}`
              : `${CATEGORIES.find((c) => c.id === category)?.label || 'Trending'} · ${items.length} videos · ${region}`}
          </span>
          {meta.source && (
            <span className="readout opacity-75">
              {meta.cached ? 'cached · ' : 'live · '}
              {meta.source.replace('_', ' ')}
            </span>
          )}
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="card p-4 border-danger/40 bg-danger/10 text-danger text-sm flex items-center gap-2.5 mb-6">
          <AlertCircle size={16} className="shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-20 gap-3">
          <Loader2 size={28} className="animate-spin text-brass" />
          <p className="text-xs text-muted lowercase">scanning youtube right now…</p>
        </div>
      )}

      {/* Empty */}
      {!loading && !error && items.length === 0 && (
        <div className="text-center py-20 text-muted card p-8">
          <Flame size={36} className="mx-auto mb-3 text-muted opacity-60" />
          <p className="text-sm lowercase mb-3">No videos found for this query.</p>
          {activeQuery && (
            <button type="button" onClick={handleClearSearch} className="btn-ghost px-3 py-1.5 text-xs">
              back to trending
            </button>
          )}
        </div>
      )}

      {/* Results Grid */}
      {!loading && items.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 pb-8">
          {items.map((video) => (
            <div key={video.video_id} className="card card-hover overflow-hidden flex flex-col">
              <a
                href={video.url}
                target="_blank"
                rel="noopener noreferrer"
                className="relative aspect-video bg-black block overflow-hidden group"
              >
                <img
                  src={video.thumbnail}
                  alt={video.title}
                  loading="lazy"
                  className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform duration-300"
                />
                {video.duration_text && (
                  <span
                    className={`absolute bottom-2 right-2 px-1.5 py-0.5 rounded text-[11px] font-mono ${
                      video.is_short
                        ? 'bg-coral text-white font-medium'
                        : 'bg-black/80 text-white'
                    }`}
                  >
                    {video.duration_text}
                  </span>
                )}
              </a>

              <div className="p-4 flex-1 flex flex-col justify-between gap-3">
                <div>
                  <h3 className="text-sm font-medium text-ink line-clamp-2 mb-1.5" title={video.title}>
                    {video.title}
                  </h3>
                  <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted">
                    {video.channel && <span className="truncate max-w-[160px] text-ink2">{video.channel}</span>}
                    {video.views_text && <span>· {video.views_text}</span>}
                    {video.published_text && <span>· {video.published_text}</span>}
                  </div>
                </div>

                <div className="flex items-center gap-2 pt-2 border-t border-rule">
                  {onClipUrl && (
                    <button
                      type="button"
                      onClick={() => onClipUrl(video.url)}
                      className="btn-primary flex-1 py-2 px-3 text-xs justify-center"
                      title="Load this video into Clip Generator"
                    >
                      <Scissors size={13} />
                      <span>clip this</span>
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={() => handleCopy(video)}
                    className="btn-ghost p-2 text-xs"
                    title="Copy YouTube URL"
                    aria-label="Copy YouTube URL"
                  >
                    {copiedId === video.video_id ? <Check size={14} className="text-ok" /> : <Copy size={14} />}
                  </button>

                  <a
                    href={video.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="btn-ghost p-2 text-xs"
                    title="Open on YouTube"
                    aria-label="Open on YouTube"
                  >
                    <ExternalLink size={14} />
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
