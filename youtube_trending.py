"""YouTube Trending & Live Search API router.

Fetches what is trending on YouTube right now (by region and category) and
supports live YouTube search so users can discover viral videos to turn into
Shorts. Works out-of-the-box without an API key via YouTube's public pages
(guarded by ``ssrf_guard``) with a ``yt-dlp`` fallback, and also supports
``YOUTUBE_API_KEY`` (YouTube Data API v3) when configured.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any
from urllib.parse import quote_plus

import httpx
from fastapi import APIRouter, Query

import ssrf_guard

logger = logging.getLogger(__name__)

router = APIRouter()

REGIONS: dict[str, str] = {
    "US": "United States",
    "GB": "United Kingdom",
    "IN": "India",
    "CA": "Canada",
    "AU": "Australia",
    "DE": "Germany",
    "FR": "France",
    "JP": "Japan",
    "KR": "South Korea",
    "BR": "Brazil",
    "MX": "Mexico",
    "NP": "Nepal",
    "ES": "Spain",
    "IT": "Italy",
    "ID": "Indonesia",
    "PH": "Philippines",
}

CATEGORIES: dict[str, dict[str, str]] = {
    "all": {"label": "All", "bp": "", "yt_cat_id": "", "search_hint": "trending"},
    "music": {"label": "Music", "bp": "4gINGgt5dG1hX2NoYXJ0cw%3D%3D", "yt_cat_id": "10", "search_hint": "trending music"},
    "gaming": {"label": "Gaming", "bp": "4gIcGhpnYW1pbmdfY29ycHVzX21vc3RfcG9wdWxhcg%3D%3D", "yt_cat_id": "20", "search_hint": "trending gaming"},
    "news": {"label": "News", "bp": "4gIOGg5uZXdzX21haW5fZmVlZA%3D%3D", "yt_cat_id": "25", "search_hint": "breaking news today"},
    "celebrity": {"label": "Celebrity & Viral", "bp": "", "yt_cat_id": "24", "search_hint": "celebrity news viral moments today"},
    "movies": {"label": "Movies", "bp": "4gIKGgh0cmFpbGVycw%3D%3D", "yt_cat_id": "30", "search_hint": "new movie trailers"},
    "shorts": {"label": "Shorts", "bp": "", "yt_cat_id": "", "search_hint": "viral youtube shorts"},
}

# YouTube search sort `sp` parameters
_SEARCH_SP: dict[str, str] = {
    "relevance": "",
    "trending": "",
    "views": "CAM%253D",
    "date": "CAI%253D",
}

_CACHE_TTL_SECONDS = 300
_CACHE_MAX_ENTRIES = 128
_cache_lock = threading.Lock()
_cache: dict[tuple[Any, ...], tuple[float, dict[str, Any]]] = {}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Cookie": "CONSENT=YES+cb.20210328-17-p0.en+FX+667",
}


def _text_of(node: Any) -> str:
    """Extract plain text from YouTube's simpleText / runs objects."""
    if not node:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, dict):
        if "simpleText" in node and isinstance(node["simpleText"], str):
            return node["simpleText"].strip()
        if "runs" in node and isinstance(node["runs"], list):
            return "".join(
                r.get("text", "") for r in node["runs"] if isinstance(r, dict)
            ).strip()
        if "content" in node and isinstance(node["content"], str):
            return node["content"].strip()
    return ""


def parse_duration_seconds(text: str) -> int | None:
    """Convert '12:34', '1:02:03', or ISO-8601 'PT12M34S' to seconds."""
    if not text:
        return None
    text = text.strip()
    if text.upper().startswith("PT"):
        m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", text.upper())
        if not m:
            return None
        h, mins, s = (int(x or 0) for x in m.groups())
        return h * 3600 + mins * 60 + s
    parts = text.split(":")
    if not all(p.isdigit() for p in parts):
        return None
    total = 0
    for p in parts:
        total = total * 60 + int(p)
    return total


def parse_view_count(text: str) -> int | None:
    """Parse '1.2M views', '450K views', or '12,345' into an integer."""
    if not text:
        return None
    cleaned = text.lower().replace(",", "").replace("views", "").replace("view", "").strip()
    m = re.search(r"(\d+(?:\.\d+)?)\s*([kmb])?", cleaned)
    if not m:
        return None
    num = float(m.group(1))
    suffix = m.group(2)
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(suffix or "", 1)
    return int(num * mult)


def format_duration(seconds: int | None) -> str:
    if seconds is None or seconds < 0:
        return ""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def extract_yt_initial_data(html: str) -> dict[str, Any] | None:
    """Extract the `ytInitialData` JSON blob from a YouTube HTML page."""
    if not html:
        return None
    markers = ("var ytInitialData = ", 'window["ytInitialData"] = ', "ytInitialData = ")
    start = -1
    for marker in markers:
        idx = html.find(marker)
        if idx != -1:
            start = idx + len(marker)
            break
    if start == -1:
        return None
    brace_start = html.find("{", start)
    if brace_start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(html, brace_start)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _best_thumbnail(thumbs: Any, video_id: str) -> str:
    if isinstance(thumbs, list) and thumbs:
        last = thumbs[-1]
        if isinstance(last, dict) and last.get("url"):
            url = str(last["url"])
            if url.startswith("//"):
                return "https:" + url
            return url
    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def _normalize_video_renderer(v: dict[str, Any]) -> dict[str, Any] | None:
    vid = v.get("videoId")
    if not vid or not isinstance(vid, str):
        return None
    title = _text_of(v.get("title"))
    if not title:
        return None
    channel = _text_of(v.get("ownerText") or v.get("shortBylineText") or v.get("longBylineText"))
    channel_url = ""
    owner_runs = (v.get("ownerText") or v.get("shortBylineText") or {}).get("runs") or []
    if owner_runs and isinstance(owner_runs[0], dict):
        ep = owner_runs[0].get("navigationEndpoint", {}).get("commandMetadata", {}).get("webCommandMetadata", {})
        if ep.get("url"):
            channel_url = "https://www.youtube.com" + ep["url"]

    views_text = _text_of(v.get("viewCountText") or v.get("shortViewCountText"))
    published_text = _text_of(v.get("publishedTimeText"))
    duration_text = _text_of(v.get("lengthText"))
    duration_seconds = parse_duration_seconds(duration_text)
    nav_url = (
        v.get("navigationEndpoint", {})
        .get("commandMetadata", {})
        .get("webCommandMetadata", {})
        .get("url", "")
    )
    is_short = "/shorts/" in nav_url or (duration_seconds is not None and duration_seconds <= 60)

    badges: list[str] = []
    for b in v.get("badges") or []:
        label = (b.get("metadataBadgeRenderer") or {}).get("label")
        if label:
            badges.append(str(label))

    thumbs = (v.get("thumbnail") or {}).get("thumbnails")
    return {
        "video_id": vid,
        "title": title,
        "channel": channel,
        "channel_url": channel_url,
        "thumbnail": _best_thumbnail(thumbs, vid),
        "url": f"https://www.youtube.com/shorts/{vid}" if is_short else f"https://www.youtube.com/watch?v={vid}",
        "views_text": views_text,
        "view_count": parse_view_count(views_text),
        "published_text": published_text,
        "duration_text": duration_text or ("Shorts" if is_short else ""),
        "duration_seconds": duration_seconds,
        "is_short": is_short,
        "badges": badges[:3],
    }


def _normalize_shorts_lockup(s: dict[str, Any]) -> dict[str, Any] | None:
    ep = s.get("onTap", {}).get("innertubeCommand", {})
    vid = (
        ep.get("reelWatchEndpoint", {}).get("videoId")
        or s.get("entityId", "").removeprefix("shorts-shelf-item-")
    )
    if not vid or not isinstance(vid, str) or len(vid) < 6:
        return None
    meta = s.get("overlayMetadata") or {}
    title = _text_of(meta.get("primaryText")) or _text_of(s.get("accessibilityText"))
    if not title:
        return None
    views_text = _text_of(meta.get("secondaryText"))
    thumbs = (s.get("thumbnailViewModel", {}).get("thumbnailViewModel", {}).get("image", {}) or {}).get("sources")
    return {
        "video_id": vid,
        "title": title,
        "channel": "",
        "channel_url": "",
        "thumbnail": _best_thumbnail(thumbs, vid),
        "url": f"https://www.youtube.com/shorts/{vid}",
        "views_text": views_text,
        "view_count": parse_view_count(views_text),
        "published_text": "",
        "duration_text": "Shorts",
        "duration_seconds": None,
        "is_short": True,
        "badges": ["Shorts"],
    }


def collect_videos_from_yt_data(data: Any, limit: int = 24) -> list[dict[str, Any]]:
    """Recursively collect normalized video items from YouTube's ytInitialData."""
    items: list[dict[str, Any]] = []
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if len(items) >= limit:
            return
        if isinstance(node, list):
            for el in node:
                walk(el)
                if len(items) >= limit:
                    return
            return
        if not isinstance(node, dict):
            return

        for key in ("videoRenderer", "gridVideoRenderer", "compactVideoRenderer"):
            if key in node and isinstance(node[key], dict):
                item = _normalize_video_renderer(node[key])
                if item and item["video_id"] not in seen:
                    seen.add(item["video_id"])
                    items.append(item)
                    if len(items) >= limit:
                        return

        if "shortsLockupViewModel" in node and isinstance(node["shortsLockupViewModel"], dict):
            item = _normalize_shorts_lockup(node["shortsLockupViewModel"])
            if item and item["video_id"] not in seen:
                seen.add(item["video_id"])
                items.append(item)
                if len(items) >= limit:
                    return

        for val in node.values():
            walk(val)
            if len(items) >= limit:
                return

    walk(data)
    return items


def _fetch_youtube_html(url: str) -> str:
    with ssrf_guard.guarded():
        with httpx.Client(headers=_HEADERS, timeout=12.0, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text


def _fetch_via_ytdlp(query: str, limit: int) -> list[dict[str, Any]]:
    import yt_dlp

    opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
        "skip_download": True,
        "noplaylist": True,
    }
    search_spec = f"ytsearch{limit}:{query}"
    with ssrf_guard.guarded(), yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(search_spec, download=False)
    entries = (info or {}).get("entries") or []
    items: list[dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        vid = e.get("id")
        title = e.get("title")
        if not vid or not title:
            continue
        dur = e.get("duration")
        dur_int = int(dur) if isinstance(dur, (int, float)) else None
        vc = e.get("view_count")
        vc_int = int(vc) if isinstance(vc, (int, float)) else None
        is_short = dur_int is not None and dur_int <= 60
        views_text = f"{vc_int:,} views" if vc_int is not None else ""
        items.append({
            "video_id": vid,
            "title": title,
            "channel": e.get("channel") or e.get("uploader") or "",
            "channel_url": e.get("channel_url") or e.get("uploader_url") or "",
            "thumbnail": _best_thumbnail(e.get("thumbnails"), vid),
            "url": f"https://www.youtube.com/shorts/{vid}" if is_short else f"https://www.youtube.com/watch?v={vid}",
            "views_text": views_text,
            "view_count": vc_int,
            "published_text": e.get("upload_date") or "",
            "duration_text": format_duration(dur_int) or ("Shorts" if is_short else ""),
            "duration_seconds": dur_int,
            "is_short": is_short,
            "badges": ["Shorts"] if is_short else [],
        })
        if len(items) >= limit:
            break
    return items


def _fetch_via_data_api(api_key: str, region: str, category: str, q: str, sort: str, limit: int) -> list[dict[str, Any]]:
    base = "https://www.googleapis.com/youtube/v3"
    items: list[dict[str, Any]] = []
    with ssrf_guard.guarded():
        with httpx.Client(timeout=12.0) as client:
            if not q:
                params: dict[str, Any] = {
                    "part": "snippet,contentDetails,statistics",
                    "chart": "mostPopular",
                    "regionCode": region,
                    "maxResults": min(limit, 50),
                    "key": api_key,
                }
                cat_id = CATEGORIES.get(category, {}).get("yt_cat_id")
                if cat_id:
                    params["videoCategoryId"] = cat_id
                resp = client.get(f"{base}/videos", params=params)
                resp.raise_for_status()
                rows = resp.json().get("items") or []
            else:
                order_map = {"views": "viewCount", "date": "date", "relevance": "relevance", "trending": "relevance"}
                params = {
                    "part": "snippet",
                    "type": "video",
                    "q": q,
                    "regionCode": region,
                    "order": order_map.get(sort, "relevance"),
                    "maxResults": min(limit, 50),
                    "key": api_key,
                }
                resp = client.get(f"{base}/search", params=params)
                resp.raise_for_status()
                search_rows = resp.json().get("items") or []
                ids = [r["id"]["videoId"] for r in search_rows if r.get("id", {}).get("videoId")]
                if not ids:
                    return []
                vresp = client.get(
                    f"{base}/videos",
                    params={
                        "part": "snippet,contentDetails,statistics",
                        "id": ",".join(ids),
                        "key": api_key,
                    },
                )
                vresp.raise_for_status()
                rows = vresp.json().get("items") or []

    for row in rows:
        vid = row.get("id")
        if isinstance(vid, dict):
            vid = vid.get("videoId")
        if not vid:
            continue
        snip = row.get("snippet") or {}
        stats = row.get("statistics") or {}
        cd = row.get("contentDetails") or {}
        dur_sec = parse_duration_seconds(cd.get("duration", ""))
        vc_raw = stats.get("viewCount")
        vc_int = int(vc_raw) if vc_raw and str(vc_raw).isdigit() else None
        is_short = dur_sec is not None and dur_sec <= 60
        thumbs = snip.get("thumbnails") or {}
        thumb = (
            (thumbs.get("high") or thumbs.get("medium") or thumbs.get("default") or {}).get("url")
            or f"https://i.ytimg.com/vi/{vid}/hqdefault.jpg"
        )
        items.append({
            "video_id": vid,
            "title": snip.get("title") or "",
            "channel": snip.get("channelTitle") or "",
            "channel_url": f"https://www.youtube.com/channel/{snip['channelId']}" if snip.get("channelId") else "",
            "thumbnail": thumb,
            "url": f"https://www.youtube.com/shorts/{vid}" if is_short else f"https://www.youtube.com/watch?v={vid}",
            "views_text": f"{vc_int:,} views" if vc_int is not None else "",
            "view_count": vc_int,
            "published_text": (snip.get("publishedAt") or "")[:10],
            "duration_text": format_duration(dur_sec) or ("Shorts" if is_short else ""),
            "duration_seconds": dur_sec,
            "is_short": is_short,
            "badges": ["Shorts"] if is_short else [],
        })
    return items[:limit]


def fetch_trending_or_search(
    region: str = "US",
    category: str = "all",
    q: str = "",
    sort: str = "trending",
    limit: int = 24,
) -> dict[str, Any]:
    region = region.upper() if region and region.upper() in REGIONS else "US"
    category = category.lower() if category and category.lower() in CATEGORIES else "all"
    sort = sort.lower() if sort and sort.lower() in _SEARCH_SP else "trending"
    q_clean = (q or "").strip()
    limit = max(1, min(int(limit), 40))

    cache_key = (region, category, q_clean.lower(), sort, limit)
    now = time.time()
    with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and (now - cached[0]) < _CACHE_TTL_SECONDS:
            return {**cached[1], "cached": True}

    items: list[dict[str, Any]] = []
    source = "youtube_feed"

    api_key = (os.environ.get("YOUTUBE_API_KEY") or "").strip()
    if api_key and category != "shorts":
        try:
            items = _fetch_via_data_api(api_key, region, category, q_clean, sort, limit)
            source = "youtube_api"
        except Exception as exc:
            logger.warning("YouTube Data API fetch failed, falling back: %s", exc)

    if not items:
        try:
            if q_clean:
                source = "youtube_search"
                sp = _SEARCH_SP.get(sort, "")
                url = f"https://www.youtube.com/results?search_query={quote_plus(q_clean)}&gl={region}"
                if sp:
                    url += f"&sp={sp}"
            elif category in ("shorts", "celebrity"):
                source = "youtube_search"
                hint = CATEGORIES[category]["search_hint"]
                url = f"https://www.youtube.com/results?search_query={quote_plus(hint)}&gl={region}&sp=CAI%253D"
            else:
                source = "youtube_feed"
                bp = CATEGORIES[category]["bp"]
                url = f"https://www.youtube.com/feed/trending?gl={region}"
                if bp:
                    url += f"&bp={bp}"

            html = _fetch_youtube_html(url)
            yt_data = extract_yt_initial_data(html)
            if yt_data:
                items = collect_videos_from_yt_data(yt_data, limit=limit)
        except Exception as exc:
            logger.warning("YouTube HTML fetch failed, falling back to yt-dlp: %s", exc)

    if not items:
        source = "yt_dlp"
        hint = q_clean or CATEGORIES[category]["search_hint"]
        try:
            items = _fetch_via_ytdlp(hint, limit=limit)
        except Exception as exc:
            logger.warning("yt-dlp fallback failed: %s", exc)
            items = []

    if sort == "views" and items:
        items.sort(key=lambda x: x.get("view_count") or 0, reverse=True)

    payload: dict[str, Any] = {
        "items": items,
        "source": source,
        "region": region,
        "category": category,
        "query": q_clean,
        "cached": False,
        "fetched_at": int(now),
    }

    if items:
        with _cache_lock:
            if len(_cache) >= _CACHE_MAX_ENTRIES:
                oldest_key = min(_cache, key=lambda k: _cache[k][0])
                _cache.pop(oldest_key, None)
            _cache[cache_key] = (now, payload)

    return payload


@router.get("/api/youtube/trending")
async def youtube_trending(
    q: str = Query("", max_length=120),
    region: str = Query("US", max_length=4),
    category: str = Query("all", max_length=20),
    sort: str = Query("trending", max_length=20),
    limit: int = Query(24, ge=1, le=40),
):
    """Return trending YouTube videos right now or live search results."""
    return fetch_trending_or_search(
        region=region,
        category=category,
        q=q,
        sort=sort,
        limit=limit,
    )
