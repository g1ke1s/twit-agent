"""
agent/tools.py — Research tool wrappers: Tavily, Exa, Jina, arXiv, HN Algolia, Twitter/X.
All calls are async with rate-limit guards.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from urllib.parse import quote_plus

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rate limit semaphores
# ---------------------------------------------------------------------------

RATE_LIMITS: dict[str, dict] = {
    "gemini_flash":        {"rpm": 15,  "sleep_between": 0.5},
    "groq_llama":          {"rpm": 30,  "sleep_between": 0.3},
    "mistral_small":       {"rpm": 5,   "sleep_between": 2.0},
    "openrouter_deepseek": {"rpm": 10,  "sleep_between": 1.0},
    "openrouter_qwen":     {"rpm": 10,  "sleep_between": 1.0},
    "tavily":              {"rps": 5,   "sleep_between": 0.5},
    "exa":                 {"rpm": 10,  "sleep_between": 1.0},
    "jina":                {"sleep_between": 0.3, "max_per_run": 5},
}

_semaphores: dict[str, asyncio.Semaphore] = {
    k: asyncio.Semaphore(1) for k in RATE_LIMITS
}


async def rate_limited_call(model_key: str, coro):
    sem = _semaphores.get(model_key, asyncio.Semaphore(1))
    async with sem:
        result = await coro
        sleep = RATE_LIMITS.get(model_key, {}).get("sleep_between", 0.0)
        if sleep:
            await asyncio.sleep(sleep)
        return result


# ---------------------------------------------------------------------------
# Jina Reader  (free, no API key)
# ---------------------------------------------------------------------------

async def jina_fetch(url: str, timeout: int = 25) -> str:
    """
    Fetch a URL via Jina Reader (r.jina.ai/{url}).
    Returns clean markdown text, capped at 4000 chars.
    """
    jina_url = f"https://r.jina.ai/{url}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                jina_url,
                headers={"Accept": "text/plain", "X-No-Cache": "true"},
            )
            resp.raise_for_status()
            text = resp.text[:4000]
            logger.debug("Jina fetched %s — %d chars", url, len(text))
            return text
    except Exception as exc:
        logger.warning("jina_fetch failed for %s: %s", url, exc)
        return ""


# ---------------------------------------------------------------------------
# Tavily  (web search)
# ---------------------------------------------------------------------------

async def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        logger.debug("TAVILY_API_KEY not set — skipping")
        return []

    payload = {
        "api_key":       api_key,
        "query":         query,
        "search_depth":  "advanced",
        "max_results":   max_results,
        "include_raw_content": False,
    }

    async def _call():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            return resp.json().get("results", [])

    try:
        results = await rate_limited_call("tavily", _call())
        logger.debug("Tavily returned %d results for: %s", len(results), query[:60])
        return results
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Exa.ai  (semantic search)
# ---------------------------------------------------------------------------

async def exa_search(query: str, num_results: int = 5) -> list[dict]:
    api_key = os.getenv("EXA_API_KEY", "")
    if not api_key:
        logger.warning("EXA_API_KEY not set — skipping")
        return []

    payload = {
        "query":         query,
        "num_results":   num_results,
        "use_autoprompt": True,
        "contents":      {"text": True},
    }

    async def _call():
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.exa.ai/search",
                json=payload,
                headers={"x-api-key": api_key},
            )
            resp.raise_for_status()
            return resp.json().get("results", [])

    try:
        results = await rate_limited_call("exa", _call())
        logger.debug("Exa returned %d results for: %s", len(results), query[:60])
        return results
    except Exception as exc:
        logger.warning("Exa search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Hacker News Algolia  (free, no key)
# ---------------------------------------------------------------------------

async def hn_search(query: str, min_score: int = 100, max_items: int = 5) -> list[dict]:
    encoded = quote_plus(query)
    url = f"https://hn.algolia.com/api/v1/search?query={encoded}&tags=story&hitsPerPage=20"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])

        results = []
        for h in hits:
            pts = h.get("points") or 0
            if pts >= min_score:
                results.append({
                    "url":     h.get("url") or f"https://news.ycombinator.com/item?id={h['objectID']}",
                    "title":   h.get("title", ""),
                    "content": h.get("story_text") or "",
                    "score":   pts,
                })
            if len(results) >= max_items:
                break
        logger.debug("HN search returned %d results for: %s", len(results), query[:60])
        return results
    except Exception as exc:
        logger.warning("HN search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# arXiv API  (free, no key)
# ---------------------------------------------------------------------------

async def arxiv_search(query: str, max_results: int = 5) -> list[dict]:
    encoded = quote_plus(query)
    url = (
        f"https://export.arxiv.org/api/query"
        f"?search_query=all:{encoded}&start=0&max_results={max_results}"
    )
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            text = resp.text

        entries = re.findall(r"<entry>(.*?)</entry>", text, re.DOTALL)
        results = []
        for entry in entries:
            title_m   = re.search(r"<title>(.*?)</title>",     entry, re.DOTALL)
            summary_m = re.search(r"<summary>(.*?)</summary>", entry, re.DOTALL)
            id_m      = re.search(r"<id>(.*?)</id>",           entry)
            if title_m and summary_m and id_m:
                results.append({
                    "url":              id_m.group(1).strip(),
                    "title":            title_m.group(1).strip(),
                    "content":          summary_m.group(1).strip()[:2000],
                    "source_type":      "arxiv",
                    "credibility_score": 8.5,
                })
        logger.debug("arXiv returned %d results for: %s", len(results), query[:60])
        return results
    except Exception as exc:
        logger.warning("arXiv search failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Twitter/X scraping
# ---------------------------------------------------------------------------

async def scrape_tweet(url: str) -> str:
    """
    Scrape tweet text using a three-stage fallback:
    1. Playwright (headless Chrome) — most reliable, requires browser binary
    2. Jina Reader — works for most public tweets via web rendering
    3. Direct HTTP with browser User-Agent — last resort

    Returns extracted text or empty string.
    """
    # Stage 1: Playwright
    playwright_result = await _scrape_with_playwright(url)
    if playwright_result:
        logger.info("Tweet scraped via Playwright: %s", url)
        return playwright_result

    # Stage 2: Jina
    jina_result = await jina_fetch(url)
    if jina_result and len(jina_result.strip()) > 20:
        logger.info("Tweet scraped via Jina: %s", url)
        return jina_result

    # Stage 3: Direct HTTP
    direct_result = await _scrape_direct(url)
    if direct_result:
        logger.info("Tweet scraped via direct HTTP: %s", url)
        return direct_result

    logger.warning("All tweet scrape methods failed for: %s", url)
    return ""


async def _scrape_with_playwright(url: str) -> str:
    """
    Use Playwright to scrape a tweet. Requires:
      pip install playwright
      playwright install chromium
    Falls back gracefully if not installed or browser unavailable.
    """
    try:
        from playwright.async_api import async_playwright  # type: ignore

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            try:
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    )
                )
                page = await context.new_page()
                await page.goto(url, wait_until="domcontentloaded", timeout=15000)
                await page.wait_for_timeout(2000)

                # Try to extract tweet text from known selectors
                selectors = [
                    "[data-testid='tweetText']",
                    "article [lang]",
                    ".tweet-text",
                ]
                for sel in selectors:
                    try:
                        el = await page.query_selector(sel)
                        if el:
                            text = await el.inner_text()
                            if text and len(text.strip()) > 10:
                                return text.strip()[:2000]
                    except Exception:
                        continue

                # Fallback: get all visible text
                body_text = await page.inner_text("body")
                return body_text[:2000] if body_text else ""
            finally:
                await browser.close()

    except ImportError:
        logger.debug("Playwright not installed — skipping")
        return ""
    except Exception as exc:
        logger.debug("Playwright scrape failed: %s", exc)
        return ""


async def _scrape_direct(url: str) -> str:
    """Direct HTTP request with a browser User-Agent."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(
                url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                },
            )
            return resp.text[:2000]
    except Exception as exc:
        logger.debug("Direct scrape failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Source scoring and deduplication
# ---------------------------------------------------------------------------

CREDIBILITY: dict[str, float] = {
    "arxiv":      8.5,
    "nber":       9.0,
    "a16z":       9.0,
    "sequoia":    9.0,
    "hn":         7.5,
    "exa":        6.5,
    "web":        6.5,
    "file":       7.0,
}


def score_source(url: str, source_type: str) -> float:
    for key, score in CREDIBILITY.items():
        if key in url.lower() or source_type == key:
            return score
    return CREDIBILITY["web"]


def deduplicate_sources(sources: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out = []
    for s in sources:
        u = s.get("url", "").rstrip("/")
        if u and u not in seen:
            seen.add(u)
            out.append(s)
    return out