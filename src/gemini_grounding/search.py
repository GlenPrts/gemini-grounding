import argparse
import os
import sys
import json
import time
import random
import logging
import requests
import concurrent.futures
from functools import lru_cache
from urllib.parse import urlparse
from fake_useragent import UserAgent
from cachetools import cached, TTLCache
from cachetools.keys import hashkey
from threading import RLock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)


def create_session():
    session = requests.Session()
    try:
        ua = UserAgent()
        user_agent = ua.random
    except Exception:
        user_agent = "GeminiGrounding/1.0"

    session.headers.update({"User-Agent": user_agent})

    retry_strategy = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"],
    )
    adapter = HTTPAdapter(
        pool_connections=20, pool_maxsize=20, max_retries=retry_strategy
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


session = create_session()

# Cache configuration
# Default TTL: 1 hour (3600 seconds)
try:
    SEARCH_CACHE_TTL = int(os.environ.get("GEMINI_CACHE_TTL", "3600"))
except ValueError:
    print(
        "Warning: Invalid GEMINI_CACHE_TTL value, defaulting to 3600 seconds.",
        file=sys.stderr,
    )
    SEARCH_CACHE_TTL = 3600

try:
    SEARCH_CACHE_MAXSIZE = int(os.environ.get("GEMINI_CACHE_MAXSIZE", "100"))
except ValueError:
    print(
        "Warning: Invalid GEMINI_CACHE_MAXSIZE value, defaulting to 100.",
        file=sys.stderr,
    )
    SEARCH_CACHE_MAXSIZE = 100

search_cache = TTLCache(maxsize=SEARCH_CACHE_MAXSIZE, ttl=SEARCH_CACHE_TTL)
search_cache_lock = RLock()


@lru_cache(maxsize=1000)
def resolve_url(url):
    """
    Resolve Google's grounding redirect URLs to their original destination.
    Uses HEAD request to minimize bandwidth. Caches results to avoid redundant requests.
    Returns the original URL if resolution succeeds, otherwise returns the input URL.
    """
    if not url.startswith(
        "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"
    ):
        return url

    proxy_base = os.environ.get("GEMINI_PROXY_URL")

    if proxy_base:
        if proxy_base.endswith("/"):
            proxy_base = proxy_base[:-1]

        proxy_url = f"{proxy_base}/{url}"

        try:
            response = session.head(
                proxy_url,
                allow_redirects=False,
                timeout=5,
                headers={"X-Proxy-Manual-Redirect": "true"},
            )

            final_url = response.headers.get("X-Final-Url")
            if final_url:
                return final_url

            if 300 <= response.status_code < 400:
                location = response.headers.get("Location")
                if location and not location.startswith(proxy_base):
                    return location

            if response.status_code == 200:
                link_header = response.headers.get("Link")
                if link_header:
                    import re

                    match = re.search(r'<([^>]+)>;\s*rel="canonical"', link_header)
                    if match:
                        return match.group(1)

        except requests.RequestException:
            pass
        except Exception as e:
            logger.warning(f"Unexpected error resolving proxy URL {url}: {e}")
            pass
    else:
        try:
            response = session.head(
                url,
                allow_redirects=True,
                timeout=5,
            )
            if response.status_code == 200:
                return response.url
        except requests.RequestException:
            pass
        except Exception as e:
            logger.warning(f"Unexpected error resolving URL {url}: {e}")
            pass

    return url


def resolve_urls_concurrently(uris):
    """
    Resolves a list of URLs concurrently using a thread pool.
    Returns a dictionary mapping original URI to resolved URI.
    """
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_uri = {executor.submit(resolve_url, uri): uri for uri in uris}
        for future in concurrent.futures.as_completed(future_to_uri):
            uri = future_to_uri[future]
            try:
                results[uri] = future.result()
            except Exception as e:
                logger.error(f"Error resolving URI {uri}: {e}")
                results[uri] = uri
    return results


def _search_cache_key(query, model, api_key, base_url, *args, **kwargs):
    """
    Generate a cache key for the search function.
    We exclude retry configurations and debug flags from the cache key.
    """
    return hashkey(query, model, base_url)


@cached(cache=search_cache, key=_search_cache_key, lock=search_cache_lock)
def _perform_search(
    query,
    model,
    api_key,
    base_url,
    retry_count,
    retry_delay,
    search_delay_min,
    search_delay_max,
    debug,
):
    """
    Internal function to perform the actual search request.
    This function is cached.
    """
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    if not query or not query.strip():
        raise ValueError("搜索查询不能为空")

    # 防止负数导致循环不执行而返回 None
    retry_count = max(0, retry_count)
    # 防止超大延迟导致服务长时间挂起
    MAX_RETRY_DELAY = 60
    retry_delay = min(max(0, retry_delay), MAX_RETRY_DELAY)

    url = f"{base_url}/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    tools = [{"googleSearch": {}}]
    payload = {
        "contents": [{"parts": [{"text": query}]}],
        "tools": tools,
        "generationConfig": {"temperature": 0.0},
    }

    if debug:
        print(json.dumps(payload, indent=2))

    if search_delay_max > search_delay_min and search_delay_max > 0:
        sleep_time = random.uniform(search_delay_min, search_delay_max)
        if sleep_time > 0:
            if debug:
                logger.info(f"Waiting {sleep_time:.2f}s before search...")
            time.sleep(sleep_time)

    for attempt in range(retry_count + 1):
        try:
            response = session.post(
                url, json=payload, headers=headers, timeout=(10, 60)
            )

            if response.status_code == 429:
                # 忽略 Retry-After，使用指数退避策略重试
                wait_time = min(retry_delay * (2**attempt) + random.uniform(0, 1), 60)

                logger.warning(f"Rate limited (429). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()

            # Process response
            full_text = ""
            all_grounding_chunks = []
            all_supports = []

            try:
                data = response.json()
                if debug:
                    print(json.dumps(data, indent=2))

                if isinstance(data, dict):
                    candidates = data.get("candidates", [])
                elif isinstance(data, list) and len(data) > 0:
                    candidates = data[0].get("candidates", [])
                else:
                    candidates = []

                if candidates:
                    candidate = candidates[0]
                    content_parts = candidate.get("content", {}).get("parts", [])
                    for part in content_parts:
                        if "text" in part:
                            full_text += part["text"]

                    grounding_metadata = candidate.get("groundingMetadata", {})
                    g_chunks = grounding_metadata.get("groundingChunks", [])
                    if g_chunks:
                        all_grounding_chunks.extend(g_chunks)

                    g_supports = grounding_metadata.get("groundingSupports", [])
                    for support in g_supports:
                        indices = support.get("groundingChunkIndices", [])
                        segment = support.get("segment", {})
                        uris = []
                        for idx in indices:
                            if idx < len(all_grounding_chunks):
                                u = all_grounding_chunks[idx].get("web", {}).get("uri")
                                if u:
                                    uris.append(u)
                        all_supports.append({"segment": segment, "uris": uris})

            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON response")
                continue

            uris_to_resolve = set()
            for chunk in all_grounding_chunks:
                web = chunk.get("web", {})
                uri = web.get("uri")
                if uri:
                    uris_to_resolve.add(uri)

            resolved_map = resolve_urls_concurrently(list(uris_to_resolve))

            final_sources = []
            url_to_id = {}
            original_url_to_id = {}
            next_id = 1

            for chunk in all_grounding_chunks:
                web = chunk.get("web", {})
                uri = web.get("uri")
                title = web.get("title")

                if uri:
                    resolved = resolved_map.get(uri, uri)
                    if resolved not in url_to_id:
                        url_to_id[resolved] = next_id
                        final_sources.append(
                            {"id": next_id, "title": title, "url": resolved}
                        )
                        next_id += 1
                    original_url_to_id[uri] = url_to_id[resolved]

            # Sort supports by endIndex in descending order to insert citations from end to start
            # This prevents index shifting issues
            all_supports.sort(
                key=lambda x: x["segment"].get("endIndex", 0), reverse=True
            )

            for support in all_supports:
                end_idx = support["segment"].get("endIndex")
                uris = support["uris"]

                # Verify end_idx is valid
                if end_idx is not None:
                    ids = []
                    for u in uris:
                        if u in original_url_to_id:
                            ids.append(original_url_to_id[u])
                    ids = sorted(list(set(ids)))

                    if ids:
                        citation = f" [{', '.join(map(str, ids))}]"
                        # Insert citation into the string at the character index
                        # Since we iterate in reverse order, previous insertions don't affect current index
                        if end_idx <= len(full_text):
                            full_text = (
                                full_text[:end_idx] + citation + full_text[end_idx:]
                            )

            return {"text": full_text, "sources": final_sources}

        except requests.RequestException as e:
            if attempt < retry_count:
                wait_time = min(retry_delay * (2**attempt) + random.uniform(0, 1), 60)
                if debug:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{retry_count + 1}): {e}. Retrying in {wait_time:.2f}s..."
                    )
                time.sleep(wait_time)
            else:
                raise e


def search(
    query,
    model=None,
    api_key=None,
    base_url=None,
    retry_count=None,
    retry_delay=None,
    search_delay_min=None,
    search_delay_max=None,
    debug=False,
):
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if base_url is None:
        base_url = os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"
        )
    if retry_count is None:
        retry_count = int(os.environ.get("GEMINI_RETRY_COUNT", "3"))
    if retry_delay is None:
        retry_delay = float(os.environ.get("GEMINI_RETRY_DELAY", "5"))
    if search_delay_min is None:
        search_delay_min = float(os.environ.get("GEMINI_SEARCH_DELAY_MIN", "0.0"))
    if search_delay_max is None:
        search_delay_max = float(os.environ.get("GEMINI_SEARCH_DELAY_MAX", "0.0"))

    return _perform_search(
        query=query,
        model=model,
        api_key=api_key,
        base_url=base_url,
        retry_count=retry_count,
        retry_delay=retry_delay,
        search_delay_min=search_delay_min,
        search_delay_max=search_delay_max,
        debug=debug,
    )


def main():
    parser = argparse.ArgumentParser(description="Google Search via Gemini API")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use (default: GEMINI_MODEL env var or gemini-2.5-flash)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print payload and exit")
    parser.add_argument("--debug", action="store_true", help="Print full response JSON")
    args = parser.parse_args()

    # Reuse environment variable loading or pass arguments directly
    try:
        if args.dry_run:
            # Just print the payload structure (simplified for dry run)
            api_key = os.environ.get("GEMINI_API_KEY", "dummy")
            base_url = os.environ.get(
                "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"
            )
            url = f"{base_url}/v1beta/models/{args.model}:generateContent"
            tools = [{"googleSearch": {}}]
            payload = {
                "contents": [{"parts": [{"text": args.query}]}],
                "tools": tools,
                "generationConfig": {"temperature": 0.0},
            }
            print(json.dumps(payload, indent=2))
            return

        result = search(args.query, model=args.model, debug=args.debug)
        print(result["text"])
        if result["sources"]:
            print("\n\n## Sources\n")
            for src in result["sources"]:
                print(f"{src['id']}. [{src['title']}]({src['url']})")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
