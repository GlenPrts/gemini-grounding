from __future__ import annotations

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
from typing import Any
from fake_useragent import UserAgent
from cachetools import TTLCache
from cachetools.keys import hashkey
from threading import RLock
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_MAX_RETRY_ATTEMPTS = 5000


class _State:
    initialized: bool = False
    session: requests.Session
    resolve_session: requests.Session
    search_cache: TTLCache
    search_cache_lock: RLock
    resolve_timeout: float = 8.0
    resolve_retry_count: int = 1
    resolve_retry_delay: float = 0.25
    resolve_concurrency: int = 4


_state = _State()


def ensure_initialized() -> None:
    """Load .env, configure logging, and create sessions/cache on first call."""
    if _state.initialized:
        return

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    _state.session = _create_session()
    _state.resolve_session = _create_session(
        retry_total=0,
        backoff_factor=0,
        status_forcelist=[],
        allowed_methods=["HEAD", "GET"],
    )

    cache_ttl = get_int_env("GEMINI_CACHE_TTL", 3600, minimum=0)
    cache_maxsize = get_int_env("GEMINI_CACHE_MAXSIZE", 100, minimum=1)
    _state.search_cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
    _state.search_cache_lock = RLock()

    _state.resolve_timeout = get_float_env("GEMINI_RESOLVE_TIMEOUT", 8.0, minimum=0.1)
    _state.resolve_retry_count = get_int_env("GEMINI_RESOLVE_RETRY_COUNT", 1, minimum=0)
    _state.resolve_retry_delay = get_float_env(
        "GEMINI_RESOLVE_RETRY_DELAY", 0.25, minimum=0.0
    )
    _state.resolve_concurrency = get_int_env("GEMINI_RESOLVE_CONCURRENCY", 4, minimum=1)

    _state.initialized = True


def get_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default

    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False

    logger.warning("Invalid %s value '%s', defaulting to %s.", name, value, default)
    return default


def get_int_env(name: str, default: int, minimum: int = 0) -> int:
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', defaulting to %s.", name, value, default)
        return default

    if parsed < minimum:
        logger.warning("%s must be >= %s, defaulting to %s.", name, minimum, default)
        return default

    return parsed


def get_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.environ.get(name)
    if value is None:
        return default

    try:
        parsed = float(value)
    except ValueError:
        logger.warning("Invalid %s value '%s', defaulting to %s.", name, value, default)
        return default

    if parsed < minimum:
        logger.warning("%s must be >= %s, defaulting to %s.", name, minimum, default)
        return default

    return parsed


def _create_session(
    retry_total: int = 3,
    backoff_factor: float = 0.5,
    status_forcelist: list[int] | None = None,
    allowed_methods: list[str] | None = None,
) -> requests.Session:
    s = requests.Session()
    try:
        ua = UserAgent()
        user_agent = ua.random
    except Exception:
        user_agent = "GeminiGrounding/1.0"

    s.headers.update({"User-Agent": user_agent})

    if status_forcelist is None:
        status_forcelist = [429, 500, 502, 503, 504]
    if allowed_methods is None:
        allowed_methods = ["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]

    retry_strategy = Retry(
        total=retry_total,
        connect=retry_total,
        read=retry_total,
        redirect=retry_total,
        status=retry_total,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=allowed_methods,
    )
    adapter = HTTPAdapter(
        pool_connections=20, pool_maxsize=20, max_retries=retry_strategy
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _resolve_wait_time(attempt: int) -> float:
    return min(
        _state.resolve_retry_delay * (2**attempt) + random.uniform(0, 0.25),
        3,
    )


def _head_for_resolve(
    url: str, allow_redirects: bool, headers: dict[str, str] | None = None
) -> requests.Response:
    request_kwargs: dict[str, Any] = {
        "allow_redirects": allow_redirects,
        "timeout": _state.resolve_timeout,
    }
    if headers is not None:
        request_kwargs["headers"] = headers

    last_exc: requests.RequestException | None = None
    for attempt in range(_state.resolve_retry_count + 1):
        try:
            return _state.resolve_session.head(url, **request_kwargs)
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < _state.resolve_retry_count:
                time.sleep(_resolve_wait_time(attempt))
                continue
            raise
    raise last_exc or requests.RequestException(f"Failed to resolve {url}")


def _extract_proxy_final_url(
    response: requests.Response, proxy_base: str
) -> str | None:
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

    return None


@lru_cache(maxsize=1000)
def resolve_url(url: str) -> str:
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
            response = _head_for_resolve(
                proxy_url,
                allow_redirects=False,
                headers={"X-Proxy-Manual-Redirect": "true"},
            )

            resolved = _extract_proxy_final_url(response, proxy_base)
            if resolved:
                return resolved

        except requests.RequestException:
            pass
        except Exception as e:
            logger.debug("Unexpected error resolving proxy URL %s: %s", url, e)

        return url

    try:
        response = _head_for_resolve(
            url,
            allow_redirects=True,
        )
        if response.status_code == 200:
            return response.url
    except requests.RequestException:
        pass
    except Exception as e:
        logger.debug("Unexpected error resolving URL %s: %s", url, e)

    return url


def resolve_urls_concurrently(uris: list[str]) -> dict[str, str]:
    if not uris:
        return {}

    results: dict[str, str] = {}
    max_workers = min(_state.resolve_concurrency, len(uris))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_uri = {executor.submit(resolve_url, uri): uri for uri in uris}
        for future in concurrent.futures.as_completed(future_to_uri):
            uri = future_to_uri[future]
            try:
                results[uri] = future.result()
            except Exception as e:
                logger.error("Error resolving URI %s: %s", uri, e)
                results[uri] = uri
    return results


def resolve_search_options(
    model: str | None = None,
    retry_count: int | None = None,
    retry_delay: float | None = None,
    search_delay_min: float | None = None,
    search_delay_max: float | None = None,
    retry_until_success: bool | None = None,
) -> dict[str, Any]:
    if model is None:
        model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    if retry_count is None:
        retry_count = get_int_env("GEMINI_RETRY_COUNT", 3, minimum=0)
    if retry_delay is None:
        retry_delay = get_float_env("GEMINI_RETRY_DELAY", 5.0, minimum=0.0)
    if search_delay_min is None:
        search_delay_min = get_float_env("GEMINI_SEARCH_DELAY_MIN", 0.0, minimum=0.0)
    if search_delay_max is None:
        search_delay_max = get_float_env("GEMINI_SEARCH_DELAY_MAX", 0.0, minimum=0.0)
    if retry_until_success is None:
        retry_until_success = get_bool_env("GEMINI_RETRY_UNTIL_SUCCESS", False)

    return {
        "model": model,
        "retry_count": retry_count,
        "retry_delay": retry_delay,
        "search_delay_min": search_delay_min,
        "search_delay_max": search_delay_max,
        "retry_until_success": retry_until_success,
    }


def _search_cache_key(
    query: str,
    model: str,
    api_key: str,
    base_url: str,
    retry_count: int,
    retry_delay: float,
    search_delay_min: float,
    search_delay_max: float,
    retry_until_success: bool,
    debug: bool,
) -> tuple:
    return hashkey(query, model, base_url, retry_until_success)


def _should_retry(attempt: int, retry_count: int, retry_until_success: bool) -> bool:
    if retry_until_success:
        return attempt < _MAX_RETRY_ATTEMPTS
    return attempt < retry_count


def _retry_wait_time(retry_delay: float, attempt: int) -> float:
    bounded_attempt = min(max(0, attempt), 6)
    return min(retry_delay * (2**bounded_attempt) + random.uniform(0, 1), 60)


def _attempt_label(attempt: int, retry_count: int, retry_until_success: bool) -> str:
    total: str | int = (
        f"∞(max {_MAX_RETRY_ATTEMPTS})" if retry_until_success else retry_count + 1
    )
    return f"{attempt + 1}/{total}"


def _perform_search(
    query: str,
    model: str,
    api_key: str,
    base_url: str,
    retry_count: int,
    retry_delay: float,
    search_delay_min: float,
    search_delay_max: float,
    retry_until_success: bool,
    debug: bool,
) -> dict[str, Any]:
    ensure_initialized()

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    if not query or not query.strip():
        raise ValueError("搜索查询不能为空")

    retry_count = max(0, retry_count)
    MAX_RETRY_DELAY = 60
    retry_delay = min(max(0, retry_delay), MAX_RETRY_DELAY)

    key = _search_cache_key(
        query,
        model,
        api_key,
        base_url,
        retry_count,
        retry_delay,
        search_delay_min,
        search_delay_max,
        retry_until_success,
        debug,
    )
    with _state.search_cache_lock:
        if key in _state.search_cache:
            return _state.search_cache[key]

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
        logger.debug("Request payload:\n%s", json.dumps(payload, indent=2))

    if search_delay_max > search_delay_min and search_delay_max > 0:
        sleep_time = random.uniform(search_delay_min, search_delay_max)
        if sleep_time > 0:
            if debug:
                logger.info("Waiting %.2fs before search...", sleep_time)
            time.sleep(sleep_time)

    attempt = 0
    while True:
        try:
            response = _state.session.post(
                url, json=payload, headers=headers, timeout=(10, 60)
            )

            if response.status_code == 429:
                if _should_retry(attempt, retry_count, retry_until_success):
                    wait_time = _retry_wait_time(retry_delay, attempt)
                    logger.warning(
                        "Rate limited (429) (attempt %s). Retrying in %.2fs...",
                        _attempt_label(attempt, retry_count, retry_until_success),
                        wait_time,
                    )
                    time.sleep(wait_time)
                    attempt += 1
                    continue

            response.raise_for_status()

            full_text = ""
            all_grounding_chunks: list[dict[str, Any]] = []
            all_supports: list[dict[str, Any]] = []

            try:
                data = response.json()
                if debug:
                    logger.debug("Response:\n%s", json.dumps(data, indent=2))

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
                        uris: list[str] = []
                        for idx in indices:
                            if idx < len(all_grounding_chunks):
                                u = all_grounding_chunks[idx].get("web", {}).get("uri")
                                if u:
                                    uris.append(u)
                        all_supports.append({"segment": segment, "uris": uris})

            except json.JSONDecodeError as exc:
                if _should_retry(attempt, retry_count, retry_until_success):
                    wait_time = _retry_wait_time(retry_delay, attempt)
                    logger.warning(
                        "Failed to decode JSON response "
                        "(attempt %s). "
                        "Retrying in %.2fs...",
                        _attempt_label(attempt, retry_count, retry_until_success),
                        wait_time,
                    )
                    time.sleep(wait_time)
                    attempt += 1
                    continue
                raise ValueError("Failed to decode JSON response") from exc

            if retry_until_success and not candidates:
                if not _should_retry(attempt, retry_count, retry_until_success):
                    raise ValueError(
                        f"Search returned no candidates after {attempt + 1} attempts (max {_MAX_RETRY_ATTEMPTS})"
                    )
                wait_time = _retry_wait_time(retry_delay, attempt)
                logger.warning(
                    "Search returned no candidates (attempt %s). Retrying in %.2fs...",
                    _attempt_label(attempt, retry_count, retry_until_success),
                    wait_time,
                )
                time.sleep(wait_time)
                attempt += 1
                continue

            uris_to_resolve: set[str] = set()
            for chunk in all_grounding_chunks:
                web = chunk.get("web", {})
                uri = web.get("uri")
                if uri:
                    uris_to_resolve.add(uri)

            resolved_map = resolve_urls_concurrently(list(uris_to_resolve))

            final_sources: list[dict[str, Any]] = []
            url_to_id: dict[str, int] = {}
            original_url_to_id: dict[str, int] = {}
            next_id = 1

            for chunk in all_grounding_chunks:
                web = chunk.get("web", {})
                uri = web.get("uri")
                title = web.get("title")

                if uri and isinstance(uri, str):
                    resolved = resolved_map.get(uri, uri)
                    if resolved not in url_to_id:
                        url_to_id[resolved] = next_id
                        final_sources.append(
                            {"id": next_id, "title": title, "url": resolved}
                        )
                        next_id += 1
                    original_url_to_id[uri] = url_to_id[resolved]

            # Sort supports by endIndex descending to insert citations without index shifting
            all_supports.sort(
                key=lambda x: x["segment"].get("endIndex", 0), reverse=True
            )

            for support in all_supports:
                end_idx = support["segment"].get("endIndex")
                uris = support["uris"]

                if end_idx is not None:
                    ids = []
                    for u in uris:
                        if u in original_url_to_id:
                            ids.append(original_url_to_id[u])
                    ids = sorted(list(set(ids)))

                    if ids:
                        citation = f" [{', '.join(map(str, ids))}]"
                        if end_idx <= len(full_text):
                            full_text = (
                                full_text[:end_idx] + citation + full_text[end_idx:]
                            )

            if retry_until_success and not full_text.strip():
                if not _should_retry(attempt, retry_count, retry_until_success):
                    raise ValueError(
                        f"Search returned empty text after {attempt + 1} attempts (max {_MAX_RETRY_ATTEMPTS})"
                    )
                wait_time = _retry_wait_time(retry_delay, attempt)
                logger.warning(
                    "Search returned empty text (attempt %s). Retrying in %.2fs...",
                    _attempt_label(attempt, retry_count, retry_until_success),
                    wait_time,
                )
                time.sleep(wait_time)
                attempt += 1
                continue

            result = {"text": full_text, "sources": final_sources}
            with _state.search_cache_lock:
                _state.search_cache[key] = result
            return result

        except requests.RequestException as e:
            if _should_retry(attempt, retry_count, retry_until_success):
                wait_time = _retry_wait_time(retry_delay, attempt)
                logger.warning(
                    "Request failed (attempt %s): %s. Retrying in %.2fs...",
                    _attempt_label(attempt, retry_count, retry_until_success),
                    e,
                    wait_time,
                )
                time.sleep(wait_time)
                attempt += 1
                continue
            raise


def search(
    query: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    retry_count: int | None = None,
    retry_delay: float | None = None,
    search_delay_min: float | None = None,
    search_delay_max: float | None = None,
    retry_until_success: bool | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    ensure_initialized()

    options = resolve_search_options(
        model=model,
        retry_count=retry_count,
        retry_delay=retry_delay,
        search_delay_min=search_delay_min,
        search_delay_max=search_delay_max,
        retry_until_success=retry_until_success,
    )

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if base_url is None:
        base_url = os.environ.get(
            "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"
        )

    return _perform_search(
        query=query,
        model=options["model"],
        api_key=api_key,
        base_url=base_url,
        retry_count=options["retry_count"],
        retry_delay=options["retry_delay"],
        search_delay_min=options["search_delay_min"],
        search_delay_max=options["search_delay_max"],
        retry_until_success=options["retry_until_success"],
        debug=debug,
    )


def main() -> None:
    ensure_initialized()

    parser = argparse.ArgumentParser(description="Google Search via Gemini API")
    parser.add_argument("--query", required=True, help="Search query")
    parser.add_argument(
        "--model",
        default=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"),
        help="Gemini model to use (default: GEMINI_MODEL env var or gemini-2.5-flash)",
    )
    parser.add_argument(
        "--retry-until-success",
        dest="retry_until_success",
        action="store_true",
        help="Retry indefinitely until a successful non-empty response is returned",
    )
    parser.add_argument(
        "--no-retry-until-success",
        dest="retry_until_success",
        action="store_false",
        help="Disable indefinite retry even if GEMINI_RETRY_UNTIL_SUCCESS is enabled",
    )
    parser.set_defaults(retry_until_success=None)
    parser.add_argument("--dry-run", action="store_true", help="Print payload and exit")
    parser.add_argument("--debug", action="store_true", help="Print full response JSON")
    args = parser.parse_args()

    try:
        if args.dry_run:
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

        result = search(
            args.query,
            model=args.model,
            retry_until_success=args.retry_until_success,
            debug=args.debug,
        )
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
