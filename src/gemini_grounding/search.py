import argparse
import os
import sys
import json
import time
import random
import requests
import concurrent.futures
from functools import lru_cache
from urllib.parse import urlparse
from fake_useragent import UserAgent

# Configure a session for connection pooling
session = requests.Session()
ua = UserAgent()
session.headers.update({"User-Agent": ua.random})


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

    # Check for proxy configuration
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

        except Exception:
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
        except Exception:
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
            except Exception:
                results[uri] = uri
    return results


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

    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")

    url = f"{base_url}/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
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
                print(f"Waiting {sleep_time:.2f}s...", file=sys.stderr)
            time.sleep(sleep_time)

    for attempt in range(retry_count + 1):
        try:
            response = session.post(url, json=payload, headers=headers)
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

            full_text_bytes = full_text.encode("utf-8")
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
                        citation_bytes = citation.encode("utf-8")
                        if end_idx <= len(full_text_bytes):
                            full_text_bytes = (
                                full_text_bytes[:end_idx]
                                + citation_bytes
                                + full_text_bytes[end_idx:]
                            )

            full_text = full_text_bytes.decode("utf-8")
            return {"text": full_text, "sources": final_sources}

        except requests.RequestException as e:
            if attempt < retry_count:
                if debug:
                    print(
                        f"\nRequest failed (attempt {attempt + 1}/{retry_count + 1}): {e}",
                        file=sys.stderr,
                    )
                    print(f"Retrying in {retry_delay} seconds...", file=sys.stderr)
                time.sleep(retry_delay)
            else:
                raise e


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
