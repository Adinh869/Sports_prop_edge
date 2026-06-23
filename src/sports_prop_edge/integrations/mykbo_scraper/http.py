"""HTTP session for mykbostats.com with throttle and backoff."""

from __future__ import annotations

import logging
import time
from typing import Any
from urllib.parse import quote_plus

import requests

logger = logging.getLogger(__name__)

MYKBO_BASE = "https://mykbostats.com"
DEFAULT_THROTTLE_SEC = 0.5
MAX_RETRIES = 4

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# MyKBO returns 406 for Accept: application/json only.
ACCEPT_JSON = "application/json,text/html,*/*"
ACCEPT_HTML = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"

_CF_MARKERS = (
    "just a moment",
    "cf-browser-verification",
    "enable javascript and cookies",
    "performing security verification",
)


def is_cloudflare_challenge(text: str) -> bool:
    lower = str(text or "").lower()
    return any(marker in lower for marker in _CF_MARKERS)


def browser_headers(*, accept_json: bool = False, referer: str | None = None) -> dict[str, str]:
    headers = dict(HEADERS)
    headers["Accept"] = ACCEPT_JSON if accept_json else ACCEPT_HTML
    if referer:
        headers["Referer"] = referer
    if accept_json:
        headers["X-Requested-With"] = "XMLHttpRequest"
        headers["Sec-Fetch-Dest"] = "empty"
        headers["Sec-Fetch-Mode"] = "cors"
        headers["Sec-Fetch-Site"] = "same-origin"
    return headers


class MyKBOHttpClient:
    """Reusable requests session with rate limiting and retries."""

    def __init__(self, *, throttle_sec: float = DEFAULT_THROTTLE_SEC) -> None:
        self.session = requests.Session()
        self.throttle_sec = throttle_sec
        self._last_request_at = 0.0
        self._warmed = False
        self.search_requests = 0
        self.game_requests = 0
        self.cloudflare_failures = 0

    def _wait_throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.throttle_sec:
            time.sleep(self.throttle_sec - elapsed)

    def warm_session(self) -> None:
        if self._warmed:
            return
        self._wait_throttle()
        self.session.get(
            f"{MYKBO_BASE}/",
            headers=browser_headers(),
            timeout=45,
        )
        self._last_request_at = time.monotonic()
        self._warmed = True

    def get(
        self,
        path_or_url: str,
        *,
        params: dict[str, Any] | None = None,
        accept_json: bool = False,
        kind: str = "html",
    ) -> requests.Response:
        self.warm_session()
        url = path_or_url if path_or_url.startswith("http") else f"{MYKBO_BASE}{path_or_url}"
        referer = MYKBO_BASE + "/"
        if path_or_url.startswith("/players"):
            referer = f"{MYKBO_BASE}/players"
        headers = browser_headers(accept_json=accept_json, referer=referer)

        # Some MyKBO routes 406 unless q is embedded in the path (probe-style URL).
        if accept_json and params and len(params) == 1 and "q" in params:
            q = quote_plus(str(params["q"]).strip())
            url = f"{MYKBO_BASE}/players/search?q={q}"
            params = None

        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            self._wait_throttle()
            try:
                response = self.session.get(url, params=params, headers=headers, timeout=45)
                self._last_request_at = time.monotonic()

                if is_cloudflare_challenge(response.text):
                    self.cloudflare_failures += 1
                    raise RuntimeError(f"Cloudflare challenge on {url}")

                if response.status_code == 406 and accept_json and attempt == 0:
                    # Retry once with looser Accept after warm session.
                    headers["Accept"] = "*/*"
                    continue

                if response.status_code in {403, 429, 503}:
                    raise RuntimeError(f"HTTP {response.status_code} on {url}")

                response.raise_for_status()
                if kind == "search":
                    self.search_requests += 1
                elif kind == "game":
                    self.game_requests += 1
                return response
            except Exception as exc:
                last_exc = exc
                backoff = min(8.0, 0.5 * (2**attempt))
                logger.warning("MyKBO GET retry %s/%s %s: %s", attempt + 1, MAX_RETRIES, url, exc)
                time.sleep(backoff)

        raise RuntimeError(f"MyKBO GET failed for {url}: {last_exc}")


_default_client: MyKBOHttpClient | None = None


def get_client() -> MyKBOHttpClient:
    global _default_client
    if _default_client is None:
        _default_client = MyKBOHttpClient()
    return _default_client


def reset_client() -> None:
    global _default_client
    _default_client = None
