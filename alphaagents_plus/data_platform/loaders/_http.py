"""Small HTTP helper with retries."""
from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)


def get_with_retry(session: requests.Session, url: str, params: dict | None = None,
                   tries: int = 4, backoff: float = 2.0, timeout: int = 30) -> requests.Response:
    """GET a URL, retrying on network errors and 429/5xx responses."""
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"HTTP {resp.status_code} for {url}")
            resp.raise_for_status()
            return resp
        except (requests.RequestException, requests.HTTPError) as err:
            last_err = err
            wait = backoff ** attempt
            log.warning("GET failed (%s). attempt %d/%d, retrying in %.0fs", err, attempt, tries, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url}: {last_err}")
