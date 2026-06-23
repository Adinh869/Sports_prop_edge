"""Legal odds API adapter placeholder (The Odds API)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class OddsApiClient:
    api_key: str | None = None
    base_url: str = "https://api.the-odds-api.com/v4"

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.getenv("ODDS_API_KEY")

    def _get(self, path: str, **params: Any) -> Any:
        if not self.api_key:
            raise RuntimeError("Missing ODDS_API_KEY")
        params["apiKey"] = self.api_key
        response = requests.get(f"{self.base_url}{path}", params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    def sports(self) -> Any:
        return self._get("/sports")

    def odds(self, sport_key: str, regions: str = "us", markets: str = "h2h,spreads,totals") -> Any:
        return self._get(f"/sports/{sport_key}/odds", regions=regions, markets=markets)
