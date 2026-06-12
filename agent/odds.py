"""
OddsPapi v4 client.

Verified against OddsPapi's published World Cup tutorial:
  * auth: query param `apiKey` (NOT a header)
  * GET /fixtures?sportId=10&from=YYYY-MM-DD&to=YYYY-MM-DD   (<=10 day window)
        fixture fields: fixtureId, startTime (ISO), participant1Name, participant2Name, hasOdds, tournamentName
  * GET /odds?fixtureId=...&bookmakers=a,b,c
        price path: bookmakerOdds[slug]["markets"][mid]["outcomes"][oid]["players"]["0"]["price" | "active"]
  * markets:  1X2 = 101 (101 Home / 102 Draw / 103 Away);  Over/Under 2.5 = 1010 (1010 Over / 1011 Under)

The parsing layer is deliberately isolated: if OddsPapi ever changes field names, only
the small extract_* helpers need touching. `python -m agent.odds <fixtureId>` dumps a raw
response for inspection.
"""
from __future__ import annotations

import datetime as dt
import sys
import time
from typing import Optional

import requests

from agent import config, model

M_1X2, O_HOME, O_DRAW, O_AWAY = "101", "101", "102", "103"
M_OU25, O_OVER, O_UNDER = "1010", "1010", "1011"

_COOLDOWN = 0.9  # OddsPapi ~0.88s cooldown between calls to the same endpoint


def _get(path: str, **params):
    params["apiKey"] = config.ODDSPAPI_KEY
    r = requests.get(f"{config.ODDSPAPI_BASE}/{path}", params=params, timeout=30)
    r.raise_for_status()
    time.sleep(_COOLDOWN)
    return r.json()


def parse_iso_utc(s: str) -> dt.datetime:
    s = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.astimezone(dt.timezone.utc)


def world_cup_fixtures(date_from: str, date_to: str) -> list[dict]:
    """Upcoming WC fixtures in a <=10 day window. Returns normalised dicts."""
    raw = _get("fixtures", sportId=10, **{"from": date_from, "to": date_to})
    out = []
    for f in raw:
        name = f.get("tournamentName") or ""
        parts = (str(f.get("participant1Name", "")) + " " + str(f.get("participant2Name", ""))).upper()
        if name != "World Cup" or "SRL" in parts:   # exclude Simulated Reality League
            continue
        out.append({
            "fixture_id": f["fixtureId"],
            "home": f["participant1Name"],
            "away": f["participant2Name"],
            "kickoff_utc": parse_iso_utc(f["startTime"]),
            "has_odds": bool(f.get("hasOdds")),
        })
    return out


def _price(books: dict, slug: str, market: str, outcome: str) -> Optional[float]:
    try:
        o = books[slug]["markets"][market]["outcomes"][outcome]["players"]["0"]
        if o.get("active", True):
            return float(o["price"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def extract_1x2(books: dict, slug: str):
    h = _price(books, slug, M_1X2, O_HOME)
    d = _price(books, slug, M_1X2, O_DRAW)
    a = _price(books, slug, M_1X2, O_AWAY)
    return (h, d, a) if None not in (h, d, a) else None


def extract_ou25(books: dict, slug: str):
    o = _price(books, slug, M_OU25, O_OVER)
    u = _price(books, slug, M_OU25, O_UNDER)
    return (o, u) if None not in (o, u) else None


def devig(prices: list[float]) -> list[float]:
    imp = [1.0 / p for p in prices]
    tot = sum(imp)
    return [x / tot for x in imp]


def sharp_consensus(books: dict, slugs: list[str]):
    """
    Average de-vigged fair probabilities across whichever sharp books are present.
    Returns (fair_1x2, fair_over25, n_books_1x2, disagreement) where disagreement is
    the spread (max-min) of P(home win) across books - a calibration input.
    """
    home_probs, draw_probs, away_probs, over_probs = [], [], [], []
    for slug in slugs:
        line = extract_1x2(books, slug)
        if line:
            fh, fd, fa = devig(list(line))
            home_probs.append(fh); draw_probs.append(fd); away_probs.append(fa)
        ou = extract_ou25(books, slug)
        if ou:
            fo, _ = devig(list(ou))
            over_probs.append(fo)

    if not home_probs:                      # fall back to every book in the payload
        return sharp_consensus(books, list(books.keys())) if slugs != list(books.keys()) else (None, None, 0, None)

    fair_1x2 = (sum(home_probs) / len(home_probs),
                sum(draw_probs) / len(draw_probs),
                sum(away_probs) / len(away_probs))
    fair_over = (sum(over_probs) / len(over_probs)) if over_probs else None
    disagreement = max(home_probs) - min(home_probs)
    return fair_1x2, fair_over, len(home_probs), disagreement


def market_view(fixture_id: str) -> dict:
    """Full market-derived view for a fixture: fair probs + fitted lambdas + quality flags."""
    odds = _get("odds", fixtureId=fixture_id, bookmakers=",".join(config.SHARP_BOOKS))
    books = odds.get("bookmakerOdds", {})
    fair_1x2, fair_over, n_books, disagreement = sharp_consensus(books, config.SHARP_BOOKS)
    if fair_1x2 is None:
        raise RuntimeError(f"No 1X2 market found for fixture {fixture_id}")
    ph, pd, pa = fair_1x2
    lh, la = model.lambdas_from_market(ph, pd, pa, fair_over)
    return {
        "fair_home": ph, "fair_draw": pd, "fair_away": pa, "fair_over25": fair_over,
        "lam_home": lh, "lam_away": la,
        "n_books": n_books, "disagreement": disagreement,
    }


if __name__ == "__main__":   # inspection helper: python -m agent.odds <fixtureId>
    import json
    fid = sys.argv[1]
    print(json.dumps(_get("odds", fixtureId=fid, bookmakers=",".join(config.SHARP_BOOKS)), indent=2)[:4000])
