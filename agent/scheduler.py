"""
Game-day scheduling (Europe/Kyiv) with a wide random send window.

The pool wants ONE submission per game day, before that day's first match. So we group the
fixtures into game days and send a single batched prediction per day, at a random time in the
window [10:00 Kyiv, first_kickoff - 3h].

Game-day grouping is gap-based: matches with no >8h gap between consecutive kickoffs belong to
the same day. This keeps an evening->next-morning slate together (WC matches in North America
run roughly 19:00 Kyiv -> 07:00 Kyiv next day) without fragile calendar-date cutting.

The random send time is derived from the game-day key, so it's stable across cron runs but
unpredictable from outside (the ~6h window kills any fixed-offset pattern).
"""
from __future__ import annotations

import hashlib
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from agent import config

KYIV = ZoneInfo("Europe/Kyiv")
UTC = ZoneInfo("UTC")

GAP_HOURS = getattr(config, "GAP_HOURS", 8)               # split game days on gaps bigger than this
DAY_LEAD_HOURS = getattr(config, "DAY_LEAD_HOURS", 3)     # deadline = 3h before the day's first match
DAY_WINDOW_FROM = getattr(config, "DAY_WINDOW_FROM", time(10, 0))   # earliest send time (Kyiv)


def _unit(key: str) -> float:
    """Stable pseudo-random value in [0,1) from a key (same across all cron runs)."""
    h = hashlib.md5(str(key).encode()).hexdigest()
    return int(h[:8], 16) / float(0x100000000)


def _ko(fx: dict) -> datetime:
    k = fx["kickoff_utc"]
    return k if k.tzinfo else k.replace(tzinfo=UTC)


def group_game_days(fixtures: list[dict], gap_hours: int = GAP_HOURS) -> list[list[dict]]:
    """Cluster fixtures into game days by time gaps. Returns clusters sorted by kickoff."""
    fxs = sorted(fixtures, key=_ko)
    days: list[list[dict]] = []
    for fx in fxs:
        if days and (_ko(fx) - _ko(days[-1][-1])) <= timedelta(hours=gap_hours):
            days[-1].append(fx)
        else:
            days.append([fx])
    return days


def first_kickoff(cluster: list[dict]) -> datetime:
    return min(_ko(f) for f in cluster)


def game_day_key(cluster: list[dict]) -> str:
    """Stable, collision-free dedup key: the Kyiv date of the day's first match."""
    return "GD:" + first_kickoff(cluster).astimezone(KYIV).strftime("%Y-%m-%d")


def send_time(cluster: list[dict]) -> datetime:
    """UTC time to send this game day's batch: random in [10:00 Kyiv, first_kickoff - 3h]."""
    first = first_kickoff(cluster)
    end = first - timedelta(hours=DAY_LEAD_HOURS)                 # deadline
    dl_kyiv = end.astimezone(KYIV)
    floor = datetime.combine(dl_kyiv.date(), DAY_WINDOW_FROM, tzinfo=KYIV).astimezone(UTC)
    start = floor
    if start >= end:                                             # very early first match -> fallback
        start = end - timedelta(hours=5)
    u = _unit(game_day_key(cluster))
    return start + timedelta(seconds=u * (end - start).total_seconds())


def due(cluster: list[dict], now_utc: datetime, already_sent: bool) -> bool:
    """Send this game day now? True once the window opened and the first match hasn't started."""
    if already_sent:
        return False
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    return send_time(cluster) <= now_utc < first_kickoff(cluster)


def kyiv_str(dt_utc: datetime) -> str:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    return dt_utc.astimezone(KYIV).strftime("%Y-%m-%d %H:%M Kyiv")
