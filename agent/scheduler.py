"""
When to send each match, in Europe/Kyiv local time.

Rule (from the brief):
  * Daytime match  - kickoff is in the awake window 10:00-24:00 Kyiv -> send 2h before kickoff.
  * Night match    - kickoff is 00:00-10:00 Kyiv (you're asleep)     -> send the day before, by 21:00.

DST is handled automatically by zoneinfo ("Europe/Kyiv"). Times are compared in UTC.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

KYIV = ZoneInfo("Europe/Kyiv")
UTC = ZoneInfo("UTC")

AWAKE_FROM = time(10, 0)     # awake window is [10:00, 24:00); before 10:00 = asleep
DAY_BEFORE_AT = time(21, 0)  # night matches are announced the day before at 21:00
LEAD_HOURS = 2               # daytime matches: lead time before kickoff


def is_awake(dt_kyiv: datetime) -> bool:
    """True if the local Kyiv time is inside the awake window 10:00-24:00."""
    return dt_kyiv.timetz().replace(tzinfo=None) >= AWAKE_FROM


def send_time(kickoff_utc: datetime) -> datetime:
    """Return the UTC datetime at which the prediction for this match should be sent."""
    if kickoff_utc.tzinfo is None:
        kickoff_utc = kickoff_utc.replace(tzinfo=UTC)
    ko = kickoff_utc.astimezone(KYIV)

    if is_awake(ko):
        st_kyiv = ko - timedelta(hours=LEAD_HOURS)
    else:
        prev_day = (ko - timedelta(days=1)).date()
        st_kyiv = datetime.combine(prev_day, DAY_BEFORE_AT, tzinfo=KYIV)

    return st_kyiv.astimezone(UTC)


def due(kickoff_utc: datetime, now_utc: datetime, already_sent: bool) -> bool:
    """
    Should we send right now? True when the send window has opened and the match
    hasn't kicked off yet. Tolerates missed cron runs (sends late but never after KO).
    """
    if already_sent:
        return False
    if kickoff_utc.tzinfo is None:
        kickoff_utc = kickoff_utc.replace(tzinfo=UTC)
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=UTC)
    return send_time(kickoff_utc) <= now_utc < kickoff_utc


def kyiv_str(dt_utc: datetime) -> str:
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=UTC)
    return dt_utc.astimezone(KYIV).strftime("%Y-%m-%d %H:%M Kyiv")
