import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from zoneinfo import ZoneInfo
from agent import scheduler as s

UTC = ZoneInfo("UTC")


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=UTC)


def test_daytime_match_sends_2h_before():
    # Mexico vs South Africa opener: 19:00 UTC = 22:00 Kyiv (awake) -> send 20:00 Kyiv = 17:00 UTC
    ko = _utc(2026, 6, 11, 19, 0)
    assert s.send_time(ko) == _utc(2026, 6, 11, 17, 0)


def test_evening_match_awake():
    # 16:00 UTC = 19:00 Kyiv (awake) -> 17:00 Kyiv = 14:00 UTC
    ko = _utc(2026, 6, 20, 16, 0)
    assert s.send_time(ko) == _utc(2026, 6, 20, 14, 0)


def test_night_match_sends_day_before_21():
    # 01:00 UTC = 04:00 Kyiv (asleep) -> day before at 21:00 Kyiv = 18:00 UTC previous day
    ko = _utc(2026, 6, 20, 1, 0)
    assert s.send_time(ko) == _utc(2026, 6, 19, 18, 0)


def test_just_after_midnight_is_night():
    # 21:30 UTC = 00:30 Kyiv next day (asleep) -> day before 21:00 Kyiv
    ko = _utc(2026, 6, 14, 21, 30)        # = 2026-06-15 00:30 Kyiv
    st = s.send_time(ko)
    assert st == _utc(2026, 6, 14, 18, 0)  # 2026-06-14 21:00 Kyiv


def test_due_window():
    ko = _utc(2026, 6, 11, 19, 0)          # send at 17:00 UTC
    assert s.due(ko, _utc(2026, 6, 11, 16, 30), already_sent=False) is False  # too early
    assert s.due(ko, _utc(2026, 6, 11, 17, 1), already_sent=False) is True    # in window
    assert s.due(ko, _utc(2026, 6, 11, 18, 30), already_sent=False) is True   # late but ok
    assert s.due(ko, _utc(2026, 6, 11, 19, 1), already_sent=False) is False   # after KO
    assert s.due(ko, _utc(2026, 6, 11, 17, 1), already_sent=True) is False    # dedup


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
