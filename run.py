#!/usr/bin/env python3
"""
Cron entrypoint (runs every ~30 min on GitHub Actions).

For every upcoming World Cup fixture, decide via the Kyiv send-time rules whether it's due
now; if so, build the prediction and send it once. State (which fixtures were sent) is a JSON
file committed back to the repo by the workflow.
"""
import datetime as dt
import json
import os
import traceback

from agent import calibration, config, odds, predict, telegram
from agent.scheduler import due


def _load_sent() -> set[str]:
    if os.path.exists(config.SENT_FILE):
        with open(config.SENT_FILE, encoding="utf-8") as f:
            return set(json.load(f).get("sent", []))
    return set()


def _save_sent(sent: set[str]) -> None:
    os.makedirs(config.STATE_DIR, exist_ok=True)
    with open(config.SENT_FILE, "w", encoding="utf-8") as f:
        json.dump({"sent": sorted(sent)}, f, ensure_ascii=False, indent=2)


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    sent = _load_sent()
    elo = calibration.load_elo()

    date_from = now.date().isoformat()
    date_to = (now.date() + dt.timedelta(days=3)).isoformat()

    try:
        fixtures = odds.world_cup_fixtures(date_from, date_to)
    except Exception as e:
        print(f"[fatal] could not fetch fixtures: {e}")
        raise

    for fx in fixtures:
        fid = fx["fixture_id"]
        if not fx["has_odds"]:
            continue
        if not due(fx["kickoff_utc"], now, already_sent=fid in sent):
            continue
        try:
            market = odds.market_view(fid)
            pred = predict.build_prediction(fx, market, weather=None, elo=elo)
            msg = predict.format_message(pred)
            telegram.send_message(msg)
            calibration.log_prediction(pred, elo)
            sent.add(fid)
            print(f"[sent] {fx['home']} - {fx['away']}  pick {pred['pick']}  conf {pred['confidence']}")
        except Exception:
            print(f"[error] {fx['home']} - {fx['away']}:\n{traceback.format_exc()}")

    _save_sent(sent)


if __name__ == "__main__":
    main()
