#!/usr/bin/env python3
"""
Cron entrypoint (runs every ~30 min on GitHub Actions).

Groups upcoming World Cup fixtures into game days and, for each game day that is due now
(per the random Kyiv send window), builds predictions for ALL its matches and sends them as
one batched message. Dedup is per game day. State is committed back to the repo by the workflow.
"""
import datetime as dt
import json
import os
import traceback

from agent import calibration, config, odds, predict, scheduler, telegram


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

    game_days = scheduler.group_game_days(fixtures)

    for cluster in game_days:
        key = scheduler.game_day_key(cluster)
        if not scheduler.due(cluster, now, already_sent=key in sent):
            continue

        preds = []
        for fx in sorted(cluster, key=lambda f: f["kickoff_utc"]):
            if not fx["has_odds"]:
                continue
            try:
                market = odds.market_view(fx["fixture_id"])
                pred = predict.build_prediction(fx, market, weather=None, elo=elo)
                preds.append(pred)
            except Exception:
                print(f"[error] {fx['home']} - {fx['away']}:\n{traceback.format_exc()}")

        if not preds:
            print(f"[skip] {key}: no matches with odds yet")
            continue

        try:
            telegram.send_message(predict.format_day_message(preds))
            for pred in preds:
                calibration.log_prediction(pred, elo)
            sent.add(key)
            print(f"[sent] {key}: {len(preds)} match(es)")
        except Exception:
            print(f"[error] sending {key}:\n{traceback.format_exc()}")

    _save_sent(sent)


if __name__ == "__main__":
    main()
