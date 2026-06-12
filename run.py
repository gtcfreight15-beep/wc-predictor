#!/usr/bin/env python3
"""
Cron entrypoint (runs every ~30 min on GitHub Actions).

Groups upcoming World Cup fixtures into game days. Scheduled (cron) runs send a game day only
when it is due inside its random Kyiv window. MANUAL runs (Run workflow) are a test button:
they force-send the SOONEST upcoming, not-yet-sent game day immediately (one day only),
ignoring the window - so you can verify the pipeline on demand. Dedup is per game day.
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


def _send_day(cluster, elo, sent, key):
    preds = []
    for fx in sorted(cluster, key=lambda f: f["kickoff_utc"]):
        if not fx["has_odds"]:
            continue
        try:
            market = odds.market_view(fx["fixture_id"])
            preds.append(predict.build_prediction(fx, market, weather=None, elo=elo))
        except Exception:
            print(f"[error] {fx['home']} - {fx['away']}:\n{traceback.format_exc()}")
    if not preds:
        print(f"[skip] {key}: no matches with odds yet")
        return False
    telegram.send_message(predict.format_day_message(preds))
    for pred in preds:
        calibration.log_prediction(pred, elo)
    sent.add(key)
    print(f"[sent] {key}: {len(preds)} match(es)")
    return True


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    force = os.environ.get("GITHUB_EVENT_NAME") == "workflow_dispatch"
    sent = _load_sent()
    elo = calibration.load_elo()

    date_from = now.date().isoformat()
    date_to = (now.date() + dt.timedelta(days=3)).isoformat()

    try:
        fixtures = odds.world_cup_fixtures(date_from, date_to)
    except Exception as e:
        print(f"[fatal] could not fetch fixtures: {e}")
        raise

    game_days = sorted(scheduler.group_game_days(fixtures), key=scheduler.first_kickoff)
    print(f"[info] fixtures: {len(fixtures)} | with odds: {sum(1 for f in fixtures if f['has_odds'])} "
          f"| window {date_from}..{date_to} | game days: {len(game_days)} | manual(force)={force}")

    forced_done = False
    for cluster in game_days:
        key = scheduler.game_day_key(cluster)
        first = scheduler.first_kickoff(cluster)
        st = scheduler.send_time(cluster)
        already = key in sent
        if force:
            eligible = (not already) and now < first and not forced_done
        else:
            eligible = scheduler.due(cluster, now, already_sent=already)
        print(f"[info] {key}: matches={len(cluster)} first={scheduler.kyiv_str(first)} "
              f"send={scheduler.kyiv_str(st)} already_sent={already} eligible={eligible}")
        if not eligible:
            continue
        try:
            if _send_day(cluster, elo, sent, key) and force:
                forced_done = True
        except Exception:
            print(f"[error] sending {key}:\n{traceback.format_exc()}")

    _save_sent(sent)


if __name__ == "__main__":
    main()
