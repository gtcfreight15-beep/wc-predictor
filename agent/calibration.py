"""
The live out-of-sample self-check.

At prediction time we log the EV pick alongside three dumb baselines computed on the same
data. After each match we score all four against the real result. A weekly audit reports
whether the EV agent is actually beating the baselines - if it isn't, you see it in numbers,
not in prose. Calibration is computed by Python from real results; the LLM never grades itself.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

from agent import config, model

STRATEGIES = ("ev_pick", "modal", "naive_fav", "elo_modal")


def _append(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def baselines(pred: dict, elo: dict | None) -> dict:
    """Predicted score for each strategy, on the same (adjusted) distribution."""
    market = pred["market"]
    fav_home = market["fair_home"] >= market["fair_away"]
    pickem = abs(market["fair_home"] - market["fair_away"]) < 0.05
    naive = [1, 1] if pickem else ([1, 0] if fav_home else [0, 1])

    elo_modal = None
    m = pred["match"]
    if elo and m["home"] in elo and m["away"] in elo:
        elh, ela = model.elo_baseline_lambdas(elo[m["home"]], elo[m["away"]])
        elo_modal = list(model.modal_score(model.dixon_coles_matrix(elh, ela)))

    return {
        "ev_pick": list(pred["pick"]),
        "modal": list(pred["modal"]),
        "naive_fav": naive,
        "elo_modal": elo_modal,
    }


def log_prediction(pred: dict, elo: dict | None) -> None:
    m = pred["match"]
    _append(config.PRED_LOG, {
        "fixture_id": m["fixture_id"],
        "home": m["home"], "away": m["away"],
        "kickoff_utc": m["kickoff_utc"].isoformat(),
        "pick": list(pred["pick"]), "pick_ev": round(pred["pick_ev"], 4),
        "confidence": pred["confidence"],
        "baselines": baselines(pred, elo),
    })


def fetch_final_score(home: str, away: str, kickoff_iso: str) -> dict | None:
    """Get a finished match's final score via a cheap Opus web_search call."""
    if config.DRY_RUN or not config.ANTHROPIC_API_KEY:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    resp = client.messages.create(
        model=config.MODEL, max_tokens=400,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 3}],
        messages=[{"role": "user", "content":
            f"Final full-time score of the {home} vs {away} World Cup 2026 match "
            f"(kickoff {kickoff_iso[:10]}). If it has finished, return ONLY JSON "
            f'{{"finished":true,"home":<int>,"away":<int>}}. If not finished or unknown, '
            f'return ONLY {{"finished":false}}. Home team is {home}.'}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    try:
        s, e = text.find("{"), text.rfind("}")
        data = json.loads(text[s:e + 1])
        return data if data.get("finished") else None
    except Exception:
        return None


def settle() -> int:
    """Score any logged predictions whose result we don't yet have. Returns count settled."""
    settled_ids = {r["fixture_id"] for r in _read(config.RESULTS_LOG)}
    n = 0
    for p in _read(config.PRED_LOG):
        if p["fixture_id"] in settled_ids:
            continue
        res = fetch_final_score(p["home"], p["away"], p["kickoff_utc"])
        if not res:
            continue
        actual = (res["home"], res["away"])
        pts = {s: model.points(tuple(p["baselines"][s]), actual)
               for s in STRATEGIES if p["baselines"].get(s) is not None}
        _append(config.RESULTS_LOG, {
            "fixture_id": p["fixture_id"], "home": p["home"], "away": p["away"],
            "actual": list(actual), "pick": p["pick"], "pick_ev": p["pick_ev"],
            "points": pts,
        })
        n += 1
    return n


def audit_text() -> str:
    rows = _read(config.RESULTS_LOG)
    if not rows:
        return "Пока нет завершённых матчей для аудита."

    totals = defaultdict(float)
    counts = defaultdict(int)
    ev_sum = realized_sum = 0.0
    for r in rows:
        for s, pts in r["points"].items():
            totals[s] += pts
            counts[s] += 1
        ev_sum += r.get("pick_ev", 0.0)
        realized_sum += r["points"].get("ev_pick", 0)

    n = len(rows)
    lines = [f"📊 <b>Самоаудит за {n} матч(ей)</b>", ""]
    label = {"ev_pick": "EV-агент", "modal": "модальный", "naive_fav": "наивный 1:0",
             "elo_modal": "Elo"}
    ranking = sorted(totals.items(), key=lambda x: x[1], reverse=True)
    for s, tot in ranking:
        avg = tot / counts[s] if counts[s] else 0
        mark = " 👑" if s == ranking[0][0] else ""
        lines.append(f"{label.get(s, s)}: <b>{tot:.0f}</b> очк. ({avg:.2f}/матч){mark}")

    edge = totals["ev_pick"] - totals.get("modal", 0)
    lines += ["", f"EV vs модальный: {edge:+.0f} очк. "
                  f"({'есть преимущество' if edge > 0 else 'нет преимущества — хитрость не окупается'})"]
    lines.append(f"Калибровка EV: обещано {ev_sum:.1f}, набрано {realized_sum:.0f} "
                 f"(model {'переоценивает' if ev_sum > realized_sum + 1 else 'в норме'})")
    return "\n".join(lines)


def load_elo() -> dict | None:
    if os.path.exists(config.ELO_FILE):
        with open(config.ELO_FILE, encoding="utf-8") as f:
            return json.load(f)
    return None
