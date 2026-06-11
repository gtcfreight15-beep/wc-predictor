"""
The LLM layer (Opus).

Division of labour (the anti-overconfidence rule):
  * Opus may ONLY return a small, source-justified multiplier on (lam_home, lam_away),
    plus 3-5 key factors, a lineup-confirmed flag and the single biggest uncertainty.
  * Opus NEVER outputs the probability vector and NEVER picks the score.
  * Python applies the capped adjustment, rebuilds the Dixon-Coles matrix, computes EV,
    picks the score, and derives a MECHANICAL confidence. The LLM can only lower confidence.
"""
from __future__ import annotations

import json
import re

from agent import config, model

# The owner's own system prompt - verbatim - is the spec for the analytical layer.
USER_SYSTEM_PROMPT = """Ты — экспертный футбольный аналитик и движок прогнозов на Чемпионат мира 2026.
Твоя задача в этом вызове: по данным о матче дать МАЛУЮ, обоснованную источником поправку к
ожидаемым голам (lambda) каждой команды и перечислить ключевые факторы.

Опорная точка — рынок (коэффициенты, очищенные от маржи). Рынок уже впитал почти всю публичную
информацию быстрее и точнее тебя — поэтому твои поправки малы по определению. Не выдумывай
травмы и составы, которых не проверял. Если новость не подтверждена источником с датой —
поправки нет (множитель 1.0).

Учитывай: подтверждённые/вероятные составы, травмы, дисквалификации, форму последних матчей,
турнирную ситуацию (не обеспечено ли уже место в плей-офф → возможна ротация), дни отдыха,
стадион/город, погоду, очные встречи."""

OUTPUT_CONTRACT = """
Сначала при необходимости используй web_search, чтобы проверить составы/травмы/форму/мотивацию
на СЕГОДНЯ. Затем верни СТРОГО один JSON-объект и ничего больше (без markdown, без префиксов):

{
  "lam_home_mult": <float 0.85..1.15>,   // поправка к ожидаемым голам хозяев
  "lam_away_mult": <float 0.85..1.15>,   // поправка к ожидаемым голам гостей
  "lineup_confirmed": <true|false>,      // подтверждены ли стартовые составы
  "factors": ["...", "...", "..."],      // 3-5 коротких пунктов
  "biggest_uncertainty": "...",          // одной строкой
  "confidence_hint": "low|medium|high",  // твоя качественная уверенность
  "sources": ["...", "..."]              // ссылки/источники проверенных новостей (если были поправки)
}

Любой множитель != 1.0 ОБЯЗАН опираться на источник из "sources" с указанием сути новости в
"factors". Нет источника → множитель строго 1.0.
""".strip()


def _clip(mult: float) -> float:
    lo, hi = 1.0 - config.MAX_LAMBDA_ADJ, 1.0 + config.MAX_LAMBDA_ADJ
    try:
        return max(lo, min(hi, float(mult)))
    except (TypeError, ValueError):
        return 1.0


def _extract_json(text: str) -> dict:
    text = re.sub(r"```(json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object in model output")
    return json.loads(text[start:end + 1])


def qualitative_adjustment(match: dict, market: dict, weather: str | None) -> dict:
    """Ask Opus for a bounded lambda nudge + factors. Returns identity adjustment in DRY_RUN."""
    if config.DRY_RUN or not config.ANTHROPIC_API_KEY:
        return {"lam_home_mult": 1.0, "lam_away_mult": 1.0, "lineup_confirmed": False,
                "factors": ["(dry-run: только рыночная база, без LLM-слоя)"],
                "biggest_uncertainty": "LLM-слой выключен", "confidence_hint": "medium",
                "sources": []}

    import anthropic
    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

    user_msg = (
        f"Матч: {match['home']} (хозяева) — {match['away']} (гости)\n"
        f"Старт (UTC): {match['kickoff_utc'].isoformat()}\n"
        f"Рыночная база (fair, без маржи): P(хозяева)={market['fair_home']:.3f}, "
        f"P(ничья)={market['fair_draw']:.3f}, P(гости)={market['fair_away']:.3f}, "
        f"P(тотал>2.5)={market.get('fair_over25')}\n"
        f"Рыночные ожидаемые голы: хозяева≈{market['lam_home']}, гости≈{market['lam_away']}\n"
        f"Погода: {weather or 'н/д'}\n\n" + OUTPUT_CONTRACT
    )

    resp = client.messages.create(
        model=config.MODEL,
        max_tokens=1500,
        system=USER_SYSTEM_PROMPT,
        tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
        messages=[{"role": "user", "content": user_msg}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    data = _extract_json(text)
    data["lam_home_mult"] = _clip(data.get("lam_home_mult", 1.0))
    data["lam_away_mult"] = _clip(data.get("lam_away_mult", 1.0))
    return data


def mechanical_confidence(market: dict, adj: dict, elo_we: float | None,
                          ev_gap: float) -> tuple[str, list[str]]:
    """Confidence from hard signals, not vibes. The LLM hint can only lower it."""
    downgrades = []
    if (market.get("disagreement") or 0) > 0.06:
        downgrades.append("букмекеры расходятся в оценке")
    # Compare like-for-like: Elo We (expected result, draw-inclusive) vs the market's
    # expected result P(home) + 0.5*P(draw) - NOT vs P(home win) alone.
    if elo_we is not None:
        market_expected = market["fair_home"] + 0.5 * market["fair_draw"]
        if abs(market_expected - elo_we) > 0.12:
            downgrades.append("рынок сильно расходится с Elo")
    if not adj.get("lineup_confirmed"):
        downgrades.append("составы не подтверждены")
    if ev_gap < 0.05:
        downgrades.append("EV топ-кандидатов почти равны")

    level = "high" if not downgrades else ("medium" if len(downgrades) == 1 else "low")
    order = {"low": 0, "medium": 1, "high": 2}
    hint = adj.get("confidence_hint", "high")
    if order.get(hint, 2) < order[level]:        # LLM may only lower, never raise
        level = hint
    return level, downgrades


def build_prediction(match: dict, market: dict, weather: str | None,
                     elo: dict | None) -> dict:
    adj = qualitative_adjustment(match, market, weather)

    lh = market["lam_home"] * adj["lam_home_mult"]
    la = market["lam_away"] * adj["lam_away_mult"]
    P = model.dixon_coles_matrix(lh, la)
    ranked = model.rank_candidates(P)
    pick, pick_ev = ranked[0]
    alts = ranked[1:3]
    ev_gap = pick_ev - ranked[1][1]

    elo_we = None
    if elo and match["home"] in elo and match["away"] in elo:
        elo_we = model.elo_win_prob(elo[match["home"]], elo[match["away"]])

    conf, downgrades = mechanical_confidence(market, adj, elo_we, ev_gap)

    return {
        "match": match, "market": market, "adj": adj,
        "lam_home": round(lh, 3), "lam_away": round(la, 3),
        "pick": pick, "pick_ev": pick_ev, "alts": alts, "ev_gap": ev_gap,
        "p_exact": model.prob_exact(pick, P),
        "p_winner": model.prob_winner_correct(pick, P),
        "modal": model.modal_score(P),
        "confidence": conf, "downgrades": downgrades, "elo_we": elo_we,
    }


def format_message(pred: dict) -> str:
    m = pred["match"]
    from agent.scheduler import kyiv_str
    ph, pa = pred["pick"]
    conf_ru = {"low": "низкая", "medium": "средняя", "high": "высокая"}[pred["confidence"]]
    alt_str = "  •  ".join(f"{h}:{a} (EV {e:.2f})" for (h, a), e in pred["alts"])
    factors = "\n".join(f"• {f}" for f in pred["adj"].get("factors", [])[:5])
    note = ""
    if pred["ev_gap"] < 0.05:
        note = "\n⚠️ EV топ-кандидатов в пределах шума — можно рискнуть на «5»."
    src = pred["adj"].get("sources") or []
    src_line = f"\nИсточники поправок: {', '.join(src[:3])}" if src else ""

    return (
        f"⚽ <b>{m['home']} — {m['away']}</b>\n"
        f"🕒 {kyiv_str(m['kickoff_utc'])}\n\n"
        f"🎯 <b>Прогноз: {ph}:{pa}</b>  (EV {pred['pick_ev']:.2f})\n"
        f"   P(точный счёт)={pred['p_exact']:.0%}, P(угадан победитель)={pred['p_winner']:.0%}\n"
        f"   Альтернативы: {alt_str}\n"
        f"   λ: хозяева {pred['lam_home']} / гости {pred['lam_away']}  "
        f"(модальный счёт {pred['modal'][0]}:{pred['modal'][1]})\n\n"
        f"<b>Факторы:</b>\n{factors}\n\n"
        f"Уверенность: <b>{conf_ru}</b>"
        + (f" ({'; '.join(pred['downgrades'])})" if pred["downgrades"] else "")
        + f"\nГлавная неопределённость: {pred['adj'].get('biggest_uncertainty', '—')}"
        + note + src_line
    )
