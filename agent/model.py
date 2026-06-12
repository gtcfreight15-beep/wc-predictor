"""
Core prediction math:
  * points()                 -> the pool's exact scoring rule
  * dixon_coles_matrix()     -> scoreline probability distribution from (lam_home, lam_away)
  * ev() / rank_candidates() -> expected points per candidate score, optimised for the pool
  * lambdas_from_market()    -> invert de-vigged 1X2 + Over/Under 2.5 into (lam_home, lam_away)
  * elo_win_prob()           -> sanity cross-check vs the market

Design rule (see README): the LLM never produces a probability vector and never picks the
score. It may only nudge (lam_home, lam_away) within a hard cap. Everything below is pure,
deterministic, and unit-tested so the EV recommendation is auditable.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

DEFAULT_RHO = -0.13      # Dixon-Coles low-score dependence (typical empirical value)
GRID = 11                # model goals 0..10 for each team
MAX_PRED = 6             # we only ever submit scores with <=6 goals per team


# --------------------------------------------------------------------------- #
# 1. The pool scoring rule                                                     #
# --------------------------------------------------------------------------- #
def _winner(h: int, a: int) -> int:
    """+1 home win, -1 away win, 0 draw."""
    return 0 if h == a else (1 if h > a else -1)


def points(pred: tuple[int, int], actual: tuple[int, int]) -> int:
    """
    Pool rules:
      5 - exact score
      3 - correct winner AND correct goal difference
      2 - correct winner AND one team's goal count exactly right
      1 - correct winner (incl. correctly called draw)
      0 - otherwise
    Note on draws: a correctly called draw always matches the goal difference (0),
    so it scores 5 (exact) or 3 (any other draw) and never 2 or 1 - matching the brief.
    """
    ph, pa = pred
    ah, aa = actual
    if ph == ah and pa == aa:
        return 5
    if _winner(ph, pa) != _winner(ah, aa):
        return 0
    if (ph - pa) == (ah - aa):          # winner already matches here
        return 3
    if ph == ah or pa == aa:
        return 2
    return 1


# --------------------------------------------------------------------------- #
# 2. Dixon-Coles scoreline distribution                                        #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=2048)
def _poisson_col(lam: float, n: int = GRID) -> tuple[float, ...]:
    return tuple(math.exp(-lam) * lam ** k / math.factorial(k) for k in range(n))


def _tau(i: int, j: int, lh: float, la: float, rho: float) -> float:
    if i == 0 and j == 0:
        return 1.0 - lh * la * rho
    if i == 0 and j == 1:
        return 1.0 + lh * rho
    if i == 1 and j == 0:
        return 1.0 + la * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def dixon_coles_matrix(lam_home: float, lam_away: float,
                       rho: float = DEFAULT_RHO, n: int = GRID) -> np.ndarray:
    """P[i, j] = probability of home i : away j. Rows = home goals, cols = away goals."""
    ph = np.array(_poisson_col(round(lam_home, 4), n))
    pa = np.array(_poisson_col(round(lam_away, 4), n))
    P = np.outer(ph, pa)
    for i in (0, 1):
        for j in (0, 1):
            P[i, j] *= _tau(i, j, lam_home, lam_away, rho)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    return P / s if s > 0 else P


# --------------------------------------------------------------------------- #
# 3. EV engine                                                                 #
# --------------------------------------------------------------------------- #
def ev(pred: tuple[int, int], P: np.ndarray) -> float:
    n = P.shape[0]
    total = 0.0
    for ah in range(n):
        for aa in range(n):
            p = P[ah, aa]
            if p:
                total += p * points(pred, (ah, aa))
    return total


def rank_candidates(P: np.ndarray, max_pred: int = MAX_PRED) -> list[tuple[tuple[int, int], float]]:
    """All candidate scores 0..max_pred each, sorted by EV desc."""
    cands = [((h, a), ev((h, a), P)) for h in range(max_pred + 1) for a in range(max_pred + 1)]
    cands.sort(key=lambda x: x[1], reverse=True)
    return cands


def modal_score(P: np.ndarray, max_pred: int = MAX_PRED) -> tuple[int, int]:
    """Most-likely exact score (baseline that EV-optimisation must beat)."""
    sub = P[: max_pred + 1, : max_pred + 1]
    h, a = np.unravel_index(np.argmax(sub), sub.shape)
    return int(h), int(a)


def prob_exact(pred: tuple[int, int], P: np.ndarray) -> float:
    return float(P[pred[0], pred[1]])


def prob_outcomes(P: np.ndarray) -> tuple[float, float, float]:
    """(P_home_win, P_draw, P_away_win) from a scoreline matrix."""
    n = P.shape[0]
    home = draw = away = 0.0
    for i in range(n):
        for j in range(n):
            if i > j:
                home += P[i, j]
            elif i == j:
                draw += P[i, j]
            else:
                away += P[i, j]
    return home, draw, away


def prob_over(P: np.ndarray, line: float = 2.5) -> float:
    n = P.shape[0]
    return float(sum(P[i, j] for i in range(n) for j in range(n) if i + j > line))


def prob_winner_correct(pred: tuple[int, int], P: np.ndarray) -> float:
    h, d, a = prob_outcomes(P)
    w = _winner(*pred)
    return h if w == 1 else (a if w == -1 else d)


# --------------------------------------------------------------------------- #
# 4. Invert market odds -> (lam_home, lam_away)                                #
# --------------------------------------------------------------------------- #
def lambdas_from_market(p_home: float, p_draw: float, p_away: float,
                        p_over25: float | None = None,
                        rho: float = DEFAULT_RHO) -> tuple[float, float]:
    """
    Fit (lam_home, lam_away) so the Dixon-Coles model reproduces the de-vigged
    market probabilities. Targets: P(home win), P(draw) and (if given) P(Over 2.5).
    Solved with a light-dependency 2-D minimiser (no scipy required).
    """
    targets = [("home", p_home), ("draw", p_draw)]
    if p_over25 is not None:
        targets.append(("over", p_over25))

    def loss(lh: float, la: float) -> float:
        P = dixon_coles_matrix(lh, la, rho)
        ph, pd, pa = prob_outcomes(P)
        po = prob_over(P, 2.5)
        got = {"home": ph, "draw": pd, "away": pa, "over": po}
        return sum((got[name] - val) ** 2 for name, val in targets)

    # coarse grid then local refine - fully deterministic and transparent
    best, best_l = None, (1.3, 1.3)
    for lh in np.arange(0.2, 3.81, 0.1):
        for la in np.arange(0.2, 3.81, 0.1):
            v = loss(lh, la)
            if best is None or v < best:
                best, best_l = v, (lh, la)
    lh0, la0 = best_l
    for lh in np.arange(max(0.05, lh0 - 0.1), lh0 + 0.101, 0.01):
        for la in np.arange(max(0.05, la0 - 0.1), la0 + 0.101, 0.01):
            v = loss(lh, la)
            if v < best:
                best, best_l = v, (lh, la)
    return round(float(best_l[0]), 3), round(float(best_l[1]), 3)


# --------------------------------------------------------------------------- #
# 5. Elo cross-check / baseline                                                #
# --------------------------------------------------------------------------- #
# Common alternate spellings -> the key convention used in state/elo.json.
_ELO_ALIASES = {
    "south korea": "korea republic",
    "republic of korea": "korea republic",
    "czech republic": "czechia",
    "united states": "usa",
    "united states of america": "usa",
    "iran": "ir iran",
    "ivory coast": "ivory coast",
    "cote d'ivoire": "ivory coast",
    "bosnia & herzegovina": "bosnia and herzegovina",
    "turkiye": "turkey",
    "türkiye": "turkey",
    "cabo verde": "cape verde",
    "curacao": "curacao",
    "curaçao": "curacao",
}


def elo_get(elo: dict | None, team: str) -> float | None:
    """Look up a team's Elo tolerantly (exact -> case-insensitive -> alias)."""
    if not elo:
        return None
    if team in elo and not team.startswith("_"):
        return elo[team]
    n = team.strip().lower()
    norm = {k.strip().lower(): v for k, v in elo.items() if not k.startswith("_")}
    if n in norm:
        return norm[n]
    alias = _ELO_ALIASES.get(n)
    return norm.get(alias) if alias else None


def elo_win_prob(elo_home: float, elo_away: float, home_adv: float = 0.0) -> float:
    """World-Football-Elo expected result for the home side (0..1, draw-inclusive)."""
    dr = (elo_home + home_adv) - elo_away
    return 1.0 / (10 ** (-dr / 400.0) + 1.0)


def elo_baseline_lambdas(elo_home: float, elo_away: float, home_adv: float = 0.0,
                         avg_total: float = 2.6) -> tuple[float, float]:
    """
    Crude Elo-only lambdas: split an assumed average total according to the Elo
    expectation. Used purely as a dumb baseline in calibration, never for the pick.
    """
    we = elo_win_prob(elo_home, elo_away, home_adv)      # 0..1
    share = 0.30 + 0.40 * we                              # map to 0.3..0.7 goal share
    return round(avg_total * share, 3), round(avg_total * (1 - share), 3)
