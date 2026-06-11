"""Self-check tests for the math. Run: python -m pytest tests/ -q  (or python tests/test_model.py)"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from agent import model as m


def test_points_brief_example():
    # Brief: actual 3:1 -> 3:1=5, 2:0=3, 4:1 & 3:2 =2, 1:0=1
    a = (3, 1)
    assert m.points((3, 1), a) == 5
    assert m.points((2, 0), a) == 3
    assert m.points((4, 1), a) == 2
    assert m.points((3, 2), a) == 2
    assert m.points((1, 0), a) == 1
    assert m.points((0, 0), a) == 0   # wrong winner (draw)
    assert m.points((1, 1), a) == 0   # wrong winner (draw)
    assert m.points((0, 2), a) == 0   # wrong winner (away)


def test_points_draws():
    assert m.points((1, 1), (1, 1)) == 5
    assert m.points((1, 1), (0, 0)) == 3   # correct draw, not exact
    assert m.points((2, 2), (0, 0)) == 3
    assert m.points((1, 1), (2, 1)) == 0   # predicted draw, actual home win
    # draws never yield 1 or 2:
    for ah in range(5):
        pts = m.points((1, 1), (ah, ah))
        assert pts in (3, 5)


def test_points_count_rule():
    # away win, one team's count correct -> 2
    assert m.points((0, 2), (1, 2)) == 2   # away count 2 matches, winner away, diff differs
    assert m.points((1, 3), (1, 2)) == 2   # home count 1 matches, winner away
    assert m.points((0, 1), (1, 2)) == 3   # away win, diff -1 == -1
    assert m.points((0, 3), (1, 2)) == 1   # away win only


def test_matrix_sums_to_one():
    for lh, la in [(1.8, 1.0), (0.7, 0.7), (2.4, 1.6), (1.3, 1.3)]:
        P = m.dixon_coles_matrix(lh, la)
        assert abs(P.sum() - 1.0) < 1e-9
        assert (P >= 0).all()


def test_lambda_round_trip():
    # Build a market from known lambdas, then recover them.
    for lh_true, la_true in [(1.8, 1.0), (1.1, 1.4), (2.2, 0.8)]:
        P = m.dixon_coles_matrix(lh_true, la_true)
        ph, pd, pa = m.prob_outcomes(P)
        po = m.prob_over(P, 2.5)
        lh, la = m.lambdas_from_market(ph, pd, pa, po)
        assert abs(lh - lh_true) < 0.12, (lh, lh_true)
        assert abs(la - la_true) < 0.12, (la, la_true)


def test_ev_beats_naive_on_its_own_distribution():
    # The EV-max candidate must have EV >= the modal score's EV (by construction).
    P = m.dixon_coles_matrix(1.7, 1.0)
    ranked = m.rank_candidates(P)
    top, top_ev = ranked[0]
    modal = m.modal_score(P)
    assert top_ev >= m.ev(modal, P) - 1e-9


def test_ev_is_a_real_expectation():
    P = m.dixon_coles_matrix(1.5, 1.2)
    # EV must lie between 0 and 5
    for (h, a), e in m.rank_candidates(P):
        assert 0.0 <= e <= 5.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tests passed.")
