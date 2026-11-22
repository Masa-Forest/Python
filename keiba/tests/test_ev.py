import numpy as np
import pandas as pd

from keiba_predictor.ev import Strategy, expected_value, kelly_fraction, select_bets, stakes


def test_expected_value():
    assert np.isclose(expected_value(0.25, 5.0), 0.25)
    assert np.isclose(expected_value(0.1, 8.0), -0.2)


def test_kelly_fraction():
    # p=0.25, オッズ5倍 → (1.25-1)/4 = 0.0625
    assert np.isclose(kelly_fraction(0.25, 5.0), 0.0625)
    assert kelly_fraction(0.1, 8.0) == 0.0          # 期待値がマイナスなら賭けない


def _pred():
    return pd.DataFrame({
        'date': pd.to_datetime(['2025-01-01'] * 4),
        'race_id': ['R1', 'R1', 'R1', 'R2'],
        'horse_no': [1, 2, 3, 1],
        'p': [0.30, 0.25, 0.01, 0.5],
        'odds': [4.0, 5.0, 200.0, 1.5],
    })


def test_select_bets_picks_best_ev_per_race():
    b = select_bets(_pred(), Strategy(min_ev=0.1, min_p=0.02, max_odds=50))
    # R1: 1番 EV 0.20, 2番 EV 0.25 → 2番だけ（1レース1頭）。3番は確率が小さく、オッズも高すぎる。R2: EV -0.25
    assert b[['race_id', 'horse_no']].values.tolist() == [['R1', 2]]


def test_stakes_flat_and_kelly():
    b = select_bets(_pred(), Strategy(min_ev=0.1))
    assert stakes(b, Strategy(), 100_000).tolist() == [100.0]
    # ケリー 0.0625 × 0.25 = 1.5625% → 上限 2% に入る → 1,562円 → 100円単位で 1,500円
    assert stakes(b, Strategy(stake='kelly'), 100_000).tolist() == [1500.0]
