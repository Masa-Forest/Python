"""期待値と買い方。

単勝の期待値（100円あたりの儲けの割合）: EV = p × オッズ − 1
  EV > 0 なら長い目で見てプラス、ただし p が正しい場合だけ。p の誤差を見込んで、しきい値は 0 より上にする。
賭け金: ケリー基準 f = (p × オッズ − 1) / (オッズ − 1) を、さらに kelly_scale 倍に小さくする（控えめなケリー）。
  全額ケリーは p の誤差に弱く、資金の上下が大きすぎるため。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


def expected_value(p, odds):
    return np.asarray(p) * np.asarray(odds) - 1.0


def kelly_fraction(p, odds):
    p, odds = np.asarray(p, float), np.asarray(odds, float)
    f = (p * odds - 1.0) / np.maximum(odds - 1.0, 1e-9)
    return np.clip(f, 0.0, 1.0)


@dataclass
class Strategy:
    """どの馬を買うか。"""
    min_ev: float = 0.10          # 期待値がこれを超える馬だけ
    min_p: float = 0.02           # 1着確率がこれ未満の馬（大穴）は買わない（確率の誤差が大きい）
    max_odds: float = 50.0        # オッズがこれを超える馬は買わない
    max_per_race: int = 1         # 1レースで買う頭数の上限（期待値の高い順）
    stake: str = 'flat'           # 'flat' = 1点 unit 円、'kelly' = 資金 × ケリー × kelly_scale
    unit: int = 100
    kelly_scale: float = 0.25
    max_fraction: float = 0.02    # 1点に使う資金の上限（ケリーのとき）

    def to_dict(self):
        return asdict(self)


def select_bets(pred: pd.DataFrame, s: Strategy, p_col: str = 'p') -> pd.DataFrame:
    """予測の表 → 買う馬だけの表（ev, kelly 列つき）。賭け金はここでは決めない。"""
    d = pred.copy()
    d['ev'] = expected_value(d[p_col], d['odds'])
    d['kelly'] = kelly_fraction(d[p_col], d['odds'])
    ok = (d['ev'] > s.min_ev) & (d[p_col] >= s.min_p) & (d['odds'] <= s.max_odds) & d['odds'].notna()
    d = d[ok].sort_values(['race_id', 'ev'], ascending=[True, False])
    d = d.groupby('race_id', sort=False).head(s.max_per_race)
    return d.sort_values(['date', 'race_id', 'horse_no']).reset_index(drop=True)


def stakes(bets: pd.DataFrame, s: Strategy, bankroll: float) -> np.ndarray:
    """その時点の資金から1点ごとの賭け金（100円単位に切り捨て）"""
    if s.stake == 'flat':
        return np.full(len(bets), float(s.unit))
    f = np.minimum(bets['kelly'].to_numpy() * s.kelly_scale, s.max_fraction)
    return np.floor(bankroll * f / 100) * 100
