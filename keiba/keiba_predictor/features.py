"""未来の情報を混ぜない特徴量。

決まり:
  - 過去成績から作る特徴量は「そのレースの開催日より前の日」の結果だけを使う。
    同じ日の他のレース（先に終わったレースも含む）の結果は使わない。発走時刻がデータにないため、
    同じ日のどのレースが先に終わったかを正しく決められないから。
  - そのレース自身の着順・オッズは過去成績に入れない。
  - 単勝オッズはモデル（LightGBM）の特徴量には入れず、後で市場の見立てとして組み合わせる（model.py）。

結果のない行（finish が空）は、過去成績の集計に入らない。予測したい出走表を同じ表に足しても、
過去の行の特徴量は変わらない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import CATEGORICAL

# 平均への縮め方（出走数が少ないうちは全体の平均に近づける）: (全体の平均, 何走分の重み)
PRIORS = {'win': (0.08, 10.0), 'top3': (0.24, 10.0), 'fin_pct': (0.5, 5.0)}

BASE = ['distance', 'horse_no', 'frame', 'age', 'weight_carried', 'horse_weight', 'weight_diff', 'field_size',
        'venue_code', 'surface_code', 'going_code', 'sex_code']
HISTORY = [
    'h_runs', 'h_win_rate', 'h_top3_rate', 'h_fin_pct_mean',
    'h_surf_runs', 'h_surf_top3_rate', 'h_surf_fin_pct_mean',
    'h_dist_runs', 'h_dist_fin_pct_mean',
    'h_last_fin_pct', 'h_last5_fin_pct', 'h_days_since', 'h_dist_change', 'h_last_mkt_p', 'h_last5_mkt_p',
    'j_rides', 'j_win_rate', 'j_top3_rate', 'j_surf_win_rate',
    't_runs', 't_win_rate', 't_top3_rate',
]
RELATIVE_SRC = ['h_win_rate', 'h_top3_rate', 'h_fin_pct_mean', 'h_last5_fin_pct', 'h_last5_mkt_p', 'j_win_rate']
RELATIVE = [f'{c}_{k}' for c in RELATIVE_SRC for k in ('dev', 'rank')]
FEATURES = BASE + HISTORY + RELATIVE

CODES = {
    'venue': {v: i for i, v in enumerate(['札幌', '函館', '福島', '新潟', '東京', '中山', '中京', '京都', '阪神', '小倉'])},
    'surface': {'芝': 0, 'ダート': 1, '障害': 2},
    'going': {'良': 0, '稍重': 1, '重': 2, '不良': 3},
    'sex': {'牡': 0, '牝': 1, 'セ': 2},
}


def _codes(s: pd.Series, name: str) -> pd.Series:
    """決まった表で番号にする（表にない値は欠損）。学習と予測で番号がずれないようにするため。"""
    return s.map(CODES[name]).astype(float)


def _result_columns(df: pd.DataFrame) -> pd.DataFrame:
    """1頭ごとの結果の値（結果がない行は ran=0、値は 0）"""
    ran = df['finish'].notna()
    out = pd.DataFrame(index=df.index)
    out['ran'] = ran.astype(float)
    out['win'] = (df['finish'] == 1).astype(float)
    out['top3'] = (df['finish'] <= 3).astype(float)
    n = df.groupby('race_id')['finish'].transform('count')
    out['fin_pct'] = ((df['finish'] - 1) / (n - 1).clip(lower=1)).fillna(0.0)   # 0 = 1着, 1 = 最下位
    return out


def _prior_totals(df: pd.DataFrame, res: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    """keys ごとに、開催日より前の日の合計（ran と cols）。同じ日の結果は含めない。"""
    tmp = pd.concat([df[keys + ['date']], res[['ran'] + cols]], axis=1)
    daily = tmp.groupby(keys + ['date'], sort=True)[['ran'] + cols].sum()
    cum = daily.groupby(level=list(range(len(keys)))).cumsum() - daily     # その日の分を引く = 前日まで
    return df[keys + ['date']].merge(cum.reset_index(), on=keys + ['date'], how='left')[['ran'] + cols]


def _rate(tot: pd.DataFrame, col: str) -> pd.Series:
    mean, k = PRIORS[col]
    return (tot[col] + mean * k) / (tot['ran'] + k)


def _mkt_p(df: pd.DataFrame) -> pd.Series:
    inv = 1.0 / df['odds']
    return inv / inv.groupby(df['race_id']).transform('sum')


def _last_runs(df: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """馬ごとに、前日までで最後に走ったレース（結果のあるもの）の情報と、直近5走の平均"""
    done = df.loc[res['ran'] == 1, ['horse_id', 'date', 'distance']].copy()
    done['fin_pct'] = res.loc[done.index, 'fin_pct']
    done['mkt_p'] = _mkt_p(df).loc[done.index]
    done = done.sort_values(['horse_id', 'date'])
    g = done.groupby('horse_id')
    done['last5_fin_pct'] = g['fin_pct'].transform(lambda s: s.rolling(5, min_periods=1).mean())
    done['last5_mkt_p'] = g['mkt_p'].transform(lambda s: s.rolling(5, min_periods=1).mean())
    done = done.rename(columns={'date': 'last_date', 'distance': 'last_distance', 'fin_pct': 'last_fin_pct',
                                'mkt_p': 'last_mkt_p'})
    done['_t'] = done['last_date']
    left = df[['horse_id', 'date']].reset_index().sort_values('date')
    m = pd.merge_asof(left, done.sort_values('_t'), left_on='date', right_on='_t', by='horse_id',
                      allow_exact_matches=False)                          # 同じ日は使わない
    return m.set_index('index').sort_index()


def add_features(races: pd.DataFrame) -> pd.DataFrame:
    """validate 済みの表に特徴量の列を足して返す（行の並びはそのまま）。"""
    df = races.copy()
    res = _result_columns(df)
    df['field_size'] = df.groupby('race_id')['horse_id'].transform('size')
    for c in CATEGORICAL:
        df[f'{c}_code'] = _codes(df[c], c)

    df['dist_band'] = (df['distance'] // 400).astype(int)

    h = _prior_totals(df, res, ['horse_id'], ['win', 'top3', 'fin_pct'])
    df['h_runs'] = h['ran'].fillna(0).to_numpy()
    df['h_win_rate'] = _rate(h, 'win').to_numpy()
    df['h_top3_rate'] = _rate(h, 'top3').to_numpy()
    df['h_fin_pct_mean'] = _rate(h, 'fin_pct').to_numpy()

    hs = _prior_totals(df, res, ['horse_id', 'surface'], ['top3', 'fin_pct'])
    df['h_surf_runs'] = hs['ran'].fillna(0).to_numpy()
    df['h_surf_top3_rate'] = _rate(hs, 'top3').to_numpy()
    df['h_surf_fin_pct_mean'] = _rate(hs, 'fin_pct').to_numpy()

    hd = _prior_totals(df, res, ['horse_id', 'dist_band'], ['fin_pct'])
    df['h_dist_runs'] = hd['ran'].fillna(0).to_numpy()
    df['h_dist_fin_pct_mean'] = _rate(hd, 'fin_pct').to_numpy()

    last = _last_runs(df, res)
    df['h_last_fin_pct'] = last['last_fin_pct'].to_numpy()
    df['h_last5_fin_pct'] = last['last5_fin_pct'].to_numpy()
    df['h_days_since'] = (df['date'] - last['last_date']).dt.days.to_numpy()
    df['h_dist_change'] = (df['distance'] - last['last_distance']).to_numpy()
    df['h_last_mkt_p'] = last['last_mkt_p'].to_numpy()
    df['h_last5_mkt_p'] = last['last5_mkt_p'].to_numpy()

    j = _prior_totals(df, res, ['jockey_id'], ['win', 'top3'])
    df['j_rides'] = j['ran'].fillna(0).to_numpy()
    df['j_win_rate'] = _rate(j, 'win').to_numpy()
    df['j_top3_rate'] = _rate(j, 'top3').to_numpy()
    js = _prior_totals(df, res, ['jockey_id', 'surface'], ['win'])
    df['j_surf_win_rate'] = _rate(js, 'win').to_numpy()

    t = _prior_totals(df, res, ['trainer_id'], ['win', 'top3'])
    df['t_runs'] = t['ran'].fillna(0).to_numpy()
    df['t_win_rate'] = _rate(t, 'win').to_numpy()
    df['t_top3_rate'] = _rate(t, 'top3').to_numpy()

    g = df.groupby('race_id')
    for c in RELATIVE_SRC:
        df[f'{c}_dev'] = df[c] - g[c].transform('mean')
        # 小さいほど良い列（着順の割合）は小さい順、それ以外は大きい順に 1, 2, ...
        df[f'{c}_rank'] = g[c].rank(ascending='fin_pct' in c, method='average')
    df['mkt_p'] = _mkt_p(df)
    return df.drop(columns=['dist_band'])
