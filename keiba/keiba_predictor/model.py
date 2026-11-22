"""1着確率のモデル。

1段目（基礎モデル）: LightGBM で「レースの中で誰が勝つか」をソフトマックスで学習する。
    競艇版（BoatRace の all_venue_v2.py）の SoftmaxGBM と同じ考え方で、出走頭数がレースごとに違っても動くようにした。
    オッズは特徴量に入れない。
2段目（市場との組み合わせ）: Benter 方式の条件付きロジット
    p ∝ exp(a · log p_model + b · log p_market)
    p_market は単勝オッズの逆数をレース内で合計1にしたもの。a, b は1段目が学習に使っていない期間で決める。
    市場はたいてい強いので、モデル単独より「市場 + モデルの差分」の方が損をしにくい。
"""

from __future__ import annotations

import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .features import FEATURES

EPS = 1e-12


def group_sizes(race_id: pd.Series) -> np.ndarray:
    """同じ race_id が続けて並んでいる前提で、各レースの頭数"""
    change = np.r_[True, race_id.to_numpy()[1:] != race_id.to_numpy()[:-1]]
    starts = np.flatnonzero(change)
    sizes = np.diff(np.r_[starts, len(race_id)])
    if len(np.unique(race_id)) != len(sizes):
        raise ValueError('同じレースの行が続けて並んでいません')
    return sizes


def group_softmax(scores: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    gid = np.repeat(np.arange(len(sizes)), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    mx = np.maximum.reduceat(scores, starts)
    e = np.exp(scores - mx[gid])
    return e / np.add.reduceat(e, starts)[gid]


def _softmax_objective(preds, data):
    """LightGBM の自作の目的関数（レース内ソフトマックスの交差エントロピー）"""
    p = group_softmax(preds, data.get_group().astype(int))
    y = data.get_label()
    return p - y, np.maximum(p * (1 - p), 1e-6)


GBM_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_data_in_leaf=100, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, num_threads=4)


class SoftmaxGBM:
    def __init__(self, features=FEATURES, rounds: int = 300, seed: int = 0, **params):
        self.features, self.rounds = list(features), rounds
        self.params = {**GBM_PARAMS, **params, 'objective': _softmax_objective, 'seed': seed}

    def fit(self, df: pd.DataFrame) -> 'SoftmaxGBM':
        sizes = group_sizes(df['race_id'])
        y = (df['finish'] == 1).astype(float).to_numpy()
        ds = lgb.Dataset(df[self.features], y, group=sizes, free_raw_data=False)
        self.booster = lgb.train(self.params, ds, self.rounds)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        raw = self.booster.predict(df[self.features], raw_score=True)
        return group_softmax(raw, group_sizes(df['race_id']))


def _blend_logits(coef, lp_model, lp_mkt):
    return coef[0] * lp_model + coef[1] * lp_mkt


def fit_blend(p_model, p_mkt, won, sizes) -> np.ndarray:
    """条件付きロジットの係数 (a, b) を最尤法で決める"""
    lm, lk = np.log(np.maximum(p_model, EPS)), np.log(np.maximum(p_mkt, EPS))

    def nll(coef):
        p = group_softmax(_blend_logits(coef, lm, lk), sizes)
        return -np.log(np.maximum(p[won], EPS)).sum()

    return minimize(nll, x0=np.array([0.5, 0.5]), method='Nelder-Mead').x


def blend(p_model, p_mkt, coef, sizes) -> np.ndarray:
    lm, lk = np.log(np.maximum(p_model, EPS)), np.log(np.maximum(p_mkt, EPS))
    return group_softmax(_blend_logits(coef, lm, lk), sizes)


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    """学習に使える行: 結果のある馬だけ、1着が1頭のレースだけ、日付・レース順に並べる"""
    d = df[df['finish'].notna() & df['odds'].notna()]
    one_winner = d.groupby('race_id')['finish'].transform(lambda s: (s == 1).sum()) == 1
    d = d[one_winner & (d.groupby('race_id')['horse_id'].transform('size') >= 2)]
    return d.sort_values(['date', 'race_id', 'horse_no']).reset_index(drop=True)


class KeibaModel:
    """1段目 + 2段目。fit は学習期間の後ろ blend_frac を使って a, b を決め、そのあと全期間で1段目を学習し直す。"""

    def __init__(self, features=FEATURES, rounds: int = 300, blend_frac: float = 0.3, seed: int = 0, **params):
        self.features, self.rounds, self.blend_frac, self.seed, self.params = features, rounds, blend_frac, seed, params

    def _gbm(self):
        return SoftmaxGBM(self.features, self.rounds, self.seed, **self.params)

    def fit(self, df: pd.DataFrame) -> 'KeibaModel':
        d = trainable(df)
        dates = np.sort(d['date'].unique())
        cut = dates[int(len(dates) * (1 - self.blend_frac))]
        early, late = d[d['date'] < cut], d[d['date'] >= cut]
        if early['race_id'].nunique() < 50 or late['race_id'].nunique() < 50:
            raise ValueError('学習データが少なすぎます（2段目を決めるのに前後それぞれ50レース以上必要）')
        pm = self._gbm().fit(early).predict(late)
        sizes = group_sizes(late['race_id'])
        self.coef = fit_blend(pm, late['mkt_p'].to_numpy(), (late['finish'] == 1).to_numpy(), sizes)
        self.gbm = self._gbm().fit(d)
        self.trained_until = d['date'].max()
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """df（特徴量つき）→ レース・馬ごとの p_model（1段目）, mkt_p（市場）, p（組み合わせ後）"""
        d = df.sort_values(['date', 'race_id', 'horse_no']).reset_index(drop=True)
        sizes = group_sizes(d['race_id'])
        out = d.copy()
        out['p_model'] = self.gbm.predict(d)
        if d['odds'].notna().all():
            out['p'] = blend(out['p_model'].to_numpy(), d['mkt_p'].to_numpy(), self.coef, sizes)
        else:
            out['p'] = out['p_model']          # オッズがまだないときは1段目だけ
        return out

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'KeibaModel':
        with open(path, 'rb') as f:
            return pickle.load(f)


def log_loss(p: np.ndarray, won: np.ndarray) -> float:
    """1着馬に付けた確率の対数損失（レースあたり平均）。小さいほど良い。"""
    return float(-np.log(np.maximum(p[won], EPS)).mean())
