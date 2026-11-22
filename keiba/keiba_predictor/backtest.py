"""時期をずらしながらの検証（ウォークフォワード）と、回収率の集計。

1. 期間を n_folds 個の区間に分け、各区間は「その区間より前」だけで学習したモデルで予測する（学習の外の予測）。
2. 学習の外の予測を前半・後半に分け、買い方（期待値のしきい値など）は前半だけで選び、後半で確かめる。
   後半は選ぶのに使わないので、後半の回収率が「この先に近い」数字。
3. 回収率にはレース単位のブートストラップで90%区間を付ける。区間が100%をまたぐなら、勝てるとは言えない。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ev import Strategy, select_bets, stakes
from .model import KeibaModel, group_sizes, log_loss, trainable


def walk_forward(feat: pd.DataFrame, n_folds: int = 4, first_train_frac: float = 0.4, **model_kwargs) -> pd.DataFrame:
    """学習の外の予測（fold 列つき）。feat は add_features 済み。"""
    d = trainable(feat)
    dates = np.sort(d['date'].unique())
    start = int(len(dates) * first_train_frac)
    edges = np.linspace(start, len(dates), n_folds + 1).astype(int)
    out = []
    for k in range(n_folds):
        lo, hi = dates[edges[k]], (dates[edges[k + 1]] if edges[k + 1] < len(dates) else None)
        train = d[d['date'] < lo]
        test = d[(d['date'] >= lo) & ((d['date'] < hi) if hi is not None else True)]
        if test.empty:
            continue
        m = KeibaModel(**model_kwargs).fit(train)
        pred = m.predict(test)
        pred['fold'] = k
        out.append(pred)
    return pd.concat(out, ignore_index=True)


def payout(bets: pd.DataFrame) -> np.ndarray:
    """単勝の払戻（100円あたり）。1着なら オッズ×100 円。"""
    return np.where(bets['finish'] == 1, bets['odds'] * 100, 0.0)


def simulate(pred: pd.DataFrame, s: Strategy, bankroll: float = 100_000.0, p_col: str = 'p') -> pd.DataFrame:
    """買う馬ごとの賭け金・払戻・資金の推移。ケリーのときは各開催日の朝の資金で賭け金を決める。"""
    bets = select_bets(pred, s, p_col)
    if bets.empty:
        return bets.assign(stake=[], ret=[], bankroll=[])
    rows = []
    for _, day in bets.groupby('date', sort=True):
        st = stakes(day, s, bankroll)
        ret = payout(day) / 100 * st
        bankroll += ret.sum() - st.sum()
        rows.append(day.assign(stake=st, ret=ret, bankroll=bankroll))
    return pd.concat(rows, ignore_index=True)


def favorite_bets(pred: pd.DataFrame) -> pd.DataFrame:
    """比べる相手: 毎レース1番人気（オッズが一番低い馬）を100円"""
    d = pred.sort_values(['race_id', 'odds', 'horse_no']).groupby('race_id').head(1)
    return d.assign(stake=100.0, ret=payout(d))


def all_bets(pred: pd.DataFrame) -> pd.DataFrame:
    """比べる相手: 全馬を100円ずつ（だいたい 1 − 控除率 になる）"""
    return pred.assign(stake=100.0, ret=payout(pred))


def summarize(ledger: pd.DataFrame, name: str = '', boot: int = 1000, seed: int = 0) -> dict:
    if ledger.empty or ledger['stake'].sum() == 0:
        return {'買い方': name, 'レース': 0, '点数': 0, '的中': 0, '的中率': np.nan, '投資': 0, '払戻': 0,
                '回収率': np.nan, '回収率90%区間': '—', '最大ドローダウン': 0}
    by_race = ledger.groupby('race_id')[['stake', 'ret']].sum()
    rng = np.random.default_rng(seed)
    st, rt = by_race['stake'].to_numpy(), by_race['ret'].to_numpy()
    bs = []
    for _ in range(boot):
        i = rng.integers(0, len(st), len(st))
        bs.append(rt[i].sum() / max(st[i].sum(), 1e-9))
    cum = (ledger['ret'] - ledger['stake']).cumsum().to_numpy()
    dd = float(np.max(np.maximum.accumulate(np.r_[0, cum]) - np.r_[0, cum]))
    hits = int((ledger['ret'] > 0).sum())
    return {'買い方': name, 'レース': int(len(by_race)), '点数': int((ledger['stake'] > 0).sum()), '的中': hits,
            '的中率': hits / max(int((ledger['stake'] > 0).sum()), 1), '投資': float(st.sum()),
            '払戻': float(rt.sum()), '回収率': float(rt.sum() / st.sum()),
            '回収率90%区間': f'{np.percentile(bs, 5):.1%}〜{np.percentile(bs, 95):.1%}', '最大ドローダウン': dd}


def prob_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    """1着馬に付けた確率の対数損失。p（組み合わせ後）が mkt_p（市場だけ）より小さければ、モデルが市場に足せている。"""
    won = (pred['finish'] == 1).to_numpy()
    sizes = group_sizes(pred['race_id'])
    uniform = np.repeat(1.0 / sizes, sizes)
    rows = [(name, log_loss(p, won)) for name, p in
            [('一様（全馬同じ）', uniform), ('市場（オッズ）', pred['mkt_p'].to_numpy()),
             ('1段目（モデルだけ）', pred['p_model'].to_numpy()), ('組み合わせ後', pred['p'].to_numpy())]]
    return pd.DataFrame(rows, columns=['確率', '対数損失'])


GRID = [Strategy(min_ev=e, min_p=mp, max_odds=mo)
        for e in (0.0, 0.05, 0.10, 0.20, 0.30) for mp in (0.02, 0.05) for mo in (20.0, 50.0)]


def choose_and_check(pred: pd.DataFrame, grid=GRID, min_bets: int = 50) -> tuple[Strategy | None, pd.DataFrame]:
    """前半で回収率が一番良い買い方を選び、後半で確かめる。後半は選ぶのに使わない。"""
    dates = np.sort(pred['date'].unique())
    mid = dates[len(dates) // 2]
    first, second = pred[pred['date'] < mid], pred[pred['date'] >= mid]
    rows, best, best_roi = [], None, -np.inf
    for s in grid:
        a = summarize(simulate(first, s))
        rows.append({'min_ev': s.min_ev, 'min_p': s.min_p, 'max_odds': s.max_odds, '前半 点数': a['点数'],
                     '前半 回収率': a['回収率']})
        if a['点数'] >= min_bets and a['回収率'] > best_roi:
            best, best_roi = s, a['回収率']
    table = pd.DataFrame(rows)
    return best, table


def run(feat: pd.DataFrame, n_folds: int = 4, **model_kwargs) -> dict:
    """検証一式。結果は dict（表は DataFrame）。"""
    pred = walk_forward(feat, n_folds=n_folds, **model_kwargs)
    best, grid_table = choose_and_check(pred)
    dates = np.sort(pred['date'].unique())
    mid = dates[len(dates) // 2]
    second = pred[pred['date'] >= mid]
    rows = [summarize(favorite_bets(second), '1番人気を全部（後半）'),
            summarize(all_bets(second), '全馬を全部（後半）')]
    if best is not None:
        rows.append(summarize(simulate(second, best), f'選んだ買い方（後半, EV>{best.min_ev:.2f}）'))
        kelly = Strategy(**{**best.to_dict(), 'stake': 'kelly'})
        led = simulate(second, kelly)
        rows.append(summarize(led, '選んだ買い方・控えめケリー（後半）'))
        final_bankroll = float(led['bankroll'].iloc[-1]) if not led.empty else 100_000.0
    else:
        final_bankroll = None
    return {'pred': pred, 'prob_metrics': prob_metrics(pred), 'grid': grid_table, 'best': best,
            'summary': pd.DataFrame(rows), 'kelly_final_bankroll': final_bankroll,
            'split_date': pd.Timestamp(mid)}
