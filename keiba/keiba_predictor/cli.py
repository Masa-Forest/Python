"""コマンドライン。

    python -m keiba_predictor sample   --out data/sample.csv
    python -m keiba_predictor backtest --data data/sample.csv
    python -m keiba_predictor train    --data data/sample.csv --model models/keiba.pkl
    python -m keiba_predictor predict  --model models/keiba.pkl --history data/sample.csv --entries 出走表.csv \
                                       --out 予測.csv
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from . import backtest
from .data import load_races, make_sample, validate
from .ev import Strategy, select_bets
from .features import add_features
from .model import KeibaModel, trainable


def _mkdir_for(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def cmd_sample(a):
    df = make_sample(n_days=a.days, races_per_day=a.races_per_day, seed=a.seed)
    _mkdir_for(a.out)
    df.assign(date=df['date'].dt.strftime('%Y-%m-%d')).to_csv(a.out, index=False)
    print(f'{a.out}: {df["race_id"].nunique():,} レース, {len(df):,} 頭')


def cmd_backtest(a):
    feat = add_features(load_races(a.data))
    r = backtest.run(feat, n_folds=a.folds, rounds=a.rounds)
    pd.set_option('display.width', 200)
    print('\n=== 確率の良さ（学習の外、小さいほど良い） ===')
    print(r['prob_metrics'].to_string(index=False, float_format='%.4f'))
    print(f'\n=== 買い方の候補（前半 = {r["split_date"]:%Y-%m-%d} より前で比べる） ===')
    print(r['grid'].to_string(index=False, float_format='%.3f'))
    print(f'\n=== 後半（{r["split_date"]:%Y-%m-%d} 以降、選ぶのに使っていない期間） ===')
    print(r['summary'].to_string(index=False, float_format='%.3f'))
    if r['best'] is None:
        print('\n前半で十分な点数のある買い方がありませんでした。')
    else:
        print(f'\n選んだ買い方: {json.dumps(r["best"].to_dict(), ensure_ascii=False)}')
        print(f'控えめケリーの資金（10万円から）: {r["kelly_final_bankroll"]:,.0f} 円')
    print('\n回収率の90%区間が100%をまたぐなら「勝てる」とは言えません。')


def cmd_train(a):
    feat = add_features(load_races(a.data))
    m = KeibaModel(rounds=a.rounds).fit(feat)
    _mkdir_for(a.model)
    m.save(a.model)
    print(f'{a.model}: {m.trained_until:%Y-%m-%d} までで学習、組み合わせの係数 a={m.coef[0]:.3f}（モデル）, '
          f'b={m.coef[1]:.3f}（市場）')


def cmd_predict(a):
    m = KeibaModel.load(a.model)
    hist = load_races(a.history)
    entries = pd.read_csv(a.entries, dtype={'race_id': str, 'horse_id': str, 'jockey_id': str, 'trainer_id': str})
    if 'finish' not in entries:
        entries['finish'] = pd.NA
    entries['finish'] = pd.NA                                     # 予測する側の結果は使わない
    both = validate(pd.concat([hist, entries], ignore_index=True))
    ids = set(entries['race_id'].astype(str))
    feat = add_features(both)
    feat = feat[feat['race_id'].isin(ids)]
    pred = m.predict(feat)
    s = Strategy(min_ev=a.min_ev, min_p=a.min_p, max_odds=a.max_odds)
    buy = select_bets(pred, s)
    pred['ev'] = pred['p'] * pred['odds'] - 1
    pred['buy'] = pred.set_index(['race_id', 'horse_id']).index.isin(buy.set_index(['race_id', 'horse_id']).index)
    cols = ['date', 'race_id', 'venue', 'horse_no', 'horse_id', 'odds', 'mkt_p', 'p_model', 'p', 'ev', 'buy']
    out = pred[cols].assign(date=pred['date'].dt.strftime('%Y-%m-%d'))
    _mkdir_for(a.out)
    out.to_csv(a.out, index=False)
    print(f'{a.out}: {pred["race_id"].nunique()} レース、買う馬 {int(pred["buy"].sum())} 頭')


def main(argv=None):
    p = argparse.ArgumentParser(prog='keiba_predictor', description='競馬の単勝予測・期待値・バックテスト')
    sub = p.add_subparsers(dest='cmd', required=True)

    s = sub.add_parser('sample', help='サンプルデータを作る')
    s.add_argument('--out', default='data/sample.csv')
    s.add_argument('--days', type=int, default=120)
    s.add_argument('--races-per-day', type=int, default=24)
    s.add_argument('--seed', type=int, default=0)
    s.set_defaults(func=cmd_sample)

    b = sub.add_parser('backtest', help='ウォークフォワードで検証する')
    b.add_argument('--data', required=True)
    b.add_argument('--folds', type=int, default=4)
    b.add_argument('--rounds', type=int, default=300)
    b.set_defaults(func=cmd_backtest)

    t = sub.add_parser('train', help='全期間で学習してモデルを保存する')
    t.add_argument('--data', required=True)
    t.add_argument('--model', default='models/keiba.pkl')
    t.add_argument('--rounds', type=int, default=300)
    t.set_defaults(func=cmd_train)

    q = sub.add_parser('predict', help='出走表を予測し、買う馬に印を付ける')
    q.add_argument('--model', required=True)
    q.add_argument('--history', required=True, help='過去の結果（特徴量を作るのに使う）')
    q.add_argument('--entries', required=True, help='予測したい出走表（列は過去の結果と同じ、finish は空でよい）')
    q.add_argument('--out', default='predictions.csv')
    q.add_argument('--min-ev', type=float, default=0.10)
    q.add_argument('--min-p', type=float, default=0.02)
    q.add_argument('--max-odds', type=float, default=50.0)
    q.set_defaults(func=cmd_predict)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
