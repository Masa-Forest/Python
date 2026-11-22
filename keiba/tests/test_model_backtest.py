import numpy as np
import pandas as pd

from keiba_predictor import backtest
from keiba_predictor.ev import Strategy
from keiba_predictor.model import KeibaModel, group_sizes, group_softmax, log_loss, trainable


def test_group_softmax_sums_to_one():
    sizes = np.array([3, 5, 2])
    p = group_softmax(np.random.default_rng(0).normal(size=10), sizes)
    starts = np.r_[0, np.cumsum(sizes)[:-1]]
    assert np.allclose(np.add.reduceat(p, starts), 1.0)


def test_model_predicts_valid_probs_and_beats_uniform(feat, tmp_path):
    d = trainable(feat)
    dates = np.sort(d['date'].unique())
    cut = dates[int(len(dates) * 0.75)]
    m = KeibaModel(rounds=80).fit(d[d['date'] < cut])
    test = d[d['date'] >= cut]
    pred = m.predict(test)
    assert np.allclose(pred.groupby('race_id')['p'].sum(), 1.0)
    assert np.allclose(pred.groupby('race_id')['p_model'].sum(), 1.0)
    won = (pred['finish'] == 1).to_numpy()
    sizes = group_sizes(pred['race_id'])
    assert log_loss(pred['p'].to_numpy(), won) < log_loss(np.repeat(1.0 / sizes, sizes), won)
    # 保存して読み込んでも同じ予測
    m.save(tmp_path / 'm.pkl')
    pred2 = KeibaModel.load(tmp_path / 'm.pkl').predict(test)
    assert np.allclose(pred['p'], pred2['p'])


def test_walk_forward_uses_only_past(feat):
    pred = backtest.walk_forward(feat, n_folds=2, rounds=50)
    assert set(pred['fold']) == {0, 1}
    assert pred['date'].min() > trainable(feat)['date'].min()   # 最初の期間は学習だけに使う


def test_simulate_accounting():
    pred = pd.DataFrame({
        'date': pd.to_datetime(['2025-01-01', '2025-01-01', '2025-01-02']),
        'race_id': ['R1', 'R1', 'R2'], 'horse_no': [1, 2, 1], 'horse_id': ['a', 'b', 'c'],
        'p': [0.5, 0.1, 0.5], 'odds': [3.0, 5.0, 3.0], 'finish': [1, 2, 4],
    })
    led = backtest.simulate(pred, Strategy(min_ev=0.0, min_p=0.0))
    # R1 の1番（EV 0.5）と R2 の1番（EV 0.5）を100円ずつ → 払戻 300円
    assert led['stake'].sum() == 200 and led['ret'].sum() == 300
    s = backtest.summarize(led)
    assert np.isclose(s['回収率'], 1.5) and s['的中'] == 1
    assert s['最大ドローダウン'] == 100


def test_run_end_to_end(feat):
    r = backtest.run(feat, n_folds=2, rounds=50)
    assert {'pred', 'prob_metrics', 'grid', 'summary'} <= set(r)
    fav = r['summary'].iloc[0]
    assert 0 < fav['回収率'] < 2
