"""未来の情報が混ざっていないことの確認"""

import numpy as np
import pandas as pd

from keiba_predictor.data import validate
from keiba_predictor.features import FEATURES, add_features


def _same(a, b):
    a, b = a[FEATURES].reset_index(drop=True), b[FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_future_rows_do_not_change_past_features(sample, feat):
    """ある日より後の着順・オッズを全部書き換えても、その日までの特徴量は変わらない"""
    dates = np.sort(sample['date'].unique())
    cut = dates[len(dates) // 2]
    fake = sample.copy()
    later = fake['date'] > cut
    rng = np.random.default_rng(0)
    fake.loc[later, 'odds'] = rng.uniform(1.5, 80, later.sum()).round(1)
    # 未来のレースの着順を逆にする
    fake.loc[later, 'finish'] = (fake[later].groupby('race_id')['finish'].transform('max') + 1 - fake.loc[later, 'finish'])
    fake_feat = add_features(validate(fake))
    keep = feat['date'] <= cut
    _same(feat[keep], fake_feat[fake_feat['date'] <= cut])
    # 書き換えた後の日は変わっている（テスト自体が効いていることの確認）
    assert not np.allclose(feat.loc[~keep, 'h_win_rate'], fake_feat.loc[fake_feat['date'] > cut, 'h_win_rate'])


def test_same_day_results_are_not_used(sample, feat):
    """同じ日の他のレースの結果を書き換えても、その日のレースの特徴量は変わらない"""
    day = np.sort(sample['date'].unique())[30]
    races = sample.loc[sample['date'] == day, 'race_id'].unique()
    target, others = races[0], races[1:]
    fake = sample.copy()
    m = fake['race_id'].isin(others)
    fake.loc[m, 'finish'] = fake[m].groupby('race_id')['finish'].transform('max') + 1 - fake.loc[m, 'finish']
    fake_feat = add_features(validate(fake))
    _same(feat[feat['race_id'] == target], fake_feat[fake_feat['race_id'] == target])


def test_own_result_is_not_used(sample, feat):
    """そのレース自身の着順を消しても特徴量は変わらない（予測したい出走表と同じ扱い）"""
    rid = sample.loc[sample['date'] == sample['date'].max(), 'race_id'].iloc[0]
    fake = sample.copy()
    fake.loc[fake['race_id'] == rid, 'finish'] = np.nan
    fake_feat = add_features(validate(fake))
    _same(feat[feat['race_id'] == rid], fake_feat[fake_feat['race_id'] == rid])


def test_first_run_has_no_history(feat):
    first = feat.sort_values('date').groupby('horse_id').head(1)
    assert (first['h_runs'] == 0).all()
    assert first['h_last_fin_pct'].isna().all()
    assert first['h_days_since'].isna().all()


def test_runs_count_matches_previous_days(sample, feat):
    h = sample['horse_id'].value_counts().index[0]
    rows = feat[feat['horse_id'] == h].sort_values('date')
    assert rows['h_runs'].tolist() == list(range(len(rows)))


def test_market_prob_sums_to_one(feat):
    assert np.allclose(feat.groupby('race_id')['mkt_p'].sum(), 1.0)
