import pandas as pd
import pytest

from keiba_predictor.data import REQUIRED, validate


def test_sample_has_columns_and_one_winner(sample):
    assert set(REQUIRED) <= set(sample.columns)
    assert (sample.groupby('race_id')['finish'].apply(lambda s: (s == 1).sum()) == 1).all()
    assert (sample['odds'] >= 1.0).all()
    # 1頭は1日に1回まで
    assert not sample.duplicated(['date', 'horse_id']).any()


def test_validate_rejects_missing_column(sample):
    with pytest.raises(ValueError, match='必要な列'):
        validate(sample.drop(columns=['odds']))


def test_validate_rejects_race_without_winner(sample):
    bad = sample.copy()
    rid = bad['race_id'].iloc[0]
    bad.loc[(bad['race_id'] == rid) & (bad['finish'] == 1), 'finish'] = 99
    with pytest.raises(ValueError, match='1着がいない'):
        validate(bad)


def test_validate_allows_entries_without_result(sample):
    entries = sample[sample['race_id'] == sample['race_id'].iloc[-1]].copy()
    entries['finish'] = pd.NA
    validate(pd.concat([sample[sample['race_id'] != entries['race_id'].iloc[0]], entries]))
