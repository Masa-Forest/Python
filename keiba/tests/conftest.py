import pytest

from keiba_predictor.data import make_sample
from keiba_predictor.features import add_features


@pytest.fixture(scope='session')
def sample():
    return make_sample(n_days=60, races_per_day=12, n_horses=700, seed=1)


@pytest.fixture(scope='session')
def feat(sample):
    return add_features(sample)
