import pandas as pd

from keiba_predictor.cli import main


def test_cli_sample_train_predict(tmp_path, capsys):
    data, model, out = tmp_path / 's.csv', tmp_path / 'm.pkl', tmp_path / 'p.csv'
    main(['sample', '--out', str(data), '--days', '40', '--races-per-day', '12'])
    df = pd.read_csv(data, dtype={'race_id': str})
    last = df['date'].max()
    hist, entries = df[df['date'] < last], df[df['date'] == last].drop(columns=['finish'])
    hist.to_csv(tmp_path / 'h.csv', index=False)
    entries.to_csv(tmp_path / 'e.csv', index=False)
    main(['train', '--data', str(tmp_path / 'h.csv'), '--model', str(model), '--rounds', '50'])
    main(['predict', '--model', str(model), '--history', str(tmp_path / 'h.csv'), '--entries',
          str(tmp_path / 'e.csv'), '--out', str(out)])
    p = pd.read_csv(out, dtype={'race_id': str})
    assert set(p['race_id']) == set(entries['race_id'])
    assert (p.groupby('race_id')['p'].sum().round(6) == 1).all()
    assert {'ev', 'buy'} <= set(p.columns)
