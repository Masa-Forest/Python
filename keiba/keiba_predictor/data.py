"""レースデータの読み込み・チェックと、サンプルデータの生成。

1行 = 1頭（1レースの1出走馬）。列は COLUMNS を参照。
finish（着順）が空の行は「結果がまだない」（予測したい出走表）か、取消・除外・中止。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 列名: 説明
COLUMNS = {
    'race_id': 'レースID（文字列。同じレースの馬は同じ値）',
    'date': '開催日（YYYY-MM-DD）',
    'venue': '競馬場（例: 東京）',
    'surface': '芝 / ダート',
    'distance': '距離（m）',
    'going': '馬場状態（良 / 稍重 / 重 / 不良）',
    'horse_id': '馬ID',
    'jockey_id': '騎手ID',
    'trainer_id': '調教師ID',
    'horse_no': '馬番',
    'frame': '枠番',
    'age': '馬齢',
    'sex': '性別（牡 / 牝 / セ）',
    'weight_carried': '斤量（kg）',
    'horse_weight': '馬体重（kg）',
    'weight_diff': '馬体重の増減（kg）',
    'odds': '単勝オッズ（倍）',
    'popularity': '単勝人気',
    'finish': '着順（結果がない・取消などは空）',
}
REQUIRED = list(COLUMNS)
CATEGORICAL = ['venue', 'surface', 'going', 'sex']


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """列の有無と型をそろえ、おかしなデータがあれば ValueError を出す。"""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f'必要な列がありません: {missing}')
    df = df.copy()
    df['race_id'] = df['race_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    for c in ['horse_id', 'jockey_id', 'trainer_id']:
        df[c] = df[c].astype(str)
    for c in ['distance', 'horse_no', 'frame', 'age', 'weight_carried', 'horse_weight', 'weight_diff', 'odds',
              'popularity', 'finish']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df.duplicated(['race_id', 'horse_id']).any():
        raise ValueError('同じレースに同じ馬が2回出ています')
    if (df.groupby('race_id')['date'].nunique() > 1).any():
        raise ValueError('1つのレースに複数の日付があります')
    if df.duplicated(['date', 'horse_id']).any():
        raise ValueError('同じ馬が同じ日に2回出ています')
    if (df['odds'].dropna() < 1.0).any():
        raise ValueError('単勝オッズに 1.0 未満があります')
    winners = df[df['finish'] == 1].groupby('race_id').size()
    done = df[df['finish'].notna()]['race_id'].unique()
    if len(set(done) - set(winners.index)):
        raise ValueError('結果があるのに1着がいないレースがあります')
    return df.sort_values(['date', 'race_id', 'horse_no']).reset_index(drop=True)


def load_races(path: str) -> pd.DataFrame:
    return validate(pd.read_csv(path, dtype={'race_id': str, 'horse_id': str, 'jockey_id': str,
                                             'trainer_id': str}))


# ---------------------------------------------------------------- サンプルデータ

VENUES = ['東京', '中山', '京都', '阪神', '中京', '新潟', '福島', '小倉', '札幌', '函館']
DISTANCES = {'芝': [1200, 1400, 1600, 1800, 2000, 2400], 'ダート': [1200, 1400, 1700, 1800, 2100]}
GOINGS = ['良', '稍重', '重', '不良']


def make_sample(n_days: int = 120, races_per_day: int = 24, n_horses: int = 1500, n_jockeys: int = 80,
                n_trainers: int = 120, takeout: float = 0.20, seed: int = 0) -> pd.DataFrame:
    """隠れた「強さ」から着順とオッズを作るサンプルデータ。

    強さ = 馬の能力（少しずつ変わる）+ 芝/ダート・距離の適性 + 騎手の腕 + 調教師 + 枠の有利不利 + 体重増減
    着順は強さ + ガンベル乱数の大きい順（Plackett-Luce）。
    オッズは「市場」の見立て（強さにノイズを足したもの）から、控除率 takeout を引いて作る。
    市場はおおむね正しいが完全ではないので、モデルが本当に学べるかを確かめられる。
    """
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, n_horses)
    turf_apt = rng.normal(0, 0.5, n_horses)          # + は芝向き、- はダート向き
    best_dist = rng.choice([1200, 1400, 1600, 1800, 2000, 2400], n_horses)
    sex = rng.choice(['牡', '牝', 'セ'], n_horses, p=[0.5, 0.42, 0.08])
    age0 = rng.integers(2, 6, n_horses)
    base_weight = rng.normal(480, 25, n_horses)
    trainer_of = rng.integers(0, n_trainers, n_horses)
    jockey_skill = rng.normal(0, 0.35, n_jockeys)
    trainer_skill = rng.normal(0, 0.2, n_trainers)
    last_run = np.full(n_horses, -999)

    rows = []
    start = pd.Timestamp('2025-01-04')
    for day in range(n_days):
        date = start + pd.Timedelta(days=7 * (day // 2) + (day % 2))   # 土日開催
        ability += rng.normal(0, 0.03, n_horses)                          # 調子・成長
        rested = np.where(day - last_run >= 2)[0]                         # 中1開催日以上あけた馬
        rng.shuffle(rested)
        pos = 0
        for r in range(races_per_day):
            n = int(rng.integers(8, 17))
            if pos + n > len(rested):
                break
            horses = rested[pos:pos + n]
            pos += n
            surface = '芝' if rng.random() < 0.5 else 'ダート'
            dist = int(rng.choice(DISTANCES[surface]))
            going = rng.choice(GOINGS, p=[0.65, 0.2, 0.1, 0.05])
            venue = VENUES[(day // 2 * 3 + r // 12) % len(VENUES)]
            jockeys = rng.choice(n_jockeys, n, replace=False)
            horse_no = np.arange(1, n + 1)
            frame = np.ceil(horse_no * 8 / n).astype(int) if n > 8 else horse_no
            wdiff = np.round(rng.normal(0, 5, n) / 2) * 2
            strength = (ability[horses]
                        + turf_apt[horses] * (1 if surface == '芝' else -1)
                        - 0.5 * np.abs(best_dist[horses] - dist) / 400
                        + jockey_skill[jockeys] + trainer_skill[trainer_of[horses]]
                        - 0.02 * (horse_no - 1)
                        - 0.02 * np.abs(wdiff))
            # 着順
            perf = strength + rng.gumbel(0, 1, n)
            finish = np.empty(n, int)
            finish[np.argsort(-perf)] = np.arange(1, n + 1)
            # 市場（強さを少しノイズ付きで見ている）
            view = strength + rng.normal(0, 0.3, n)
            q = np.exp(view - view.max())
            q /= q.sum()
            odds = np.maximum(1.0, np.floor((1 - takeout) / q * 10) / 10)
            pop = np.empty(n, int)
            pop[np.argsort(odds, kind='stable')] = np.arange(1, n + 1)
            age = age0[horses] + day // 100
            for i, h in enumerate(horses):
                rows.append(dict(
                    race_id=f'{date:%Y%m%d}{r + 1:02d}', date=f'{date:%Y-%m-%d}', venue=venue, surface=surface,
                    distance=dist, going=going, horse_id=f'H{h:05d}', jockey_id=f'J{jockeys[i]:03d}',
                    trainer_id=f'T{trainer_of[h]:03d}', horse_no=int(horse_no[i]), frame=int(frame[i]),
                    age=int(age[i]), sex=sex[h], weight_carried=55.0 if sex[h] != '牝' else 53.0,
                    horse_weight=float(np.round(base_weight[h] + rng.normal(0, 4))), weight_diff=float(wdiff[i]),
                    odds=float(odds[i]), popularity=int(pop[i]), finish=int(finish[i])))
            last_run[horses] = day
    return validate(pd.DataFrame(rows))
