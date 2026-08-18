from __future__ import annotations
import numpy as np
import pandas as pd


def fixture_features(teams: pd.DataFrame, fixtures: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    future = fixtures[(fixtures['finished'] == False) & fixtures['event'].notna()].copy()
    future['event'] = pd.to_numeric(future['event'], errors='coerce').astype('Int64')
    events = sorted(future['event'].dropna().astype(int).unique())[:horizon]
    strength = teams.set_index('id')['strength'].to_dict()
    rows = []
    for tid in teams['id'].astype(int):
        fs = future[((future['team_h'] == tid) | (future['team_a'] == tid)) & future['event'].isin(events)]
        diffs = []
        for _, r in fs.iterrows():
            opp = int(r['team_a'] if int(r['team_h']) == tid else r['team_h'])
            opp_strength = float(strength.get(opp, 100))
            home = int(r['team_h']) == tid
            # Lower = easier. Home advantage is deliberately modest.
            diffs.append(opp_strength - (6.0 if home else 0.0))
        rows.append({'team': tid, 'fixture_difficulty': float(np.mean(diffs)) if diffs else 100.0,
                     'fixtures_in_horizon': len(diffs)})
    return pd.DataFrame(rows)


def score(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    p = players.copy()
    ff = fixture_features(teams, fixtures, horizon)
    p = p.merge(ff, on='team', how='left')
    numeric = ['form','points_per_game','ep_next','ep_this','selected_by_percent','minutes','ict_index',
               'chance_of_playing_next_round','expected_goals','expected_assists','expected_goal_involvements',
               'goals_scored','assists','clean_sheets','bonus','bps']
    for c in numeric:
        if c not in p.columns: p[c] = 0.0
        p[c] = pd.to_numeric(p[c], errors='coerce').fillna(0.0)

    chance = p['chance_of_playing_next_round'].replace(0, 100).clip(0,100) / 100.0
    status_penalty = np.select([p['status'].eq('i'), p['status'].eq('s'), p['status'].eq('u')], [0.55, 0.0, 0.65], default=1.0)
    availability = chance * status_penalty

    # v2 transparent ensemble. Official ep_next is a strong prior; underlying xGI and
    # historical FPL output are used as independent stabilizers. We avoid look-ahead by
    # never constructing a feature from future fixtures' results.
    xgi = p['expected_goal_involvements'].clip(lower=0)
    form = p['form'].clip(lower=0)
    ppg = p['points_per_game'].clip(lower=0)
    ict = (p['ict_index'] / 10.0).clip(lower=0)
    ep = p['ep_next'].clip(lower=0)
    base = 0.45*ep + 0.20*ppg + 0.15*form + 0.10*ict + 0.10*np.sqrt(xgi + 1.0)
    fixture_adj = np.clip((55.0 - p['fixture_difficulty'])/120.0, -0.18, 0.18)

    # Defensive-position floor/ceiling stabilizers.
    pos = p['element_type']
    pos_adj = np.select([pos.eq(1), pos.eq(2), pos.eq(3), pos.eq(4)], [0.96, 1.02, 1.00, 1.02], default=1.0)
    p['xpts'] = np.maximum(0.05, base * (0.78 + 0.22*availability) * (1+fixture_adj) * status_penalty * pos_adj)
    p['value'] = p['xpts'] / p['price'].replace(0, np.nan)

    # Minutes probability is intentionally conservative at the start of a season.
    recent_minutes_signal = np.clip(p['minutes']/1800.0, 0, 1)
    p['minutes_prob'] = np.clip(0.68 + 0.18*recent_minutes_signal + 0.14*chance, 0.05, 0.99)
    p.loc[p['minutes'] >= 900, 'minutes_prob'] = np.maximum(p.loc[p['minutes'] >= 900, 'minutes_prob'], 0.88)
    p.loc[p['status'].isin(['i','u']), 'minutes_prob'] *= 0.65
    p.loc[p['status'].eq('s'), 'minutes_prob'] = 0.02
    p['risk'] = np.clip(100*(1-p['minutes_prob']) + np.where(p['status']!='a', 12, 0), 0, 100)

    # Ceiling proxy: useful for captain selection.
    p['ceiling'] = p['xpts'] * (1.25 + np.clip(xgi,0,4)*0.08)
    p['captain_score'] = p['xpts'] * p['minutes_prob'] + 0.25*p['ceiling']
    return p
