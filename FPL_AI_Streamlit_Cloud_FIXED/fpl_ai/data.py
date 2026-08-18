from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import requests
import pandas as pd

BASE='https://fantasy.premierleague.com/api'
HEADERS={'User-Agent':'Mozilla/5.0 (FPL-AI/Final)'}


def get_json(url:str, timeout:int=30)->Any:
    r=requests.get(url,timeout=timeout,headers=HEADERS)
    r.raise_for_status(); return r.json()


def fetch_current():
    b=get_json(f'{BASE}/bootstrap-static/')
    fixtures=get_json(f'{BASE}/fixtures/')
    p=pd.DataFrame(b['elements']); t=pd.DataFrame(b['teams']); pos=pd.DataFrame(b['element_types'])
    team_cols=['id','name','short_name','strength','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away']
    p=p.merge(t[team_cols],left_on='team',right_on='id',suffixes=('','_team'))
    p=p.merge(pos[['id','singular_name_short']],left_on='element_type',right_on='id',suffixes=('','_pos'))
    p['price']=p['now_cost']/10
    for c in ['form','points_per_game','ep_next','ep_this','selected_by_percent','total_points','minutes','creativity','influence','threat','ict_index','chance_of_playing_next_round','expected_goals','expected_assists','expected_goal_involvements','goals_scored','assists','clean_sheets','bonus','bps']:
        if c in p: p[c]=pd.to_numeric(p[c],errors='coerce')
    p['ownership']=p['selected_by_percent']
    return b,p,t,pd.DataFrame(fixtures)


def save_snapshot(root:Path,b,p,t,f):
    d=root/'data'/'snapshots'; d.mkdir(parents=True,exist_ok=True)
    (d/'bootstrap.json').write_text(json.dumps(b),encoding='utf8')
    p.to_csv(d/'players.csv',index=False); t.to_csv(d/'teams.csv',index=False); f.to_csv(d/'fixtures.csv',index=False)


def load_snapshot(root:Path):
    d=root/'data'/'snapshots'
    if not (d/'bootstrap.json').exists(): raise FileNotFoundError('No local snapshot.')
    b=json.loads((d/'bootstrap.json').read_text(encoding='utf8'))
    return b,pd.read_csv(d/'players.csv'),pd.read_csv(d/'teams.csv'),pd.read_csv(d/'fixtures.csv')
