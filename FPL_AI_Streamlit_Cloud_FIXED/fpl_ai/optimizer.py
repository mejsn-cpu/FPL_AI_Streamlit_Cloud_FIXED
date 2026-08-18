from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds


def optimize_squad(players: pd.DataFrame, budget: float = 100.0, objective: str = 'xpts'):
    p = players.reset_index(drop=True).copy()
    n = len(p)
    if objective == 'value':
        score = p['value'].fillna(0).to_numpy(float)
    elif objective == 'balanced':
        score = (p['xpts'] * (1 - 0.18*p['risk']/100) + 0.08*np.log1p(p['ownership'].fillna(0))).to_numpy(float)
    else:
        score = p['xpts'].fillna(0).to_numpy(float)
    c = -score
    A=[]; lb=[]; ub=[]
    A.append(p['price'].to_numpy(float)); lb.append(-np.inf); ub.append(float(budget))
    A.append(np.ones(n)); lb.append(15); ub.append(15)
    for pos, num in [(1,2),(2,5),(3,5),(4,3)]:
        A.append((p['element_type']==pos).astype(float)); lb.append(num); ub.append(num)
    for tid in sorted(p['team'].unique()):
        A.append((p['team']==tid).astype(float)); lb.append(-np.inf); ub.append(3)
    cons=LinearConstraint(np.array(A),np.array(lb),np.array(ub))
    res=milp(c,integrality=np.ones(n),bounds=Bounds(0,1),constraints=cons,options={'time_limit':45,'mip_rel_gap':0.0005})
    if not res.success:
        raise RuntimeError(f'Optimizer failed: {res.message}')
    return p.loc[res.x > .5].copy(), float(-res.fun)


def best_xi(squad: pd.DataFrame):
    best=None
    for d in range(3,6):
        for m in range(2,6):
            f=5-d-m
            if not (1 <= f <= 3): continue
            g=squad[squad.element_type==1].nlargest(1,'xpts')
            de=squad[squad.element_type==2].nlargest(d,'xpts')
            mi=squad[squad.element_type==3].nlargest(m,'xpts')
            fw=squad[squad.element_type==4].nlargest(f,'xpts')
            if min(len(g),len(de),len(mi),len(fw)) == 0: continue
            xi=pd.concat([g,de,mi,fw]).copy()
            val=float(xi['xpts'].sum())
            if best is None or val>best[0]: best=(val,xi,d,m,f)
    return best


def optimize_with_locked(players, locked_ids, budget=100.0):
    locked=set(int(x) for x in locked_ids)
    p=players.copy()
    if not locked: return optimize_squad(p,budget)
    # Feasibility check: locked players must obey max 3 club and position counts.
    lock=p[p.id.isin(locked)]
    if len(lock)!=len(locked): raise ValueError('Locked player ID not found in current data.')
    for tid,c in lock.team.value_counts().items():
        if c>3: raise ValueError('More than 3 locked players from one club.')
    for pos,lim in [(1,2),(2,5),(3,5),(4,3)]:
        if (lock.element_type==pos).sum()>lim: raise ValueError('Too many locked players in one position.')
    # Large bonus forces the optimizer to retain locks while preserving exact constraints.
    p=p.copy(); p['_locked']=p.id.isin(locked)
    p['_objective']=p.xpts + p['_locked'].astype(float)*1000
    original=p.xpts.copy(); p.xpts=p['_objective']
    result=optimize_squad(p,budget,'xpts')
    result=result[result.id.isin(set(result.id))].copy()
    result['xpts']=result.id.map(dict(zip(players.id, players.xpts)))
    return result, float(result.xpts.sum())
