from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_players(players: pd.DataFrame, n: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng=np.random.default_rng(seed)
    x=np.maximum(players.xpts.to_numpy(float),0.05)
    risk=np.clip(players.risk.to_numpy(float),0,100)
    pmin=np.clip(players.minutes_prob.to_numpy(float),0.02,0.99)
    # Lognormal is a useful positive, heavy-tailed approximation for FPL outcomes.
    sigma=np.clip(0.55 + risk/220, 0.45, 1.0)
    mu=np.log(x) - 0.5*sigma*sigma
    sims=rng.lognormal(mu,sigma,size=(n,len(players)))
    mins=rng.random((n,len(players))) < pmin
    sims*=mins
    return pd.DataFrame({
        'id':players.id.to_numpy(), 'mean':sims.mean(0),
        'p10':np.quantile(sims,.10,axis=0), 'p50':np.quantile(sims,.50,axis=0),
        'p90':np.quantile(sims,.90,axis=0), 'p10plus':(sims>=10).mean(0),
        'p15plus':(sims>=15).mean(0), 'blank':(sims<2).mean(0)
    })


def simulate_captain(players, n=20000, seed=42):
    rng=np.random.default_rng(seed)
    x=np.maximum(players.xpts.to_numpy(float),0.05)
    pmin=np.clip(players.minutes_prob.to_numpy(float),.02,.99)
    sigma=.55
    mu=np.log(x)-.5*sigma*sigma
    base=rng.lognormal(mu,sigma,size=(n,len(players)))
    base*=rng.random((n,len(players)))<pmin
    doubled=2*base
    return pd.DataFrame({'id':players.id.to_numpy(),'captain_mean':doubled.mean(0),'captain_p12':(doubled>=12).mean(0),'captain_p16':(doubled>=16).mean(0)})
