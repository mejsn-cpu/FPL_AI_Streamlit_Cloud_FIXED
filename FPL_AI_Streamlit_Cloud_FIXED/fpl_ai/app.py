from __future__ import annotations
from pathlib import Path
import streamlit as st
import pandas as pd
from .data import fetch_current, save_snapshot, load_snapshot
from .model import score
from .optimizer import optimize_squad, best_xi, optimize_with_locked
from .sim import simulate_players, simulate_captain

ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='FPL AI 2026/27',page_icon='⚽',layout='wide')

@st.cache_data(ttl=900)
def load_online():
    b,p,t,f=fetch_current(); save_snapshot(ROOT,b,p,t,f); return b,p,t,f

st.title('⚽ FPL AI — 2026/27')
st.caption('Live FPL data • player model • exact squad optimization • Monte Carlo • captain analysis')

with st.sidebar:
    st.header('⚙️ Model')
    horizon=st.slider('Fixture horizon',1,10,5)
    budget=st.number_input('Budget (£m)',90.0,100.0,100.0,0.1)
    objective=st.selectbox('Squad objective',['xpts','balanced','value'])
    sims=st.select_slider('Monte Carlo simulations',[5000,10000,20000,50000],value=20000)
    refresh=st.button('🔄 Načíst aktuální data')

if refresh: load_online.clear()
try:
    b,p,t,f=load_online()
    source='LIVE'
except Exception as e:
    try:
        b,p,t,f=load_snapshot(ROOT); source='LOCAL SNAPSHOT'
        st.warning('Online FPL API není dostupné. Používám poslední uložený snapshot.')
    except Exception:
        st.error('FPL API není dostupné a není k dispozici lokální snapshot. Připoj aplikaci k internetu a obnov stránku.')
        st.stop()

s=score(p,t,f,horizon)
next_gw=next((e['id'] for e in b.get('events',[]) if e.get('is_next')), None)

m1,m2,m3,m4=st.columns(4)
m1.metric('Next GW',next_gw or '—'); m2.metric('Players',len(s)); m3.metric('Budget',f'£{budget:.1f}m'); m4.metric('Data',source)

# Tabs
rank_tab,squad_tab,cap_tab,sim_tab,guide_tab=st.tabs(['🏆 Players','🧩 Squad','🧢 Captain','🎲 Simulation','ℹ️ Jak to používat'])

with rank_tab:
    st.subheader('Player ranking')
    pos=st.multiselect('Pozice',['GK','DEF','MID','FWD'],default=['GK','DEF','MID','FWD'])
    mapping={1:'GK',2:'DEF',3:'MID',4:'FWD'}
    teams_sel=st.multiselect('Tým',sorted(s.short_name.dropna().unique()))
    q=s[s.element_type.map(mapping).isin(pos)].copy()
    if teams_sel: q=q[q.short_name.isin(teams_sel)]
    q=q.sort_values('xpts',ascending=False)
    out=q[['web_name','singular_name_short','short_name','price','xpts','value','minutes_prob','risk','ownership','fixture_difficulty']].head(60).copy()
    out.columns=['Hráč','Pozice','Tým','Cena','xPts','xPts/£','Start %','Risk','Ownership %','Fixture difficulty']
    st.dataframe(out,use_container_width=True,hide_index=True)
    st.download_button('⬇️ Export ranking CSV',out.to_csv(index=False).encode('utf-8-sig'),'fpl_player_ranking.csv','text/csv')

with squad_tab:
    st.subheader('Optimalizovaný 15členný squad')
    squad,total=optimize_squad(s,budget,objective)
    xi=best_xi(squad)
    q=squad[['web_name','singular_name_short','short_name','price','xpts','value','minutes_prob','risk']].copy()
    q.columns=['Hráč','Pozice','Tým','Cena','xPts','xPts/£','Start %','Risk']
    st.dataframe(q.sort_values(['Pozice','xPts'],ascending=[True,False]),use_container_width=True,hide_index=True)
    c1,c2,c3=st.columns(3); c1.metric('Squad cost',f"£{squad.price.sum():.1f}m"); c2.metric('Expected points',f'{squad.xpts.sum():.1f}'); c3.metric('Bank',f"£{budget-squad.price.sum():.1f}m")
    if xi:
        st.markdown(f'### Doporučená XI — {xi[0]:.1f} xPts')
        xi_df=xi[1][['web_name','singular_name_short','short_name','xpts','minutes_prob','risk']].copy(); xi_df.columns=['Hráč','Pozice','Tým','xPts','Start %','Risk']
        st.dataframe(xi_df,use_container_width=True,hide_index=True)
        st.success(f'Formace {xi[2]}-{xi[3]}-{xi[4]}')
    st.download_button('⬇️ Export squad CSV',squad.to_csv(index=False).encode('utf-8-sig'),'fpl_optimal_squad.csv','text/csv')

with cap_tab:
    st.subheader('Captain optimizer')
    eligible=s[s.element_type.isin([3,4])].sort_values('captain_score',ascending=False).head(15).copy()
    cs=simulate_captain(eligible,max(5000,min(sims,20000)))
    cap=eligible[['id','web_name','short_name','price','xpts','ceiling','minutes_prob','ownership','risk']].merge(cs,on='id')
    cap['rank_score']=cap.captain_mean*(1+0.12*(1-cap.ownership/100))
    cap=cap.sort_values('rank_score',ascending=False)
    display=cap[['web_name','short_name','xpts','captain_mean','captain_p12','captain_p16','minutes_prob','ownership','risk']].copy()
    display.columns=['Hráč','Tým','xPts','Captain EV','P(12+)','P(16+)','Start %','Ownership %','Risk']
    st.dataframe(display,use_container_width=True,hide_index=True)
    if len(cap): st.success(f"🤝 Model captain: **{cap.iloc[0].web_name}**")

with sim_tab:
    st.subheader('Monte Carlo risk & ceiling')
    pool=s.sort_values('xpts',ascending=False).head(80)
    mc=simulate_players(pool,sims)
    mc=mc.merge(pool[['id','web_name','short_name','price','xpts']],on='id').sort_values('mean',ascending=False)
    mc.columns=['ID','Mean','P10','P50','P90','P(10+)','P(15+)','Blank','Hráč','Tým','Cena','Model xPts']
    st.dataframe(mc[['Hráč','Tým','Cena','Model xPts','Mean','P10','P50','P90','P(10+)','P(15+)','Blank']],use_container_width=True,hide_index=True)

with guide_tab:
    st.subheader('Co tato verze skutečně dělá')
    st.markdown('''
**Model není tipovač.** Z aktuálních FPL dat vytváří transparentní baseline xPts, zohledňuje dostupnost, formu, oficiální EP, ICT, xGI a obtížnost nadcházejících fixtures.

**Squad optimizer** řeší matematicky £100m, 15 hráčů, 2 GK / 5 DEF / 5 MID / 3 FWD a max. 3 hráče z klubu.

**Captain optimizer** odděluje očekávaný výnos od stropu a minutes risk.

**Monte Carlo** ukazuje distribuci výsledků, nikoliv jen průměr.

### Co model zatím NEPŘEDSTÍRÁ
Není to kouzelná predikce. Před prvními zápasy sezóny nemáme skutečné 2026/27 match-level sample. Proto model používá aktuální oficiální FPL signály a fixture context. Jakmile se odehrají GW, historické rolling features budou mít stále větší váhu.

### Jak bych podle něj hrál
Na začátku sezóny bych defaultně používal **balanced** objective, držel nízké minutes risk a optimalizoval horizont 5 GW. Po každém GW bych kontroloval hlavně změny v minutes probability, xGI, fixture strength a transfer value.
''')

st.caption('FPL AI je analytický nástroj. Predikce nejsou garancí výsledků.')
