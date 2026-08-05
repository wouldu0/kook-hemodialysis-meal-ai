# -*- coding: utf-8 -*-
"""
bench_lever_reward_FOOK.py — RL 보상(레버 조정량) 실현가능성 확인.
  1) F.adjust() 1회 비용 측정 → 에폭당 비용 추정 (배치 1095)
  2) 보상 함수 프로토타입: 앵커 보존율 × 통과여부
  3) 보상이 메뉴조합에 따라 실제로 변별력 있는지(분산) 확인 — 상수면 RL 학습 불가
"""
import time, random
import numpy as np, pandas as pd
from collections import Counter
import FOOK_adjust_levers as F

F.NUT = F.load_all()
diet = pd.read_csv('data/FOOK_meals_for_model.csv')
W = 60
b = F.meal_bounds(W)


def snap(inst):
    return [(i['menu'], i['ing'], round(i['amt'], 4)) for i in inst]

def gram_change(a, b_):
    ca, cb = Counter(), Counter()
    for m, ing, amt in a: ca[(m, ing)] += amt
    for m, ing, amt in b_: cb[(m, ing)] += amt
    return sum(abs(cb[k] - ca[k]) for k in set(ca) | set(cb))

def passes(t, bd):
    return (bd['Elo'] <= t['E'] <= bd['Ehi'] and bd['Plo'] <= t['protein'] <= bd['Phi'] and
            t['K'] <= bd['Kmax'] and t['P'] <= bd['Pmax'] and t['Na_season'] <= bd['Namax'])


def reward(menus, anchor):
    """레버가 앵커(유저 메뉴)를 얼마나 덜 건드렸나. 영양 미통과면 0 (레버가 포기한 걸 보상하면 안 됨)."""
    orig = F.expand(menus)
    _, after, inst, _ = F.adjust(list(menus), b, anchor=anchor)
    if not passes(after, b):
        return 0.0, 0.0, 0.0
    ao = [s for s in snap(orig) if s[0] == anchor]
    af = [s for s in snap(inst) if s[0] == anchor]
    a_g = sum(x[2] for x in ao) or 1.0
    a_keep = 1 - min(1.0, gram_change(ao, af) / a_g)          # 앵커 보존율
    o_g = sum(i['amt'] for i in orig) or 1.0
    o_keep = 1 - min(1.0, gram_change(snap(orig), snap(inst)) / o_g)   # 전체 보존율
    return 0.7 * a_keep + 0.3 * o_keep, a_keep, o_keep


# ---- 1) 속도 ----
meals = [[r.iloc[c] for c in range(5) if isinstance(r.iloc[c], str)] for _, r in diet.iterrows()]
sample = meals[:200]
t0 = time.perf_counter()
for m in sample:
    F.adjust(list(m), b, anchor=m[2])
dt = (time.perf_counter() - t0) / len(sample)
print(f'=== 속도 ===')
print(f'  adjust() 1회: {dt*1000:.1f} ms')
print(f'  배치 1095 = 에폭당 {dt*1095:.1f} s  → 1000에폭 {dt*1095*1000/60:.0f} 분')
print(f'  (참고: 모방 에폭 ~0.5s)')

# ---- 2)(3) 보상 변별력 ----
t0 = time.perf_counter()
rs, aks, oks = [], [], []
for m in meals[:300]:
    r_, ak, ok = reward(m, m[2])   # slot2(주찬)를 유저 앵커로 가정
    rs.append(r_); aks.append(ak); oks.append(ok)
rs, aks, oks = map(np.array, (rs, aks, oks))
print(f'\n=== 보상 변별력 (실제 300끼, 앵커=slot2 주찬) ===')
print(f'  보상       평균 {rs.mean():.3f}  std {rs.std():.3f}  min {rs.min():.3f}  max {rs.max():.3f}')
print(f'  앵커보존율 평균 {aks.mean():.3f}  std {aks.std():.3f}')
print(f'  전체보존율 평균 {oks.mean():.3f}  std {oks.std():.3f}')
print(f'  통과(보상>0) 비율: {100*(rs>0).mean():.0f}%')
print(f'  >>> std가 0에 가까우면 RL 학습신호 없음. 0.1+ 면 변별력 있음.')

# ---- 4) 캐시 효용: 같은 메뉴조합 반복률 ----
keys = [tuple(sorted(m)) for m in meals]
print(f'\n=== 캐시 ===')
print(f'  고유 메뉴조합 {len(set(keys))} / {len(keys)}끼 → 실제데이터 중복률 {100*(1-len(set(keys))/len(keys)):.0f}%')
print(f'  (생성 중엔 조합공간이 커서 초기 히트율 낮음. 수렴할수록 상승)')
