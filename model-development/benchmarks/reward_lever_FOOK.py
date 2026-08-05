# -*- coding: utf-8 -*-
"""
reward_lever_FOOK.py — RL 보상: "레버가 유저 메뉴를 얼마나 덜 뜯어고쳤나"

설계:
  reward = (통과영양수/5) × (0.5 + 0.5 × 보존율)
    - 영양 달성도가 바닥을 깔고(레버가 성공해야 함), 보존은 그 위의 보너스.
    - "안 고쳐서 보존율 높은데 영양 실패"인 조합이 이길 수 없음.
    - 부분점수(통과영양수)로 실패 조합들 사이에도 기울기 존재 (하드 0이면 64%가 무신호).
  보존율 = 0.7 × 앵커(유저메뉴) 보존 + 0.3 × 전체 보존
    - 앵커 우선: 유저가 고른 메뉴의 맛을 지키는 게 이 RL의 존재이유.

영양 자체는 레버가 보장하므로 RL은 "레버 부담이 적은 궁합"만 학습 → 역할 중복 없음.
자연스러움(슬롯 적합성)은 모델의 use_beta(incidence matrix)가 담당.

사용:
  import FOOK_adjust_levers as F; F.NUT = F.load_all()
  import reward_lever_FOOK as R; R.init(weight=60)
  r = R.batch_reward([[menu1..menu5], ...], anchors=[m, ...])
"""
from collections import Counter
import numpy as np
import FOOK_adjust_levers as F

_B = None          # meal_bounds (체중 고정)
_CACHE = {}        # (menus, anchor) -> reward  (수렴할수록 히트율 상승)
W_ALL = 0.7        # 앵커 보존 가중
W_ANY = 0.3        # 전체 보존 가중
NUT_KEYS = ('E', 'protein', 'K', 'P', 'Na_season')


def init(weight=60):
    global _B
    _B = F.meal_bounds(weight)
    _CACHE.clear()
    return _B


def snap(inst):
    return [(i['menu'], i['ing'], round(i['amt'], 4)) for i in inst]


def gram_change(a, b):
    ca, cb = Counter(), Counter()
    for m, ing, amt in a: ca[(m, ing)] += amt
    for m, ing, amt in b: cb[(m, ing)] += amt
    return sum(abs(cb[k] - ca[k]) for k in set(ca) | set(cb))


def pass_flags(t, b):
    """5영양 각각 통과 여부. 나트륨은 조미료(첨가염)만."""
    return (b['Elo'] <= t['E'] <= b['Ehi'],
            b['Plo'] <= t['protein'] <= b['Phi'],
            t['K'] <= b['Kmax'],
            t['P'] <= b['Pmax'],
            t['Na_season'] <= b['Namax'])


def _keep(orig_snap, final_snap):
    g = sum(x[2] for x in orig_snap)
    if g <= 0:
        return 1.0
    return 1.0 - min(1.0, gram_change(orig_snap, final_snap) / g)


def meal_reward(menus, anchor, detail=False):
    """menus: 5메뉴 list. anchor: 유저 지정메뉴(menus에 포함). -> reward in [0,1]"""
    key = (tuple(menus), anchor)
    if not detail and key in _CACHE:
        return _CACHE[key]

    orig = F.expand(list(menus))
    if not orig:                       # 레시피 없는 조합 = 학습 불가 조합
        return (0.0, {}) if detail else 0.0
    _, after, inst, _ = F.adjust(list(menus), _B, anchor=anchor)

    flags = pass_flags(after, _B)
    pass_frac = sum(flags) / 5.0

    os_, fs_ = snap(orig), snap(inst)
    ao = [s for s in os_ if s[0] == anchor]
    af = [s for s in fs_ if s[0] == anchor]
    a_keep = _keep(ao, af) if ao else 1.0     # 앵커 보존율
    o_keep = _keep(os_, fs_)                  # 전체 보존율
    preserve = W_ALL * a_keep + W_ANY * o_keep

    r = pass_frac * (0.5 + 0.5 * preserve)
    if detail:
        return r, dict(pass_frac=pass_frac, flags=flags, a_keep=a_keep,
                       o_keep=o_keep, preserve=preserve, after=after)
    _CACHE[key] = r
    return r


def batch_reward(menus_list, anchors):
    return np.array([meal_reward(m, a) for m, a in zip(menus_list, anchors)],
                    dtype=np.float32)


def cache_stats():
    return len(_CACHE)
