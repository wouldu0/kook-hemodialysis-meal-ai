# -*- coding: utf-8 -*-
"""
reward_lever_v2_FOOK.py — STEP1: 최종 서비스 판정과 정렬된 보상함수(R0~R4 ablation).

기존 reward_lever_FOOK.py(R0)는 그대로 두고(수정 안 함), 여기에 새 버전을 만든다.
프로덕션 함수(passes/unrealistic_reason/게이트 함수/adjust)는 복제하지 않고 그대로 import해서
쓴다 - app_core_FOOK을 import하면 그 모듈이 로드하는 production RL 모델도 같이 로드되지만
(부수효과), 여기서 그 모델 자체는 쓰지 않고 순수 함수(passes/_has_ingredient_clash/...)만
가져다 쓴다.

R0: 기존 reward_lever_FOOK.meal_reward (원본 그대로, 비교기준)
R1: R0 + 최종 final_pass 보너스 (서비스 게이트 전부 통과했는지 이진 신호 추가)
R2: R1 + 연속형 영양위반 패널티(calorie/protein/potassium/phosphorus/sodium)
R3: R2 + raw P 가중 패널티 강화(phosphorus_weight 별도 스윕)
R4: R3 + 현실성/보존/재료겹침 패널티(이진 게이트 위반에 대한 명시적 패널티)

가중치는 예시값을 맹목적으로 쓰지 않고, log_reward_components.py로 실측한 분포를 보고
calibrate_weights()에서 스케일을 맞춘다.
"""
import os, sys
sys.path.insert(0, r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code')
sys.path.insert(0, r'E:\final')
from collections import Counter
import numpy as np

import FOOK_adjust_levers as F           # 실제 레버(수정 안 함, import만)
import reward_lever_FOOK as R0MOD        # 기존 보상(원본 그대로)

# app_core_FOOK을 import하면 production 모델도 로드되는 부수효과가 있지만, 여기서는
# passes()/게이트 함수 4개만 재사용한다(복제 아님, 실제 함수 그대로 import).
import app_core_FOOK as core

NUT_KEYS = ('E', 'protein', 'K', 'P', 'Na_season')


def upper_violation(value, upper):
    return max(0.0, (value - upper) / max(upper, 1e-8))


def lower_violation(value, lower):
    return max(0.0, (lower - value) / max(lower, 1e-8))


def range_violation(value, lower, upper):
    return lower_violation(value, lower) + upper_violation(value, upper)


def snap(inst):
    return [(i['menu'], i['ing'], round(i['amt'], 4)) for i in inst]


def gram_change(a, b):
    ca, cb = Counter(), Counter()
    for m, ing, amt in a: ca[(m, ing)] += amt
    for m, ing, amt in b: cb[(m, ing)] += amt
    return sum(abs(cb[k] - ca[k]) for k in set(ca) | set(cb))


def _keep(orig_snap, final_snap):
    g = sum(x[2] for x in orig_snap)
    if g <= 0:
        return 1.0
    return 1.0 - min(1.0, gram_change(orig_snap, final_snap) / g)


def eval_meal(menus, anchor, requested_ingredient=None, bounds=None):
    """menus(5개)+anchor를 실제 서비스 함수(F.adjust/passes/게이트)로 전부 평가해
    보상계산에 필요한 모든 원재료(components)를 딕셔너리로 반환한다.
    bounds가 None이면 R0MOD._B(전역, R0MOD.init()으로 세팅된 것)를 쓴다 - 기존 학습 코드와
    동일한 방식."""
    b = bounds if bounds is not None else R0MOD._B
    orig = F.expand(list(menus))
    if not orig:
        return None   # 레시피 없는 조합 = 학습 불가(R0와 동일 처리)
    before, after, inst, _ = F.adjust(list(menus), b, anchor=anchor)

    # --- 서비스와 동일한(엄밀한 <, 미만) 5영양 판정 ---
    flags_strict = {
        '열량': b['Elo'] <= after['E'] <= b['Ehi'],
        '단백질': b['Plo'] <= after['protein'] <= b['Phi'],
        '칼륨': after['K'] < b['Kmax'],
        '인': after['P'] < b['Pmax'],
        '나트륨': after['Na_season'] <= b['Namax'],
    }
    pass_frac = sum(flags_strict.values()) / 5.0

    # --- 연속형 위반도 ---
    calorie_violation = range_violation(after['E'], b['Elo'], b['Ehi'])
    protein_violation = range_violation(after['protein'], b['Plo'], b['Phi'])
    potassium_violation = upper_violation(after['K'], b['Kmax'])
    phosphorus_violation = upper_violation(after['P'], b['Pmax'])
    sodium_violation = upper_violation(after['Na_season'], b['Namax'])

    # --- 서비스 게이트(현실성/재료겹침/과다군) - 실제 함수 그대로 ---
    unreal = F.unrealistic_reason(inst)
    clash = core._has_ingredient_clash(menus)
    overload = core._has_seafood_overload(menus)
    p_overload = core._has_high_p_overload(menus)

    # --- 최종 서비스 성공 여부(day 연속성=dup_today는 단일 샘플이라 항상 False로 취급) ---
    final_pass = (pass_frac == 1.0 and unreal is None and not clash
                  and not overload and not p_overload)

    # --- 보존(기존 R0 방식 재사용) ---
    os_, fs_ = snap(orig), snap(inst)
    ao = [s for s in os_ if s[0] == anchor]
    af = [s for s in fs_ if s[0] == anchor]
    a_keep = _keep(ao, af) if ao else 1.0
    o_keep = _keep(os_, fs_)
    preserve = R0MOD.W_ALL * a_keep + R0MOD.W_ANY * o_keep

    final_menus = list(dict.fromkeys(i['menu'] for i in inst))
    requested_menu_lost = (anchor is not None) and (anchor not in final_menus)

    requested_ingredient_lost = False
    if requested_ingredient:
        anchor_ings = [i['ing'] for i in inst if i['menu'] == anchor]
        requested_ingredient_lost = not any(requested_ingredient in ing for ing in anchor_ings)

    return {
        'after': after, 'inst': inst, 'flags_strict': flags_strict, 'pass_frac': pass_frac,
        'calorie_violation': calorie_violation, 'protein_violation': protein_violation,
        'potassium_violation': potassium_violation, 'phosphorus_violation': phosphorus_violation,
        'sodium_violation': sodium_violation,
        'unrealistic_amount': unreal is not None, 'ingredient_clash': clash,
        'overload': overload, 'p_overload': p_overload, 'final_pass': final_pass,
        'a_keep': a_keep, 'o_keep': o_keep, 'preserve': preserve,
        'requested_menu_lost': requested_menu_lost, 'requested_ingredient_lost': requested_ingredient_lost,
    }


# ============================================================
# 가중치 — log_reward_components_FOOK.py 실측(BASE 웜스타트 시점 500건 원시후보) 기준 보정.
# 예시값(10.0/1.0/1.5/2.0/3.0 등)을 그대로 쓰지 않았다. 근거:
#   - r0_reward(기존 보상) 실측 분포: mean=0.65, range=[0.11, 0.96] -> R1~R4의 추가항이 이
#     스케일을 완전히 뭉개거나(과대) 무시되지(과소) 않도록 비슷한 자릿수로 맞춤.
#   - final_pass율 실측 21.0% (아직 학습 전 BASE라 낮음) -> final_pass_bonus를 r0_reward의
#     최댓값(0.96)과 비슷한 1.0으로 잡아 "완전 통과가 부분점수 전부를 능가"하게 함(우선순위1).
#   - 위반(violation) 실측 평균이 calorie=0.011/protein=0.034/potassium=0.017/phosphorus=0.014/
#     sodium=0.0004로 전부 작음(0~0.05대) -> 예시의 1.0~2.0 가중을 그대로 곱하면 기여가
#     0.01~0.07 수준으로 미미. r0_reward 스케일(0.1~1.0대)에서 의미있게 느껴지도록 예시 대비
#     동일 상대순서(calorie<protein<potassium≈sodium, phosphorus는 별도 스윕)를 유지하되
#     절대크기를 0.5~1.0 대역으로 올림.
#   - unrealistic_amount 실측 35.8%(꽤 흔함) -> 예시의 final_pass_bonus 대비 상대비율(3.0/10.0=0.3)
#     을 유지해 0.4(약간 상향, 흔한 문제라 더 강하게 억제).
#   - menu_lost/ingredient_lost/clash/overload도 예시의 상대비율(각 30%,30%,20%,20% of
#     final_pass_bonus)을 유지.
# phosphorus_weight는 STEP1-3에서 별도 스윕(R3) — 예시 {2,4,6,8}은 그들의 sodium_w=2.0 기준
# 스케일이라, 내 calibration의 potassium_w=sodium_w=1.0 기준으로 비례 축소해 {1,2,3,4}로 스윕.
# ============================================================
WEIGHTS = {
    'final_pass_bonus': 1.0,
    'calorie_w': 0.5,
    'protein_w': 0.75,
    'potassium_w': 1.0,
    'phosphorus_w': 1.0,            # R3에서 {1.0, 2.0, 3.0, 4.0}로 스윕
    'sodium_w': 1.0,
    'unrealistic_penalty': 0.4,
    'menu_lost_penalty': 0.3,
    'ingredient_lost_penalty': 0.3,
    'clash_penalty': 0.2,
    'overload_penalty': 0.2,
}


def reward_R0(menus, anchor, requested_ingredient=None, bounds=None):
    """기존 reward_lever_FOOK.meal_reward 그대로(비교기준)."""
    return R0MOD.meal_reward(menus, anchor)


def reward_R1(menus, anchor, requested_ingredient=None, bounds=None, w=None):
    w = w or WEIGHTS
    c = eval_meal(menus, anchor, requested_ingredient, bounds)
    if c is None:
        return 0.0
    r0 = c['pass_frac'] * (0.5 + 0.5 * c['preserve'])
    return r0 + w['final_pass_bonus'] * (1.0 if c['final_pass'] else 0.0)


def reward_R2(menus, anchor, requested_ingredient=None, bounds=None, w=None):
    w = w or WEIGHTS
    c = eval_meal(menus, anchor, requested_ingredient, bounds)
    if c is None:
        return 0.0
    r0 = c['pass_frac'] * (0.5 + 0.5 * c['preserve'])
    r = r0 + w['final_pass_bonus'] * (1.0 if c['final_pass'] else 0.0)
    r -= w['calorie_w'] * c['calorie_violation']
    r -= w['protein_w'] * c['protein_violation']
    r -= w['potassium_w'] * c['potassium_violation']
    r -= w['phosphorus_w'] * c['phosphorus_violation']
    r -= w['sodium_w'] * c['sodium_violation']
    return r


def reward_R3(menus, anchor, requested_ingredient=None, bounds=None, w=None, phosphorus_weight=None):
    w = dict(w or WEIGHTS)
    if phosphorus_weight is not None:
        w['phosphorus_w'] = phosphorus_weight
    c = eval_meal(menus, anchor, requested_ingredient, bounds)
    if c is None:
        return 0.0
    r0 = c['pass_frac'] * (0.5 + 0.5 * c['preserve'])
    r = r0 + w['final_pass_bonus'] * (1.0 if c['final_pass'] else 0.0)
    r -= w['calorie_w'] * c['calorie_violation']
    r -= w['protein_w'] * c['protein_violation']
    r -= w['potassium_w'] * c['potassium_violation']
    r -= w['phosphorus_w'] * c['phosphorus_violation']
    r -= w['sodium_w'] * c['sodium_violation']
    return r


def reward_R4(menus, anchor, requested_ingredient=None, bounds=None, w=None, phosphorus_weight=None):
    w = dict(w or WEIGHTS)
    if phosphorus_weight is not None:
        w['phosphorus_w'] = phosphorus_weight
    c = eval_meal(menus, anchor, requested_ingredient, bounds)
    if c is None:
        return 0.0
    r0 = c['pass_frac'] * (0.5 + 0.5 * c['preserve'])
    r = r0 + w['final_pass_bonus'] * (1.0 if c['final_pass'] else 0.0)
    r -= w['calorie_w'] * c['calorie_violation']
    r -= w['protein_w'] * c['protein_violation']
    r -= w['potassium_w'] * c['potassium_violation']
    r -= w['phosphorus_w'] * c['phosphorus_violation']
    r -= w['sodium_w'] * c['sodium_violation']
    r -= w['unrealistic_penalty'] * (1.0 if c['unrealistic_amount'] else 0.0)
    r -= w['menu_lost_penalty'] * (1.0 if c['requested_menu_lost'] else 0.0)
    r -= w['ingredient_lost_penalty'] * (1.0 if c['requested_ingredient_lost'] else 0.0)
    r -= w['clash_penalty'] * (1.0 if c['ingredient_clash'] else 0.0)
    r -= w['overload_penalty'] * (1.0 if (c['overload'] or c['p_overload']) else 0.0)
    return r


REWARD_FUNCS = {'R0': reward_R0, 'R1': reward_R1, 'R2': reward_R2, 'R3': reward_R3, 'R4': reward_R4}
