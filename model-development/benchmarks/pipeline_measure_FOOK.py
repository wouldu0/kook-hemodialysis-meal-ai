# -*- coding: utf-8 -*-
"""
pipeline_measure_FOOK.py — 생성모델 → 레버 파이프라인 측정 (코어 검증 + RL 승산 재확인)

흐름:
  1) 학습된 생성모델로 한끼 N개 생성 (seed=랜덤 실제한끼, 샘플링)
  2) 생성 한끼 3개=하루로 묶어 남은예산 레버 조정
  3) 측정: 하루 5영양 통과율 + 레버 조정량(선택영향 vs 불가피)
  4) 대조군: 같은 seed(실제 한끼)도 동일 측정 → 생성 vs 실제 비교

실행 (foodbert):
  conda activate foodbert
  set TF_USE_LEGACY_KERAS=1
  cd /d E:\\final
  python pipeline_measure_FOOK.py --days 120 --temp 0.8
"""
import os, sys, argparse, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

CODE = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code'
DATA = r'E:\final\data'


# ---------- 1) 생성 (cwd=CODE) ----------
def generate_meals(n_meals, temp, rng_seed):
    os.chdir(CODE)
    sys.path.insert(0, CODE)
    import tensorflow as tf
    from util import (nutrition_preprocessor, diet_sequence_preprocessor,
                      food_to_token, diet_to_incidence)
    import generate_FOOK as G

    feature = pd.read_csv(os.path.join(DATA, 'FOOK_nutrition.csv'))
    nutrient_data, food_dict = nutrition_preprocessor(feature_data=feature)()
    diet = pd.read_csv(os.path.join(DATA, 'FOOK_meals_for_model.csv'))
    dsp = diet_sequence_preprocessor(sequence_data=diet, DB_quality='correct2', integrate=False)
    diet = dsp(nutrient_data)
    diet_np = food_to_token(diet, nutrient_data, empty_delete=True, num_empty=3)
    incidence = diet_to_incidence(diet_np, food_dict)

    gen = G.restore(food_dict, nutrient_data, incidence, batch_size=diet_np.shape[0])

    np.random.seed(rng_seed)
    idx = np.random.choice(diet_np.shape[0], size=n_meals, replace=True)
    seeds = diet_np.numpy()[idx]
    out = G.generate(gen, seeds, temp=temp)

    def to_menus(rows):
        return [[food_dict[int(t)] for t in r if int(t) not in (0, 825, 826)] for r in rows]
    return to_menus(out), to_menus(seeds)


# ---------- 2) 레버 측정 (cwd=E:/final) ----------
def snap(inst):
    return [(i['menu'], i['ing'], round(i['amt'], 4)) for i in inst]

def gram_change(a, b):
    ca, cb = Counter(), Counter()
    for m, ing, amt in a: ca[(m, ing)] += amt
    for m, ing, amt in b: cb[(m, ing)] += amt
    return sum(abs(cb[k] - ca[k]) for k in set(ca) | set(cb))


def measure(meal_days, F, W=60):
    """meal_days: list of days, each = [meal1(5메뉴), meal2, meal3]. 남은예산 흐름."""
    dt = F.day_targets(W)
    lever_g = defaultdict(float)
    n_days = 0; all_ok = 0; cnt = [0]*5
    n_meals = 0; meal_na_ok = 0
    tot_orig = tot_change = 0.0

    for day in meal_days:
        consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0, 'Na_season': 0}
        for mi, menus in enumerate(day):
            b = F.meal_bounds(W, consumed, meals_left=3 - mi)
            inst = F.expand(menus)
            orig = snap(inst); orig_g = sum(i['amt'] for i in inst)

            def step(fn, key, *a):
                nonlocal cur
                fn(inst, *a); s2 = snap(inst)
                lever_g[key] += gram_change(cur, s2); cur = s2
            cur = orig
            step(F.lever_kimchi, '김치교체(선택)')
            step(F.lever_sodium, '조미료(불가피)')
            step(F.lever_sodium_extra, '가공품(불가피)', F.NA_TOTAL_CEIL)
            for _ in range(2):                           # adjust()와 동일 (하루누적 기준 2회 최적)
                step(F.lever_potassium, '칼륨대체(선택)', b['Kmax'])
                step(F.lever_phosphorus, '인대체(선택)', b['Pmax'])
                step(F.lever_protein, '단백질(양)', b['Plo'], b['Phi'])
                step(F.lever_sodium, '조미료(불가피)')   # adjust()와 동일: 열량 레버 "앞"
                step(F.lever_calorie, '열량(양)', b['Elo'], b['Ehi'])

            t = F.totals(inst)
            for k in consumed: consumed[k] += t[k]
            tot_orig += orig_g; tot_change += gram_change(orig, snap(inst))
            n_meals += 1
            if t['Na_season'] <= b['Namax']: meal_na_ok += 1
        # 하루 판정 (나트륨=조미료 첨가염만)
        k = (dt['Elo'] <= consumed['E'] <= dt['Ehi'], dt['Plo'] <= consumed['protein'] <= dt['Phi'],
             consumed['K'] <= dt['Kmax'], consumed['P'] <= dt['Pmax'], consumed['Na_season'] <= dt['Namax'])
        for i in range(5): cnt[i] += k[i]
        all_ok += all(k); n_days += 1

    sel = sum(lever_g[x] for x in ['김치교체(선택)', '칼륨대체(선택)', '인대체(선택)'])
    ina = sum(lever_g[x] for x in ['조미료(불가피)', '가공품(불가피)'])
    amt = sum(lever_g[x] for x in ['단백질(양)', '열량(양)'])
    tot = sel + ina + amt or 1.0
    return dict(n_days=n_days, all_ok=all_ok, cnt=cnt, n_meals=n_meals, meal_na_ok=meal_na_ok,
                sel=100*sel/tot, ina=100*ina/tot, amt=100*amt/tot,
                change_ratio=100*tot_change/max(tot_orig, 1))


def report(name, r):
    labels = ['열량', '단백질', '칼륨', '인', '나트륨']
    print(f'\n=== {name} ({r["n_days"]}일) ===')
    for i, l in enumerate(labels):
        print(f'  {l:<5} {r["cnt"][i]:>4}/{r["n_days"]}')
    print(f'  하루 5개 전부 충족: {r["all_ok"]}/{r["n_days"]} ({100*r["all_ok"]/r["n_days"]:.0f}%)')
    print(f'  끼니 나트륨: {r["meal_na_ok"]}/{r["n_meals"]}')
    print(f'  조정량(원본대비): {r["change_ratio"]:.0f}%  | 선택영향 {r["sel"]:.0f}% · 불가피 {r["ina"]:.0f}% · 양 {r["amt"]:.0f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=120)
    ap.add_argument('--temp', type=float, default=0.8)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    n_meals = args.days * 3

    print(f'생성 중... ({n_meals}끼 = {args.days}일)')
    gen_meals, seed_meals = generate_meals(n_meals, args.temp, args.seed)

    os.chdir(r'E:\final')
    import FOOK_adjust_levers as F
    print('식약청 DB 로딩...')
    F.NUT = F.load_all()

    def to_days(flat):
        return [flat[i*3:i*3+3] for i in range(len(flat)//3)]
    gen_days = to_days(gen_meals)
    seed_days = to_days(seed_meals)

    r_gen = measure(gen_days, F)
    r_seed = measure(seed_days, F)
    report('생성 한끼 → 레버', r_gen)
    report('실제 한끼(seed) → 레버 [대조군]', r_seed)
    print('\n해석: 생성이 실제와 비슷하면 "생성이 레버가치 유지" 성공. 선택영향% = RL 줄일여지.')


if __name__ == '__main__':
    main()
