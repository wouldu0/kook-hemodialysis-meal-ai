# -*- coding: utf-8 -*-
"""
measure_lever_adjustment_FOOK.py
레버(재료 대체·양 조정)가 한 끼마다 원본을 얼마나 바꾸는지 정량화.
목적: RL('덜 고쳐도 되는 궁합' 선택)이 승산 있는지 판단.

핵심 질문 = 레버 조정량을 레버별로 쪼갠다:
  · 불가피(반찬 뭘 골라도 해야 함): 조미료 x1/2, 가공품 감소
  · 선택영향(반찬 잘 고르면 줄 수 있음): 칼륨/인 재료대체, 김치교체
선택영향 비중이 크면 -> RL 승산 O. 대부분 불가피면 -> RL 스킵.

사용법 (cmd):
  cd /d E:\\final
  chcp 65001
  set PYTHONIOENCODING=utf-8
  python measure_lever_adjustment_FOOK.py

선택: 빠른 테스트로 앞 N끼만  ->  python measure_lever_adjustment_FOOK.py 30
결과는 콘솔 + FOOK_lever_adjustment_report.txt (utf-8) 로 저장.
"""
import sys, csv
from collections import Counter, defaultdict
import FOOK_adjust_levers as F


def snap(inst):
    return [(i['menu'], i['ing'], round(i['amt'], 4)) for i in inst]


def gram_change(a, b):
    """두 스냅샷 사이 바뀐 총 그램수 (양변화+제거+추가+대체 모두 포함)."""
    ca, cb = Counter(), Counter()
    for m, ing, amt in a:
        ca[(m, ing)] += amt
    for m, ing, amt in b:
        cb[(m, ing)] += amt
    return sum(abs(cb[k] - ca[k]) for k in set(ca) | set(cb))


def subst_count(a, b):
    """대체된 재료 수 (제거된 재료 종류 수 = 교체 횟수 근사)."""
    ka = {(m, ing) for m, ing, _ in a}
    kb = {(m, ing) for m, ing, _ in b}
    return len(ka - kb)   # 사라진 재료 = 대체/제거된 것


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    print("영양/식약청 DB 로딩중... (수십초 걸릴 수 있음)")
    F.NUT = F.load_all()

    rows = list(csv.reader(open('data/FOOK_diet_1_12_kor.csv', encoding='cp949')))
    W = 60

    # 레버별 누적 그램변화
    lever_g = defaultdict(float)
    lever_sub = defaultdict(int)
    tot_orig_g = 0.0
    tot_change_g = 0.0
    n_meals = 0
    per_meal_ratio = []

    for r in rows[1:]:
        slots = [r[1:6], r[6:11], r[11:16]]
        meals = [[c.strip() for c in sl if c.strip() and c.strip() != 'empty'] for sl in slots]
        if not all(meals):
            continue
        consumed = {'E': 0, 'protein': 0, 'K': 0, 'P': 0, 'Na': 0}
        for mi, menus in enumerate(meals):
            b = F.meal_bounds(W, consumed, meals_left=3 - mi)
            inst = F.expand(menus)
            orig_g = sum(i['amt'] for i in inst)
            s = snap(inst)

            def step(fn, key, *args):
                nonlocal s
                fn(inst, *args)
                s2 = snap(inst)
                lever_g[key] += gram_change(s, s2)
                lever_sub[key] += subst_count(s, s2)
                s = s2

            # adjust()와 동일 순서
            step(F.lever_kimchi, '김치교체(선택)')
            step(F.lever_sodium, '조미료x0.5(불가피)')
            step(F.lever_sodium_extra, '가공품감소(불가피)', b['Namax'])
            for _ in range(2):
                step(F.lever_potassium, '칼륨대체(선택)', b['Kmax'])
                step(F.lever_phosphorus, '인대체/감소(선택)', b['Pmax'])
                step(F.lever_protein, '단백질스케일(양)', b['Plo'], b['Phi'])
                step(F.lever_calorie, '열량스케일(양)', b['Elo'], b['Ehi'])

            final_g_change = gram_change(snap_orig(menus), s)
            for k in consumed:
                consumed[k] += F.totals(inst)[k]
            tot_orig_g += orig_g
            tot_change_g += final_g_change
            if orig_g > 0:
                per_meal_ratio.append(final_g_change / orig_g)
            n_meals += 1
        if limit and n_meals >= limit:
            break

    # 선택영향 vs 불가피 집계
    sel_keys = ['김치교체(선택)', '칼륨대체(선택)', '인대체/감소(선택)']
    ina_keys = ['조미료x0.5(불가피)', '가공품감소(불가피)']
    amt_keys = ['단백질스케일(양)', '열량스케일(양)']
    tot_lever_g = sum(lever_g.values()) or 1.0
    sel_g = sum(lever_g[k] for k in sel_keys)
    ina_g = sum(lever_g[k] for k in ina_keys)
    amt_g = sum(lever_g[k] for k in amt_keys)

    lines = []
    lines.append(f"측정 끼니 수: {n_meals}  (체중 {W}kg, 남은예산 방식)")
    lines.append(f"끼니당 평균 원본량: {tot_orig_g/max(n_meals,1):.1f} g")
    lines.append(f"끼니당 평균 조정량(바뀐 그램): {tot_change_g/max(n_meals,1):.1f} g "
                 f"(원본 대비 {100*tot_change_g/max(tot_orig_g,1):.1f}%)")
    lines.append("")
    lines.append("레버별 총 그램변화 / 대체횟수:")
    for k in sorted(lever_g, key=lambda x: -lever_g[x]):
        lines.append(f"  {k:<20} {lever_g[k]:>10.0f} g ({100*lever_g[k]/tot_lever_g:4.1f}%)"
                     f"   대체 {lever_sub[k]}회")
    lines.append("")
    lines.append("=== RL 판단 핵심 ===")
    lines.append(f"  선택영향(반찬 잘고르면 줄일 수 있음): {100*sel_g/tot_lever_g:4.1f}%  "
                 f"[김치·칼륨·인 대체]")
    lines.append(f"  불가피(뭘 골라도 해야 함)      : {100*ina_g/tot_lever_g:4.1f}%  "
                 f"[조미료·가공품]")
    lines.append(f"  양스케일(밥·단백찬 양조절)     : {100*amt_g/tot_lever_g:4.1f}%")
    lines.append("")
    lines.append("해석: '선택영향'이 클수록 RL('덜 고칠 궁합 선택')이 줄일 여지가 큼.")
    lines.append("      대부분 '불가피'면 RL 넣어도 조정량 별로 안 줄어듦 -> 스킵 권장.")

    report = "\n".join(lines)
    print("\n" + report)
    with open('FOOK_lever_adjustment_report.txt', 'w', encoding='utf-8') as f:
        f.write(report + "\n")
    print("\n(저장됨: FOOK_lever_adjustment_report.txt)")


# 원본 스냅샷 (대체 전) — 최종 변화량 계산용
def snap_orig(menus):
    return snap(F.expand(menus))


if __name__ == '__main__':
    main()
