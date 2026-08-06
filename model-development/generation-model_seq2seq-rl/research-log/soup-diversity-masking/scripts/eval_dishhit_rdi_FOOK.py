# -*- coding: utf-8 -*-
"""
eval_dishhit_rdi_FOOK.py — 정량 평가: Dish-hit rate(앵커 제외) + RDI + 동시충족률 + 영양소별 + 앵커보존율
여러 시드로 반복해 평균±표준편차 제시. (선행 논문 지표를 FOOK 구조에 맞게 적용)

※ 한계 명시(2026-07-26, 방법론 검토 반영):
  - 이 평가는 모델 학습에 쓴 것과 같은 1,095끼 데이터에서 시나리오를 뽑는 "내부 평가"다.
    별도로 떼어둔 학습/검증/테스트 분리가 없음 — "독립 테스트셋 평가"라고 부르면 안 됨.
  - 복원추출(같은 식단이 여러 번 뽑힐 수 있음)을 그대로 씀 — 정보량이 300보다 작을 수 있음.
  - 60kg·첫 끼(이전 섭취 없음) 조건에 한정된 결과 — 다른 체중/하루 누적 상황엔 일반화 불가.
  - 최적화 모듈 평가는 "시스템이 자신의 설계 규칙을 얼마나 만족했는가"이지 그 자체로 "임상적
    우수성"을 뜻하지 않음 — 별도 임상영양사 정성평가로 보완해야 함.

Dish-hit rate = 모델이 실제로 생성한 4개 슬롯(앵커 제외) 중 올바른 메뉴 유형이 배치된 비율
                (앵커 슬롯은 실제 데이터를 그대로 고정한 것이라 계산에서 뺀다 — 포함하면 20%가
                 자동 정답 처리되어 부풀려짐)
RDI score      = 5대 영양(열량·단백질·칼륨·인·나트륨) 중 충족한 기준 수 (0~5)
동시충족률      = 5개 기준을 전부 충족한 식단의 비율
영양소별 충족률 = 각 기준을 개별적으로 충족한 비율
앵커 보존율     = 유저 지정 메뉴의 재료 구성 유지 정도 (reward_lever_FOOK.a_keep)

같은 (시드 식단, 앵커 슬롯) 조건을 세 모델(모방/RL/RL+최적화)에 동일 적용 — 시나리오 차이를
통제한 대응(paired) 비교다. ("완전히 동일한 확률 표본"까지는 아님 — 모델별 생성 분포가 달라
난수 소비 방식도 달라질 수 있음.)

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python eval_dishhit_rdi_FOOK.py --n 300 --seeds 11,22,33,44,55
"""
import os, sys, copy, argparse, io
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf

from Model import Sequence_Generator
from train_FOOK import build_data
from eval_rl_FOOK import load_gen, generate, SPECIAL

FINAL = os.path.abspath(os.path.join('..', '..', '..'))
sys.path.insert(0, FINAL)

CLASS_TO_SLOT = {'밥': 0, '일품밥': 0, '일품(간식)': 0, '면': 0,
                 '국': 1, '수프(간식)': 1,
                 '볶음': 2, '조림': 2, '구이': 2, '찜': 2, '튀김': 2, '전': 2,
                 '샐러드': 3, '반찬': 3, '곡류(간식)': 3, '과일(간식)': 3,
                 '음료(간식)': 3, '우유(간식)': 3, '김치': 4}
NUT_LABELS = ('열량', '단백질', '칼륨', '인', '나트륨')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--seeds', type=str, default='11,22,33,44,55')
    ap.add_argument('--weight', type=int, default=60)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    cwd = os.getcwd()
    os.chdir(FINAL)
    import FOOK_adjust_levers as F
    import reward_lever_FOOK as R
    print('식약청 DB + app_core 로딩...')
    F.NUT = F.load_all()
    R.init(weight=args.weight)
    import app_core_FOOK as core   # MENU_CLASS에 군 메뉴까지 업데이트된 상태로 로딩
    os.chdir(cwd)

    nutrient_data, food_dict, diet_np, incidence = build_data()
    bs = int(diet_np.shape[0])
    b = F.meal_bounds(args.weight)   # 첫 끼 기준(고정 밴드) — 세 행 동일 기준 비교

    gen_imit, _ = load_gen('./results_FOOK/checkpoints', food_dict, nutrient_data, incidence, bs)
    gen_rl, _ = load_gen('./results_sweep_FOOK/i002', food_dict, nutrient_data, incidence, bs)

    def pass_flags_correct(t, b):
        # 인은 원값(P) 기준 — 이게 실제 앱(app_core_FOOK.passes)이 최종 통과를 판정하는 방식과
        # 같다(2026-07-27 확정). Peff(흡수보정)는 lever_phosphorus 내부에서 "어떤 재료로 대체할지"
        # 고르는 보조지표로만 쓰이고, 최종 임상 통과 여부의 게이트는 아니다. 여기서도 그대로 맞춘다.
        # (7/26엔 이 필드를 Peff로 바꿨었는데, 안전판정을 검증 안 된 흡수보정 가중치에 맡기는 셈이라
        #  원값으로 되돌림 — 대신 앱은 가공식품·유제품·견과류 개수를 별도로 캡(HIGH_P_MAX=2)해서
        #  첨가인 편중을 막는다. 이 개수 캡까지는 이 평가 스크립트가 아직 반영 못 함 — 알려진 한계.)
        return (b['Elo'] <= t['E'] <= b['Ehi'],
                b['Plo'] <= t['protein'] <= b['Phi'],
                t['K'] <= b['Kmax'],
                t['P'] <= b['Pmax'],
                t['Na_season'] <= b['Namax'])

    def evaluate(gen, seed, apply_lever):
        rng = np.random.default_rng(seed)
        idx = rng.choice(bs, size=args.n, replace=True)
        seeds_all = diet_np.numpy()[idx]
        anchors_all = rng.integers(0, 5, size=args.n)
        np.random.seed(seed)
        toks = generate(gen, seeds_all, anchors_all, food_dict)

        hits, gen_slots_total = 0, 0     # 앵커 제외(모델이 실제로 생성한 4개 슬롯만)
        rdi_scores, all5_hits, per_nut = [], 0, np.zeros(5)
        a_keeps = []
        skipped = 0
        for i in range(args.n):
            menus = [food_dict[int(t)] if int(t) not in SPECIAL else None for t in toks[i]]
            if any(m is None for m in menus):
                skipped += 1
                continue
            a_slot = int(anchors_all[i])
            for slot, m in enumerate(menus):
                if slot == a_slot:
                    continue    # 앵커는 실제 데이터 그대로 고정한 것 -> Dish-hit 계산에서 제외
                cls = F.MENU_CLASS.get(m)
                expect_slot = CLASS_TO_SLOT.get(cls, 3)
                gen_slots_total += 1
                if expect_slot == slot:
                    hits += 1
            anchor = menus[a_slot]
            if apply_lever:
                _, after, _, _ = F.adjust(list(menus), b, anchor=anchor)
                t = after
                _, det = R.meal_reward(menus, anchor, detail=True)
                a_keeps.append(det['a_keep'])
            else:
                t = F.totals(F.expand(list(menus)))
            flags = pass_flags_correct(t, b)
            rdi_scores.append(sum(flags))
            per_nut += np.array(flags, dtype=float)
            if all(flags):
                all5_hits += 1

        n_valid = args.n - skipped
        return {
            'dish_hit': hits / gen_slots_total if gen_slots_total else 0.0,
            'rdi': float(np.mean(rdi_scores)) if rdi_scores else 0.0,
            'all5_rate': all5_hits / n_valid if n_valid else 0.0,
            'per_nut': (per_nut / n_valid) if n_valid else per_nut,
            'anchor_keep': float(np.mean(a_keeps)) if a_keeps else None,
        }

    def run_all_seeds(gen, apply_lever):
        runs = [evaluate(gen, s, apply_lever) for s in seeds]
        agg = {}
        for k in ('dish_hit', 'rdi', 'all5_rate'):
            vals = [r[k] for r in runs]
            agg[k] = (np.mean(vals), np.std(vals))
        ak_vals = [r['anchor_keep'] for r in runs if r['anchor_keep'] is not None]
        agg['anchor_keep'] = (np.mean(ak_vals), np.std(ak_vals)) if ak_vals else (None, None)
        per_nut_stack = np.stack([r['per_nut'] for r in runs])
        agg['per_nut'] = (per_nut_stack.mean(axis=0), per_nut_stack.std(axis=0))
        return agg

    print(f'\n※ 내부 평가(학습 데이터와 동일 풀에서 시나리오 추출) · seed {len(seeds)}회 반복(n={args.n}/회) '
          f'· 60kg 첫 끼 조건 고정')
    rows = []
    if gen_imit:
        rows.append(('Seq2Seq(모방)', run_all_seeds(gen_imit, apply_lever=False)))
    if gen_rl:
        rows.append(('Seq2Seq+RL', run_all_seeds(gen_rl, apply_lever=False)))
        rows.append(('+ 최종 최적화 적용', run_all_seeds(gen_rl, apply_lever=True)))

    def fmt(mean_std, pct=False):
        m, s = mean_std
        if m is None:
            return '-'
        return f'{m*100:.1f}±{s*100:.1f}%' if pct else f'{m:.2f}±{s:.2f}'

    print(f'\n{"":<20} {"Dish-hit(4슬롯)":>18} {"RDI(0~5)":>14} {"5개동시":>16} {"앵커보존":>16}')
    print('-' * 88)
    for name, a in rows:
        print(f"{name:<20} {fmt(a['dish_hit'], True):>18} {fmt(a['rdi']):>14} "
              f"{fmt(a['all5_rate'], True):>16} {fmt(a['anchor_keep']):>16}")

    print(f'\n{"영양소별 충족률(평균±표준편차)":<20}')
    for name, a in rows:
        m, s = a['per_nut']
        print(f'{name:<20}', '  '.join(f'{lb} {m[j]*100:.1f}±{s[j]*100:.1f}%' for j, lb in enumerate(NUT_LABELS)))


if __name__ == '__main__':
    main()
