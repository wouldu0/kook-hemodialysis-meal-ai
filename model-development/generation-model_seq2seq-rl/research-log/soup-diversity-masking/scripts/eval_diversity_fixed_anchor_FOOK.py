# -*- coding: utf-8 -*-
"""
eval_diversity_fixed_anchor_FOOK.py — 고정 조건(동일 환자·동일 seed 식단·동일 앵커) 다양성 평가 v2

질문: "임상 영양 최적화 때문에 특정 메뉴만 반복 생성되는가?"

v1과 달라진 점(검증 지적 반영):
  - "+최적화" 단계를 자체 구현(generate+adjust 1회)이 아니라 실제 프로덕션 함수
    app_core_FOOK.make_meal()을 그대로 재사용한다 — 48개 후보 생성 → SLOT_OK 카테고리 마스크·
    밥/국/김치 중복금지·재료겹침(clash)·해산물군 과다·고인비율군 과다·비현실적 양 필터링 →
    passes() 전부 통과 시 즉시 채택, 48개 다 실패하면 최고점수(부분통과+현실성+비겹침) 후보 채택.
    이 로직을 다시 손으로 구현하면 미묘하게 달라질 위험이 있어서, 대신 app_core_FOOK.diet_np를
    "고정 seed 식단만 반복된 배열"로 바꿔치기해서 make_meal 내부의 랜덤 seed 추첨이 항상 같은
    행을 뽑게 만든다(마스킹·선택 로직 자체는 원본 그대로, seed만 고정) — 아래 _patched_diet_np() 참고.
  - 조건·시드별로 따로 집계(seed 5개 각각 지표 계산 후 평균±표준편차) — 이전엔 5시드를 풀링해서
    점추정치 하나만 냈음.
  - raw count를 CSV로 남긴다(감사용) — {SCRIPT_DIR}/eval_diversity_out/ 아래.

주의(참고용, 별도 확인 필요): make_meal의 채택 기준 passes()(app_core_FOOK.py)는 인(P)을
  원값 기준으로 판정한다 — Peff가 아니다. eval_dishhit_rdi_FOOK.py에서 쓴 Peff 기준과 다르다.
  이건 "실제 앱이 뭘 기준으로 삼는지" 자체를 다시 확인해야 할 사안이라 이 스크립트에서는 손대지
  않고 그대로 둔다(다양성 측정과는 별개 이슈).

조건 선정(사전 등록 — 결과를 본 뒤 유리한 조건을 고르지 않는다):
  국(slot1)/주찬(slot2)/부찬(slot3)을 각각 앵커로 하는 조건 3개.
  세 조건 모두 학습데이터 0번째 행(첫 번째 식단)을 seed로 쓰고, 앵커로 지정하는 슬롯의 실제
  메뉴명을 그대로 앵커로 쓴다 — "첫 번째 적격 식단을 쓴다"는 규칙을 그대로 따른 것이라 조건
  선정에 임의성이 없다.

집계 원칙:
  - 세 조건을 합쳐서 계산하지 않는다. 조건별로 각각 계산한 뒤 마지막에 평균만 낸다.
  - 앵커 슬롯은 고정값이라 다양성 계산에서 제외한다 — 생성되는 나머지 4슬롯만 본다.
  - 시드 5개(11,22,33,44,55)를 각각 따로 돌려 지표를 계산한 뒤 평균±표준편차로 보고한다.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python eval_diversity_fixed_anchor_FOOK.py
"""
import os, sys, io, csv
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from collections import Counter

from train_FOOK import build_data
from eval_rl_FOOK import load_gen, generate

FINAL = os.path.abspath(os.path.join('..', '..', '..'))
sys.path.insert(0, FINAL)

ANCHOR_SLOTS = [('국', 1), ('주찬', 2), ('부찬', 3)]   # (조건 이름, 슬롯 인덱스) — 사전 등록
SLOT_NAMES = {0: '밥', 1: '국', 2: '주찬', 3: '부찬', 4: '김치'}
SEEDS = [11, 22, 33, 44, 55]
N = 300
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_diversity_out')


def slot_stats(counter):
    total = sum(counter.values())
    top1 = max(counter.values()) / total
    top5 = sum(v for _, v in counter.most_common(5)) / total
    probs = np.array(list(counter.values())) / total
    entropy = float(-(probs * np.log2(probs)).sum())
    return {'uniq': len(counter), 'top1': top1, 'top5': top5, 'entropy': entropy}


def raw_stage(gen, seed_row, anchor_slot, food_dict, seed):
    """Seq2Seq(모방)/Seq2Seq+RL 단계: 레버·48후보 없이 순수 생성 1회씩."""
    seeds_all = np.tile(seed_row, (N, 1))
    anchors_all = np.full(N, anchor_slot)
    np.random.seed(seed)
    toks = generate(gen, seeds_all, anchors_all, food_dict)
    gen_slots = [i for i in range(5) if i != anchor_slot]
    tuples_, counters = [], {s: Counter() for s in gen_slots}
    for bi in range(N):
        menus = [food_dict[int(t)] for t in toks[bi]]
        rest = tuple(menus[s] for s in gen_slots)
        tuples_.append(rest)
        for pos, s in enumerate(gen_slots):
            counters[s][rest[pos]] += 1
    return tuples_, counters


def optimized_stage(core, anchor_menu, anchor_slot, b, seed):
    """+최적화 단계: 실제 프로덕션 make_meal()을 그대로 재사용(48후보+필터+선택 로직 원본 그대로).
    core.diet_np가 미리 고정 seed 식단으로 바꿔치기된 상태에서 호출한다(_patched_diet_np 참고)."""
    np.random.seed(seed)
    gen_slots = [i for i in range(5) if i != anchor_slot]
    tuples_, counters = [], {s: Counter() for s in gen_slots}
    for _ in range(N):
        cand, note, bb, anchor, warn = core.make_meal(menu=anchor_menu, W=60, bounds=b)
        menus, inst, after, ok = cand
        final = final_slot_menus(menus, inst)
        rest = tuple(final[s] for s in gen_slots)
        tuples_.append(rest)
        for pos, s in enumerate(gen_slots):
            counters[s][rest[pos]] += 1
    return tuples_, counters


def final_slot_menus(orig_menus, inst):
    """레버(F.adjust, make_meal 내부에서 이미 적용됨) 이후 각 슬롯에 실제로 남은 메뉴명.
    ingredient 교체로 메뉴 자체가 통째로 바뀐 경우(예: 김치→저염김치)까지 반영한다.
    app_core_FOOK.meal_result()의 final_menus 계산과 동일한 원리."""
    present = set(i['menu'] for i in inst)
    kept = [m if m in present else None for m in orig_menus]
    added = [m for m in dict.fromkeys(i['menu'] for i in inst) if m not in orig_menus]
    added_iter = iter(added)
    return [m if m is not None else next(added_iter, m) for m in kept]


def per_seed_metrics(tuples_, counters, gen_slots):
    uniq_diet_ratio = len(set(tuples_)) / len(tuples_)
    per_slot = {s: slot_stats(counters[s]) for s in gen_slots}
    return uniq_diet_ratio, per_slot


def mean_std(vals):
    return float(np.mean(vals)), float(np.std(vals))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import FOOK_adjust_levers as F
    import reward_lever_FOOK as R
    print('식약청 DB 로딩...')
    F.NUT = F.load_all()
    R.init(weight=60)
    os.chdir(cwd)

    nutrient_data, food_dict, diet_np, incidence = build_data()
    bs = int(diet_np.shape[0])
    b = F.meal_bounds(60)

    gen_imit, _ = load_gen('./results_FOOK/checkpoints', food_dict, nutrient_data, incidence, bs)
    gen_rl, _ = load_gen('./results_sweep_FOOK/i002', food_dict, nutrient_data, incidence, bs)

    seed_row = diet_np.numpy()[0]   # 사전 등록: 학습데이터 0번째 행, 세 조건 공통
    seed_menus = [food_dict[int(t)] for t in seed_row]
    print(f'고정 seed 식단(0번 행): {seed_menus}')
    # {조건명: (슬롯인덱스, 앵커메뉴명)}
    conditions = [(name, slot, seed_menus[slot + 1]) for name, slot in ANCHOR_SLOTS]
    print('조건:', conditions)

    # ── 실제 make_meal() 재사용을 위해 app_core_FOOK 로딩 후 diet_np를 고정 seed로 바꿔치기 ──
    print('\napp_core_FOOK(실제 프로덕션 파이프라인) 로딩... (수십 초 소요)')
    os.chdir(FINAL)
    import app_core_FOOK as core
    os.chdir(cwd)
    core.diet_np = tf.constant(np.tile(seed_row, (core.diet_np.shape[0], 1)), dtype=core.diet_np.dtype)
    print('core.diet_np를 고정 seed로 교체 완료 (make_meal 내부의 랜덤 seed 추첨이 항상 이 행을 뽑음)')

    raw_rows = []   # CSV: stage, condition, seed, slot, menu, count
    summary = {}    # (stage, cond) -> {'udr': [5개], 'per_slot': {slot: {'uniq':[], 'top1':[], 'top5':[], 'entropy':[]}}}

    def ensure_summary(stage, cond, gen_slots):
        key = (stage, cond)
        if key not in summary:
            summary[key] = {'udr': [], 'per_slot': {s: {'uniq': [], 'top1': [], 'top5': [], 'entropy': []}
                                                      for s in gen_slots}}
        return summary[key]

    stages_raw = [('Seq2Seq(모방)', gen_imit), ('Seq2Seq+RL', gen_rl)]
    for stage_name, gen in stages_raw:
        for cond_name, anchor_slot, anchor_menu in conditions:
            gen_slots = [i for i in range(5) if i != anchor_slot]
            entry = ensure_summary(stage_name, cond_name, gen_slots)
            for seed in SEEDS:
                tuples_, counters = raw_stage(gen, seed_row, anchor_slot, food_dict, seed)
                udr, per_slot = per_seed_metrics(tuples_, counters, gen_slots)
                entry['udr'].append(udr)
                for s in gen_slots:
                    st = per_slot[s]
                    entry['per_slot'][s]['uniq'].append(st['uniq'])
                    entry['per_slot'][s]['top1'].append(st['top1'])
                    entry['per_slot'][s]['top5'].append(st['top5'])
                    entry['per_slot'][s]['entropy'].append(st['entropy'])
                    for menu, cnt in counters[s].items():
                        raw_rows.append([stage_name, cond_name, seed, SLOT_NAMES[s], menu, cnt])
            print(f'  [{stage_name}] {cond_name}-앵커 5시드 완료')

    stage_name = '+최적화'
    for cond_name, anchor_slot, anchor_menu in conditions:
        gen_slots = [i for i in range(5) if i != anchor_slot]
        entry = ensure_summary(stage_name, cond_name, gen_slots)
        for seed in SEEDS:
            tuples_, counters = optimized_stage(core, anchor_menu, anchor_slot, b, seed)
            udr, per_slot = per_seed_metrics(tuples_, counters, gen_slots)
            entry['udr'].append(udr)
            for s in gen_slots:
                st = per_slot[s]
                entry['per_slot'][s]['uniq'].append(st['uniq'])
                entry['per_slot'][s]['top1'].append(st['top1'])
                entry['per_slot'][s]['top5'].append(st['top5'])
                entry['per_slot'][s]['entropy'].append(st['entropy'])
                for menu, cnt in counters[s].items():
                    raw_rows.append([stage_name, cond_name, seed, SLOT_NAMES[s], menu, cnt])
        print(f'  [{stage_name}] {cond_name}-앵커 5시드 완료')

    # ── CSV 저장 (감사용 raw count) ──
    raw_csv = os.path.join(OUT_DIR, 'raw_counts.csv')
    with open(raw_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'condition', 'seed', 'slot', 'menu', 'count'])
        w.writerows(raw_rows)
    print(f'\nraw count 저장: {raw_csv} ({len(raw_rows)}행)')

    # ── 요약 CSV + 콘솔 출력 ──
    summary_csv = os.path.join(OUT_DIR, 'summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['stage', 'condition', 'metric', 'slot', 'mean', 'std', 'n_seeds', 'raw_values'])
        for (stage, cond), entry in summary.items():
            m, s = mean_std(entry['udr'])
            w.writerow([stage, cond, 'unique_diet_ratio', '', m, s, len(entry['udr']),
                        ';'.join(f'{v:.4f}' for v in entry['udr'])])
            for slot, vals in entry['per_slot'].items():
                for metric in ('uniq', 'top1', 'top5', 'entropy'):
                    m, s = mean_std(vals[metric])
                    w.writerow([stage, cond, metric, SLOT_NAMES[slot], m, s, len(vals[metric]),
                                ';'.join(f'{v:.4f}' for v in vals[metric])])
    print(f'요약(평균±표준편차) 저장: {summary_csv}')

    print('\n===== 콘솔 요약 (조건·시드별 평균±표준편차) =====')
    for stage_name in ['Seq2Seq(모방)', 'Seq2Seq+RL', '+최적화']:
        print(f'\n----- {stage_name} -----')
        udr_means = []
        for cond_name, anchor_slot, _ in conditions:
            entry = summary[(stage_name, cond_name)]
            m, s = mean_std(entry['udr'])
            udr_means.append(m)
            gen_slots = [i for i in range(5) if i != anchor_slot]
            detail = '  '.join(
                f'{SLOT_NAMES[sl]} 고유{mean_std(entry["per_slot"][sl]["uniq"])[0]:.1f}±'
                f'{mean_std(entry["per_slot"][sl]["uniq"])[1]:.1f} '
                f'top1 {mean_std(entry["per_slot"][sl]["top1"])[0]*100:.1f}±'
                f'{mean_std(entry["per_slot"][sl]["top1"])[1]*100:.1f}%'
                for sl in gen_slots)
            print(f'{cond_name}-앵커  고유식단비율 {m*100:.1f}±{s*100:.1f}%   {detail}')
        print(f'  → 3조건 평균 고유식단비율: {np.mean(udr_means)*100:.1f}%')


if __name__ == '__main__':
    main()
