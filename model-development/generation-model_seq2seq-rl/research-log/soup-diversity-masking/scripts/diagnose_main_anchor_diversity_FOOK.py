# -*- coding: utf-8 -*-
"""
diagnose_main_anchor_diversity_FOOK.py — 주찬 앵커 고정 시나리오, 국·부찬 다양성 저하 진단 스크립트

목적: "사용자가 먹고 싶은 주찬을 앵커로 고정했을 때, 함께 생성되는 국·부찬이 어느 단계에서
      특정 메뉴로 쏠리는가?"를 4단계(생성직후/레버보정후/게이트통과/최종선택)로 나눠 추적한다.

※ 기존 서비스 코드(app_core_FOOK.py)는 변경하지 않는다 — 여기서는 app_core_FOOK의 함수를
  그대로 재사용(gen_batch, F.adjust, passes 구성요소, _has_ingredient_clash 등)해서 매 후보를
  개별 추적만 한다. make_meal()과 동일한 순서·기준으로 처리하되, make_meal()은 첫 완전통과에서
  바로 return하는 반면 여기서는 48개를 전부 생성해 각 단계 분포를 남긴다(선택 결과 자체는
  make_meal()과 동일한 규칙 — 첫 완전통과 후보, 없으면 부분점수 최고 후보 — 으로 판정한다).

조건(사전 등록 — 결과를 본 뒤 유리한 조건을 고르지 않음):
  주찬 앵커 3개를 "조리형태·주재료가 서로 다른 대표 주찬"으로 선정(인 함량 기준 대신, 실제
  서비스에서 고르일 법한 유형 다양성 기준으로 변경, 2026-07-27):
    고등어구이  (생선·구이류) -> 행 55  (잡곡밥/버섯전골/고등어구이/새우살브로콜리볶음/배추김치, 13회 등장)
    제육불고기  (육류·고추장양념) -> 행 152 (백미밥/호박된장찌개/제육불고기/봄동겉절이/깍두기, 5회 등장)
    두부양념조림 (두부·콩류·조림) -> 행 36  (백미밥/시래기된장국/두부양념조림/치커리양파무침/배추김치, 16회 등장)
  각 앵커의 seed 식단(encoder 입력)은 그 메뉴가 실제로 주찬 자리에 등장한 "첫 번째" 학습데이터
  행을 그대로 쓴다(임의 선택 아님). 세 조건은 섞지 않고 각각 계산한 뒤 마지막에 평균만 낸다.

밥은 의도된 표준화 슬롯(JAPGOK_RICE→WHITE_RICE 강제 치환)이라 다양성 분석에서 제외.
주찬은 고정 앵커라 제외. 김치는 저염 치환 규칙 영향이 커서 참고 지표로만 별도 저장.
핵심 분석 대상은 국·부찬.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python diagnose_main_anchor_diversity_FOOK.py
"""
import os, sys, io, csv
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from collections import Counter

FINAL = r'E:\final'
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, FINAL)

OUT_DIR = os.path.join(CODE, 'diagnose_main_anchor_out')
SLOT_NAMES = ['밥', '국', '주찬', '부찬', '김치']
N = 200
TRIES = 48
SEED = 11

# (조건명, 주찬 앵커 메뉴, 특성설명, seed 행 인덱스)
ANCHOR_CONDITIONS = [
    ('생선구이', '고등어구이', '생선·구이류', 55),
    ('육류', '제육불고기', '육류·고추장양념', 152),
    ('두부콩류', '두부양념조림', '두부·콩류·조림', 36),
]
ANCHOR_SLOT = 2   # 주찬


def final_slot_menus(orig_menus, inst):
    """레버(F.adjust) 적용 후 각 슬롯에 실제로 남은 메뉴명 (통째 교체까지 반영)."""
    present = set(i['menu'] for i in inst)
    kept = [m if m in present else None for m in orig_menus]
    added = [m for m in dict.fromkeys(i['menu'] for i in inst) if m not in orig_menus]
    it = iter(added)
    return [m if m is not None else next(it, m) for m in kept]


def process_candidate(core, menus, anchor_menu, gun_s, b):
    menus = list(menus)
    if gun_s is not None:
        menus[gun_s] = anchor_menu
    generated = menus[:]                                    # 생성 직후 5슬롯
    _, after, inst, _ = core.F.adjust(menus, b, anchor=anchor_menu)
    adjusted = final_slot_menus(menus, inst)                 # 보정 후 5슬롯(통째 교체 반영)

    e_pass = b['Elo'] <= after['E'] <= b['Ehi']
    pr_pass = b['Plo'] <= after['protein'] <= b['Phi']
    k_pass = after['K'] < b['Kmax']
    p_pass = after['P'] < b['Pmax']                          # 원값 P 기준(2026-07-27 확정)
    na_pass = after['Na_season'] <= b['Namax']
    nutrition_pass = e_pass and pr_pass and k_pass and p_pass and na_pass

    unreal = core.F.unrealistic_reason(inst)
    realistic_pass = unreal is None
    duplicate_pass = not core._has_ingredient_clash(menus)
    seafood_pass = not core._has_seafood_overload(menus)
    high_p_pass = not core._has_high_p_overload(menus)
    realism_pass = realistic_pass and duplicate_pass and seafood_pass and high_p_pass

    all_pass = nutrition_pass and realism_pass

    score = sum([e_pass, pr_pass, k_pass, p_pass, na_pass])
    if realistic_pass: score += 0.5
    score += 0.3                      # dup_today: 하루 시퀀싱 없음 -> 항상 미중복 취급(원본 기본값과 동일)
    if duplicate_pass: score += 0.3
    if seafood_pass: score += 0.3
    if high_p_pass: score += 0.3

    return {
        'raw_bap': generated[0], 'generated_soup': generated[1], 'raw_anchor': generated[2],
        'generated_side': generated[3], 'generated_kimchi': generated[4],
        'adjusted_bap': adjusted[0], 'adjusted_soup': adjusted[1], 'adjusted_anchor': adjusted[2],
        'adjusted_side': adjusted[3], 'adjusted_kimchi': adjusted[4],
        'soup_changed': generated[1] != adjusted[1],
        'side_changed': generated[3] != adjusted[3],
        'kimchi_changed': generated[4] != adjusted[4],
        'soup_before': generated[1], 'soup_after': adjusted[1],
        'side_before': generated[3], 'side_after': adjusted[3],
        'kimchi_before': generated[4], 'kimchi_after': adjusted[4],
        'energy_pass': e_pass, 'protein_pass': pr_pass, 'potassium_pass': k_pass,
        'phosphorus_pass': p_pass, 'sodium_pass': na_pass, 'nutrition_pass': nutrition_pass,
        'realistic_amount_pass': realistic_pass, 'duplicate_pass': duplicate_pass,
        'seafood_pass': seafood_pass, 'high_p_pass': high_p_pass, 'realism_pass': realism_pass,
        'all_pass': all_pass, 'score': score,
    }


def run_call(core, anchor_menu, b, call_id, cond_name):
    tok_anchor = anchor_menu if anchor_menu in core.name2idx else None
    gun_s = core.gun_slot.get(anchor_menu) if (anchor_menu in core.gun_names) else None
    raw_candidates = core.gen_batch(tok_anchor, n=TRIES, temp=0.8)

    rows = []
    passing_ids = []
    best_id, best_score = None, -1
    for cand_id, menus in enumerate(raw_candidates):
        rec = process_candidate(core, menus, anchor_menu, gun_s, b)
        rec['call_id'] = call_id
        rec['candidate_id'] = cand_id
        rec['anchor_condition'] = cond_name
        rec['anchor_menu'] = anchor_menu
        if rec['all_pass']:
            passing_ids.append(cand_id)
        if rec['score'] > best_score:
            best_id, best_score = cand_id, rec['score']
        rows.append(rec)

    if passing_ids:
        selected_id = passing_ids[0]                 # make_meal()과 동일: 첫 완전통과 즉시 채택
        selection_type = '첫 완전 통과 후보'
    else:
        selected_id = best_id
        selection_type = '완전 통과 후보 없음-부분점수 최고 후보'

    num_passed = len(passing_ids)
    uniq_soups = len({rows[i]['adjusted_soup'] for i in passing_ids}) if passing_ids else 0
    uniq_sides = len({rows[i]['adjusted_side'] for i in passing_ids}) if passing_ids else 0
    for r in rows:
        r['selected'] = (r['candidate_id'] == selected_id)
        r['selection_type'] = selection_type
        r['num_passed_in_call'] = num_passed
        r['unique_passed_soups'] = uniq_soups
        r['unique_passed_sides'] = uniq_sides
    return rows


def slot_stats(counter):
    total = sum(counter.values())
    if total == 0:
        return {'n': 0, 'uniq': 0, 'top1_menu': '', 'top1_ratio': 0.0, 'top5_ratio': 0.0, 'entropy': 0.0}
    top1_menu, top1_cnt = counter.most_common(1)[0]
    top1 = top1_cnt / total
    top5 = sum(v for _, v in counter.most_common(5)) / total
    probs = np.array(list(counter.values())) / total
    entropy = float(-(probs * np.log2(probs)).sum())
    return {'n': total, 'uniq': len(counter), 'top1_menu': top1_menu, 'top1_ratio': top1,
            'top5_ratio': top5, 'entropy': entropy}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    os.chdir(cwd)

    b = core.F.meal_bounds(60)
    all_trace_rows = []
    stage_dist_rows = []
    call_summary_rows = []

    orig_diet_np = core.diet_np   # 조건마다 바꿔치기 -> 마지막에 복원

    for cond_name, anchor_menu, desc, row_idx in ANCHOR_CONDITIONS:
        seed_row = orig_diet_np.numpy()[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in seed_row]
        assert decoded[3] == anchor_menu, f'seed 행 검증 실패: {decoded} vs 기대 앵커 {anchor_menu}'
        core.diet_np = tf.constant(np.tile(seed_row, (orig_diet_np.shape[0], 1)), dtype=orig_diet_np.dtype)
        print(f'[{cond_name}] 앵커={anchor_menu}({desc}) seed행={row_idx} 확인됨: {decoded}')

        np.random.seed(SEED)
        cond_rows = []
        for call_id in range(N):
            cond_rows.extend(run_call(core, anchor_menu, b, call_id, cond_name))
        all_trace_rows.extend(cond_rows)
        print(f'  {N}회 호출 완료 (후보 {len(cond_rows)}개)')

        # ── stage_distribution ──
        stages = {
            '생성직후': ('generated_soup', 'generated_side', 'generated_kimchi', None),
            '레버보정후': ('adjusted_soup', 'adjusted_side', 'adjusted_kimchi', None),
            '전체조건통과후보': ('adjusted_soup', 'adjusted_side', 'adjusted_kimchi', 'all_pass'),
            '최종선택': ('adjusted_soup', 'adjusted_side', 'adjusted_kimchi', 'selected'),
        }
        for stage_name, (soup_k, side_k, kim_k, filt) in stages.items():
            for slot_label, key in [('국', soup_k), ('부찬', side_k), ('김치(참고)', kim_k)]:
                c = Counter(r[key] for r in cond_rows if (filt is None or r[filt]))
                st = slot_stats(c)
                stage_dist_rows.append([cond_name, stage_name, slot_label, st['n'], st['uniq'],
                                         st['top1_menu'], st['top1_ratio'], st['top5_ratio'], st['entropy']])

        # ── call_summary ──
        by_call = {}
        for r in cond_rows:
            by_call.setdefault(r['call_id'], []).append(r)
        for call_id, rs in by_call.items():
            n_gen = len(rs)
            n_pass = rs[0]['num_passed_in_call']
            uniq_soups = rs[0]['unique_passed_soups']
            uniq_sides = rs[0]['unique_passed_sides']
            pass_combos = {(r['adjusted_soup'], r['adjusted_side']) for r in rs if r['all_pass']}
            sel = next(r for r in rs if r['selected'])
            call_summary_rows.append([cond_name, call_id, n_gen, n_pass, uniq_soups, uniq_sides,
                                       len(pass_combos), sel['adjusted_soup'], sel['adjusted_side'],
                                       sel['selection_type']])

    core.diet_np = orig_diet_np   # 원복

    # ── ① candidate_trace_main_anchor.csv ──
    trace_csv = os.path.join(OUT_DIR, 'candidate_trace_main_anchor.csv')
    fieldnames = ['anchor_condition', 'anchor_menu', 'call_id', 'candidate_id',
                  'raw_bap', 'generated_soup', 'raw_anchor', 'generated_side', 'generated_kimchi',
                  'adjusted_bap', 'adjusted_soup', 'adjusted_anchor', 'adjusted_side', 'adjusted_kimchi',
                  'soup_changed', 'side_changed', 'kimchi_changed',
                  'soup_before', 'soup_after', 'side_before', 'side_after', 'kimchi_before', 'kimchi_after',
                  'energy_pass', 'protein_pass', 'potassium_pass', 'phosphorus_pass', 'sodium_pass',
                  'nutrition_pass', 'realistic_amount_pass', 'duplicate_pass', 'seafood_pass',
                  'high_p_pass', 'realism_pass', 'all_pass', 'score',
                  'selected', 'selection_type', 'num_passed_in_call',
                  'unique_passed_soups', 'unique_passed_sides']
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_trace_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f'\n① {trace_csv} ({len(all_trace_rows)}행)')

    # ── ② stage_distribution_main_anchor.csv ──
    stage_csv = os.path.join(OUT_DIR, 'stage_distribution_main_anchor.csv')
    with open(stage_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['anchor_condition', 'stage', 'slot', 'n', 'unique_menus', 'top1_menu',
                    'top1_ratio', 'top5_ratio', 'entropy'])
        w.writerows(stage_dist_rows)
    print(f'② {stage_csv} ({len(stage_dist_rows)}행)')

    # ── ③ transition_summary_main_anchor.csv ──
    trans_csv = os.path.join(OUT_DIR, 'transition_summary_main_anchor.csv')
    trans_rows = []
    for cond_name, _, _, _ in ANCHOR_CONDITIONS:
        rs = [r for r in all_trace_rows if r['anchor_condition'] == cond_name]
        soup_trans = Counter((r['soup_before'], r['soup_after']) for r in rs)
        side_trans = Counter((r['side_before'], r['side_after']) for r in rs)
        soup_gate = Counter((r['adjusted_soup'], r['all_pass']) for r in rs)
        side_gate = Counter((r['adjusted_side'], r['all_pass']) for r in rs)
        sel_from_pass = Counter((r['adjusted_soup'], r['adjusted_side']) for r in rs if r['selected'])
        for (b4, af), cnt in soup_trans.most_common():
            trans_rows.append([cond_name, '생성국->보정국', b4, af, b4 == af, cnt])
        for (b4, af), cnt in side_trans.most_common():
            trans_rows.append([cond_name, '생성부찬->보정부찬', b4, af, b4 == af, cnt])
        for (m, passed), cnt in soup_gate.most_common():
            trans_rows.append([cond_name, '보정국->통과여부', m, str(passed), None, cnt])
        for (m, passed), cnt in side_gate.most_common():
            trans_rows.append([cond_name, '보정부찬->통과여부', m, str(passed), None, cnt])
        for (soup, side), cnt in sel_from_pass.most_common():
            trans_rows.append([cond_name, '통과조합->최종선택', f'{soup}|{side}', '', None, cnt])
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['anchor_condition', 'transition', 'from_또는_메뉴', 'to_또는_통과여부',
                    '변경없음', 'count'])
        w.writerows(trans_rows)
    print(f'③ {trans_csv} ({len(trans_rows)}행)')

    # ── ④ call_summary_main_anchor.csv ──
    call_csv = os.path.join(OUT_DIR, 'call_summary_main_anchor.csv')
    with open(call_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['anchor_condition', 'call_id', 'n_generated', 'n_full_pass',
                    'unique_soups_in_pass', 'unique_sides_in_pass', 'unique_soup_side_combos_in_pass',
                    'selected_soup', 'selected_side', 'selection_type'])
        w.writerows(call_summary_rows)
        # 조건별 요약행(평균/중앙값/최소/최대)
        import statistics as st
        for cond_name, _, _, _ in ANCHOR_CONDITIONS:
            rs = [r for r in call_summary_rows if r[0] == cond_name]
            npass = [r[3] for r in rs]
            usoup = [r[4] for r in rs]
            uside = [r[5] for r in rs]
            for label, vals in [('평균', None), ('중앙값', None), ('최솟값', None), ('최댓값', None)]:
                pass
            w.writerow([cond_name, '요약_평균', '', round(np.mean(npass), 2),
                        round(np.mean(usoup), 2), round(np.mean(uside), 2), '', '', '', ''])
            w.writerow([cond_name, '요약_중앙값', '', st.median(npass),
                        st.median(usoup), st.median(uside), '', '', '', ''])
            w.writerow([cond_name, '요약_최솟값', '', min(npass), min(usoup), min(uside), '', '', '', ''])
            w.writerow([cond_name, '요약_최댓값', '', max(npass), max(usoup), max(uside), '', '', '', ''])
    print(f'④ {call_csv} ({len(call_summary_rows)}행 + 조건별 요약)')

    return all_trace_rows, stage_dist_rows, call_summary_rows


if __name__ == '__main__':
    main()
