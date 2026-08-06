# -*- coding: utf-8 -*-
"""
phosphorus_rawP_fix_experiment_FOOK.py — raw P vs Peff 불일치를 해결하는 수정안 A/B(A만 구현)/C를
소규모 paired A/B/C로 검증. 두부·콩류 앵커. C_mask100+RL(epoch260). ★ 코드 수정 없음.

Baseline = adjust() 수기 복제(직전 작업에서 F.adjust()와 5/5 완전일치 검증된 방식) 그대로.
A        = Baseline 최종상태 복사 → F.lever_phosphorus() 1회 더 호출(Peff 기준 그대로, 대조군).
C        = Baseline 최종상태 복사 → raw P(passes()와 동일 raw P<Pmax) 직접 재검사 →
           초과시 "T0 대비 인 증가분이 가장 큰(+분량도 증가한) 메뉴"를 최소량만 비례축소(최대1회,
           amt_floor_of() 하한 준수, 반복 없음) → 5영양 재검사.

실행: conda activate foodbert; set TF_USE_LEGACY_KERAS=1; python phosphorus_rawP_fix_experiment_FOOK.py
"""
import os, sys, io, csv, copy
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

from Model import Encoder, Decoder
from train_FOOK_soupmask_1000 import build_data, SOUP_POS

OUT_DIR = os.path.join(CODE, 'phosphorus_rawP_fix_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_MENU = '두부양념조림'
SEED_ROWS = [11, 12, 6, 36, 7]
N_CALLS = 20
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
MARGIN = 0.01   # 부동소수점 경계 안전마진(mg) — 게이트 tolerance 아님, 보정계산에만 사용


def load_model(ckpt_dir, num_tokens):
    kwargs = {'num_tokens': num_tokens, 'embed_dim': 128, 'fc_dim': 64,
              'fully-connected_layer': 'GRU', 'attention': True}
    enc = Encoder(**kwargs, batch_size=10)
    dec = Decoder(**kwargs, batch_size=10)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    assert ck, f'체크포인트 없음: {ckpt_dir}'
    tf.train.Checkpoint(encoder=enc, decoder=dec).restore(ck).expect_partial()
    print('로딩:', ck)
    return enc, dec


def gen_batch_slots(core, encoder, decoder, num_tokens, mask_id, food_dict,
                     fixed_seed_row_7tok, anchor_token, n, temp):
    seeds = np.tile(fixed_seed_row_7tok, (n, 1)).astype(np.int64)
    seeds[:, SOUP_POS] = mask_id
    fixed = {2: anchor_token}
    seeds[:, 3] = anchor_token
    seeds_tf = tf.constant(seeds, dtype=tf.int32)
    enc_hidden0 = tf.zeros([n, encoder.units])
    enc_output, enc_hidden = encoder(seeds_tf, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int); res[:, 0] = seeds[:, 0]; res[:, -1] = 826
    used = [set(fixed.values()) for _ in range(n)]
    used_grp = [{core.TOK_GRP[t] for t in fixed.values() if t in core.TOK_GRP} for _ in range(n)]
    for j in range(5):
        outputs, dec_hidden, _ = decoder(seeds_tf[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for bi in range(n):
            if j in fixed:
                res[bi, j + 1] = fixed[j]; continue
            p = probs[bi].copy()
            for t in core.SPECIAL:
                if t < num_tokens: p[t] = 0.0
            if mask_id is not None: p[mask_id] = 0.0
            for t in core.BLOCK_TOK:
                if t < num_tokens: p[t] = 0.0
            for t in used[bi]:
                if t < len(p): p[t] = 0.0
            slot_ok = core.SLOT_OK[j]
            if num_tokens > len(slot_ok):
                slot_ok = np.append(slot_ok, np.zeros(num_tokens - len(slot_ok)))
            masked = p * slot_ok
            for gi in used_grp[bi]:
                idx = np.array([g for g in core.GRP_TOK[gi] if g < num_tokens])
                if len(idx): masked[idx] = 0.0
            if masked.sum() > 0:
                p = masked
            p = np.clip(p, 1e-12, None); p = p ** (1.0 / temp); p /= p.sum()
            tok = int(np.random.choice(len(p), p=p))
            res[bi, j + 1] = tok; used[bi].add(tok)
            gi = core.TOK_GRP.get(tok)
            if gi is not None:
                used_grp[bi].add(gi)
    menus_list = [[food_dict[int(t)] for t in r if int(t) not in core.SPECIAL and t != mask_id] for r in res]
    return menus_list


def run_baseline(F, menus, anchor, b, na_target):
    """adjust() 수기 복제(직전 작업 검증필: F.adjust()와 5/5 완전일치). T0(=expand직후)의 inst
    깊은복사, 최종상태(T15) inst, 그리고 매 레버 스텝의 (lever,pass,totals) 경량 기록을 반환
    (culprit-lever 판정용, 원인별 그룹 분류에 재사용)."""
    inst = F.expand(list(menus))
    F.SWAP_LOG.clear()
    t0_inst = copy.deepcopy(inst)
    steps = [('expand', None, F.totals(inst))]
    F.lever_kimchi(inst); steps.append(('kimchi', None, F.totals(inst)))
    F.lever_sodium(inst); steps.append(('sodium', 'pre', F.totals(inst)))
    F.lever_sodium_extra(inst, na_target); steps.append(('sodium_extra', 'pre', F.totals(inst)))
    for pass_i in range(2):
        F.lever_potassium(inst, b['Kmax'], anchor=anchor); steps.append(('potassium', pass_i + 1, F.totals(inst)))
        F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo']); steps.append(('phosphorus', pass_i + 1, F.totals(inst)))
        F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor); steps.append(('protein', pass_i + 1, F.totals(inst)))
        F.lever_sodium(inst); steps.append(('sodium', pass_i + 1, F.totals(inst)))
        F.lever_sodium_extra(inst, na_target); steps.append(('sodium_extra', pass_i + 1, F.totals(inst)))
        F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                         kmax=b['Kmax'], pmax=b['Pmax']); steps.append(('calorie', pass_i + 1, F.totals(inst)))
    return t0_inst, inst, steps


def classify_cause(steps, PMAX):
    """직전 작업(phosphorus_lever_step_diagnosis_FOOK.py)과 동일 로직: 마지막으로 raw P
    통과했던 스텝 바로 다음(culprit)의 레버명을 원인으로 판정."""
    raw_pass_seq = [t['P'] < PMAX for _, _, t in steps]
    true_idx = [i for i, v in enumerate(raw_pass_seq) if v]
    if not true_idx:
        return 'never_passed', None
    L = true_idx[-1]
    if L >= len(steps) - 1:
        return 'always_passed', None
    culprit_lever = steps[L + 1][0]
    return culprit_lever, L


def menu_p_amt(inst):
    """entry.get('orig_menu', menu) 기준(치환·이름변경도 원 정체성으로 묶음) 메뉴별 P합/amt합."""
    p_by = {}
    amt_by = {}
    cur_name = {}
    for i in inst:
        identity = i.get('orig_menu', i['menu'])
        p_by[identity] = p_by.get(identity, 0.0) + (i['amt'] / 100 * i['P'] if i['P'] is not None else 0.0)
        amt_by[identity] = amt_by.get(identity, 0.0) + i['amt']
        cur_name[identity] = i['menu']   # 현재(치환후) 이름 — 보정 적용시 이걸로 매칭
    return p_by, amt_by, cur_name


def apply_variant_A(F, inst, b, anchor):
    inst2 = copy.deepcopy(inst)
    F.lever_phosphorus(inst2, b['Pmax'], anchor=anchor, plo=b['Plo'])
    return inst2


def apply_variant_C(F, inst, b, t0_inst):
    """raw P 직접 재검사 + 최소보정(최대1회). 반환: (inst2, fix_attempted, fix_success,
    target_menu, amt_before, amt_after, reason)."""
    inst2 = copy.deepcopy(inst)
    PMAX = b['Pmax']
    t = F.totals(inst2)
    raw_p = t['P']
    if raw_p < PMAX:
        return inst2, False, None, None, None, None, None

    p0_by, amt0_by, _ = menu_p_amt(t0_inst)
    p1_by, amt1_by, cur_name = menu_p_amt(inst2)

    # 스케일업 시그니처: T0 대비 P도 늘고 amt도 늘어난 메뉴 중 P증가 최대
    cands = []
    for ident in p1_by:
        dp = p1_by[ident] - p0_by.get(ident, 0.0)
        damt = amt1_by[ident] - amt0_by.get(ident, 0.0)
        if dp > 0 and damt > 1e-6:
            cands.append((dp, ident))
    if not cands:   # 폴백: 스케일업 시그니처 없으면 그냥 P증가 최대인 메뉴
        for ident in p1_by:
            dp = p1_by[ident] - p0_by.get(ident, 0.0)
            if dp > 0:
                cands.append((dp, ident))
    if not cands:
        return inst2, True, False, None, None, None, 'no_target_menu_found'

    cands.sort(reverse=True)
    target_ident = cands[0][1]
    target_name = cur_name[target_ident]

    menu_items = [i for i in inst2 if i['menu'] == target_name]
    P_menu = sum(i['amt'] / 100 * i['P'] for i in menu_items if i['P'] is not None)
    Amt_menu = sum(i['amt'] for i in menu_items)
    if P_menu <= 0 or Amt_menu <= 0:
        return inst2, True, False, target_name, Amt_menu, Amt_menu, 'target_menu_zero_P_or_amt'

    excess = raw_p - PMAX + MARGIN
    capped_reduction = min(excess, P_menu)
    reduction_ratio = capped_reduction / P_menu

    # amt_floor_of() 하한 준수(기존 함수 그대로 호출, 읽기전용)
    max_allowed_ratio = reduction_ratio
    for i in menu_items:
        floor_i = F.amt_floor_of(i)
        if i['amt'] > 0:
            allowed = 1.0 - (floor_i / i['amt'])
            max_allowed_ratio = min(max_allowed_ratio, max(0.0, allowed))
    final_ratio = min(reduction_ratio, max_allowed_ratio)

    amt_before = Amt_menu
    for i in menu_items:
        i['amt'] *= (1.0 - final_ratio)
    amt_after = sum(i['amt'] for i in menu_items)

    t_after = F.totals(inst2)
    fix_success = t_after['P'] < PMAX
    reason = None if fix_success else ('floor_capped_insufficient' if final_ratio < reduction_ratio else 'still_over_after_full_reduction')
    return inst2, True, fix_success, target_name, amt_before, amt_after, reason


def nutrient_flags(t, b):
    raw_pass = t['P'] < b['Pmax']
    eff_pass = t['Peff'] < b['Pmax']
    protein_low = t['protein'] < b['Plo']
    protein_high = t['protein'] > b['Phi']
    calorie_low = t['E'] < b['Elo']
    calorie_high = t['E'] > b['Ehi']
    na_pass = t['Na_season'] <= b['Namax']
    k_pass = t['K'] < b['Kmax']
    protein_pass = not protein_low and not protein_high
    calorie_pass = not calorie_low and not calorie_high
    all_pass = raw_pass and protein_pass and calorie_pass and na_pass and k_pass
    return {'raw_pass': raw_pass, 'eff_pass': eff_pass, 'protein_low': protein_low, 'protein_high': protein_high,
            'calorie_low': calorie_low, 'calorie_high': calorie_high, 'na_pass': na_pass, 'k_pass': k_pass,
            'protein_pass': protein_pass, 'calorie_pass': calorie_pass, 'all_pass': all_pass}


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    os.chdir(cwd)
    b = F.meal_bounds(60)
    PMAX = b['Pmax']
    na_target = b.get('Na_total_target', F.NA_TOTAL_MEAL)

    _, food_dict, diet_np, _, mask_id = build_data(with_mask=True)
    num_tokens = len(food_dict)
    orig_diet_np_np = diet_np.numpy()
    enc, dec = load_model(RL_CKPT_DIR, num_tokens)
    anchor_token = core.name2idx[ANCHOR_MENU]

    print(f'생성 시작: 두부콩류 {len(SEED_ROWS)}seed x {N_CALLS}call x {TRIES} = {len(SEED_ROWS)*N_CALLS*TRIES}후보(예정)')

    candidates = []
    cid = 0
    for sid, row_idx in enumerate(SEED_ROWS):
        base_row = orig_diet_np_np[row_idx].copy()
        np.random.seed(RNG_SEED)
        for call_id in range(N_CALLS):
            menus_list = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                          base_row, anchor_token, TRIES, TEMP)
            for menus in menus_list:
                if len(menus) != 5:
                    continue
                cid += 1
                t0_inst, base_inst, steps = run_baseline(F, menus, ANCHOR_MENU, b, na_target)
                inst_A = apply_variant_A(F, base_inst, b, ANCHOR_MENU)
                inst_C, fix_att, fix_ok, tgt_menu, amt_b, amt_a, reason = apply_variant_C(F, base_inst, b, t0_inst)

                t_base = F.totals(base_inst)
                t_A = F.totals(inst_A)
                t_C = F.totals(inst_C)
                t0_totals = F.totals(t0_inst)
                before_pass = t0_totals['P'] < PMAX
                after_pass = t_base['P'] < PMAX
                culprit_lever, culprit_idx = classify_cause(steps, PMAX) if (before_pass and not after_pass) else (None, None)
                cause_group = None
                if before_pass and not after_pass:
                    if culprit_lever == 'protein':
                        cause_group = 'lever_protein_직후'
                    elif culprit_lever == 'calorie':
                        cause_group = 'lever_calorie_직후'
                    else:
                        cause_group = f'기타({culprit_lever})'

                candidates.append({
                    'candidate_id': cid, 'seed_id': sid, 'call_id': call_id, 'menus': menus,
                    'before_pass': before_pass, 'after_pass_baseline': after_pass,
                    'group': 'F' if (before_pass and not after_pass) else ('pass' if (before_pass and after_pass) else ('G' if not after_pass else 'recovered')),
                    'cause_group': cause_group,
                    't0': t0_totals, 't_base': t_base, 't_A': t_A, 't_C': t_C,
                    'fix_attempted': fix_att, 'fix_success': fix_ok, 'target_menu': tgt_menu,
                    'amt_before': amt_b, 'amt_after': amt_a, 'reason': reason,
                    'base_inst': base_inst,
                })
        print(f'  seed {sid}(row{row_idx}) 완료, 누적 {len(candidates)}건')

    print(f'\n총 생성: {len(candidates)}건')

    for c in candidates:
        c['flags_base'] = nutrient_flags(c['t_base'], b)
        c['flags_A'] = nutrient_flags(c['t_A'], b)
        c['flags_C'] = nutrient_flags(c['t_C'], b)

    n_total = len(candidates)
    n_F = sum(1 for c in candidates if c['group'] == 'F')
    print(f"그룹: F={n_F} pass={sum(1 for c in candidates if c['group']=='pass')} "
          f"G={sum(1 for c in candidates if c['group']=='G')} recovered={sum(1 for c in candidates if c['group']=='recovered')}")

    # ── ① phosphorus_rawP_fix_candidate_trace.csv (long format: candidate x variant) ──
    trace_rows = []
    for c in candidates:
        for variant, t_after, flags in [('Baseline', c['t_base'], c['flags_base']),
                                          ('A', c['t_A'], c['flags_A']),
                                          ('C', c['t_C'], c['flags_C'])]:
            t_before = c['t_base']  # baseline 최종상태가 A/C 각 변형의 '보정 전' 기준점
            is_C = (variant == 'C')
            fix_attempted = c['fix_attempted'] if is_C else False
            fix_success = c['fix_success'] if is_C else None
            new_reason = c['reason'] if is_C else None
            adjusted_menu = c['target_menu'] if is_C else None
            amt_before = c['amt_before'] if is_C else None
            amt_after = c['amt_after'] if is_C else None
            amt_reduced = (amt_before - amt_after) if (is_C and amt_before is not None) else None
            raw_p_reduction = (t_before['P'] - t_after['P']) if variant != 'Baseline' else 0.0
            protein_reduction = (t_before['protein'] - t_after['protein']) if variant != 'Baseline' else 0.0
            calorie_reduction = (t_before['E'] - t_after['E']) if variant != 'Baseline' else 0.0
            trace_rows.append({
                'candidate_id': c['candidate_id'], 'anchor_type': '두부콩류', 'seed_id': c['seed_id'],
                'call_id': c['call_id'], 'variant': variant,
                'raw_P_before': t_before['P'], 'raw_P_after': t_after['P'],
                'Peff_before': t_before['Peff'], 'Peff_after': t_after['Peff'],
                'protein_before': t_before['protein'], 'protein_after': t_after['protein'],
                'calorie_before': t_before['E'], 'calorie_after': t_after['E'],
                'raw_P_pass': flags['raw_pass'], 'Peff_pass': flags['eff_pass'],
                'nutrition_all_pass': flags['all_pass'],
                'fix_attempted': fix_attempted, 'fix_success': fix_success,
                'adjusted_menu': adjusted_menu, 'amount_before': amt_before, 'amount_after': amt_after,
                'amount_reduced': amt_reduced, 'raw_P_reduction': raw_p_reduction,
                'protein_reduction': protein_reduction, 'calorie_reduction': calorie_reduction,
                'new_failure_reason': new_reason,
            })
    trace_csv = os.path.join(OUT_DIR, 'phosphorus_rawP_fix_candidate_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    print(f'① {trace_csv} ({len(trace_rows)}행 = {n_total}후보 x 3변형)')

    # ── 배치(seed,call)별 선택 시뮬레이션(첫 all_pass 즉시채택, 없으면 부분점수 최고) + 다양성 ──
    def score_of(flags):
        s = sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])
        return s

    def batch_selection(batch, variant_key, flags_key):
        sel = None
        for c in batch:
            if c[flags_key]['all_pass']:
                sel = c; break
        if sel is None:
            best_score, best_c = -1, None
            for c in batch:
                s = score_of(c[flags_key])
                if s > best_score:
                    best_score, best_c = s, c
            sel = best_c
        return sel

    batches = {}
    for c in candidates:
        batches.setdefault((c['seed_id'], c['call_id']), []).append(c)

    variant_stats = {}
    for variant, flags_key in [('Baseline', 'flags_base'), ('A', 'flags_A'), ('C', 'flags_C')]:
        zero_cnt = 0
        rice_c, soup_c, main_c, side_c, kim_c = Counter(), Counter(), Counter(), Counter(), Counter()
        for key, batch in batches.items():
            has_pass = any(c[flags_key]['all_pass'] for c in batch)
            if not has_pass:
                zero_cnt += 1
            sel = batch_selection(batch, variant, flags_key)
            if sel is not None:
                m = sel['menus']
                rice_c[m[0]] += 1; soup_c[m[1]] += 1; main_c[m[2]] += 1; side_c[m[3]] += 1; kim_c[m[4]] += 1
        variant_stats[variant] = {
            'zero_candidate_rate': zero_cnt / len(batches), 'final_selection_success_rate': 1 - zero_cnt / len(batches),
            'rice_unique': len(rice_c), 'soup_unique': len(soup_c), 'main_unique': len(main_c),
            'side_unique': len(side_c), 'kimchi_unique': len(kim_c),
        }
    print('배치(=call)수:', len(batches))
    for v, s in variant_stats.items():
        print(f"  {v}: zero_candidate_rate={s['zero_candidate_rate']*100:.1f}% "
              f"부찬다양성={s['side_unique']} 국다양성={s['soup_unique']} 주찬다양성={s['main_unique']} 김치다양성={s['kimchi_unique']}")

    # ── ② phosphorus_rawP_fix_summary.csv ──
    summary_rows = []
    for variant, flags_key, t_key in [('Baseline', 'flags_base', 't_base'), ('A', 'flags_A', 't_A'), ('C', 'flags_C', 't_C')]:
        flags_list = [c[flags_key] for c in candidates]
        n = len(flags_list)
        raw_pass_rate = sum(f['raw_pass'] for f in flags_list) / n
        eff_pass_rate = sum(f['eff_pass'] for f in flags_list) / n
        mismatch_rate = sum(1 for f in flags_list if f['raw_pass'] != f['eff_pass']) / n
        all_pass_rate = sum(f['all_pass'] for f in flags_list) / n
        f_fail_rate = sum(1 for c in candidates if c['before_pass'] and not c[flags_key]['raw_pass']) / n
        protein_low_rate = sum(f['protein_low'] for f in flags_list) / n
        protein_high_rate = sum(f['protein_high'] for f in flags_list) / n
        calorie_low_rate = sum(f['calorie_low'] for f in flags_list) / n
        calorie_high_rate = sum(f['calorie_high'] for f in flags_list) / n
        na_fail_rate = sum(1 for f in flags_list if not f['na_pass']) / n
        k_fail_rate = sum(1 for f in flags_list if not f['k_pass']) / n
        if variant == 'C':
            attempted = [c for c in candidates if c['fix_attempted']]
            fix_attempt_count = len(attempted)
            fix_success_rate = (sum(1 for c in attempted if c['fix_success']) / len(attempted)) if attempted else None
            mean_amt_reduced = (sum((c['amt_before'] - c['amt_after']) for c in attempted if c['amt_before'] is not None) /
                                len([c for c in attempted if c['amt_before'] is not None])) if attempted else None
            mean_rawP_reduction = (sum((c['t_base']['P'] - c['t_C']['P']) for c in attempted) / len(attempted)) if attempted else None
        else:
            fix_attempt_count = 0; fix_success_rate = None; mean_amt_reduced = None; mean_rawP_reduction = None
        vs = variant_stats[variant]
        summary_rows.append({
            'variant': variant, 'candidate_count': n, 'raw_P_pass_rate': raw_pass_rate,
            'Peff_pass_rate': eff_pass_rate, 'rawP_Peff_mismatch_rate': mismatch_rate,
            'nutrition_all_pass_rate': all_pass_rate, 'F_failure_rate': f_fail_rate,
            'protein_low_fail_rate': protein_low_rate, 'protein_high_fail_rate': protein_high_rate,
            'calorie_low_fail_rate': calorie_low_rate, 'calorie_high_fail_rate': calorie_high_rate,
            'sodium_fail_rate': na_fail_rate, 'potassium_fail_rate': k_fail_rate,
            'zero_candidate_rate': vs['zero_candidate_rate'],
            'final_selection_success_rate': vs['final_selection_success_rate'],
            'fix_attempt_count': fix_attempt_count, 'fix_success_rate': fix_success_rate,
            'mean_amount_reduced': mean_amt_reduced, 'mean_raw_P_reduction': mean_rawP_reduction,
            'rice_diversity': vs['rice_unique'], 'soup_diversity': vs['soup_unique'],
            'main_diversity': vs['main_unique'], 'side_diversity': vs['side_unique'],
            'kimchi_diversity': vs['kimchi_unique'],
        })
    summary_csv = os.path.join(OUT_DIR, 'phosphorus_rawP_fix_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f'② {summary_csv}')
    for r in summary_rows:
        print(f"  [{r['variant']}] rawP통과={r['raw_P_pass_rate']*100:.1f}% Peff통과={r['Peff_pass_rate']*100:.1f}% "
              f"불일치={r['rawP_Peff_mismatch_rate']*100:.1f}% 5영양전부={r['nutrition_all_pass_rate']*100:.1f}% "
              f"F발생={r['F_failure_rate']*100:.1f}% 단백하한={r['protein_low_fail_rate']*100:.1f}% "
              f"열량하한={r['calorie_low_fail_rate']*100:.1f}% 후보0개율={r['zero_candidate_rate']*100:.1f}%")

    # ── ③ phosphorus_rawP_fix_cause_breakdown.csv ──
    def group_members(name):
        if name == 'lever_protein_직후':
            return [c for c in candidates if c['cause_group'] == 'lever_protein_직후']
        if name == 'lever_calorie_직후':
            return [c for c in candidates if c['cause_group'] == 'lever_calorie_직후']
        if name == 'rawP실패_Peff통과':
            return [c for c in candidates if not c['flags_base']['raw_pass'] and c['flags_base']['eff_pass']]
        if name == 'rawP실패_Peff도실패':
            return [c for c in candidates if not c['flags_base']['raw_pass'] and not c['flags_base']['eff_pass']]
        if name == '두부양념조림_스케일업':
            return [c for c in candidates if c['fix_attempted'] and c['target_menu'] == ANCHOR_MENU]
        if name == '기타메뉴_스케일업':
            return [c for c in candidates if c['fix_attempted'] and c['target_menu'] is not None and c['target_menu'] != ANCHOR_MENU]
        return []

    cause_rows = []
    for gname in ['lever_protein_직후', 'lever_calorie_직후', 'rawP실패_Peff통과', 'rawP실패_Peff도실패',
                  '두부양념조림_스케일업', '기타메뉴_스케일업']:
        members = group_members(gname)
        n_m = len(members)
        if n_m == 0:
            continue
        for variant, flags_key in [('Baseline', 'flags_base'), ('A', 'flags_A'), ('C', 'flags_C')]:
            rescued = sum(1 for c in members if c[flags_key]['raw_pass'])
            full_pass = sum(1 for c in members if c[flags_key]['all_pass'])
            protein_regress = sum(1 for c in members if c['flags_base']['protein_pass'] and not c[flags_key]['protein_pass'])
            calorie_regress = sum(1 for c in members if c['flags_base']['calorie_pass'] and not c[flags_key]['calorie_pass'])
            cause_rows.append({
                'variant': variant, 'cause_group': gname, 'candidate_count': n_m,
                'rescued_count': rescued, 'rescued_rate': rescued / n_m,
                'nutrition_full_pass_count': full_pass,
                'protein_regression_count': protein_regress, 'calorie_regression_count': calorie_regress,
            })
    cause_csv = os.path.join(OUT_DIR, 'phosphorus_rawP_fix_cause_breakdown.csv')
    with open(cause_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(cause_rows[0].keys()))
        w.writeheader(); w.writerows(cause_rows)
    print(f'③ {cause_csv} ({len(cause_rows)}행)')

    return F, b, PMAX, candidates, variant_stats, summary_rows, cause_rows, OUT_DIR


if __name__ == '__main__':
    main()
