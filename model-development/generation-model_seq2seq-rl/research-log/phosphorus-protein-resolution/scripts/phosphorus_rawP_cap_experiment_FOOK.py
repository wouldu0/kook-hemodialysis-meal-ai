# -*- coding: utf-8 -*-
"""
phosphorus_rawP_cap_experiment_FOOK.py — 수정안 B(스케일업 전 raw P 예산으로 증량배율 제한)를
Baseline/B80/B90/B100 paired A/B로 검증. 두부·콩류 앵커. C_mask100+RL(epoch260). ★ 코드 수정 없음.

설계:
  공유 프리픽스(후보당 1회) = expand → lever_kimchi → lever_sodium → lever_sodium_extra
  (ROT[0] 오염 방지 위해 김치 레버는 후보당 정확히 1번만 호출, 4개 variant가 그 결과를 공유)
  이후 각 variant가 독립적으로 패스루프 2회를 실행:
    Baseline: lever_potassium → lever_phosphorus → lever_protein(원본) → lever_sodium →
              lever_sodium_extra → lever_calorie(원본)
    B80/90/100: lever_protein/lever_calorie 자리에 capped_lever_protein/capped_lever_calorie
              (메뉴 선택 로직은 원본과 동일 복제, 증량(scale_menu 비율>1) 직전에만 raw P cap 삽입)

cap 미적용 근거(코드로 확인, 실행 중 재검증):
  - add_oil()이 추가하는 참기름/콩기름은 P=0(원본 데이터) → 밀도 cap이 자연히 무제한
  - add_snack()은 호출부에서 이미 kmax/pmax로 raw P 절대예산을 자체 체크함(FOOK_adjust_levers.py
    969행 부근 `cur['P']+s[3]<=pmax`) → 별도 cap 불필요
  따라서 cap은 lever_protein의 두 scale_menu 호출과 lever_calorie의 밥 증량 분기에만 삽입.
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

OUT_DIR = os.path.join(CODE, 'phosphorus_rawP_cap_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_MENU = '두부양념조림'
SEED_ROWS = [11, 12, 6, 36, 7]
N_CALLS = 20
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
MARGIN = 0.01
VARIANTS = ['Baseline', 'B80', 'B90', 'B100']
FRAC = {'B80': 0.80, 'B90': 0.90, 'B100': 1.0}   # B100은 margin 방식(아래서 분기)


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


def run_shared_prefix(F, menus, na_target):
    inst = F.expand(list(menus))
    F.SWAP_LOG.clear()
    t0_inst = copy.deepcopy(inst)
    F.lever_kimchi(inst)
    F.lever_sodium(inst)
    F.lever_sodium_extra(inst, na_target)
    return t0_inst, inst


def _cap_and_scale(F, inst, menu, ratio, PMAX, frac, lever_tag, pass_no, log, abnormal_log, candidate_id):
    """ratio>1(증량)일 때만 raw P cap 적용 후 F.scale_menu() 호출(원본 함수 그대로 재사용)."""
    if ratio <= 1.0 + 1e-9:
        F.scale_menu(inst, menu, ratio)
        return
    current_raw_P = F.totals(inst)['P']
    headroom = PMAX - current_raw_P
    if frac >= 1.0:
        allowed_increase = max(0.0, headroom - MARGIN)
    else:
        allowed_increase = max(0.0, headroom * frac)
    menu_items = [i for i in inst if i['menu'] == menu]
    P_menu = sum(i['amt'] / 100 * i['P'] for i in menu_items if i['P'] is not None)
    Amt_menu = sum(i['amt'] for i in menu_items)
    requested_grams = Amt_menu * (ratio - 1.0)
    if Amt_menu <= 0:
        abnormal_log.append({'candidate_id': candidate_id, 'lever': lever_tag, 'menu': menu, 'reason': 'zero_amt'})
        F.scale_menu(inst, menu, ratio)
        return
    density = P_menu / Amt_menu
    if density <= 1e-9:
        allowed_grams = float('inf')
    else:
        allowed_grams = allowed_increase / density
    applied_grams = max(0.0, min(requested_grams, allowed_grams))
    final_ratio = 1.0 + applied_grams / Amt_menu
    capped = applied_grams < requested_grams - 1e-6
    F.scale_menu(inst, menu, final_ratio)
    log.append({
        'candidate_id': candidate_id, 'pass_no': pass_no, 'lever': lever_tag, 'menu': menu,
        'raw_P_headroom_before': headroom, 'allowed_P_increase': allowed_increase,
        'menu_raw_P_per_gram': density, 'requested_additional_grams': requested_grams,
        'allowed_additional_grams_by_P': allowed_grams, 'applied_additional_grams': applied_grams,
        'capped': capped, 'predicted_raw_P_increase_without_cap': requested_grams * density if density < float('inf') else None,
        'actual_raw_P_increase_with_cap': applied_grams * density if density < float('inf') else None,
    })


def capped_lever_protein(F, inst, lo, hi, anchor, PMAX, frac, pass_no, log, abnormal_log, candidate_id):
    t = F.totals(inst)['protein']
    if lo <= t <= hi:
        return
    pm = {}
    for i in inst:
        if i['protein']:
            pm[i['menu']] = pm.get(i['menu'], 0.0) + i['amt'] / 100 * i['protein']
    if not pm:
        return
    target = (lo + hi) / 2
    for m in sorted([x for x in pm if x != anchor], key=pm.get, reverse=True):
        cur = pm[m]
        new = target - (t - cur)
        if cur > 0 and new > 0 and 0.3 <= new / cur <= 2.0:
            _cap_and_scale(F, inst, m, new / cur, PMAX, frac, 'protein', pass_no, log, abnormal_log, candidate_id)
            return
    top = max(pm, key=pm.get)
    cur_top = pm[top]
    new_top = target - (t - cur_top)
    if cur_top > 0 and new_top > 0:
        ratio = max(0.3, min(new_top / cur_top, 2.0))
        _cap_and_scale(F, inst, top, ratio, PMAX, frac, 'protein', pass_no, log, abnormal_log, candidate_id)


def capped_lever_calorie(F, inst, lo, hi, anchor, allow_snack, kmax, pmax_arg, PMAX, frac, pass_no, log, abnormal_log, candidate_id):
    e = F.totals(inst)['E']
    if e > hi:
        # 감소 분기 — 원본 그대로(감량은 raw P를 못 늘리므로 cap 불필요)
        rice_pool, rice_cur = F._pick_pool([i for i in inst if i['menu'] in F.RICE and i['E']], anchor, e - hi)
        if rice_cur > 0:
            new = max(rice_cur - (e - hi), rice_cur * F.RICE_FLOOR)
            f = new / rice_cur
            for i in rice_pool:
                i['amt'] = max(i['amt'] * f, F.amt_floor_of(i))
        e2 = F.totals(inst)['E']
        if e2 > hi:
            oil_pool, oil_cur = F._pick_pool([i for i in inst if i['group'] == '유지류' and i['E']], anchor, e2 - hi)
            if oil_cur > 0:
                new = max(oil_cur - (e2 - hi), 0)
                f = new / oil_cur
                for i in oil_pool:
                    i['amt'] = max(i['amt'] * f, F.amt_floor_of(i))
    elif e < lo and e > 0:
        rice = [i for i in inst if i['menu'] in F.RICE and i['E']]
        non = [i for i in rice if i['menu'] != anchor]
        pool = non if non else rice
        cur = sum(i['amt'] / 100 * i['E'] for i in pool)
        if cur > 0:
            new = min(cur + (lo - e), cur * 1.3)
            ratio = new / cur
            # 밥 증량 — 원본은 여러 (메뉴,재료) 인스턴스를 한 풀로 묶어 같은 비율 적용. cap도 풀 전체 기준.
            if ratio > 1.0 + 1e-9:
                current_raw_P = F.totals(inst)['P']
                headroom = PMAX - current_raw_P
                allowed_increase = max(0.0, headroom - MARGIN) if frac >= 1.0 else max(0.0, headroom * frac)
                P_pool = sum(i['amt'] / 100 * i['P'] for i in pool if i['P'] is not None)
                Amt_pool = sum(i['amt'] for i in pool)
                requested_grams = Amt_pool * (ratio - 1.0)
                if Amt_pool <= 0:
                    abnormal_log.append({'candidate_id': candidate_id, 'lever': 'calorie_rice', 'menu': 'RICE_POOL', 'reason': 'zero_amt'})
                    final_ratio = ratio
                else:
                    density = P_pool / Amt_pool
                    allowed_grams = float('inf') if density <= 1e-9 else allowed_increase / density
                    applied_grams = max(0.0, min(requested_grams, allowed_grams))
                    final_ratio = 1.0 + applied_grams / Amt_pool
                    capped = applied_grams < requested_grams - 1e-6
                    log.append({
                        'candidate_id': candidate_id, 'pass_no': pass_no, 'lever': 'calorie_rice', 'menu': 'RICE_POOL',
                        'raw_P_headroom_before': headroom, 'allowed_P_increase': allowed_increase,
                        'menu_raw_P_per_gram': density, 'requested_additional_grams': requested_grams,
                        'allowed_additional_grams_by_P': allowed_grams, 'applied_additional_grams': applied_grams,
                        'capped': capped, 'predicted_raw_P_increase_without_cap': requested_grams * density if density < float('inf') else None,
                        'actual_raw_P_increase_with_cap': applied_grams * density if density < float('inf') else None,
                    })
            else:
                final_ratio = ratio
            for i in pool:
                i['amt'] *= final_ratio
        e2 = F.totals(inst)['E']
        if e2 < lo:
            F.add_oil(inst, lo - e2)   # 원본 그대로(참기름/콩기름 P=0 — cap 불필요, 실측으로 검증)
            e3 = F.totals(inst)['E']
            if e3 < lo and allow_snack:
                F.add_snack(inst, lo - e3, kmax=kmax, pmax=pmax_arg)  # 원본 그대로(이미 raw P 예산 체크 내장)


def run_pass_loop(F, inst, anchor, b, na_target, variant, PMAX, cap_log, abnormal_log, candidate_id):
    frac = FRAC.get(variant)
    for pass_i in range(2):
        pass_no = pass_i + 1
        F.lever_potassium(inst, b['Kmax'], anchor=anchor)
        F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        if variant == 'Baseline':
            F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor)
        else:
            capped_lever_protein(F, inst, b['Plo'], b['Phi'], anchor, PMAX, frac, pass_no, cap_log, abnormal_log, candidate_id)
        F.lever_sodium(inst)
        F.lever_sodium_extra(inst, na_target)
        if variant == 'Baseline':
            F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                             kmax=b['Kmax'], pmax=b['Pmax'])
        else:
            capped_lever_calorie(F, inst, b['Elo'], b['Ehi'], anchor, (pass_i == 1), b['Kmax'], b['Pmax'],
                                  PMAX, frac, pass_no, cap_log, abnormal_log, candidate_id)
    return inst


def nutrient_flags(t, b):
    raw_pass = t['P'] < b['Pmax']
    eff_pass = t['Peff'] < b['Pmax']
    protein_low = t['protein'] < b['Plo']; protein_high = t['protein'] > b['Phi']
    calorie_low = t['E'] < b['Elo']; calorie_high = t['E'] > b['Ehi']
    na_pass = t['Na_season'] <= b['Namax']; k_pass = t['K'] < b['Kmax']
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

    print(f'생성 시작: 두부콩류 {len(SEED_ROWS)}seed x {N_CALLS}call x {TRIES} = {len(SEED_ROWS)*N_CALLS*TRIES}후보(예정) x 4variant')

    candidates = []
    cap_log_all = []
    abnormal_log_all = []
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
                t0_inst, prefix_inst = run_shared_prefix(F, menus, na_target)
                t0 = F.totals(t0_inst)
                results = {}
                for variant in VARIANTS:
                    inst_v = copy.deepcopy(prefix_inst)
                    cap_log = []
                    inst_v = run_pass_loop(F, inst_v, ANCHOR_MENU, b, na_target, variant, PMAX,
                                            cap_log, abnormal_log_all, cid)
                    t_v = F.totals(inst_v)
                    for e in cap_log:
                        e['variant'] = variant
                    cap_log_all.extend(cap_log)
                    results[variant] = {'t': t_v, 'flags': nutrient_flags(t_v, b), 'cap_log': cap_log}
                candidates.append({'candidate_id': cid, 'seed_id': sid, 'call_id': call_id, 'menus': menus,
                                    't0': t0, 'before_pass': t0['P'] < PMAX, 'results': results})
        print(f'  seed {sid}(row{row_idx}) 완료, 누적 {len(candidates)}건')

    print(f'\n총 생성: {len(candidates)}건, cap 이벤트: {len(cap_log_all)}건, 비정상 로그: {len(abnormal_log_all)}건')

    # ── 후보별 파생 필드 ──
    for c in candidates:
        for v in VARIANTS:
            r = c['results'][v]
            cap_events = r['cap_log']
            protein_events = [e for e in cap_events if e['lever'] == 'protein']
            calorie_events = [e for e in cap_events if e['lever'] == 'calorie_rice']
            protein_cap_triggered = any(e['capped'] for e in protein_events)
            calorie_cap_triggered = any(e['capped'] for e in calorie_events)
            primary = None
            all_events = protein_events + calorie_events
            if all_events:
                primary = max(all_events, key=lambda e: e['requested_additional_grams'])
            r['protein_cap_triggered'] = protein_cap_triggered
            r['calorie_cap_triggered'] = calorie_cap_triggered
            r['cap_triggered_lever'] = ('protein' if protein_cap_triggered else ('calorie' if calorie_cap_triggered else 'none'))
            r['primary_event'] = primary
            r['F_failure'] = c['before_pass'] and not r['flags']['raw_pass']

    n_total = len(candidates)

    # ── ① phosphorus_rawP_cap_trace.csv ──
    trace_rows = []
    for c in candidates:
        for idx, variant in enumerate(VARIANTS):
            r = c['results'][variant]
            t = r['t']; flags = r['flags']; primary = r['primary_event']
            trace_rows.append({
                'candidate_id': c['candidate_id'], 'seed_id': c['seed_id'], 'call_id': c['call_id'],
                'candidate_index': idx, 'variant': variant, 'anchor_type': '두부콩류',
                'raw_P_before_adjust': c['t0']['P'], 'raw_P_after_adjust': t['P'],
                'Peff_before_adjust': c['t0']['Peff'], 'Peff_after_adjust': t['Peff'],
                'Pmax': PMAX,
                'protein_before': c['t0']['protein'], 'protein_after': t['protein'],
                'calorie_before': c['t0']['E'], 'calorie_after': t['E'],
                'potassium_after': t['K'], 'sodium_after': t['Na_season'],
                'raw_P_pass': flags['raw_pass'], 'Peff_pass': flags['eff_pass'],
                'protein_pass': flags['protein_pass'], 'calorie_pass': flags['calorie_pass'],
                'nutrition_all_pass': flags['all_pass'], 'F_failure': r['F_failure'],
                'protein_cap_triggered': r['protein_cap_triggered'], 'calorie_cap_triggered': r['calorie_cap_triggered'],
                'cap_triggered_lever': r['cap_triggered_lever'],
                'adjusted_menu': primary['menu'] if primary else None,
                'requested_additional_grams': primary['requested_additional_grams'] if primary else None,
                'allowed_additional_grams_by_P': primary['allowed_additional_grams_by_P'] if primary else None,
                'applied_additional_grams': primary['applied_additional_grams'] if primary else None,
                'prevented_additional_grams': (primary['requested_additional_grams'] - primary['applied_additional_grams']) if primary else None,
                'raw_P_headroom_before': primary['raw_P_headroom_before'] if primary else None,
                'allowed_P_increase': primary['allowed_P_increase'] if primary else None,
                'menu_raw_P_per_gram': primary['menu_raw_P_per_gram'] if primary else None,
                'predicted_raw_P_increase_without_cap': primary['predicted_raw_P_increase_without_cap'] if primary else None,
                'actual_raw_P_increase_with_cap': primary['actual_raw_P_increase_with_cap'] if primary else None,
                'protein_shortfall_after': max(0.0, b['Plo'] - t['protein']),
                'calorie_shortfall_after': max(0.0, b['Elo'] - t['E']),
                'final_failure_reason': (None if flags['all_pass'] else
                                          ('raw_P' if not flags['raw_pass'] else
                                           ('protein_low' if flags['protein_low'] else
                                            ('protein_high' if flags['protein_high'] else
                                             ('calorie_low' if flags['calorie_low'] else
                                              ('calorie_high' if flags['calorie_high'] else
                                               ('sodium' if not flags['na_pass'] else 'potassium'))))))),
            })
    trace_csv = os.path.join(OUT_DIR, 'phosphorus_rawP_cap_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    print(f'① {trace_csv} ({len(trace_rows)}행)')

    # ── 배치(seed,call) 선택 시뮬레이션 ──
    def score_of(flags):
        return sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])

    batches = {}
    for c in candidates:
        batches.setdefault((c['seed_id'], c['call_id']), []).append(c)

    variant_batch_stats = {}
    for variant in VARIANTS:
        zero_cnt = 0
        rice_c, soup_c, main_c, side_c, kim_c = Counter(), Counter(), Counter(), Counter(), Counter()
        unique_meals = set()
        for key, batch in batches.items():
            has_pass = any(c['results'][variant]['flags']['all_pass'] for c in batch)
            if not has_pass:
                zero_cnt += 1
            sel = None
            for c in batch:
                if c['results'][variant]['flags']['all_pass']:
                    sel = c; break
            if sel is None:
                best_score, best_c = -1, None
                for c in batch:
                    s = score_of(c['results'][variant]['flags'])
                    if s > best_score:
                        best_score, best_c = s, c
                sel = best_c
            m = sel['menus']
            rice_c[m[0]] += 1; soup_c[m[1]] += 1; main_c[m[2]] += 1; side_c[m[3]] += 1; kim_c[m[4]] += 1
            unique_meals.add(tuple(m))
        variant_batch_stats[variant] = {
            'zero_candidate_rate': zero_cnt / len(batches),
            'final_generation_success_rate': 1 - zero_cnt / len(batches),
            'rice_diversity': len(rice_c), 'soup_diversity': len(soup_c), 'main_diversity': len(main_c),
            'side_diversity': len(side_c), 'kimchi_diversity': len(kim_c), 'unique_meal_count': len(unique_meals),
        }
    print('배치수:', len(batches))
    for v, s in variant_batch_stats.items():
        print(f"  {v}: 후보0개율={s['zero_candidate_rate']*100:.1f}% 최종생성성공률={s['final_generation_success_rate']*100:.1f}% "
              f"부찬다양성={s['side_diversity']} 국다양성={s['soup_diversity']} 고유식단={s['unique_meal_count']}")

    # ── ② phosphorus_rawP_cap_summary.csv ──
    summary_rows = []
    for variant in VARIANTS:
        flags_list = [c['results'][variant]['flags'] for c in candidates]
        n = len(flags_list)
        raw_pass_rate = sum(f['raw_pass'] for f in flags_list) / n
        eff_pass_rate = sum(f['eff_pass'] for f in flags_list) / n
        mismatch_rate = sum(1 for f in flags_list if f['raw_pass'] != f['eff_pass']) / n
        protein_pass_rate = sum(f['protein_pass'] for f in flags_list) / n
        calorie_pass_rate = sum(f['calorie_pass'] for f in flags_list) / n
        all_pass_rate = sum(f['all_pass'] for f in flags_list) / n
        f_fail_rate = sum(1 for c in candidates if c['results'][variant]['F_failure']) / n
        protein_low_rate = sum(f['protein_low'] for f in flags_list) / n
        calorie_low_rate = sum(f['calorie_low'] for f in flags_list) / n
        cap_events_v = [c['results'][variant]['primary_event'] for c in candidates if c['results'][variant]['primary_event']]
        cap_trigger_rate = sum(1 for c in candidates if c['results'][variant]['cap_triggered_lever'] != 'none') / n
        base_flags = [c['results']['Baseline']['flags'] for c in candidates]
        # cap이 실제로 raw P 실패를 막은 건수: Baseline에서 raw P 실패였는데 이 variant에서 raw P 통과
        cap_prevented = sum(1 for c in candidates if not c['results']['Baseline']['flags']['raw_pass'] and c['results'][variant]['flags']['raw_pass'])
        new_protein_fail = sum(1 for c in candidates if c['results']['Baseline']['flags']['protein_pass'] and not c['results'][variant]['flags']['protein_pass'])
        new_calorie_fail = sum(1 for c in candidates if c['results']['Baseline']['flags']['calorie_pass'] and not c['results'][variant]['flags']['calorie_pass'])
        vs = variant_batch_stats[variant]
        summary_rows.append({
            'variant': variant, 'candidate_count': n, 'raw_P_pass_rate': raw_pass_rate,
            'Peff_pass_rate': eff_pass_rate, 'rawP_Peff_mismatch_rate': mismatch_rate,
            'protein_pass_rate': protein_pass_rate, 'calorie_pass_rate': calorie_pass_rate,
            'nutrition_all_pass_rate': all_pass_rate, 'F_failure_rate': f_fail_rate,
            'protein_low_fail_rate': protein_low_rate, 'calorie_low_fail_rate': calorie_low_rate,
            'zero_candidate_rate': vs['zero_candidate_rate'],
            'final_generation_success_rate': vs['final_generation_success_rate'],
            'cap_trigger_rate': cap_trigger_rate,
            'cap_prevented_P_failure_count': cap_prevented,
            'new_protein_failure_count': new_protein_fail, 'new_calorie_failure_count': new_calorie_fail,
            'unique_meal_count': vs['unique_meal_count'],
        })
    summary_csv = os.path.join(OUT_DIR, 'phosphorus_rawP_cap_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f'② {summary_csv}')
    for r in summary_rows:
        print(f"  [{r['variant']}] rawP={r['raw_P_pass_rate']*100:.1f}% protein={r['protein_pass_rate']*100:.1f}% "
              f"calorie={r['calorie_pass_rate']*100:.1f}% 5영양전부={r['nutrition_all_pass_rate']*100:.1f}% "
              f"F발생={r['F_failure_rate']*100:.1f}% capTrig={r['cap_trigger_rate']*100:.1f}% "
              f"신규단백실패={r['new_protein_failure_count']} 신규열량실패={r['new_calorie_failure_count']}")

    # ── ③ phosphorus_rawP_cap_transition.csv ──
    trans_defs = {
        'baseline_rawP_fail_to_B_rawP_pass': lambda c, v: (not c['results']['Baseline']['flags']['raw_pass']) and c['results'][v]['flags']['raw_pass'],
        'baseline_all_fail_to_B_all_pass': lambda c, v: (not c['results']['Baseline']['flags']['all_pass']) and c['results'][v]['flags']['all_pass'],
        'baseline_all_pass_to_B_protein_low': lambda c, v: c['results']['Baseline']['flags']['all_pass'] and c['results'][v]['flags']['protein_low'],
        'baseline_all_pass_to_B_calorie_low': lambda c, v: c['results']['Baseline']['flags']['all_pass'] and c['results'][v]['flags']['calorie_low'],
        'baseline_F_to_B_all_pass': lambda c, v: c['results']['Baseline']['F_failure'] and c['results'][v]['flags']['all_pass'],
        'baseline_F_to_B_rawP_pass_protein_fail': lambda c, v: c['results']['Baseline']['F_failure'] and c['results'][v]['flags']['raw_pass'] and not c['results'][v]['flags']['protein_pass'],
        'baseline_F_to_B_rawP_pass_calorie_fail': lambda c, v: c['results']['Baseline']['F_failure'] and c['results'][v]['flags']['raw_pass'] and not c['results'][v]['flags']['calorie_pass'],
        'both_fail': lambda c, v: (not c['results']['Baseline']['flags']['all_pass']) and (not c['results'][v]['flags']['all_pass']),
        'reason_moved_phosphorus_to_protein': lambda c, v: (not c['results']['Baseline']['flags']['raw_pass']) and c['results'][v]['flags']['raw_pass'] and not c['results'][v]['flags']['protein_pass'] and c['results']['Baseline']['flags']['protein_pass'],
        'reason_moved_phosphorus_to_calorie': lambda c, v: (not c['results']['Baseline']['flags']['raw_pass']) and c['results'][v]['flags']['raw_pass'] and not c['results'][v]['flags']['calorie_pass'] and c['results']['Baseline']['flags']['calorie_pass'],
    }
    trans_rows = []
    for v in ['B80', 'B90', 'B100']:
        for name, fn in trans_defs.items():
            cnt = sum(1 for c in candidates if fn(c, v))
            trans_rows.append({'variant': v, 'transition_type': name, 'candidate_count': cnt, 'candidate_rate': cnt / n_total})
    trans_csv = os.path.join(OUT_DIR, 'phosphorus_rawP_cap_transition.csv')
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
        w.writeheader(); w.writerows(trans_rows)
    print(f'③ {trans_csv} ({len(trans_rows)}행)')

    # ── ④ phosphorus_rawP_cap_failure_types.csv (B1~B6) ──
    def classify_failure(c, v):
        base = c['results']['Baseline']['flags']
        cur = c['results'][v]['flags']
        r = c['results'][v]
        if cur['all_pass']:
            return None  # 실패 아님
        if base['all_pass'] is False and base['raw_pass'] is True:
            return 'B6_원래실패(P무관)'
        if cur['protein_low']:
            primary = r['primary_event']
            if r['protein_cap_triggered'] and primary and primary['applied_additional_grams'] / max(primary['requested_additional_grams'], 1e-9) < 0.2:
                return 'B1_인예산협소_증량거의불가'
            elif r['protein_cap_triggered']:
                return 'B2_일부증량_단백여전히미달'
            else:
                return 'B_기타_단백미달(cap무관)'
        if cur['protein_pass'] and cur['calorie_low']:
            return 'B3_단백충족_열량미달'
        if cur['protein_pass'] and cur['calorie_pass'] and (not cur['na_pass'] or not cur['k_pass']):
            return 'B4_단백열량충족_기타영양실패'
        if not cur['raw_pass']:
            return 'B5_rawP여전히초과'
        return 'B_기타'

    failtype_rows = []
    for v in ['B80', 'B90', 'B100']:
        fails = [(c, classify_failure(c, v)) for c in candidates]
        fails = [(c, t) for c, t in fails if t is not None]
        n_fail = len(fails)
        by_type = {}
        for c, t in fails:
            by_type.setdefault(t, []).append(c)
        for t, members in by_type.items():
            headrooms = [PMAX - c['t0']['P'] for c in members]
            pshort = [c['results'][v]['t']['protein'] for c in members]
            pshort = [max(0.0, b['Plo'] - x) for x in pshort]
            cshort = [max(0.0, b['Elo'] - c['results'][v]['t']['E']) for c in members]
            failtype_rows.append({
                'variant': v, 'failure_type': t, 'candidate_count': len(members),
                'candidate_rate': len(members) / n_total,
                'mean_raw_P_headroom': sum(headrooms) / len(headrooms),
                'mean_protein_shortfall': sum(pshort) / len(pshort),
                'mean_calorie_shortfall': sum(cshort) / len(cshort),
            })
    failtype_csv = os.path.join(OUT_DIR, 'phosphorus_rawP_cap_failure_types.csv')
    with open(failtype_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(failtype_rows[0].keys()))
        w.writeheader(); w.writerows(failtype_rows)
    print(f'④ {failtype_csv} ({len(failtype_rows)}행)')

    # ── cap 이벤트/비정상 로그 원본도 참고용 저장 ──
    if cap_log_all:
        with open(os.path.join(OUT_DIR, 'cap_events_raw_log.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(cap_log_all[0].keys()))
            w.writeheader(); w.writerows(cap_log_all)
    if abnormal_log_all:
        with open(os.path.join(OUT_DIR, 'cap_abnormal_log.csv'), 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=list(abnormal_log_all[0].keys()))
            w.writeheader(); w.writerows(abnormal_log_all)

    return F, b, PMAX, candidates, cap_log_all, abnormal_log_all, summary_rows, trans_rows, failtype_rows, OUT_DIR


if __name__ == '__main__':
    main()
