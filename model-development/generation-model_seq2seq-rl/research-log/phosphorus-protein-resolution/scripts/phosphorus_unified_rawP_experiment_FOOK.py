# -*- coding: utf-8 -*-
"""
phosphorus_unified_rawP_experiment_FOOK.py — lever_phosphorus() 내부 판정기준을 Peff에서
raw P로 전부 통일한 "Unified-rawP" 최소 수정안을 Baseline과 paired 비교. 두부·콩류 앵커.
C_mask100+RL(epoch260). ★ 코드 수정 없음 — 진단용 복사본에서만 실험.

Baseline      : 원본 lever_phosphorus() 그대로(Peff 기준 진입/수렴/최종판정), 원본 lever_protein/
                lever_calorie 그대로(B90 cap 미적용), 원본 2회 패스 구조 그대로.
Unified-rawP  : lever_phosphorus()의 4개 Peff/p_abs() 사용처를 raw P로 치환한 복제본
                (unified_lever_phosphorus_rawP)만 대체. 그 외 전부 Baseline과 동일.

Peff 사용처(원본 코드 확인, 4곳 전부 raw P로 치환):
  1) 루프 진입/수렴 조건: totals(inst)['Peff'] < pmax  → totals(inst)['P'] < pmax
  2) 대체후보 비교: p_abs(i['P'],...) vs p_abs(nd['P'],...) → i['P'] vs nd['P'] (원값)
  3) 양감소 대상 랭킹: max(cand, key=...p_abs(x['P'],...)) → max(cand, key=...x['P'])
  4) 루프소진 최종반환: totals(inst)['Peff'] < pmax → totals(inst)['P'] < pmax
  (SUBS_P/DRIED/same_category/is_processed_name/menu_has_ingredient/is_sole_solid_ingredient/
   reducible/reduce_amt/rename_menu_for_swap/SWAP_LOG/P_SWAP_MIN_GAIN 등 그 외 로직·임계값·
   메뉴선택 순서는 전부 원본 그대로 재사용, 무변경)
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

OUT_DIR = os.path.join(CODE, 'phosphorus_unified_rawP_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_MENU = '두부양념조림'
SEED_ROWS = [11, 12, 6, 36, 7]
N_CALLS = 20
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
VARIANTS = ['Baseline', 'Unified-rawP']


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


def unified_lever_phosphorus_rawP(F, inst, pmax, anchor=None, plo=0):
    """lever_phosphorus 복제 — Peff/p_abs() 4곳을 raw P로 치환. 그 외 전부 원본과 동일."""
    ing_nut, base_fresh, ing2kw, kw_rep, subs, _ = F.NUT
    for _ in range(25):
        if F.totals(inst)['P'] < pmax:                       # (1) 원본: Peff
            return True
        non = [i for i in inst if i['menu'] != anchor]

        best = None
        for i in non:
            if i['P'] is None:
                continue
            if i['group'] == '조미료류':
                continue
            if any(d in i['ing'] for d in F.DRIED):
                continue
            if F.is_sole_solid_ingredient(non, i['menu'], exclude=i):
                continue
            for sub in F.SUBS_P.get(ing2kw.get(i['ing']), []):
                rep = kw_rep.get(sub); nd = ing_nut.get(rep) if rep else None
                if not nd or nd['P'] is None or not F.same_category(nd['group'], i['group']):
                    continue
                if rep.split(',')[0].strip() == i['ing'].split(',')[0].strip():
                    continue
                if F.is_processed_name(rep, nd['group']) and not F.is_processed(i):
                    continue
                if F.menu_has_ingredient(non, i['menu'], rep, exclude=i):
                    continue
                effP_i = i['P']                                # (2) 원본: p_abs(i['P'],...)
                effP_nd = nd['P']                               # (2) 원본: p_abs(nd['P'],...)
                if effP_nd < effP_i:
                    ip, npr = (i['protein'] or 0), (nd['protein'] or 0)
                    if ip == 0 or npr >= ip * 0.75:
                        g = i['amt'] / 100 * (effP_i - effP_nd)
                        if g >= F.P_SWAP_MIN_GAIN and (best is None or g > best[0]):
                            best = (g, i, rep, nd)
                    break
        if best:
            _, i, rep, nd = best
            F.SWAP_LOG.append((i['menu'], i['ing'], rep, 'P', i['P'], nd['P']))
            old_ing, old_menu = i['ing'], i['menu']
            i['ing'] = rep
            for k in ('E', 'protein', 'P', 'K', 'Na', 'group'):
                i[k] = nd[k]
            F.rename_menu_for_swap(inst, old_menu, old_ing, rep)
            continue

        cand = [i for i in non if i['P'] and i['amt'] > 1 and F.reducible(i)]
        if cand:
            F.reduce_amt(max(cand, key=lambda x: x['amt'] / 100 * x['P']), 0.7)   # (3) 원본: p_abs
            continue

        anc = [i for i in inst if i['menu'] == anchor and i['amt'] > 1 and F.reducible(i)]
        if anc and F.totals(inst)['protein'] > plo:
            for i in anc:
                F.reduce_amt(i, 0.85)
            continue

        return False
    return F.totals(inst)['P'] < pmax                            # (4) 원본: Peff


def run_pass_loop_with_steps(F, inst, anchor, b, na_target, variant):
    """패스루프 실행 + 매 레버 직후 totals() 스냅샷(핑퐁·pass1/2 변화 탐지용)."""
    steps = []

    def snap(name, pass_no):
        steps.append((name, pass_no, F.totals(inst)))

    for pass_i in range(2):
        pass_no = pass_i + 1
        F.lever_potassium(inst, b['Kmax'], anchor=anchor); snap('potassium', pass_no)
        if variant == 'Baseline':
            F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        else:
            unified_lever_phosphorus_rawP(F, inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        snap('phosphorus', pass_no)
        F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor); snap('protein', pass_no)
        F.lever_sodium(inst); snap('sodium', pass_no)
        F.lever_sodium_extra(inst, na_target); snap('sodium_extra', pass_no)
        F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                         kmax=b['Kmax'], pmax=b['Pmax'])
        snap('calorie', pass_no)
    return inst, steps


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

    # ── 검증: Baseline 경로(F.lever_phosphorus 원본 호출)가 F.adjust()와 일치하는지 5건 확인 ──
    _, food_dict0, diet_np0, _, mask_id0 = build_data(with_mask=True)
    num_tokens0 = len(food_dict0)
    orig_diet_np0 = diet_np0.numpy()
    enc0, dec0 = load_model(RL_CKPT_DIR, num_tokens0)
    anchor_token0 = core.name2idx[ANCHOR_MENU]

    print('=== 검증: Baseline 경로 vs F.adjust() 5건 ===')
    val_ok = 0
    for vi in range(5):
        np.random.seed(999 + vi)
        menus_v = gen_batch_slots(core, enc0, dec0, num_tokens0, mask_id0, food_dict0,
                                   orig_diet_np0[SEED_ROWS[0]], anchor_token0, 4, TEMP)[0]
        if len(menus_v) != 5:
            continue
        F.ROT[0] = 0
        t0_v, prefix_v = run_shared_prefix(F, menus_v, na_target)
        inst_base, _ = run_pass_loop_with_steps(F, copy.deepcopy(prefix_v), ANCHOR_MENU, b, na_target, 'Baseline')
        manual_final = F.totals(inst_base)
        F.ROT[0] = 0
        _, adj_after, _, _ = F.adjust(list(menus_v), b, anchor=ANCHOR_MENU)
        ok = all(abs(manual_final[k] - adj_after[k]) < 1e-6 for k in ('E', 'protein', 'P', 'K', 'Na_season'))
        val_ok += int(ok)
        print(f'  검증{vi}: manual_P={manual_final["P"]:.3f} adjust_P={adj_after["P"]:.3f} 일치={ok}')
    print(f'검증 결과: {val_ok}/5 일치\n')
    F.ROT[0] = 0

    print(f'생성 시작: 두부콩류 {len(SEED_ROWS)}seed x {N_CALLS}call x {TRIES} = {len(SEED_ROWS)*N_CALLS*TRIES}후보(예정) x 2variant')

    candidates = []
    cid = 0
    for sid, row_idx in enumerate(SEED_ROWS):
        base_row = orig_diet_np0[row_idx].copy()
        np.random.seed(RNG_SEED)
        for call_id in range(N_CALLS):
            menus_list = gen_batch_slots(core, enc0, dec0, num_tokens0, mask_id0, food_dict0,
                                          base_row, anchor_token0, TRIES, TEMP)
            for menus in menus_list:
                if len(menus) != 5:
                    continue
                cid += 1
                t0_inst, prefix_inst = run_shared_prefix(F, menus, na_target)
                t0 = F.totals(t0_inst)
                results = {}
                for variant in VARIANTS:
                    inst_v = copy.deepcopy(prefix_inst)
                    inst_v, steps = run_pass_loop_with_steps(F, inst_v, ANCHOR_MENU, b, na_target, variant)
                    t_v = F.totals(inst_v)
                    results[variant] = {'t': t_v, 'flags': nutrient_flags(t_v, b), 'steps': steps}
                candidates.append({'candidate_id': cid, 'seed_id': sid, 'call_id': call_id, 'menus': menus,
                                    't0': t0, 'before_pass': t0['P'] < PMAX, 'results': results})
        print(f'  seed {sid}(row{row_idx}) 완료, 누적 {len(candidates)}건')

    print(f'\n총 생성: {len(candidates)}건')

    for c in candidates:
        for v in VARIANTS:
            r = c['results'][v]
            r['F_failure'] = c['before_pass'] and not r['flags']['raw_pass']
            steps = r['steps']
            raw_seq = [(name, pn, t['P'] < PMAX) for name, pn, t in steps]
            pingpong = 0
            for i in range(len(raw_seq) - 1):
                if raw_seq[i][0] == 'phosphorus' and raw_seq[i][2] is True and raw_seq[i + 1][0] == 'protein' and raw_seq[i + 1][2] is False:
                    pingpong += 1
            r['pingpong_events'] = pingpong
            r['pingpong'] = pingpong > 0
            pass1_end = [t for name, pn, t in steps if name == 'calorie' and pn == 1][0]
            pass2_end = [t for name, pn, t in steps if name == 'calorie' and pn == 2][0]
            r['rawP_pass1_end'] = pass1_end['P']
            r['rawP_pass2_end'] = pass2_end['P']
            r['rawP_pass1_to_pass2_delta'] = pass2_end['P'] - pass1_end['P']

    n_total = len(candidates)

    # ── 배치 선택 시뮬레이션(다양성, 후보0개율) ──
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

    # ── ① phosphorus_unified_rawP_trace.csv ──
    trace_rows = []
    for c in candidates:
        for variant in VARIANTS:
            r = c['results'][variant]; t = r['t']; flags = r['flags']
            trace_rows.append({
                'candidate_id': c['candidate_id'], 'seed_id': c['seed_id'], 'call_id': c['call_id'],
                'variant': variant, 'anchor_type': '두부콩류',
                'raw_P_before_adjust': c['t0']['P'], 'raw_P_after_adjust': t['P'],
                'Peff_before_adjust': c['t0']['Peff'], 'Peff_after_adjust': t['Peff'], 'Pmax': PMAX,
                'protein_before': c['t0']['protein'], 'protein_after': t['protein'],
                'calorie_before': c['t0']['E'], 'calorie_after': t['E'],
                'potassium_after': t['K'], 'sodium_after': t['Na_season'],
                'raw_P_pass': flags['raw_pass'], 'Peff_pass': flags['eff_pass'],
                'protein_pass': flags['protein_pass'], 'calorie_pass': flags['calorie_pass'],
                'nutrition_all_pass': flags['all_pass'], 'F_failure': r['F_failure'],
                'pingpong_events': r['pingpong_events'], 'pingpong': r['pingpong'],
                'rawP_pass1_end': r['rawP_pass1_end'], 'rawP_pass2_end': r['rawP_pass2_end'],
                'rawP_pass1_to_pass2_delta': r['rawP_pass1_to_pass2_delta'],
            })
    trace_csv = os.path.join(OUT_DIR, 'phosphorus_unified_rawP_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    print(f'① {trace_csv} ({len(trace_rows)}행)')

    # ── ② phosphorus_unified_rawP_summary.csv ──
    summary_rows = []
    for variant in VARIANTS:
        flags_list = [c['results'][variant]['flags'] for c in candidates]
        n = len(flags_list)
        raw_pass_rate = sum(f['raw_pass'] for f in flags_list) / n
        eff_pass_rate = sum(f['eff_pass'] for f in flags_list) / n
        mismatch_rate = sum(1 for f in flags_list if f['raw_pass'] != f['eff_pass']) / n
        all_pass_rate = sum(f['all_pass'] for f in flags_list) / n
        f_fail_rate = sum(1 for c in candidates if c['results'][variant]['F_failure']) / n
        protein_low_rate = sum(f['protein_low'] for f in flags_list) / n
        calorie_low_rate = sum(f['calorie_low'] for f in flags_list) / n
        pingpong_rate = sum(1 for c in candidates if c['results'][variant]['pingpong']) / n
        mean_pass_delta = sum(c['results'][variant]['rawP_pass1_to_pass2_delta'] for c in candidates) / n
        new_all_fail = sum(1 for c in candidates if c['results']['Baseline']['flags']['all_pass'] and not c['results'][variant]['flags']['all_pass'])
        vs = variant_batch_stats[variant]
        summary_rows.append({
            'variant': variant, 'candidate_count': n, 'raw_P_pass_rate': raw_pass_rate,
            'Peff_pass_rate': eff_pass_rate, 'rawP_Peff_mismatch_rate': mismatch_rate,
            'nutrition_all_pass_rate': all_pass_rate, 'F_failure_rate': f_fail_rate,
            'protein_low_fail_rate': protein_low_rate, 'calorie_low_fail_rate': calorie_low_rate,
            'zero_candidate_rate': vs['zero_candidate_rate'],
            'final_generation_success_rate': vs['final_generation_success_rate'],
            'pingpong_rate': pingpong_rate, 'mean_rawP_pass1_to_pass2_delta': mean_pass_delta,
            'new_all_pass_to_fail_count': new_all_fail, 'new_all_pass_to_fail_rate': new_all_fail / n,
            'rice_diversity': vs['rice_diversity'], 'soup_diversity': vs['soup_diversity'],
            'main_diversity': vs['main_diversity'], 'side_diversity': vs['side_diversity'],
            'kimchi_diversity': vs['kimchi_diversity'], 'unique_meal_count': vs['unique_meal_count'],
        })
    summary_csv = os.path.join(OUT_DIR, 'phosphorus_unified_rawP_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f'② {summary_csv}')
    for r in summary_rows:
        print(f"  [{r['variant']}] rawP={r['raw_P_pass_rate']*100:.1f}% 5영양전부={r['nutrition_all_pass_rate']*100:.1f}% "
              f"F발생={r['F_failure_rate']*100:.1f}% 단백저={r['protein_low_fail_rate']*100:.1f}% "
              f"열량저={r['calorie_low_fail_rate']*100:.1f}% 핑퐁={r['pingpong_rate']*100:.1f}% "
              f"pass1to2Δ평균={r['mean_rawP_pass1_to_pass2_delta']:.2f}mg 후보0개율={r['zero_candidate_rate']*100:.1f}% "
              f"최종생성성공률={r['final_generation_success_rate']*100:.1f}% 신규전부실패={r['new_all_pass_to_fail_count']}건")

    # ── ③ paired 전환 요약(간단) ──
    trans_rows = []
    b_res = lambda c: c['results']['Baseline']
    u_res = lambda c: c['results']['Unified-rawP']
    defs = {
        'baseline_rawP_fail_to_unified_rawP_pass': lambda c: (not b_res(c)['flags']['raw_pass']) and u_res(c)['flags']['raw_pass'],
        'baseline_all_fail_to_unified_all_pass': lambda c: (not b_res(c)['flags']['all_pass']) and u_res(c)['flags']['all_pass'],
        'baseline_all_pass_to_unified_protein_low': lambda c: b_res(c)['flags']['all_pass'] and u_res(c)['flags']['protein_low'],
        'baseline_all_pass_to_unified_calorie_low': lambda c: b_res(c)['flags']['all_pass'] and u_res(c)['flags']['calorie_low'],
        'baseline_F_to_unified_all_pass': lambda c: b_res(c)['F_failure'] and u_res(c)['flags']['all_pass'],
        'baseline_F_to_unified_rawP_pass_protein_fail': lambda c: b_res(c)['F_failure'] and u_res(c)['flags']['raw_pass'] and not u_res(c)['flags']['protein_pass'],
        'both_fail': lambda c: (not b_res(c)['flags']['all_pass']) and (not u_res(c)['flags']['all_pass']),
        'pingpong_resolved': lambda c: b_res(c)['pingpong'] and not u_res(c)['pingpong'],
        'pingpong_newly_introduced': lambda c: (not b_res(c)['pingpong']) and u_res(c)['pingpong'],
    }
    for name, fn in defs.items():
        cnt = sum(1 for c in candidates if fn(c))
        trans_rows.append({'transition_type': name, 'candidate_count': cnt, 'candidate_rate': cnt / n_total})
    trans_csv = os.path.join(OUT_DIR, 'phosphorus_unified_rawP_transition.csv')
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
        w.writeheader(); w.writerows(trans_rows)
    print(f'③ {trans_csv} ({len(trans_rows)}행)')
    for r in trans_rows:
        print(f"  {r['transition_type']}: {r['candidate_count']}건({r['candidate_rate']*100:.1f}%)")

    return F, b, PMAX, candidates, summary_rows, trans_rows, OUT_DIR


if __name__ == '__main__':
    main()
