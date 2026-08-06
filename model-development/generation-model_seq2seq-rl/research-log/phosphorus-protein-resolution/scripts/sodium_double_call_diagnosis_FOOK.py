# -*- coding: utf-8 -*-
"""
sodium_double_call_diagnosis_FOOK.py — adjust() 내 lever_sodium/lever_sodium_extra 반복호출
(pre-loop 1회 + pass1 1회 + pass2 1회 = 각 3회)이 의도된 재검증인지 불필요한 중복인지 진단.
두부·콩류 앵커. C_mask100+RL(epoch260). ★ 코드 수정 없음 — 실험용 진단 스크립트에서만 검증.

호출 지점(FOOK_adjust_levers.py, adjust() 1111~1143행, 코드 그대로 복제):
  call1 = lever_sodium(pre-loop, 1119행)       call2 = lever_sodium_extra(pre-loop, 1120행)
  call3 = lever_sodium(pass1, 1135행)          call4 = lever_sodium_extra(pass1, 1138행)
  call5 = lever_sodium(pass2, 1135행 반복)     call6 = lever_sodium_extra(pass2, 1138행 반복)

Baseline : call1~6 전부(원본 그대로)
S1       : call1·2(pre-loop) 제거, call3~6 유지
S2       : call1~4 유지, call5·6(pass2) 제거
S3       : 각 호출 지점에서 함수 자신의 조기종료 조건(season_na>SALT_MG / totals(inst)['Na']>na_target)을
           호출 전에 미리 확인해 필요할 때만 호출(함수 내부에 이미 동일 조건이 있어 이론상 Baseline과
           동일할 것으로 예상 — 실측으로 확인만 함)
"""
import os, sys, io, csv, copy, time
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

OUT_DIR = os.path.join(CODE, 'sodium_double_call_diagnosis_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_MENU = '두부양념조림'
SEED_ROWS = [11, 12, 6, 36, 7]
N_CALLS = 20
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
VARIANTS = ['Baseline', 'S1', 'S2', 'S3']


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


def menu_amt_snapshot(inst):
    m = {}
    for i in inst:
        m[(i['menu'], i['ing'])] = i['amt']
    return m


def diff_menu_change(before_snap, after_inst):
    """가장 큰 amt변화를 보인 메뉴와 변화량(절대값 합) 반환."""
    per_menu_delta = {}
    for i in after_inst:
        key = (i['menu'], i['ing'])
        prev = before_snap.get(key, i['amt'])
        d = i['amt'] - prev
        per_menu_delta.setdefault(i['menu'], 0.0)
        per_menu_delta[i['menu']] += abs(d)
    if not per_menu_delta or max(per_menu_delta.values()) < 1e-9:
        return None, 0.0
    top_menu = max(per_menu_delta, key=per_menu_delta.get)
    return top_menu, per_menu_delta[top_menu]


def call_sodium(F, inst, variant, call_log, candidate_id, pass_no, call_index, b):
    """call_index: 1=preloop_sodium 2=preloop_sodium_extra 3/5=pass_sodium 4/6=pass_sodium_extra"""
    is_extra = call_index in (2, 4, 6)
    fname = 'lever_sodium_extra' if is_extra else 'lever_sodium'
    skip = False
    if variant == 'S1' and call_index in (1, 2):
        skip = True
    if variant == 'S2' and call_index in (5, 6):
        skip = True
    before_t = F.totals(inst)
    before_snap = menu_amt_snapshot(inst)
    added_salt_before = F.season_na(inst)

    if not skip:
        if variant == 'S3':
            if is_extra:
                if F.totals(inst)['Na'] > b.get('Na_total_target', F.NA_TOTAL_MEAL):
                    F.lever_sodium_extra(inst, b.get('Na_total_target', F.NA_TOTAL_MEAL))
                else:
                    skip = True   # 조건부 생략(함수 자체도 동일 조건으로 no-op 했을 것)
            else:
                if F.season_na(inst) > F.SALT_MG:
                    F.lever_sodium(inst)
                else:
                    skip = True
        else:
            if is_extra:
                F.lever_sodium_extra(inst, b.get('Na_total_target', F.NA_TOTAL_MEAL))
            else:
                F.lever_sodium(inst)

    after_t = F.totals(inst)
    added_salt_after = F.season_na(inst)
    changed_menu, amount_delta = diff_menu_change(before_snap, inst)
    changed = (changed_menu is not None)

    call_log.append({
        'candidate_id': candidate_id, 'pass_number': pass_no, 'sodium_call_index': call_index,
        'sodium_function_name': fname, 'variant': variant, 'skipped': skip,
        'Na_season_before': before_t['Na_season'], 'Na_season_after': after_t['Na_season'],
        'total_sodium_before': before_t['Na'], 'total_sodium_after': after_t['Na'],
        'added_salt_before': added_salt_before, 'added_salt_after': added_salt_after,
        'calories_before': before_t['E'], 'calories_after': after_t['E'],
        'protein_before': before_t['protein'], 'protein_after': after_t['protein'],
        'potassium_before': before_t['K'], 'potassium_after': after_t['K'],
        'phosphorus_before': before_t['P'], 'phosphorus_after': after_t['P'],
        'changed_menu': changed_menu, 'changed_ingredient': None, 'amount_delta': amount_delta,
        'swap_log': 'no_swap(amt_only)', 'function_return_value': None,
        'changed': changed,
    })


def run_variant(F, prefix_inst, anchor, b, na_target, variant, candidate_id, call_log):
    inst = copy.deepcopy(prefix_inst)
    call_sodium(F, inst, variant, call_log, candidate_id, 0, 1, b)
    call_sodium(F, inst, variant, call_log, candidate_id, 0, 2, b)
    for pass_i in range(2):
        pass_no = pass_i + 1
        F.lever_potassium(inst, b['Kmax'], anchor=anchor)
        F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor)
        sodium_idx = 3 if pass_i == 0 else 5
        extra_idx = 4 if pass_i == 0 else 6
        call_sodium(F, inst, variant, call_log, candidate_id, pass_no, sodium_idx, b)
        call_sodium(F, inst, variant, call_log, candidate_id, pass_no, extra_idx, b)
        F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                         kmax=b['Kmax'], pmax=b['Pmax'])
    return inst


def nutrient_flags(t, b):
    raw_pass = t['P'] < b['Pmax']
    protein_low = t['protein'] < b['Plo']; protein_high = t['protein'] > b['Phi']
    calorie_low = t['E'] < b['Elo']; calorie_high = t['E'] > b['Ehi']
    na_pass = t['Na_season'] <= b['Namax']; k_pass = t['K'] < b['Kmax']
    protein_pass = not protein_low and not protein_high
    calorie_pass = not calorie_low and not calorie_high
    all_pass = raw_pass and protein_pass and calorie_pass and na_pass and k_pass
    return {'raw_pass': raw_pass, 'protein_low': protein_low, 'protein_high': protein_high,
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
    na_target = b.get('Na_total_target', F.NA_TOTAL_MEAL)

    _, food_dict, diet_np, _, mask_id = build_data(with_mask=True)
    num_tokens = len(food_dict)
    orig_diet_np_np = diet_np.numpy()
    enc, dec = load_model(RL_CKPT_DIR, num_tokens)
    anchor_token = core.name2idx[ANCHOR_MENU]

    # ── 검증: Baseline 경로 vs F.adjust() 5건 ──
    print('=== 검증: Baseline 경로 vs F.adjust() 5건 ===')
    val_ok = 0
    for vi in range(5):
        np.random.seed(999 + vi)
        menus_v = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                   orig_diet_np_np[SEED_ROWS[0]], anchor_token, 4, TEMP)[0]
        if len(menus_v) != 5:
            continue
        F.ROT[0] = 0
        inst_v = F.expand(list(menus_v)); F.SWAP_LOG.clear(); F.lever_kimchi(inst_v)
        tmp_log = []
        inst_base = run_variant(F, inst_v, ANCHOR_MENU, b, na_target, 'Baseline', -1, tmp_log)
        manual_final = F.totals(inst_base)
        F.ROT[0] = 0
        _, adj_after, _, _ = F.adjust(list(menus_v), b, anchor=ANCHOR_MENU)
        ok = all(abs(manual_final[k] - adj_after[k]) < 1e-6 for k in ('E', 'protein', 'P', 'K', 'Na_season'))
        val_ok += int(ok)
        print(f'  검증{vi}: manual_Na_season={manual_final["Na_season"]:.3f} adjust_Na_season={adj_after["Na_season"]:.3f} 일치={ok}')
    print(f'검증 결과: {val_ok}/5 일치\n')
    F.ROT[0] = 0

    print(f'생성 시작: 두부콩류 {len(SEED_ROWS)}seed x {N_CALLS}call x {TRIES} = {len(SEED_ROWS)*N_CALLS*TRIES}후보(예정) x 4variant')

    candidates = []
    call_log = []
    cid = 0
    t_start = time.time()
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
                inst0 = F.expand(list(menus))
                F.SWAP_LOG.clear()
                F.lever_kimchi(inst0)   # 공유 프리픽스: expand+kimchi만(나트륨 호출은 variant별로 갈림)
                t0 = F.totals(inst0)
                results = {}
                for variant in VARIANTS:
                    inst_v = run_variant(F, inst0, ANCHOR_MENU, b, na_target, variant, cid, call_log)
                    t_v = F.totals(inst_v)
                    results[variant] = {'t': t_v, 'flags': nutrient_flags(t_v, b)}
                candidates.append({'candidate_id': cid, 'seed_id': sid, 'call_id': call_id, 'menus': menus,
                                    't0': t0, 'results': results})
        print(f'  seed {sid}(row{row_idx}) 완료, 누적 {len(candidates)}건')
    elapsed = time.time() - t_start
    print(f'\n총 생성: {len(candidates)}건, 소요시간 {elapsed:.1f}초, call_log {len(call_log)}행')

    n_total = len(candidates)

    # ── ① sodium_lever_call_trace.csv ──
    for r in call_log:
        r['sodium_pass_before'] = r['Na_season_before'] <= b['Namax']
        r['sodium_pass_after'] = r['Na_season_after'] <= b['Namax']
    trace_csv = os.path.join(OUT_DIR, 'sodium_lever_call_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(call_log[0].keys()))
        w.writeheader(); w.writerows(call_log)
    print(f'① {trace_csv} ({len(call_log)}행)')

    # ── 배치 통계 ──
    def score_of(flags):
        return sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])

    batches = {}
    for c in candidates:
        batches.setdefault((c['seed_id'], c['call_id']), []).append(c)

    variant_batch_stats = {}
    for variant in VARIANTS:
        zero_cnt = 0
        rice_c, soup_c, main_c, side_c, kim_c = Counter(), Counter(), Counter(), Counter(), Counter()
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
        variant_batch_stats[variant] = {
            'zero_candidate_rate': zero_cnt / len(batches),
            'final_generation_success_rate': 1 - zero_cnt / len(batches),
            'rice_diversity': len(rice_c), 'soup_diversity': len(soup_c), 'main_diversity': len(main_c),
            'side_diversity': len(side_c), 'kimchi_diversity': len(kim_c),
        }

    # ── ② sodium_lever_variant_summary.csv ──
    summary_rows = []
    calls_by_variant = {}
    for r in call_log:
        calls_by_variant.setdefault(r['variant'], []).append(r)
    for variant in VARIANTS:
        flags_list = [c['results'][variant]['flags'] for c in candidates]
        n = len(flags_list)
        na_pass_rate = sum(f['na_pass'] for f in flags_list) / n
        all_pass_rate = sum(f['all_pass'] for f in flags_list) / n
        vcalls = calls_by_variant[variant]
        n_calls_total = len(vcalls)
        n_executed = sum(1 for r in vcalls if not r['skipped'])
        n_changed = sum(1 for r in vcalls if not r['skipped'] and r['changed'])
        n_noop = n_executed - n_changed
        vs = variant_batch_stats[variant]
        # sodium 함수별 실행횟수
        sodium_calls = [r for r in vcalls if r['sodium_function_name'] == 'lever_sodium']
        extra_calls = [r for r in vcalls if r['sodium_function_name'] == 'lever_sodium_extra']
        summary_rows.append({
            'variant': variant, 'candidate_count': n,
            'sodium_pass_rate': na_pass_rate, 'nutrition_all_pass_rate': all_pass_rate,
            'zero_candidate_rate': vs['zero_candidate_rate'],
            'final_generation_success_rate': vs['final_generation_success_rate'],
            'lever_sodium_call_count': sum(1 for r in sodium_calls if not r['skipped']),
            'lever_sodium_extra_call_count': sum(1 for r in extra_calls if not r['skipped']),
            'total_calls_executed': n_executed, 'total_calls_skipped': n_calls_total - n_executed,
            'calls_that_changed_state': n_changed, 'calls_that_were_noop': n_noop,
            'noop_rate_among_executed': n_noop / n_executed if n_executed else None,
            'rice_diversity': vs['rice_diversity'], 'soup_diversity': vs['soup_diversity'],
            'main_diversity': vs['main_diversity'], 'side_diversity': vs['side_diversity'],
            'kimchi_diversity': vs['kimchi_diversity'],
        })
    summary_csv = os.path.join(OUT_DIR, 'sodium_lever_variant_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f'② {summary_csv}')
    for r in summary_rows:
        print(f"  [{r['variant']}] 나트륨통과={r['sodium_pass_rate']*100:.1f}% 5영양전부={r['nutrition_all_pass_rate']*100:.1f}% "
              f"후보0개율={r['zero_candidate_rate']*100:.1f}% 실행={r['total_calls_executed']} "
              f"변화없음비율={r['noop_rate_among_executed']*100:.1f}%")

    # ── ③ sodium_lever_transition.csv ──
    b_res = lambda c: c['results']['Baseline']
    trans_defs = {
        'call2_rescues_sodium(pass_after_call2_only)': None,  # 아래 개별 계산
    }
    trans_rows = []
    for variant in ['S1', 'S2', 'S3']:
        v_res = lambda c, variant=variant: c['results'][variant]
        defs = {
            'baseline_sodium_pass_to_v_sodium_fail': lambda c: b_res(c)['flags']['na_pass'] and not v_res(c)['flags']['na_pass'],
            'baseline_sodium_fail_to_v_sodium_pass': lambda c: (not b_res(c)['flags']['na_pass']) and v_res(c)['flags']['na_pass'],
            'baseline_all_pass_to_v_all_fail': lambda c: b_res(c)['flags']['all_pass'] and not v_res(c)['flags']['all_pass'],
            'baseline_all_pass_to_v_protein_fail': lambda c: b_res(c)['flags']['all_pass'] and not v_res(c)['flags']['protein_pass'],
            'baseline_all_pass_to_v_calorie_fail': lambda c: b_res(c)['flags']['all_pass'] and not v_res(c)['flags']['calorie_pass'],
            'baseline_all_pass_to_v_potassium_fail': lambda c: b_res(c)['flags']['all_pass'] and not v_res(c)['flags']['k_pass'],
            'baseline_all_pass_to_v_phosphorus_fail': lambda c: b_res(c)['flags']['all_pass'] and not v_res(c)['flags']['raw_pass'],
            'identical_final_state': lambda c: (abs(b_res(c)['t']['Na_season'] - v_res(c)['t']['Na_season']) < 1e-6 and
                                                 abs(b_res(c)['t']['E'] - v_res(c)['t']['E']) < 1e-6 and
                                                 abs(b_res(c)['t']['protein'] - v_res(c)['t']['protein']) < 1e-6),
        }
        for name, fn in defs.items():
            cnt = sum(1 for c in candidates if fn(c))
            trans_rows.append({'variant': variant, 'transition_type': name, 'candidate_count': cnt,
                                'candidate_rate': cnt / n_total})
    trans_csv = os.path.join(OUT_DIR, 'sodium_lever_transition.csv')
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
        w.writeheader(); w.writerows(trans_rows)
    print(f'③ {trans_csv} ({len(trans_rows)}행)')
    for r in trans_rows:
        print(f"  [{r['variant']}] {r['transition_type']}: {r['candidate_count']}건({r['candidate_rate']*100:.2f}%)")

    return F, b, candidates, call_log, summary_rows, trans_rows, elapsed, OUT_DIR


if __name__ == '__main__':
    main()
