# -*- coding: utf-8 -*-
"""
lever_system_audit_FOOK.py — adjust() 전체 레버 구조를 audit(진단만, 수정 없음).
두부·콩류 앵커 2,400건. 매 레버 호출 직후 스냅샷을 남겨 noop율·목표영양소 구제율·
레버 간 핑퐁·passes() 기준 불일치를 계측한다. ★ 코드 수정 없음.

스냅샷 시점(16개, adjust() 1111~1143행 실제 순서 그대로):
  S0 expand  S1 kimchi  S2 sodium(pre)  S3 sodium_extra(pre)
  [pass1] S4 potassium S5 phosphorus S6 protein S7 sodium S8 sodium_extra S9 calorie
  [pass2] S10 potassium S11 phosphorus S12 protein S13 sodium S14 sodium_extra S15 calorie(최종)
"""
import os, sys, io, csv, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf

FINAL = r'E:\final'
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, FINAL)

from Model import Encoder, Decoder
from train_FOOK_soupmask_1000 import build_data, SOUP_POS

OUT_DIR = os.path.join(CODE, 'lever_system_audit_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_MENU = '두부양념조림'
SEED_ROWS = [11, 12, 6, 36, 7]
N_CALLS = 20
TRIES = 24
TEMP = 0.8
RNG_SEED = 11

STEPS = [
    ('S0', 'expand', None), ('S1', 'kimchi', None), ('S2', 'sodium', 'pre'), ('S3', 'sodium_extra', 'pre'),
    ('S4', 'potassium', 1), ('S5', 'phosphorus', 1), ('S6', 'protein', 1),
    ('S7', 'sodium', 1), ('S8', 'sodium_extra', 1), ('S9', 'calorie', 1),
    ('S10', 'potassium', 2), ('S11', 'phosphorus', 2), ('S12', 'protein', 2),
    ('S13', 'sodium', 2), ('S14', 'sodium_extra', 2), ('S15', 'calorie', 2),
]
IDX = {name: i for i, (name, *_ ) in enumerate(STEPS)}


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
    return {(i['menu'], i['ing']): i['amt'] for i in inst}


def diff_menu_change(before_snap, after_inst):
    per_menu = {}
    for i in after_inst:
        key = (i['menu'], i['ing'])
        prev = before_snap.get(key, i['amt'])
        per_menu.setdefault(i['menu'], 0.0)
        per_menu[i['menu']] += abs(i['amt'] - prev)
    if not per_menu or max(per_menu.values()) < 1e-9:
        return None, 0.0
    top = max(per_menu, key=per_menu.get)
    return top, per_menu[top]


def nut_flags(t, b):
    raw_pass = t['P'] < b['Pmax']
    protein_pass = b['Plo'] <= t['protein'] <= b['Phi']
    calorie_pass = b['Elo'] <= t['E'] <= b['Ehi']
    na_pass = t['Na_season'] <= b['Namax']
    k_pass = t['K'] < b['Kmax']
    return {'phosphorus': raw_pass, 'protein': protein_pass, 'calorie': calorie_pass,
            'sodium': na_pass, 'potassium': k_pass,
            'all_pass': raw_pass and protein_pass and calorie_pass and na_pass and k_pass}


def run_full_steps(F, menus, anchor, b, na_target):
    inst = F.expand(list(menus))
    F.SWAP_LOG.clear()
    records = []

    def snap(step_name, lever, pass_no, before_snap):
        t = F.totals(inst)
        changed_menu, changed_amt = diff_menu_change(before_snap, inst)
        records.append({'step': step_name, 'lever': lever, 'pass_no': pass_no, 't': t,
                         'changed_menu': changed_menu, 'changed_amount': changed_amt,
                         'flags': nut_flags(t, b)})

    before = menu_amt_snapshot(inst)
    snap('S0', 'expand', None, before)

    before = menu_amt_snapshot(inst); F.lever_kimchi(inst); snap('S1', 'kimchi', None, before)
    before = menu_amt_snapshot(inst); F.lever_sodium(inst); snap('S2', 'sodium', 'pre', before)
    before = menu_amt_snapshot(inst); F.lever_sodium_extra(inst, na_target); snap('S3', 'sodium_extra', 'pre', before)

    step_no = 4
    for pass_i in range(2):
        pass_no = pass_i + 1
        before = menu_amt_snapshot(inst); F.lever_potassium(inst, b['Kmax'], anchor=anchor)
        snap(f'S{step_no}', 'potassium', pass_no, before); step_no += 1
        before = menu_amt_snapshot(inst); F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        snap(f'S{step_no}', 'phosphorus', pass_no, before); step_no += 1
        before = menu_amt_snapshot(inst); F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor)
        snap(f'S{step_no}', 'protein', pass_no, before); step_no += 1
        before = menu_amt_snapshot(inst); F.lever_sodium(inst)
        snap(f'S{step_no}', 'sodium', pass_no, before); step_no += 1
        before = menu_amt_snapshot(inst); F.lever_sodium_extra(inst, na_target)
        snap(f'S{step_no}', 'sodium_extra', pass_no, before); step_no += 1
        before = menu_amt_snapshot(inst)
        F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                         kmax=b['Kmax'], pmax=b['Pmax'])
        snap(f'S{step_no}', 'calorie', pass_no, before); step_no += 1
    return records


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

    # 검증
    print('=== 검증: 수기 step 복제 vs F.adjust() 5건 ===')
    val_ok = 0
    for vi in range(5):
        np.random.seed(999 + vi)
        menus_v = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                   orig_diet_np_np[SEED_ROWS[0]], anchor_token, 4, TEMP)[0]
        if len(menus_v) != 5:
            continue
        F.ROT[0] = 0
        recs = run_full_steps(F, menus_v, ANCHOR_MENU, b, na_target)
        manual_final = recs[-1]['t']
        F.ROT[0] = 0
        _, adj_after, _, _ = F.adjust(list(menus_v), b, anchor=ANCHOR_MENU)
        ok = all(abs(manual_final[k] - adj_after[k]) < 1e-6 for k in ('E', 'protein', 'P', 'K', 'Na_season'))
        val_ok += int(ok)
        print(f'  검증{vi}: 일치={ok}')
    print(f'검증 결과: {val_ok}/5 일치\n')
    F.ROT[0] = 0

    print(f'생성 시작: 두부콩류 {len(SEED_ROWS)}seed x {N_CALLS}call x {TRIES} = {len(SEED_ROWS)*N_CALLS*TRIES}후보(예정)')
    all_candidates = []
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
                records = run_full_steps(F, menus, ANCHOR_MENU, b, na_target)
                all_candidates.append({'candidate_id': cid, 'seed_id': sid, 'call_id': call_id,
                                        'menus': menus, 'records': records})
        print(f'  seed {sid}(row{row_idx}) 완료, 누적 {len(all_candidates)}건')
    print(f'\n총 생성: {len(all_candidates)}건')

    # ── ① lever_interaction_step_trace.csv ──
    trace_rows = []
    for c in all_candidates:
        recs = c['records']
        for i, r in enumerate(recs):
            t = r['t']
            if i == 0:
                d = {k: 0.0 for k in ('E', 'protein', 'K', 'P', 'Peff', 'Na', 'Na_season')}
                prev_flags = r['flags']
            else:
                prev_t = recs[i - 1]['t']; prev_flags = recs[i - 1]['flags']
                d = {k: t[k] - prev_t[k] for k in ('E', 'protein', 'K', 'P', 'Peff', 'Na', 'Na_season')}
            newly_fixed = [n for n in ('phosphorus', 'protein', 'calorie', 'sodium', 'potassium')
                           if (not prev_flags[n]) and r['flags'][n]]
            newly_broken = [n for n in ('phosphorus', 'protein', 'calorie', 'sodium', 'potassium')
                            if prev_flags[n] and (not r['flags'][n])]
            noop = (r['changed_menu'] is None)
            trace_rows.append({
                'candidate_id': c['candidate_id'], 'anchor_type': '두부콩류', 'pass_number': r['pass_no'],
                'step_order': r['step'], 'lever_name': r['lever'],
                'calories': t['E'], 'protein': t['protein'], 'potassium': t['K'],
                'phosphorus_raw': t['P'], 'phosphorus_effective': t['Peff'],
                'sodium_total': t['Na'], 'sodium_season': t['Na_season'],
                'nutrition_all_pass': r['flags']['all_pass'],
                'changed_state': not noop, 'changed_menu': r['changed_menu'], 'changed_amount': r['changed_amount'],
                'delta_calorie': d['E'], 'delta_protein': d['protein'], 'delta_potassium': d['K'],
                'delta_phosphorus_raw': d['P'], 'delta_phosphorus_effective': d['Peff'], 'delta_sodium': d['Na_season'],
                'noop': noop, 'newly_fixed_nutrient': '+'.join(newly_fixed) or None,
                'newly_broken_nutrient': '+'.join(newly_broken) or None,
            })
    trace_csv = os.path.join(OUT_DIR, 'lever_interaction_step_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    print(f'① {trace_csv} ({len(trace_rows)}행)')

    return F, b, all_candidates, trace_rows, OUT_DIR


if __name__ == '__main__':
    main()
