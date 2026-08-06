# -*- coding: utf-8 -*-
"""
phosphorus_lever_step_diagnosis_FOOK.py — F_레버후신규실패 유형을 패스별·레버별 스냅샷으로 검증.
두부·콩류 앵커. C_mask100+RL(epoch260) 체크포인트, 실제 make_meal()/adjust() 로직을 그대로
재현(순서·기준·레버·대체재풀 전부 불변경). ★ 코드 수정 없음 — app_core_FOOK.py/
FOOK_adjust_levers.py는 import해서 개별 레버 함수만 순서대로 호출(읽기 전용, 진단 래퍼).

adjust()의 실제 내부 순서(FOOK_adjust_levers.py:1111-1143 코드 확인, 그대로 복제):
  expand() → lever_kimchi → lever_sodium → lever_sodium_extra   (패스 루프 진입 전, 1회)
  for pass in [1,2]:
      lever_potassium → lever_phosphorus → lever_protein → lever_sodium
      → lever_sodium_extra → lever_calorie(allow_snack= pass==2에서만 True)
  반환 = totals(최종 inst)

스냅샷 시점(사용자 요청 T0~T13은 pass별 6개 레버만 가정했으나, 실제 코드엔 패스진입 전
3개 레버(kimchi/sodium/sodium_extra)가 더 있어 그대로 포함함 — "실제 호출 순서를 그대로
기록"이라는 지시에 따름):
  T0(expand직후) → T1(kimchi) → T2(sodium,pre) → T3(sodium_extra,pre)
  → [pass1] T4(potassium) T5(phosphorus) T6(protein) T7(sodium) T8(sodium_extra) T9(calorie)
  → [pass2] T10(potassium) T11(phosphorus) T12(protein) T13(sodium) T14(sodium_extra) T15(calorie,최종)

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  python phosphorus_lever_step_diagnosis_FOOK.py
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

OUT_DIR = os.path.join(CODE, 'phosphorus_lever_step_diagnosis_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_MENU = '두부양념조림'
SEED_ROWS = [11, 12, 6, 36, 7]
N_CALLS = 6         # 소규모(5seed x 6call x 24 = 720후보) — F/pass/G 각 최소치 확보용
TRIES = 24
TEMP = 0.8
RNG_SEED = 11

SAMPLE_TARGET = {'F': 50, 'pass': 20, 'G': 20}

STEP_SEQUENCE = [
    ('T0', 'expand', None),
    ('T1', 'kimchi', None),
    ('T2', 'sodium', 'pre'),
    ('T3', 'sodium_extra', 'pre'),
    ('T4', 'potassium', 1), ('T5', 'phosphorus', 1), ('T6', 'protein', 1),
    ('T7', 'sodium', 1), ('T8', 'sodium_extra', 1), ('T9', 'calorie', 1),
    ('T10', 'potassium', 2), ('T11', 'phosphorus', 2), ('T12', 'protein', 2),
    ('T13', 'sodium', 2), ('T14', 'sodium_extra', 2), ('T15', 'calorie', 2),
]


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
    """side_dish_diagnosis_FOOK.py의 gen_batch_traced()와 동일 로직(복제, 무수정)."""
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


def slot_breakdown(inst, slot_menu, lowna_pool, snack_names):
    """side_dish_slot_phosphorus_diagnosis_FOOK.py와 동일 알고리즘(검증됨, 불일치 0건)."""
    name_to_slot = {v: k for k, v in slot_menu.items()}
    slot_p = {s: 0.0 for s in range(5)}
    slot_amt = {s: 0.0 for s in range(5)}
    slot_protein = {s: 0.0 for s in range(5)}
    snack_p = 0.0
    unknown_p = 0.0
    for i in inst:
        p_contrib = (i['amt'] / 100 * i['P']) if i['P'] is not None else 0.0
        pr_contrib = (i['amt'] / 100 * i['protein']) if i['protein'] is not None else 0.0
        identity = i.get('orig_menu', i['menu'])
        s = name_to_slot.get(identity)
        if s is not None:
            slot_p[s] += p_contrib; slot_amt[s] += i['amt']; slot_protein[s] += pr_contrib
            continue
        if i['menu'] in lowna_pool:
            slot_p[4] += p_contrib; slot_amt[4] += i['amt']; slot_protein[4] += pr_contrib
            continue
        if i['menu'] in snack_names:
            snack_p += p_contrib
            continue
        unknown_p += p_contrib
    return slot_p, slot_amt, slot_protein, snack_p, unknown_p


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    os.chdir(cwd)
    b = F.meal_bounds(60)
    PMAX = b['Pmax']

    F._build_snack_pool()
    snack_names = {m for m, *_ in F.SNACK_POOL}
    lowna_pool = set(F.LOWNA_POOL)
    na_target = b.get('Na_total_target', F.NA_TOTAL_MEAL)

    # ── 검증: 수기 step-by-step 복제가 F.adjust()와 동일한 최종상태를 내는지 5건 샌드박스 확인 ──
    _, food_dict0, diet_np0, _, mask_id0 = build_data(with_mask=True)
    num_tokens0 = len(food_dict0)
    orig_diet_np0 = diet_np0.numpy()
    enc0, dec0 = load_model(RL_CKPT_DIR, num_tokens0)
    anchor_token0 = core.name2idx[ANCHOR_MENU]

    def run_manual_steps(menus, anchor):
        """expand()→kimchi→sodium→sodium_extra→[potassium→phosphorus→protein→sodium→
        sodium_extra→calorie]x2. 각 스텝 뒤 (step_key, inst_snapshot_totals) 기록."""
        inst = F.expand(list(menus))
        F.SWAP_LOG.clear()
        records = []

        def snap(step_key, lever, pass_no):
            t = F.totals(inst)
            records.append((step_key, lever, pass_no, t, copy.deepcopy(inst)))

        snap('T0', 'expand', None)
        F.lever_kimchi(inst); snap('T1', 'kimchi', None)
        F.lever_sodium(inst); snap('T2', 'sodium', 'pre')
        F.lever_sodium_extra(inst, na_target); snap('T3', 'sodium_extra', 'pre')
        step_no = 4
        for pass_i in range(2):
            F.lever_potassium(inst, b['Kmax'], anchor=anchor); snap(f'T{step_no}', 'potassium', pass_i + 1); step_no += 1
            F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo']); snap(f'T{step_no}', 'phosphorus', pass_i + 1); step_no += 1
            F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor); snap(f'T{step_no}', 'protein', pass_i + 1); step_no += 1
            F.lever_sodium(inst); snap(f'T{step_no}', 'sodium', pass_i + 1); step_no += 1
            F.lever_sodium_extra(inst, na_target); snap(f'T{step_no}', 'sodium_extra', pass_i + 1); step_no += 1
            F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                             kmax=b['Kmax'], pmax=b['Pmax']); snap(f'T{step_no}', 'calorie', pass_i + 1); step_no += 1
        return records

    print('=== 검증: 수기 step 복제 vs F.adjust() 최종상태 일치 확인 (5건) ===')
    print('    (ROT[0]는 lever_kimchi가 쓰는 전역 회전카운터 — 두 방식을 같은 후보에 번갈아 돌리면')
    print('     이 카운터가 두 번 진행되어 김치 대체재가 달라지므로, 검증 목적으로만 호출 전 0으로')
    print('     맞춰 격리함. 본 실행(F/pass/G 720건)은 run_manual_steps만 단일 호출하므로 이 문제 없음.)')
    val_ok = 0
    for vi in range(5):
        np.random.seed(999 + vi)
        menus_v = gen_batch_slots(core, enc0, dec0, num_tokens0, mask_id0, food_dict0,
                                   orig_diet_np0[SEED_ROWS[0]], anchor_token0, 4, TEMP)[0]
        if len(menus_v) != 5:
            continue
        F.ROT[0] = 0
        recs = run_manual_steps(menus_v, ANCHOR_MENU)
        manual_final = recs[-1][3]
        F.ROT[0] = 0
        adj_before, adj_after, adj_inst, _ = F.adjust(list(menus_v), b, anchor=ANCHOR_MENU)
        ok = all(abs(manual_final[k] - adj_after[k]) < 1e-6 for k in ('E', 'protein', 'P', 'K', 'Na_season'))
        val_ok += int(ok)
        print(f'  검증{vi}: manual_P={manual_final["P"]:.3f} adjust_P={adj_after["P"]:.3f} 일치={ok}')
    print(f'검증 결과: {val_ok}/5 일치\n')
    F.ROT[0] = 0   # 본 실행 진입 전 리셋(검증 루프가 건드린 값 정리, 본 실행은 단일 호출이라 값 자체는 무관하나 명확성 위해)

    # ── 본 실행: 두부콩류 후보 생성 + 수기 step trace ──
    all_candidates = []   # (candidate_id, seed_id, call_id, menus, records)
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
                records = run_manual_steps(menus, ANCHOR_MENU)
                all_candidates.append((cid, sid, row_idx, call_id, menus, records))
    print(f'생성 완료: {len(all_candidates)}건')

    # ── 분류: F / pass / G ──
    classified = []
    for cid_, sid, row_idx, call_id, menus, records in all_candidates:
        t0_P = records[0][3]['P']
        tfinal_P = records[-1][3]['P']
        before_pass = t0_P < PMAX
        after_pass = tfinal_P < PMAX
        if before_pass and after_pass:
            grp = 'pass'
        elif before_pass and not after_pass:
            grp = 'F'
        elif (not before_pass) and (not after_pass):
            grp = 'G'
        else:
            grp = 'recovered'   # before_pass=False, after_pass=True — 요청범위 밖(정상 레버 성공), 별도 표기만
        classified.append({'candidate_id': cid_, 'seed_id': sid, 'seed_row_idx': row_idx, 'call_id': call_id,
                            'menus': menus, 'records': records, 'group': grp})

    counts = {}
    for c in classified:
        counts[c['group']] = counts.get(c['group'], 0) + 1
    print('그룹별 건수:', counts)

    F_list = [c for c in classified if c['group'] == 'F']
    pass_list = [c for c in classified if c['group'] == 'pass']
    G_list = [c for c in classified if c['group'] == 'G']
    print(f"F={len(F_list)}  pass={len(pass_list)}  G={len(G_list)}  (요청 최소: F>=30(가능30~50), pass>=20, G>=20)")

    slot_menus = {c['candidate_id']: {k: c['menus'][k] for k in range(5)} for c in classified}

    def row_for_step(cand, step_idx):
        step_key, lever, pass_no, t, inst_snap = cand['records'][step_idx]
        slot_menu = slot_menus[cand['candidate_id']]
        slot_p, slot_amt, slot_protein, snack_p, _ = slot_breakdown(inst_snap, slot_menu, lowna_pool, snack_names)
        raw_pass = t['P'] < PMAX
        eff_pass = t['Peff'] < PMAX
        protein_pass = b['Plo'] <= t['protein'] <= b['Phi']
        calorie_pass = b['Elo'] <= t['E'] <= b['Ehi']
        na_pass = t['Na_season'] <= b['Namax']
        k_pass = t['K'] < b['Kmax']
        nutrition_all_pass = raw_pass and protein_pass and calorie_pass and na_pass and k_pass
        if step_idx > 0:
            prev_t = cand['records'][step_idx - 1][3]
            dP = t['P'] - prev_t['P']; dPr = t['protein'] - prev_t['protein']; dE = t['E'] - prev_t['E']
        else:
            dP = dPr = dE = 0.0
        return {
            'candidate_id': cand['candidate_id'], 'failure_group': cand['group'], 'step': step_key,
            'pass_number': pass_no, 'lever_name': lever,
            'calories': t['E'], 'protein': t['protein'], 'sodium': t['Na_season'], 'potassium': t['K'],
            'phosphorus_raw_P': t['P'], 'phosphorus_effective_Peff': t['Peff'], 'phosphorus_limit': PMAX,
            'phosphorus_raw_pass': raw_pass, 'phosphorus_effective_pass': eff_pass,
            'protein_pass': protein_pass, 'calorie_pass': calorie_pass, 'nutrition_all_pass': nutrition_all_pass,
            'main_dish_name': slot_menu[2], 'main_dish_amount': slot_amt[2],
            'main_dish_phosphorus': slot_p[2], 'main_dish_protein': slot_protein[2],
            'soup_phosphorus': slot_p[1], 'side_phosphorus': slot_p[3], 'rice_phosphorus': slot_p[0],
            'kimchi_phosphorus': slot_p[4], 'snack_phosphorus': snack_p,
            'phosphorus_delta_from_previous': dP, 'protein_delta_from_previous': dPr,
            'calorie_delta_from_previous': dE,
        }

    # ── ① phosphorus_lever_step_trace.csv (표본만: F<=50, pass<=20, G<=20) ──
    sample_F = F_list[:SAMPLE_TARGET['F']]
    sample_pass = pass_list[:SAMPLE_TARGET['pass']]
    sample_G = G_list[:SAMPLE_TARGET['G']]
    trace_rows = []
    for cand in sample_F + sample_pass + sample_G:
        for si in range(len(cand['records'])):
            trace_rows.append(row_for_step(cand, si))
    trace_csv = os.path.join(OUT_DIR, 'phosphorus_lever_step_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    print(f'① {trace_csv} ({len(trace_rows)}행, F{len(sample_F)}/pass{len(sample_pass)}/G{len(sample_G)}건)')

    # ── 이벤트 탐지 (F전량 대상) ──
    def detect_events(cand):
        recs = cand['records']
        raw_pass_seq = [r[3]['P'] < PMAX for r in recs]
        true_idx = [i for i, v in enumerate(raw_pass_seq) if v]
        L = true_idx[-1]  # 마지막으로 raw P 통과했던 인덱스
        culprit_idx = L + 1
        step_key, lever, pass_no, t, _ = recs[culprit_idx]
        prev_t = recs[L][3]
        P_inc = t['P'] - prev_t['P']; protein_inc = t['protein'] - prev_t['protein']; E_inc = t['E'] - prev_t['E']
        later_phos_idx = [i for i in range(culprit_idx + 1, len(recs)) if recs[i][1] == 'phosphorus']
        later_phos_exists = len(later_phos_idx) > 0
        later_phos_success = any(recs[i][3]['P'] < PMAX for i in later_phos_idx) if later_phos_exists else False
        no_phos_recheck_after = not any(recs[i][1] == 'phosphorus' for i in range(culprit_idx + 1, len(recs)))
        # 패턴 C: pass1 종료(T9)가 실패, pass2 phosphorus(T11)에서 복구, 이후 pass2에서 재실패
        pass1_end_fail = not raw_pass_seq[9]     # T9 인덱스=9
        pass2_phos_recovered = raw_pass_seq[11]  # T11 인덱스=11
        is_C = pass1_end_fail and pass2_phos_recovered and pass_no == 2 and step_key not in ('T10', 'T11')
        if is_C:
            pattern = 'C_pass1실패_pass2phos복구_pass2재실패'
        elif lever == 'protein':
            pattern = 'A_phosphorus직후통과_protein직후실패'
        elif lever == 'calorie':
            pattern = 'B_phosphorus직후통과_calorie직후실패'
        else:
            eff_final = recs[-1][3]['Peff'] < PMAX
            raw_final = recs[-1][3]['P'] < PMAX
            if eff_final and not raw_final:
                pattern = 'D_Peff통과_rawP계속실패'
            else:
                pattern = f'E_기타({lever})'
        main_delta = row_for_step(cand, culprit_idx)['main_dish_phosphorus'] - row_for_step(cand, L)['main_dish_phosphorus']
        main_is_dominant = (P_inc > 0) and (main_delta / P_inc >= 0.5)
        return {
            'candidate_id': cand['candidate_id'], 'last_pass_step': step_key_of(recs, L),
            'culprit_step': step_key, 'culprit_lever': lever, 'culprit_pass_number': pass_no,
            'phosphorus_increase': P_inc, 'protein_increase': protein_inc, 'calorie_increase': E_inc,
            'later_phosphorus_lever_ran': later_phos_exists, 'later_phosphorus_succeeded': later_phos_success,
            'no_phosphorus_recheck_after_culprit': no_phos_recheck_after,
            'raw_P_pass_at_final_return': recs[-1][3]['P'] < PMAX,
            'peff_pass_at_final_return': recs[-1][3]['Peff'] < PMAX,
            'raw_vs_peff_mismatch_at_final': (recs[-1][3]['P'] < PMAX) != (recs[-1][3]['Peff'] < PMAX),
            'main_dish_phosphorus_delta_at_culprit': main_delta,
            'main_dish_dominant_cause': main_is_dominant,
            'pattern_type': pattern,
        }

    def step_key_of(recs, idx):
        return recs[idx][0]

    transition_rows = [detect_events(c) for c in F_list]
    trans_csv = os.path.join(OUT_DIR, 'phosphorus_failure_transition.csv')
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(transition_rows[0].keys()))
        w.writeheader(); w.writerows(transition_rows)
    print(f'② {trans_csv} ({len(transition_rows)}행, F 전량)')

    # ── ③ phosphorus_failure_cause_summary.csv (핵심판정 수치) ──
    n_F = len(F_list)
    n_protein_culprit = sum(1 for r in transition_rows if r['culprit_lever'] == 'protein')
    n_calorie_culprit = sum(1 for r in transition_rows if r['culprit_lever'] == 'calorie')
    n_both = sum(1 for r in transition_rows if r['culprit_lever'] in ('protein', 'calorie'))
    n_p2_phos_then_reup = sum(1 for c in F_list
                               if c['records'][11][3]['P'] < PMAX and c['records'][15][3]['P'] >= PMAX)
    n_mismatch = sum(1 for r in transition_rows if r['raw_vs_peff_mismatch_at_final'])
    n_main_dominant = sum(1 for r in transition_rows if r['main_dish_dominant_cause'])
    n_no_recheck = sum(1 for r in transition_rows if r['no_phosphorus_recheck_after_culprit'])
    pattern_counts = {}
    for r in transition_rows:
        key = r['pattern_type'].split('_')[0]
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    summary_rows = [
        {'metric': 'F유형 후보 수', 'value': n_F, 'ratio': None},
        {'metric': 'lever_protein 직후 신규 P실패', 'value': n_protein_culprit, 'ratio': n_protein_culprit / n_F},
        {'metric': 'lever_calorie 직후 신규 P실패', 'value': n_calorie_culprit, 'ratio': n_calorie_culprit / n_F},
        {'metric': 'protein+calorie 합산 설명', 'value': n_both, 'ratio': n_both / n_F},
        {'metric': 'pass2 phosphorus 직후 통과했다가 최종 재실패', 'value': n_p2_phos_then_reup, 'ratio': n_p2_phos_then_reup / n_F},
        {'metric': 'raw P vs Peff 최종판정 불일치', 'value': n_mismatch, 'ratio': n_mismatch / n_F},
        {'metric': '두부양념조림(주찬) 스케일업이 지배적 원인(delta 50%+)', 'value': n_main_dominant, 'ratio': n_main_dominant / n_F},
        {'metric': 'culprit 이후 인 재검증 기회 없음(구조적)', 'value': n_no_recheck, 'ratio': n_no_recheck / n_F},
    ]
    for k, v in sorted(pattern_counts.items()):
        summary_rows.append({'metric': f'패턴{k}', 'value': v, 'ratio': v / n_F})

    cause_csv = os.path.join(OUT_DIR, 'phosphorus_failure_cause_summary.csv')
    with open(cause_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['metric', 'value', 'ratio'])
        w.writeheader(); w.writerows(summary_rows)
    print(f'③ {cause_csv}')
    for r in summary_rows:
        ratio_s = f"{r['ratio']*100:.1f}%" if r['ratio'] is not None else '-'
        print(f"  {r['metric']}: {r['value']}건 ({ratio_s})")

    return {
        'b': b, 'PMAX': PMAX, 'F_list': F_list, 'pass_list': pass_list, 'G_list': G_list,
        'transition_rows': transition_rows, 'summary_rows': summary_rows, 'OUT_DIR': OUT_DIR,
        'val_ok': val_ok,
    }


if __name__ == '__main__':
    main()
