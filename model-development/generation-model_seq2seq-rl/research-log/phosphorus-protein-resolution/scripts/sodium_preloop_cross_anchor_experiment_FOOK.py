# -*- coding: utf-8 -*-
"""
sodium_preloop_cross_anchor_experiment_FOOK.py — S1(pre-loop lever_sodium/lever_sodium_extra
제거)이 두부콩류 외 생선구이·육류 앵커에서도 재현되는지 paired 검증. ★ 코드 수정 없음.

Baseline: adjust() 그대로(pre-loop 나트륨 호출 포함)
S1      : pre-loop lever_sodium(1119행)·lever_sodium_extra(1120행)만 제거, pass1/pass2 내부
          나트륨 호출(1135/1138행, 루프 2회 실행)은 전부 유지
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

OUT_DIR = os.path.join(CODE, 'sodium_preloop_cross_anchor_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_CONFIGS = [
    ('두부콩류', '두부양념조림', [11, 12, 6, 36, 7], 20),
    ('생선구이', '고등어구이', [11, 12, 6, 36, 7], 10),
    ('육류', '제육불고기', [11, 12, 6, 36, 7], 10),
]
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
VARIANTS = ['Baseline', 'S1']


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


def run_variant(F, prefix_inst, anchor, b, na_target, variant):
    inst = copy.deepcopy(prefix_inst)
    t_start = time.perf_counter()
    if variant == 'Baseline':
        F.lever_sodium(inst)
        F.lever_sodium_extra(inst, na_target)
    # S1: pre-loop 두 호출 생략
    for pass_i in range(2):
        F.lever_potassium(inst, b['Kmax'], anchor=anchor)
        F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor)
        F.lever_sodium(inst)
        F.lever_sodium_extra(inst, na_target)
        F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                         kmax=b['Kmax'], pmax=b['Pmax'])
    elapsed = time.perf_counter() - t_start
    return inst, elapsed


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

    all_candidates = []
    trace_rows = []
    cid_global = 0
    for anchor_name, anchor_menu, seed_rows, n_calls in ANCHOR_CONFIGS:
        anchor_token = core.name2idx[anchor_menu]
        cand_this_anchor = []
        for sid, row_idx in enumerate(seed_rows):
            base_row = orig_diet_np_np[row_idx].copy()
            np.random.seed(RNG_SEED)
            for call_id in range(n_calls):
                menus_list = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                              base_row, anchor_token, TRIES, TEMP)
                for menus in menus_list:
                    if len(menus) != 5:
                        continue
                    cid_global += 1
                    inst0 = F.expand(list(menus))
                    F.SWAP_LOG.clear()
                    F.lever_kimchi(inst0)
                    results = {}
                    for variant in VARIANTS:
                        inst_v, elapsed = run_variant(F, inst0, anchor_menu, b, na_target, variant)
                        t_v = F.totals(inst_v)
                        results[variant] = {'t': t_v, 'flags': nutrient_flags(t_v, b), 'elapsed': elapsed}
                        trace_rows.append({
                            'candidate_id': cid_global, 'anchor_type': anchor_name, 'seed_id': sid,
                            'call_id': call_id, 'variant': variant,
                            'calories': t_v['E'], 'protein': t_v['protein'], 'potassium': t_v['K'],
                            'phosphorus_raw': t_v['P'], 'phosphorus_effective': t_v['Peff'],
                            'sodium_total': t_v['Na'], 'sodium_season': t_v['Na_season'],
                            'nutrition_all_pass': results[variant]['flags']['all_pass'],
                            'elapsed_sec': elapsed,
                        })
                    c = {'candidate_id': cid_global, 'anchor_type': anchor_name, 'seed_id': sid,
                         'call_id': call_id, 'menus': menus, 'results': results}
                    cand_this_anchor.append(c)
                    all_candidates.append(c)
        print(f'[{anchor_name}] 완료: {len(cand_this_anchor)}건')

    n_total = len(all_candidates)
    print(f'\n총 생성: {n_total}건 (앵커 3종 합계)')

    trace_csv = os.path.join(OUT_DIR, 'sodium_preloop_cross_anchor_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    print(f'저장: {trace_csv} ({len(trace_rows)}행)')

    # ── 배치(seed,call) 단위 선택 시뮬레이션(앵커별) ──
    def score_of(flags):
        return sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])

    def batches_for(anchor_name):
        d = {}
        for c in all_candidates:
            if c['anchor_type'] != anchor_name:
                continue
            d.setdefault((c['seed_id'], c['call_id']), []).append(c)
        return d

    summary_rows = []
    for anchor_name, anchor_menu, seed_rows, n_calls in ANCHOR_CONFIGS:
        anchor_cands = [c for c in all_candidates if c['anchor_type'] == anchor_name]
        batches = batches_for(anchor_name)
        for variant in VARIANTS:
            flags_list = [c['results'][variant]['flags'] for c in anchor_cands]
            n = len(flags_list)
            na_pass_rate = sum(f['na_pass'] for f in flags_list) / n
            protein_pass_rate = sum(f['protein_pass'] for f in flags_list) / n
            calorie_pass_rate = sum(f['calorie_pass'] for f in flags_list) / n
            k_pass_rate = sum(f['k_pass'] for f in flags_list) / n
            raw_pass_rate = sum(f['raw_pass'] for f in flags_list) / n
            all_pass_rate = sum(f['all_pass'] for f in flags_list) / n
            protein_low_rate = sum(f['protein_low'] for f in flags_list) / n
            mean_elapsed = sum(c['results'][variant]['elapsed'] for c in anchor_cands) / n

            zero_cnt = 0
            rice_c, soup_c, main_c, side_c, kim_c = Counter(), Counter(), Counter(), Counter(), Counter()
            final_pass_count = 0
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
                else:
                    final_pass_count += 1
                m = sel['menus']
                rice_c[m[0]] += 1; soup_c[m[1]] += 1; main_c[m[2]] += 1; side_c[m[3]] += 1; kim_c[m[4]] += 1

            n_batches = len(batches)
            summary_rows.append({
                'anchor_type': anchor_name, 'variant': variant, 'candidate_count': n,
                'sodium_pass_rate': na_pass_rate, 'protein_pass_rate': protein_pass_rate,
                'calorie_pass_rate': calorie_pass_rate, 'potassium_pass_rate': k_pass_rate,
                'phosphorus_pass_rate': raw_pass_rate, 'nutrition_all_pass_rate': all_pass_rate,
                'protein_low_fail_rate': protein_low_rate,
                'zero_candidate_rate': zero_cnt / n_batches,
                'final_generation_success_rate': 1 - zero_cnt / n_batches,
                'final_selected_all_pass_rate': final_pass_count / n_batches,
                'rice_diversity': len(rice_c), 'soup_diversity': len(soup_c), 'main_diversity': len(main_c),
                'side_diversity': len(side_c), 'kimchi_diversity': len(kim_c),
                'mean_adjust_elapsed_sec': mean_elapsed,
                'call_count_reduction_pct': 33.3 if variant == 'S1' else 0.0,
            })
    summary_csv = os.path.join(OUT_DIR, 'sodium_preloop_cross_anchor_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader(); w.writerows(summary_rows)
    print(f'저장: {summary_csv}')
    for r in summary_rows:
        print(f"  [{r['anchor_type']}/{r['variant']}] Na통과={r['sodium_pass_rate']*100:.1f}% "
              f"protein통과={r['protein_pass_rate']*100:.1f}% 5영양전부={r['nutrition_all_pass_rate']*100:.1f}% "
              f"후보0개율={r['zero_candidate_rate']*100:.1f}% 최종생성성공률={r['final_generation_success_rate']*100:.1f}% "
              f"최종선택5영양통과율={r['final_selected_all_pass_rate']*100:.1f}%")

    # ── 전환 분석(앵커별) ──
    trans_rows = []
    for anchor_name, *_ in ANCHOR_CONFIGS:
        anchor_cands = [c for c in all_candidates if c['anchor_type'] == anchor_name]
        n = len(anchor_cands)
        b_res = lambda c: c['results']['Baseline']
        s1_res = lambda c: c['results']['S1']
        defs = {
            'baseline_all_pass_to_s1_fail': lambda c: b_res(c)['flags']['all_pass'] and not s1_res(c)['flags']['all_pass'],
            'baseline_fail_to_s1_all_pass': lambda c: (not b_res(c)['flags']['all_pass']) and s1_res(c)['flags']['all_pass'],
            'baseline_sodium_pass_to_s1_fail': lambda c: b_res(c)['flags']['na_pass'] and not s1_res(c)['flags']['na_pass'],
            'baseline_protein_pass_to_s1_protein_low': lambda c: b_res(c)['flags']['protein_pass'] and s1_res(c)['flags']['protein_low'],
            'baseline_all_pass_to_s1_protein_low_specifically': lambda c: b_res(c)['flags']['all_pass'] and s1_res(c)['flags']['protein_low'],
        }
        for name, fn in defs.items():
            cnt = sum(1 for c in anchor_cands if fn(c))
            trans_rows.append({'anchor_type': anchor_name, 'transition_type': name,
                                'candidate_count': cnt, 'candidate_rate': cnt / n})
    trans_csv = os.path.join(OUT_DIR, 'sodium_preloop_cross_anchor_transition.csv')
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trans_rows[0].keys()))
        w.writeheader(); w.writerows(trans_rows)
    print(f'저장: {trans_csv}')
    for r in trans_rows:
        print(f"  [{r['anchor_type']}] {r['transition_type']}: {r['candidate_count']}건({r['candidate_rate']*100:.2f}%)")

    return F, b, all_candidates, summary_rows, trans_rows, OUT_DIR


if __name__ == '__main__':
    main()
