# -*- coding: utf-8 -*-
"""
protein_phosphorus_cross_anchor_FOOK.py — B90+Unified-rawP의 교차앵커(생선구이·육류) 재현성
검증. Baseline vs B90+Unified만 비교(B90 단독·S1 미포함). ★ 코드 수정 없음, 직전 실험의
검증된 로직(capped_lever_protein/calorie, unified_lever_phosphorus_rawP) 그대로 재사용.
"""
import os, sys, io, csv, copy, time
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

OUT_DIR = os.path.join(CODE, 'protein_phosphorus_cross_anchor_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_CONFIGS = [
    ('생선구이', '고등어구이', [11, 12, 6, 36, 7], 20),
    ('육류', '제육불고기', [11, 12, 6, 36, 7], 20),
]
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
FRAC = 0.90
VARIANTS = ['Baseline', 'B90+Unified']


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
    F.lever_kimchi(inst)
    F.lever_sodium(inst)
    F.lever_sodium_extra(inst, na_target)
    return inst


def _cap_and_scale(F, inst, menu, ratio, PMAX, frac, log):
    if ratio <= 1.0 + 1e-9:
        F.scale_menu(inst, menu, ratio)
        return
    current_raw_P = F.totals(inst)['P']
    headroom = PMAX - current_raw_P
    allowed_increase = max(0.0, headroom * frac)
    menu_items = [i for i in inst if i['menu'] == menu]
    P_menu = sum(i['amt'] / 100 * i['P'] for i in menu_items if i['P'] is not None)
    Amt_menu = sum(i['amt'] for i in menu_items)
    if Amt_menu <= 0:
        F.scale_menu(inst, menu, ratio)
        return
    requested_grams = Amt_menu * (ratio - 1.0)
    density = P_menu / Amt_menu
    allowed_grams = float('inf') if density <= 1e-9 else allowed_increase / density
    applied_grams = max(0.0, min(requested_grams, allowed_grams))
    final_ratio = 1.0 + applied_grams / Amt_menu
    F.scale_menu(inst, menu, final_ratio)
    capped = applied_grams < requested_grams - 1e-6
    log.append({'requested_ratio': ratio, 'applied_ratio': final_ratio, 'capped': capped})


def capped_lever_protein(F, inst, lo, hi, anchor, PMAX, frac, log):
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
            _cap_and_scale(F, inst, m, new / cur, PMAX, frac, log)
            return
    top = max(pm, key=pm.get)
    cur_top = pm[top]
    new_top = target - (t - cur_top)
    if cur_top > 0 and new_top > 0:
        ratio = max(0.3, min(new_top / cur_top, 2.0))
        _cap_and_scale(F, inst, top, ratio, PMAX, frac, log)


def capped_lever_calorie(F, inst, lo, hi, anchor, allow_snack, kmax, pmax_arg, PMAX, frac, log):
    e = F.totals(inst)['E']
    if e > hi:
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
            if ratio > 1.0 + 1e-9:
                current_raw_P = F.totals(inst)['P']
                headroom = PMAX - current_raw_P
                allowed_increase = max(0.0, headroom * frac)
                P_pool = sum(i['amt'] / 100 * i['P'] for i in pool if i['P'] is not None)
                Amt_pool = sum(i['amt'] for i in pool)
                if Amt_pool <= 0:
                    final_ratio = ratio
                else:
                    requested_grams = Amt_pool * (ratio - 1.0)
                    density = P_pool / Amt_pool
                    allowed_grams = float('inf') if density <= 1e-9 else allowed_increase / density
                    applied_grams = max(0.0, min(requested_grams, allowed_grams))
                    final_ratio = 1.0 + applied_grams / Amt_pool
                    capped = applied_grams < requested_grams - 1e-6
                    log.append({'requested_ratio': ratio, 'applied_ratio': final_ratio, 'capped': capped})
            else:
                final_ratio = ratio
            for i in pool:
                i['amt'] *= final_ratio
        e2 = F.totals(inst)['E']
        if e2 < lo:
            F.add_oil(inst, lo - e2)
            e3 = F.totals(inst)['E']
            if e3 < lo and allow_snack:
                F.add_snack(inst, lo - e3, kmax=kmax, pmax=pmax_arg)


def unified_lever_phosphorus_rawP(F, inst, pmax, anchor=None, plo=0):
    ing_nut, base_fresh, ing2kw, kw_rep, subs, _ = F.NUT
    for _ in range(25):
        if F.totals(inst)['P'] < pmax:
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
                effP_i = i['P']; effP_nd = nd['P']
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
            F.reduce_amt(max(cand, key=lambda x: x['amt'] / 100 * x['P']), 0.7)
            continue
        anc = [i for i in inst if i['menu'] == anchor and i['amt'] > 1 and F.reducible(i)]
        if anc and F.totals(inst)['protein'] > plo:
            for i in anc:
                F.reduce_amt(i, 0.85)
            continue
        return False
    return F.totals(inst)['P'] < pmax


def run_variant(F, prefix_inst, anchor, b, na_target, PMAX, variant, cap_log):
    inst = copy.deepcopy(prefix_inst)
    for pass_i in range(2):
        F.lever_potassium(inst, b['Kmax'], anchor=anchor)
        if variant == 'Baseline':
            F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        else:
            unified_lever_phosphorus_rawP(F, inst, b['Pmax'], anchor=anchor, plo=b['Plo'])

        if variant == 'Baseline':
            F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor)
        else:
            capped_lever_protein(F, inst, b['Plo'], b['Phi'], anchor, PMAX, FRAC, cap_log)

        F.lever_sodium(inst)
        F.lever_sodium_extra(inst, na_target)

        if variant == 'Baseline':
            F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                             kmax=b['Kmax'], pmax=b['Pmax'])
        else:
            capped_lever_calorie(F, inst, b['Elo'], b['Ehi'], anchor, (pass_i == 1), b['Kmax'], b['Pmax'],
                                  PMAX, FRAC, cap_log)
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
    PMAX = b['Pmax']
    na_target = b.get('Na_total_target', F.NA_TOTAL_MEAL)

    _, food_dict, diet_np, _, mask_id = build_data(with_mask=True)
    num_tokens = len(food_dict)
    orig_diet_np_np = diet_np.numpy()
    enc, dec = load_model(RL_CKPT_DIR, num_tokens)

    all_candidates = []
    for anchor_name, anchor_menu, seed_rows, n_calls in ANCHOR_CONFIGS:
        anchor_token = core.name2idx[anchor_menu]

        # 검증
        print(f'=== [{anchor_name}] 검증: Baseline 경로 vs F.adjust() 5건 ===')
        val_ok = 0
        for vi in range(5):
            np.random.seed(999 + vi)
            menus_v = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                       orig_diet_np_np[seed_rows[0]], anchor_token, 4, TEMP)[0]
            if len(menus_v) != 5:
                continue
            F.ROT[0] = 0
            prefix_v = run_shared_prefix(F, menus_v, na_target)
            cap_log_tmp = []
            inst_base = run_variant(F, prefix_v, anchor_menu, b, na_target, PMAX, 'Baseline', cap_log_tmp)
            manual_final = F.totals(inst_base)
            F.ROT[0] = 0
            _, adj_after, _, _ = F.adjust(list(menus_v), b, anchor=anchor_menu)
            ok = all(abs(manual_final[k] - adj_after[k]) < 1e-6 for k in ('E', 'protein', 'P', 'K', 'Na_season'))
            val_ok += int(ok)
        print(f'  검증 결과: {val_ok}/5 일치')
        F.ROT[0] = 0

        cid = 0
        cand_this_anchor = []
        t_gen_start = time.time()
        for sid, row_idx in enumerate(seed_rows):
            base_row = orig_diet_np_np[row_idx].copy()
            np.random.seed(RNG_SEED)
            for call_id in range(n_calls):
                menus_list = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                              base_row, anchor_token, TRIES, TEMP)
                for menus in menus_list:
                    if len(menus) != 5:
                        continue
                    cid += 1
                    prefix_inst = run_shared_prefix(F, menus, na_target)
                    results = {}
                    for variant in VARIANTS:
                        cap_log = []
                        t_adj_start = time.perf_counter()
                        inst_v = run_variant(F, prefix_inst, anchor_menu, b, na_target, PMAX, variant, cap_log)
                        elapsed = time.perf_counter() - t_adj_start
                        t_v = F.totals(inst_v)
                        unreal = F.unrealistic_reason(inst_v)
                        cap_applied = any(e['capped'] or abs(e['applied_ratio'] - e['requested_ratio']) > 1e-6 for e in cap_log) if cap_log else False
                        cap_ratio = (sum(e['applied_ratio'] for e in cap_log) / len(cap_log)) if cap_log else None
                        results[variant] = {'t': t_v, 'flags': nutrient_flags(t_v, b), 'unrealistic': unreal,
                                             'elapsed': elapsed, 'cap_applied': cap_applied, 'cap_ratio': cap_ratio,
                                             'cap_events': len(cap_log)}
                    c = {'candidate_id': cid, 'anchor_type': anchor_name, 'seed_id': sid, 'call_id': call_id,
                         'menus': menus, 'results': results}
                    cand_this_anchor.append(c)
                    all_candidates.append(c)
        print(f'[{anchor_name}] 생성 완료: {len(cand_this_anchor)}건, {time.time()-t_gen_start:.1f}초')

    n_total = len(all_candidates)
    print(f'\n총 생성: {n_total}건 (앵커 2종 합계)')

    # ── trace CSV ──
    trace_rows = []
    batches = {}
    for c in all_candidates:
        batches.setdefault((c['anchor_type'], c['seed_id'], c['call_id']), []).append(c)

    def score_of(flags):
        return sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])

    selected_ids = {v: set() for v in VARIANTS}
    fallback_ids = {v: set() for v in VARIANTS}
    for key, batch in batches.items():
        for variant in VARIANTS:
            sel = None
            for c in batch:
                if c['results'][variant]['flags']['all_pass']:
                    sel = c; break
            fb = False
            if sel is None:
                fb = True
                best_score, best_c = -1, None
                for c in batch:
                    s = score_of(c['results'][variant]['flags'])
                    if s > best_score:
                        best_score, best_c = s, c
                sel = best_c
            selected_ids[variant].add(sel['candidate_id'])
            if fb:
                fallback_ids[variant].add(sel['candidate_id'])

    for c in all_candidates:
        for variant in VARIANTS:
            r = c['results'][variant]; t = r['t']; flags = r['flags']
            trace_rows.append({
                'anchor_type': c['anchor_type'], 'seed': c['seed_id'], 'call_id': c['call_id'],
                'candidate_id': c['candidate_id'], 'variant': variant,
                'nutrition_all_pass': flags['all_pass'],
                'selected': c['candidate_id'] in selected_ids[variant],
                'fallback_used': c['candidate_id'] in fallback_ids[variant],
                'calories': t['E'], 'protein': t['protein'], 'potassium': t['K'],
                'phosphorus_raw': t['P'], 'phosphorus_effective': t['Peff'],
                'sodium_total': t['Na'], 'sodium_season': t['Na_season'],
                'protein_low': flags['protein_low'], 'calorie_low': flags['calorie_low'],
                'phosphorus_high': not flags['raw_pass'],
                'unrealistic_reason': r['unrealistic'],
                'cap_applied': r['cap_applied'], 'cap_ratio': r['cap_ratio'],
                'soup': c['menus'][1], 'main': c['menus'][2], 'side': c['menus'][3], 'kimchi': c['menus'][4],
                'elapsed_sec': r['elapsed'],
            })
    trace_csv = os.path.join(OUT_DIR, 'protein_phosphorus_cross_anchor_trace.csv')
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(trace_rows[0].keys()))
        w.writeheader(); w.writerows(trace_rows)
    print(f'저장: {trace_csv} ({len(trace_rows)}행)')

    return F, b, all_candidates, batches, OUT_DIR


if __name__ == '__main__':
    main()
