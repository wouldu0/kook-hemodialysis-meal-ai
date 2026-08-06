# -*- coding: utf-8 -*-
"""
protein_phosphorus_final_compare_FOOK.py — protein-phosphorus 상호작용에 대한 마지막 paired
비교: Baseline / B90 / B90+Unified-rawP. 두부·콩류 앵커 2,400건. ★ 코드 수정 없음.

B90        : lever_protein/lever_calorie의 증량(scale_menu 비율>1)만 남은 raw P 예산의 90%로 제한
             (직전 B90 실험 로직 그대로 재사용). lever_phosphorus는 원본(Peff 기준) 그대로.
B90+Unified: B90 cap + lever_phosphorus 내부 판정(진입/수렴/최종반환) 4곳을 raw P로 치환
             (직전 Unified-rawP 실험 로직 그대로 재사용).

sodium/potassium/kimchi 로직 무변경, pass1/pass2 sodium 호출 유지, S1(pre-loop 나트륨 제거)은
이번 실험에 섞지 않음(원본 pre-loop 그대로 유지).
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

OUT_DIR = os.path.join(CODE, 'protein_phosphorus_final_compare_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

ANCHOR_MENU = '두부양념조림'
SEED_ROWS = [11, 12, 6, 36, 7]
N_CALLS = 20
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
MARGIN = 0.01
FRAC = 0.90
VARIANTS = ['Baseline', 'B90', 'B90+Unified']


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


# ── B90: capped protein/calorie (직전 B90 실험과 동일 로직) ──
def _cap_and_scale(F, inst, menu, ratio, PMAX, frac):
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


def capped_lever_protein(F, inst, lo, hi, anchor, PMAX, frac):
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
            _cap_and_scale(F, inst, m, new / cur, PMAX, frac)
            return
    top = max(pm, key=pm.get)
    cur_top = pm[top]
    new_top = target - (t - cur_top)
    if cur_top > 0 and new_top > 0:
        ratio = max(0.3, min(new_top / cur_top, 2.0))
        _cap_and_scale(F, inst, top, ratio, PMAX, frac)


def capped_lever_calorie(F, inst, lo, hi, anchor, allow_snack, kmax, pmax_arg, PMAX, frac):
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


# ── Unified-rawP: lever_phosphorus 복제, Peff/p_abs() 4곳을 raw P로 치환 ──
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


def run_variant(F, prefix_inst, anchor, b, na_target, PMAX, variant):
    inst = copy.deepcopy(prefix_inst)
    for pass_i in range(2):
        F.lever_potassium(inst, b['Kmax'], anchor=anchor)
        if variant == 'Baseline':
            F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        elif variant == 'B90':
            F.lever_phosphorus(inst, b['Pmax'], anchor=anchor, plo=b['Plo'])
        else:  # B90+Unified
            unified_lever_phosphorus_rawP(F, inst, b['Pmax'], anchor=anchor, plo=b['Plo'])

        if variant == 'Baseline':
            F.lever_protein(inst, b['Plo'], b['Phi'], anchor=anchor)
        else:
            capped_lever_protein(F, inst, b['Plo'], b['Phi'], anchor, PMAX, FRAC)

        F.lever_sodium(inst)
        F.lever_sodium_extra(inst, na_target)

        if variant == 'Baseline':
            F.lever_calorie(inst, b['Elo'], b['Ehi'], anchor=anchor, allow_snack=(pass_i == 1),
                             kmax=b['Kmax'], pmax=b['Pmax'])
        else:
            capped_lever_calorie(F, inst, b['Elo'], b['Ehi'], anchor, (pass_i == 1), b['Kmax'], b['Pmax'],
                                  PMAX, FRAC)
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
    anchor_token = core.name2idx[ANCHOR_MENU]

    # 검증: Baseline 경로 vs F.adjust()
    print('=== 검증: Baseline 경로 vs F.adjust() 5건 ===')
    val_ok = 0
    for vi in range(5):
        np.random.seed(999 + vi)
        menus_v = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                   orig_diet_np_np[SEED_ROWS[0]], anchor_token, 4, TEMP)[0]
        if len(menus_v) != 5:
            continue
        F.ROT[0] = 0
        prefix_v = run_shared_prefix(F, menus_v, na_target)
        inst_base = run_variant(F, prefix_v, ANCHOR_MENU, b, na_target, PMAX, 'Baseline')
        manual_final = F.totals(inst_base)
        F.ROT[0] = 0
        _, adj_after, _, _ = F.adjust(list(menus_v), b, anchor=ANCHOR_MENU)
        ok = all(abs(manual_final[k] - adj_after[k]) < 1e-6 for k in ('E', 'protein', 'P', 'K', 'Na_season'))
        val_ok += int(ok)
        print(f'  검증{vi}: 일치={ok}')
    print(f'검증 결과: {val_ok}/5 일치\n')
    F.ROT[0] = 0

    print(f'생성 시작: 두부콩류 {len(SEED_ROWS)}seed x {N_CALLS}call x {TRIES} = {len(SEED_ROWS)*N_CALLS*TRIES}후보(예정) x 3variant')
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
                prefix_inst = run_shared_prefix(F, menus, na_target)
                results = {}
                for variant in VARIANTS:
                    inst_v = run_variant(F, prefix_inst, ANCHOR_MENU, b, na_target, PMAX, variant)
                    t_v = F.totals(inst_v)
                    unreal = F.unrealistic_reason(inst_v)
                    results[variant] = {'t': t_v, 'flags': nutrient_flags(t_v, b),
                                         'unrealistic': unreal, 'inst': inst_v}
                candidates.append({'candidate_id': cid, 'seed_id': sid, 'call_id': call_id,
                                    'menus': menus, 'results': results})
        print(f'  seed {sid}(row{row_idx}) 완료, 누적 {len(candidates)}건')
    print(f'\n총 생성: {len(candidates)}건')

    n_total = len(candidates)

    # ── 후보 단위 요약(참고용) ──
    cand_summary_rows = []
    for variant in VARIANTS:
        flags_list = [c['results'][variant]['flags'] for c in candidates]
        n = len(flags_list)
        raw_pass_rate = sum(f['raw_pass'] for f in flags_list) / n
        protein_pass_rate = sum(f['protein_pass'] for f in flags_list) / n
        protein_low_rate = sum(f['protein_low'] for f in flags_list) / n
        all_pass_rate = sum(f['all_pass'] for f in flags_list) / n
        unreal_rate = sum(1 for c in candidates if c['results'][variant]['unrealistic'] is not None) / n
        mean_protein = sum(c['results'][variant]['t']['protein'] for c in candidates) / n
        cand_summary_rows.append({
            'variant': variant, 'candidate_count': n, 'candidate_level_raw_P_pass_rate': raw_pass_rate,
            'candidate_level_protein_pass_rate': protein_pass_rate,
            'candidate_level_protein_low_rate': protein_low_rate,
            'candidate_level_all_pass_rate': all_pass_rate,
            'candidate_level_unrealistic_rate': unreal_rate,
            'candidate_level_mean_protein': mean_protein,
        })
    with open(os.path.join(OUT_DIR, 'candidate_level_reference_summary.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(cand_summary_rows[0].keys()))
        w.writeheader(); w.writerows(cand_summary_rows)
    print('\n=== 후보 단위(참고용) ===')
    for r in cand_summary_rows:
        print(f"  [{r['variant']}] rawP통과={r['candidate_level_raw_P_pass_rate']*100:.1f}% "
              f"protein통과={r['candidate_level_protein_pass_rate']*100:.1f}% "
              f"protein_low={r['candidate_level_protein_low_rate']*100:.1f}% "
              f"5영양전부={r['candidate_level_all_pass_rate']*100:.1f}% "
              f"비현실={r['candidate_level_unrealistic_rate']*100:.1f}% "
              f"평균단백={r['candidate_level_mean_protein']:.2f}g")

    # ── 배치(seed,call) 단위 최종선택 시뮬레이션 (핵심 서비스 지표) ──
    def score_of(flags):
        return sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])

    batches = {}
    for c in candidates:
        batches.setdefault((c['seed_id'], c['call_id']), []).append(c)
    n_batches = len(batches)

    service_rows = []
    selected_records = {v: [] for v in VARIANTS}
    for variant in VARIANTS:
        zero_cnt = 0
        rice_c, soup_c, main_c, side_c, kim_c = Counter(), Counter(), Counter(), Counter(), Counter()
        unique_meals = set()
        final_all_pass = 0
        final_protein_low = 0
        protein_vals = []
        protein_shortfalls = []
        unreal_count = 0
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
            r = sel['results'][variant]
            selected_records[variant].append((key, sel['candidate_id'], r))
            if r['flags']['all_pass']:
                final_all_pass += 1
            if r['flags']['protein_low']:
                final_protein_low += 1
            protein_vals.append(r['t']['protein'])
            if r['t']['protein'] < b['Plo']:
                protein_shortfalls.append(b['Plo'] - r['t']['protein'])
            if r['unrealistic'] is not None:
                unreal_count += 1
            m = sel['menus']
            rice_c[m[0]] += 1; soup_c[m[1]] += 1; main_c[m[2]] += 1; side_c[m[3]] += 1; kim_c[m[4]] += 1
            unique_meals.add(tuple(m))

        service_rows.append({
            'variant': variant,
            'final_generation_success_rate': 1 - zero_cnt / n_batches,
            'zero_candidate_rate': zero_cnt / n_batches,
            'final_selected_all_pass_rate': final_all_pass / n_batches,
            'final_selected_protein_low_rate': final_protein_low / n_batches,
            'final_selected_mean_protein': sum(protein_vals) / len(protein_vals),
            'final_selected_mean_protein_shortfall_below_22g': (sum(protein_shortfalls) / len(protein_shortfalls)) if protein_shortfalls else 0.0,
            'final_selected_below_22g_count': len(protein_shortfalls),
            'final_selected_unrealistic_rate': unreal_count / n_batches,
            'rice_diversity': len(rice_c), 'soup_diversity': len(soup_c), 'main_diversity': len(main_c),
            'side_diversity': len(side_c), 'kimchi_diversity': len(kim_c), 'unique_meal_count': len(unique_meals),
        })

    # Baseline 대비 신규파괴율(배치 단위: Baseline 선택 all_pass True인데 variant 선택은 False)
    base_sel = {k: (cid_, r) for k, cid_, r in selected_records['Baseline']}
    for row in service_rows:
        variant = row['variant']
        v_sel = {k: (cid_, r) for k, cid_, r in selected_records[variant]}
        broke = 0; base_pass_n = 0
        for key in batches:
            _, br = base_sel[key]
            if br['flags']['all_pass']:
                base_pass_n += 1
                _, vr = v_sel[key]
                if not vr['flags']['all_pass']:
                    broke += 1
        row['baseline_pass_batches'] = base_pass_n
        row['newly_broken_by_variant_count'] = broke
        row['newly_broken_by_variant_rate'] = broke / base_pass_n if base_pass_n else None

    service_csv = os.path.join(OUT_DIR, 'protein_phosphorus_final_service_summary.csv')
    with open(service_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(service_rows[0].keys()))
        w.writeheader(); w.writerows(service_rows)
    print(f'\n저장: {service_csv} (배치 {n_batches}개 기준)')
    print('\n=== 배치(call) 단위 서비스 지표 ===')
    for r in service_rows:
        print(f"  [{r['variant']}] 최종생성성공률={r['final_generation_success_rate']*100:.1f}% "
              f"후보0개율={r['zero_candidate_rate']*100:.1f}% "
              f"최종선택5영양통과율={r['final_selected_all_pass_rate']*100:.1f}% "
              f"최종protein_low={r['final_selected_protein_low_rate']*100:.1f}% "
              f"평균단백={r['final_selected_mean_protein']:.2f}g "
              f"22g미만평균부족={r['final_selected_mean_protein_shortfall_below_22g']:.2f}g "
              f"비현실={r['final_selected_unrealistic_rate']*100:.1f}% "
              f"부찬다양성={r['side_diversity']} 국다양성={r['soup_diversity']} "
              f"신규파괴율={r['newly_broken_by_variant_rate']*100 if r['newly_broken_by_variant_rate'] is not None else 0:.1f}%")

    return F, b, candidates, service_rows, cand_summary_rows, OUT_DIR


if __name__ == '__main__':
    main()
