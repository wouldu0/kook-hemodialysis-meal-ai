# -*- coding: utf-8 -*-
"""
side_dish_slot_phosphorus_diagnosis_FOOK.py — 두부·콩류 앵커의 인(phosphorus) 초과가
밥/국/주찬/부찬/김치 중 어느 슬롯 때문인지 슬롯별 기여량으로 분해. C_mask100+RL(epoch260)
체크포인트, 실제 make_meal()/adjust() 로직을 그대로 재현(순서·기준·레버·대체재풀 전부 불변경).

★ 코드 수정 없음: app_core_FOOK.py / FOOK_adjust_levers.py는 import해서 함수만 호출(읽기 전용).
   계측(슬롯 태깅)은 이 스크립트 안에서만 수행.

슬롯 추적 방법 (FOOK_adjust_levers.py 코드 확인 결과 기반, 추정 아님):
  - before: F.expand(menus)를 F.adjust()와 별도로 직접 호출(순수함수, 부작용 없음) → 5개 슬롯
    메뉴명(서로 겹치지 않음, 생성단계에서 보장)으로 직접 매칭.
  - after : F.adjust()가 반환한 inst의 각 항목에 대해
      1) entry.get('orig_menu', entry['menu'])가 5개 슬롯 메뉴명 중 하나와 일치 → 그 슬롯
         (rename_menu_for_swap이 orig_menu를 보존하므로 재료교체·이름변경도 추적됨, 코드확인:1111라인 근처)
      2) entry['menu']가 F.LOWNA_POOL(6종) 중 하나 → 김치 슬롯(lever_kimchi의 유일한 통째교체
         대상이 KIMCHI_SIDES→LOWNA_POOL뿐임을 코드로 확인)
      3) entry['menu']가 간식 후보 풀(F.SNACK_POOL, add_snack()이 만듦) 중 하나 → snack(5슬롯 밖)
      4) 위 셋 다 아니면 → unknown (임의로 특정 슬롯에 합치지 않음, 건수 보고)
    각 후보마다 슬롯 합계 == totals(inst)['P'] 전수 검증(불일치시 후보ID·오차 기록).
  - F.SWAP_LOG(레버가 실제 수행한 재료교체 이력, 'K'/'P' 태그 포함)를 adjust() 직후 캡처해
    "어느 스왑이 인 레버 소관인지"를 코드 로그로 확정. 양감소·스케일링(로그 없음)은 unknown.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python side_dish_slot_phosphorus_diagnosis_FOOK.py
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

OUT_DIR = os.path.join(CODE, 'side_dish_slot_phosphorus_diagnosis_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')

# 두부콩류 주력(2,400) + 생선구이·육류 비교군(각 480) — 풀스케일 아님, 원인확인용 소규모
ANCHOR_CONDITIONS = [
    ('두부콩류', '두부양념조림', [11, 12, 6, 36, 7], 20),
    ('생선구이', '고등어구이', [11, 12], 10),
    ('육류', '제육불고기', [11, 12], 10),
]
TRIES = 24
TEMP = 0.8
RNG_SEED = 11
SLOT_NAMES = ['rice', 'soup', 'main', 'side', 'kimchi']  # 인덱스 0..4


def entropy_of(counter):
    vals = np.array(list(counter.values()), dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


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
    """기존 side_dish_diagnosis_FOOK.py의 gen_batch_traced()와 동일 로직(복제, 무수정)."""
    seeds = np.tile(fixed_seed_row_7tok, (n, 1)).astype(np.int64)
    seeds[:, SOUP_POS] = mask_id
    fixed = {2: anchor_token}
    seeds[:, 3] = anchor_token

    seeds_tf = tf.constant(seeds, dtype=tf.int32)
    enc_hidden0 = tf.zeros([n, encoder.units])
    enc_output, enc_hidden = encoder(seeds_tf, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int)
    res[:, 0] = seeds[:, 0]; res[:, -1] = 826
    used = [set(fixed.values()) for _ in range(n)]
    used_grp = [{core.TOK_GRP[t] for t in fixed.values() if t in core.TOK_GRP} for _ in range(n)]

    for j in range(5):
        outputs, dec_hidden, _ = decoder(seeds_tf[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for bi in range(n):
            if j in fixed:
                res[bi, j + 1] = fixed[j]
                continue
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
            p = np.clip(p, 1e-12, None)
            p = p ** (1.0 / temp)
            p /= p.sum()
            tok = int(np.random.choice(len(p), p=p))
            res[bi, j + 1] = tok
            used[bi].add(tok)
            gi = core.TOK_GRP.get(tok)
            if gi is not None:
                used_grp[bi].add(gi)
    menus_list = [[food_dict[int(t)] for t in r if int(t) not in core.SPECIAL and t != mask_id] for r in res]
    return menus_list


def slot_phosphorus_before(orig_inst, slot_menu):
    """slot_menu = {slot_idx: menu_name}. 5개 이름이 서로 겹치지 않음(생성단계 보장)."""
    name_to_slot = {v: k for k, v in slot_menu.items()}
    out = {s: 0.0 for s in range(5)}
    unknown = 0.0
    for i in orig_inst:
        s = name_to_slot.get(i['menu'])
        contrib = (i['amt'] / 100 * i['P']) if i['P'] is not None else 0.0
        if s is not None:
            out[s] += contrib
        else:
            unknown += contrib  # 원칙적으로 발생 불가(orig_inst는 5개 메뉴로만 구성) — 발생시 보고
    return out, unknown


def slot_phosphorus_after(final_inst, slot_menu, lowna_pool, snack_names):
    name_to_slot = {v: k for k, v in slot_menu.items()}
    out = {s: 0.0 for s in range(5)}
    snack = 0.0
    unknown = 0.0
    unknown_menus = set()
    for i in final_inst:
        contrib = (i['amt'] / 100 * i['P']) if i['P'] is not None else 0.0
        identity = i.get('orig_menu', i['menu'])
        s = name_to_slot.get(identity)
        if s is not None:
            out[s] += contrib
            continue
        if i['menu'] in lowna_pool:
            out[4] += contrib   # 김치 슬롯 귀속(코드확인: lever_kimchi의 유일한 통째교체 대상)
            continue
        if i['menu'] in snack_names:
            snack += contrib
            continue
        unknown += contrib
        unknown_menus.add(i['menu'])
    return out, snack, unknown, unknown_menus


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    os.chdir(cwd)
    b = F.meal_bounds(60)
    PMAX = b['Pmax']

    # 간식 후보 풀 확보(읽기 전용 — add_snack()이 내부에서 채우는 것과 동일 함수 호출)
    F._build_snack_pool()
    snack_names = {m for m, *_ in F.SNACK_POOL}
    lowna_pool = set(F.LOWNA_POOL)
    kimchi_sides = set(F.KIMCHI_SIDES)

    _, food_dict, diet_np, _, mask_id = build_data(with_mask=True)
    num_tokens = len(food_dict)
    orig_diet_np_np = diet_np.numpy()

    enc, dec = load_model(RL_CKPT_DIR, num_tokens)

    trace_rows = []
    slot_mismatch = []   # (candidate_id, phase, diff)
    cid_counter = 0

    for cond_name, anchor_menu, seed_rows, n_calls in ANCHOR_CONDITIONS:
        anchor_token = core.name2idx[anchor_menu]
        for sid, row_idx in enumerate(seed_rows):
            base_row = orig_diet_np_np[row_idx].copy()
            np.random.seed(RNG_SEED)
            for call_id in range(n_calls):
                menus_list = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                              base_row, anchor_token, TRIES, TEMP)
                for menus in menus_list:
                    if len(menus) != 5:
                        continue
                    cid_counter += 1
                    slot_menu = {k: menus[k] for k in range(5)}

                    orig_inst = F.expand(list(menus))
                    before_slots, before_unknown = slot_phosphorus_before(orig_inst, slot_menu)
                    p_total_before = sum(before_slots.values()) + before_unknown

                    F.SWAP_LOG.clear()
                    before, after, inst, p_ok = F.adjust(list(menus), b, anchor=anchor_menu)
                    swap_log_snapshot = list(F.SWAP_LOG)

                    after_slots, after_snack, after_unknown, unk_menus = slot_phosphorus_after(
                        inst, slot_menu, lowna_pool, snack_names)
                    p_total_after_slots = sum(after_slots.values()) + after_snack + after_unknown

                    # 검증: 슬롯합 vs totals(inst)['P']
                    real_total_after = after['P']
                    diff_after = abs(p_total_after_slots - real_total_after)
                    if diff_after > 1e-6:
                        slot_mismatch.append((cid_counter, 'after', diff_after, sorted(unk_menus)))
                    real_total_before = before['P']
                    diff_before = abs(p_total_before - real_total_before)
                    if diff_before > 1e-6:
                        slot_mismatch.append((cid_counter, 'before', diff_before, []))

                    nut_flags = (b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                                 after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax'])
                    nutrition_all_pass = all(nut_flags)
                    other4_pass = nut_flags[0] and nut_flags[1] and nut_flags[2] and nut_flags[4]

                    p_excess_before = max(0.0, p_total_before - PMAX)
                    p_excess_after = max(0.0, real_total_after - PMAX)

                    # 메뉴 변경 여부(슬롯별) — orig_menu 역추적 기반
                    slot_changed = {}
                    slot_final_name = {}
                    for s in range(5):
                        orig_name = slot_menu[s]
                        # 최종 메뉴명: 이 슬롯으로 귀속된 inst 항목들의 실제 'menu' 필드(치환 후 이름)
                        names_here = [i['menu'] for i in inst
                                      if i.get('orig_menu', i['menu']) == orig_name
                                      or (s == 4 and i['menu'] in lowna_pool)]
                        final_name = names_here[0] if names_here else None
                        slot_final_name[s] = final_name
                        slot_changed[s] = (final_name is not None and final_name != orig_name)

                    # SWAP_LOG에서 이 후보의 슬롯별 인/칼륨 스왑 이력(메뉴명 기준 매칭)
                    def swaps_for(menu_name):
                        return [s for s in swap_log_snapshot if s[0] == menu_name]

                    triggered = {}
                    for s in range(5):
                        orig_name = slot_menu[s]
                        fname = slot_final_name[s]
                        sw = swaps_for(orig_name) or (swaps_for(fname) if fname else [])
                        if any(x[3] == 'P' for x in sw):
                            triggered[s] = 'lever_phosphorus'
                        elif any(x[3] == 'K' for x in sw):
                            triggered[s] = 'lever_potassium'
                        elif s == 4 and slot_changed[s] and fname in lowna_pool:
                            triggered[s] = 'lever_kimchi'
                        elif slot_changed[s]:
                            triggered[s] = 'unknown(amt_or_scale)'
                        else:
                            triggered[s] = 'none'

                    row = {
                        'anchor_type': cond_name, 'anchor_menu': anchor_menu, 'seed_id': sid,
                        'seed_row_idx': row_idx, 'call_id': call_id, 'candidate_id': cid_counter,
                        'rice': menus[0], 'soup': menus[1], 'main_dish': menus[2], 'side_dish': menus[3],
                        'kimchi': menus[4],
                        'phosphorus_total_before': p_total_before, 'phosphorus_total_after': real_total_after,
                        'phosphorus_limit': PMAX,
                        'phosphorus_excess_before': p_excess_before, 'phosphorus_excess_after': p_excess_after,
                        'phosphorus_remaining_budget_before': PMAX - p_total_before,
                        'phosphorus_remaining_budget_after': PMAX - real_total_after,
                        'rice_phosphorus_before': before_slots[0], 'soup_phosphorus_before': before_slots[1],
                        'main_phosphorus_before': before_slots[2], 'side_phosphorus_before': before_slots[3],
                        'kimchi_phosphorus_before': before_slots[4],
                        'rice_phosphorus_after': after_slots[0], 'soup_phosphorus_after': after_slots[1],
                        'main_phosphorus_after': after_slots[2], 'side_phosphorus_after': after_slots[3],
                        'kimchi_phosphorus_after': after_slots[4],
                        'snack_phosphorus_after': after_snack, 'unknown_phosphorus_after': after_unknown,
                        'rice_phosphorus_share': after_slots[0] / real_total_after if real_total_after else 0.0,
                        'soup_phosphorus_share': after_slots[1] / real_total_after if real_total_after else 0.0,
                        'main_phosphorus_share': after_slots[2] / real_total_after if real_total_after else 0.0,
                        'side_phosphorus_share': after_slots[3] / real_total_after if real_total_after else 0.0,
                        'kimchi_phosphorus_share': after_slots[4] / real_total_after if real_total_after else 0.0,
                        'phosphorus_delta_rice': after_slots[0] - before_slots[0],
                        'phosphorus_delta_soup': after_slots[1] - before_slots[1],
                        'phosphorus_delta_main': after_slots[2] - before_slots[2],
                        'phosphorus_delta_side': after_slots[3] - before_slots[3],
                        'phosphorus_delta_kimchi': after_slots[4] - before_slots[4],
                        'rice_menu_changed': slot_changed[0], 'soup_menu_changed': slot_changed[1],
                        'main_menu_changed': slot_changed[2], 'side_menu_changed': slot_changed[3],
                        'kimchi_menu_changed': slot_changed[4],
                        'side_dish_final': slot_final_name[3], 'kimchi_final': slot_final_name[4],
                        'triggered_lever_rice': triggered[0], 'triggered_lever_soup': triggered[1],
                        'triggered_lever_main': triggered[2], 'triggered_lever_side': triggered[3],
                        'triggered_lever_kimchi': triggered[4],
                        'protein_after': after['protein'], 'calorie_after': after['E'],
                        'sodium_after': after['Na_season'], 'potassium_after': after['K'],
                        'other4_pass_before_phos_removed': other4_pass,
                        'nutrition_all_pass': nutrition_all_pass,
                        'p_ok_lever_reported': p_ok,
                    }
                    trace_rows.append(row)
        print(f'[{cond_name}] 완료 ({len([r for r in trace_rows if r["anchor_type"]==cond_name])}건)')

    print(f'\n총 후보: {len(trace_rows)}건')
    print(f'슬롯합 불일치: {len(slot_mismatch)}건', slot_mismatch[:5] if slot_mismatch else '')

    trace_csv = os.path.join(OUT_DIR, 'side_dish_slot_phosphorus_trace.csv')
    fieldnames = list(trace_rows[0].keys())
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(trace_rows)
    print(f'저장: {trace_csv} ({len(trace_rows)}행)')

    mismatch_csv = os.path.join(OUT_DIR, 'slot_sum_mismatch_log.csv')
    with open(mismatch_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['candidate_id', 'phase', 'abs_diff', 'unknown_menus'])
        for cid, phase, diff, unk in slot_mismatch:
            w.writerow([cid, phase, diff, ';'.join(unk)])
    print(f'저장: {mismatch_csv} ({len(slot_mismatch)}행, 검증용)')

    return trace_rows, slot_mismatch, PMAX


if __name__ == '__main__':
    main()
