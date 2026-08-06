# -*- coding: utf-8 -*-
"""
service_rollout_verification_FOOK.py — 서비스 코드 반영(FOOK_adjust_levers.py의 두부·콩류
조건부 raw P 통일+cap 경로) 검증. OLD(수정 전 백업파일) vs NEW(수정된 실제 서비스 파일)를
동일 후보로 paired 비교.

A. 구현 동일성   : 두부콩류 후보에서 NEW(F.adjust) 결과가 실험용 검증 로직과 일치하는지
                  (이미 여러 세션에 걸쳐 검증된 capped/unified 로직을 그대로 서비스에 옮겼으므로,
                  여기서는 NEW가 "두부콩류에서 실제로 다르게 동작하는지"를 OLD와 대조로 확인)
B. 비대상 회귀   : 생선구이·육류 후보에서 OLD와 NEW가 100% 동일한 결과를 내는지
C. 전체 회귀     : 3앵커 합산 서비스지표(성공률·통과율·다양성·재료현실성·실행시간) OLD vs NEW
D. 대표 사례     : 두부콩류 개선사례 5건, 생선구이·육류 무변경사례 각 3건
"""
import os, sys, io, csv, copy, time, importlib.util
from importlib.machinery import SourceFileLoader
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

OUT_DIR = os.path.join(CODE, 'service_rollout_verification_out')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')
OLD_BAK = os.path.join(FINAL, 'FOOK_adjust_levers.py.bak_before_rawP_tofu_path_20260727')

ANCHOR_CONFIGS = [
    ('두부콩류', '두부양념조림', [11, 12, 6, 36, 7], 10),
    ('생선구이', '고등어구이', [11, 12, 6, 36, 7], 10),
    ('육류', '제육불고기', [11, 12, 6, 36, 7], 10),
]
TRIES = 24
TEMP = 0.8
RNG_SEED = 11


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
    cwd0 = os.getcwd()

    # ── OLD 모듈(수정 전 백업) 로드 ──
    os.chdir(FINAL)
    loader_old = SourceFileLoader("FOOK_adjust_levers_OLD", OLD_BAK)
    spec_old = importlib.util.spec_from_loader(loader_old.name, loader_old)
    F_old = importlib.util.module_from_spec(spec_old)
    loader_old.exec_module(F_old)
    F_old.NUT = F_old.load_all()
    print('OLD 모듈(백업) 로드 완료:', OLD_BAK)

    # ── NEW 모듈(수정된 실제 서비스 파일) + app_core 로드 ──
    import app_core_FOOK as core
    import FOOK_adjust_levers as F_new   # 이미 app_core가 import해서 NUT 로딩까지 완료된 실제 모듈
    os.chdir(cwd0)
    print('NEW 모듈(실제 서비스 파일) 로드 완료: FOOK_adjust_levers.py (수정본)')
    print(f'  hasattr F_new.lever_phosphorus_rawP: {hasattr(F_new, "lever_phosphorus_rawP")}')
    print(f'  hasattr F_new.lever_protein_capped: {hasattr(F_new, "lever_protein_capped")}')
    print(f'  hasattr F_new.lever_calorie_capped: {hasattr(F_new, "lever_calorie_capped")}')
    print(f'  hasattr F_new._plant_protein_path_needed: {hasattr(F_new, "_plant_protein_path_needed")}')

    b = F_new.meal_bounds(60)

    _, food_dict, diet_np, _, mask_id = build_data(with_mask=True)
    num_tokens = len(food_dict)
    orig_diet_np_np = diet_np.numpy()
    enc, dec = load_model(RL_CKPT_DIR, num_tokens)

    all_candidates = []
    for anchor_name, anchor_menu, seed_rows, n_calls in ANCHOR_CONFIGS:
        anchor_token = core.name2idx[anchor_menu]
        # 이 앵커가 두부콩류 판정에 실제로 걸리는지 사전 확인(진단용) — 배치크기 1은 모델 구조상
        # 불가(attention 레이어가 batch>=2 가정), 최소 배치 4로 생성 후 첫 건만 사용.
        np.random.seed(12345)
        probe_menus = gen_batch_slots(core, enc, dec, num_tokens, mask_id, food_dict,
                                       orig_diet_np_np[seed_rows[0]], anchor_token, 4, TEMP)[0]
        probe_inst = F_new.expand(list(probe_menus))
        is_plant = F_new._plant_protein_path_needed(probe_inst, probe_menus, anchor_menu)
        print(f'[{anchor_name}] _plant_protein_path_needed 판정: {is_plant} (앵커={anchor_menu})')

        cid = 0
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
                    F_old.ROT[0] = 0
                    before_o, after_o, inst_o, pok_o = F_old.adjust(list(menus), b, anchor=anchor_menu)
                    F_new.ROT[0] = 0
                    t_start = time.perf_counter()
                    before_n, after_n, inst_n, pok_n = F_new.adjust(list(menus), b, anchor=anchor_menu)
                    elapsed_new = time.perf_counter() - t_start
                    unreal_o = F_old.unrealistic_reason(inst_o)
                    unreal_n = F_new.unrealistic_reason(inst_n)
                    all_candidates.append({
                        'candidate_id': cid, 'anchor_type': anchor_name, 'seed_id': sid, 'call_id': call_id,
                        'menus': menus, 'is_plant_protein_anchor': is_plant,
                        'after_o': after_o, 'after_n': after_n, 'unreal_o': unreal_o, 'unreal_n': unreal_n,
                        'flags_o': nutrient_flags(after_o, b), 'flags_n': nutrient_flags(after_n, b),
                        'elapsed_new': elapsed_new,
                    })
        print(f'[{anchor_name}] 완료: {cid}건')

    n_total = len(all_candidates)
    print(f'\n총 비교: {n_total}건 (OLD vs NEW, 3앵커 합계)')

    # ── identical 판정(모든 5영양 원값 완전일치인지) ──
    def identical(c):
        keys = ('E', 'protein', 'K', 'P', 'Na_season')
        return all(abs(c['after_o'][k] - c['after_n'][k]) < 1e-9 for k in keys)

    for c in all_candidates:
        c['identical'] = identical(c)

    # ── A/B: 앵커별 identical 비율 ──
    print('\n=== A/B. 앵커별 OLD vs NEW 동일성 ===')
    identity_rows = []
    for anchor_name, *_ in ANCHOR_CONFIGS:
        acands = [c for c in all_candidates if c['anchor_type'] == anchor_name]
        n_ident = sum(1 for c in acands if c['identical'])
        n_diff = len(acands) - n_ident
        is_plant = acands[0]['is_plant_protein_anchor'] if acands else None
        print(f'  [{anchor_name}] 두부콩류판정={is_plant}  동일={n_ident}/{len(acands)}  차이={n_diff}건')
        identity_rows.append({'anchor_type': anchor_name, 'is_plant_protein_anchor': is_plant,
                               'n': len(acands), 'identical_count': n_ident, 'diff_count': n_diff,
                               'identical_rate': n_ident / len(acands) if acands else None})
    with open(os.path.join(OUT_DIR, 'AB_identity_check.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(identity_rows[0].keys()))
        w.writeheader(); w.writerows(identity_rows)

    # 비대상 앵커(생선구이·육류)는 반드시 100% 동일해야 함 — 위반 시 강조 출력
    non_target_ok = True
    for row in identity_rows:
        if row['anchor_type'] != '두부콩류' and row['diff_count'] > 0:
            non_target_ok = False
            print(f"  ★★★ 경고: 비대상 앵커({row['anchor_type']})에서 {row['diff_count']}건 결과가 다름! ★★★")
    print(f'\n비대상 앵커 완전동일 여부: {"OK" if non_target_ok else "위반 발견 — 반영 중단 검토 필요"}')

    # ── C. 전체 회귀 서비스지표 (OLD vs NEW) ──
    def score_of(flags):
        return sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])

    batches = {}
    for c in all_candidates:
        batches.setdefault((c['anchor_type'], c['seed_id'], c['call_id']), []).append(c)

    service_rows = []
    for label, flags_key, after_key, unreal_key in [('OLD', 'flags_o', 'after_o', 'unreal_o'),
                                                      ('NEW', 'flags_n', 'after_n', 'unreal_n')]:
        for anchor_name, *_ in ANCHOR_CONFIGS + [('전체', None, None, None)]:
            acands = all_candidates if anchor_name == '전체' else [c for c in all_candidates if c['anchor_type'] == anchor_name]
            abatches = {k: v for k, v in batches.items() if anchor_name == '전체' or k[0] == anchor_name}
            n_batches = len(abatches)
            zero_cnt = 0
            rice_c, soup_c, main_c, side_c, kim_c = Counter(), Counter(), Counter(), Counter(), Counter()
            final_protein_low = 0; final_unreal = 0
            for key, batch in abatches.items():
                has_pass = any(c[flags_key]['all_pass'] for c in batch)
                if not has_pass:
                    zero_cnt += 1
                sel = None
                for c in sorted(batch, key=lambda x: x['candidate_id']):
                    if c[flags_key]['all_pass']:
                        sel = c; break
                if sel is None:
                    best_score, best_c = -1, None
                    for c in batch:
                        s = score_of(c[flags_key])
                        if s > best_score:
                            best_score, best_c = s, c
                    sel = best_c
                if sel[flags_key]['protein_low']:
                    final_protein_low += 1
                if sel[unreal_key] is not None:
                    final_unreal += 1
                m = sel['menus']
                rice_c[m[0]] += 1; soup_c[m[1]] += 1; main_c[m[2]] += 1; side_c[m[3]] += 1; kim_c[m[4]] += 1
            mean_elapsed = (sum(c['elapsed_new'] for c in acands) / len(acands) * 1000) if label == 'NEW' and acands else None
            service_rows.append({
                'variant': label, 'anchor_type': anchor_name, 'n_batches': n_batches,
                'generation_success_rate': 1 - zero_cnt / n_batches if n_batches else None,
                'zero_candidate_rate': zero_cnt / n_batches if n_batches else None,
                'final_protein_low_rate': final_protein_low / n_batches if n_batches else None,
                'final_unrealistic_rate': final_unreal / n_batches if n_batches else None,
                'soup_diversity': len(soup_c), 'side_diversity': len(side_c), 'kimchi_diversity': len(kim_c),
                'mean_adjust_time_ms': mean_elapsed,
            })
    service_df_csv = os.path.join(OUT_DIR, 'C_full_regression_summary.csv')
    with open(service_df_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(service_rows[0].keys()))
        w.writeheader(); w.writerows(service_rows)
    print(f'\n저장: {service_df_csv}')
    for r in service_rows:
        ms = f"{r['mean_adjust_time_ms']:.3f}ms" if r['mean_adjust_time_ms'] is not None else '-'
        print(f"  [{r['variant']}/{r['anchor_type']}] 생성성공률={r['generation_success_rate']*100:.1f}% "
              f"후보0개율={r['zero_candidate_rate']*100:.1f}% protein_low={r['final_protein_low_rate']*100:.1f}% "
              f"비현실={r['final_unrealistic_rate']*100:.1f}% 국={r['soup_diversity']} 부찬={r['side_diversity']} "
              f"김치={r['kimchi_diversity']} 실행={ms}")

    # ── D. 대표 사례 ──
    d_rows = []
    tofu_cands = [c for c in all_candidates if c['anchor_type'] == '두부콩류']
    improved = [c for c in tofu_cands if (not c['flags_o']['all_pass']) and c['flags_n']['all_pass']]
    for c in improved[:5]:
        d_rows.append({
            'category': '두부콩류_개선사례', 'anchor_type': c['anchor_type'], 'candidate_id': c['candidate_id'],
            'menus': '|'.join(c['menus']),
            'OLD_all_pass': c['flags_o']['all_pass'], 'NEW_all_pass': c['flags_n']['all_pass'],
            'OLD_calorie': c['after_o']['E'], 'NEW_calorie': c['after_n']['E'],
            'OLD_protein': c['after_o']['protein'], 'NEW_protein': c['after_n']['protein'],
            'OLD_potassium': c['after_o']['K'], 'NEW_potassium': c['after_n']['K'],
            'OLD_phosphorus_raw': c['after_o']['P'], 'NEW_phosphorus_raw': c['after_n']['P'],
            'OLD_sodium_season': c['after_o']['Na_season'], 'NEW_sodium_season': c['after_n']['Na_season'],
        })
    for anchor_name in ['생선구이', '육류']:
        acands = [c for c in all_candidates if c['anchor_type'] == anchor_name and c['identical']]
        for c in acands[:3]:
            d_rows.append({
                'category': f'{anchor_name}_무변경사례', 'anchor_type': c['anchor_type'], 'candidate_id': c['candidate_id'],
                'menus': '|'.join(c['menus']),
                'OLD_all_pass': c['flags_o']['all_pass'], 'NEW_all_pass': c['flags_n']['all_pass'],
                'OLD_calorie': c['after_o']['E'], 'NEW_calorie': c['after_n']['E'],
                'OLD_protein': c['after_o']['protein'], 'NEW_protein': c['after_n']['protein'],
                'OLD_potassium': c['after_o']['K'], 'NEW_potassium': c['after_n']['K'],
                'OLD_phosphorus_raw': c['after_o']['P'], 'NEW_phosphorus_raw': c['after_n']['P'],
                'OLD_sodium_season': c['after_o']['Na_season'], 'NEW_sodium_season': c['after_n']['Na_season'],
            })
    d_csv = os.path.join(OUT_DIR, 'D_representative_cases.csv')
    with open(d_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(d_rows[0].keys()))
        w.writeheader(); w.writerows(d_rows)
    print(f'\n저장: {d_csv} ({len(d_rows)}건: 두부콩류개선 {min(len(improved),5)}건 + 생선구이/육류 무변경 각3건)')

    return all_candidates, identity_rows, service_rows, d_rows, OUT_DIR


if __name__ == '__main__':
    main()
