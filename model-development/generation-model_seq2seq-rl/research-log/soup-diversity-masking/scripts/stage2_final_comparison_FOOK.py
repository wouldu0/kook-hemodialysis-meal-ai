# -*- coding: utf-8 -*-
"""
stage2_final_comparison_FOOK.py — A_baseline / C_mask100 / C_mask100+RL 세 모델을 동일하게
실제 make_meal() 전체 파이프라인으로 비교(stage1과 동일 방법론, RL 체크포인트만 추가).

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python stage2_final_comparison_FOOK.py
"""
import os, sys, io, csv, copy, itertools
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from collections import Counter
from scipy.spatial.distance import jensenshannon

FINAL = r'E:\final'
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, FINAL)

from Model import Encoder, Decoder
from train_FOOK_soupmask_1000 import build_data, SOUP_POS
from stage1_full_make_meal_verify_FOOK import (gen_batch_generic, make_meal_generic,
                                                ANCHOR_CONDITIONS, SEED_ROWS, N_CALLS, TRIES, TEMP, RNG_SEED)

OUT_DIR = os.path.join(CODE, 'stage2_final_out')
CKPT_ROOT = os.path.join(CODE, 'checkpoints_masking_1000')
RL_CKPT_DIR = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')
RL_CKPT_DIR_LAST = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_last')


def load_model_from(ckpt_dir, num_tokens):
    kwargs = {'num_tokens': num_tokens, 'embed_dim': 128, 'fc_dim': 64,
              'fully-connected_layer': 'GRU', 'attention': True}
    enc = Encoder(**kwargs, batch_size=10)
    dec = Decoder(**kwargs, batch_size=10)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    if not ck:
        raise RuntimeError(f'체크포인트 없음: {ckpt_dir}')
    tf.train.Checkpoint(encoder=enc, decoder=dec).restore(ck).expect_partial()
    print(f'  로딩: {ckpt_dir} -> {ck}')
    return enc, dec


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    import reward_lever_FOOK as R
    R.init(weight=60)
    os.chdir(cwd)
    b = F.meal_bounds(60)

    _, food_dict_827, diet_np, _, _ = build_data(with_mask=False)
    _, food_dict_828, _, _, mask_id828 = build_data(with_mask=True)
    orig_diet_np_np = diet_np.numpy()

    rl_dir = RL_CKPT_DIR if tf.train.latest_checkpoint(RL_CKPT_DIR) else RL_CKPT_DIR_LAST
    print('RL 체크포인트 소스:', rl_dir)

    MODELS = {
        'A_baseline': {'vocab': 827, 'fd': food_dict_827, 'mask_id': None, 'soup_masked': False,
                       'ckpt_dir': os.path.join(CKPT_ROOT, 'A_baseline', 'checkpoints')},
        'C_mask100': {'vocab': 828, 'fd': food_dict_828, 'mask_id': mask_id828, 'soup_masked': True,
                      'ckpt_dir': os.path.join(CKPT_ROOT, 'C_mask100', 'checkpoints')},
        'C_mask100_RL': {'vocab': 828, 'fd': food_dict_828, 'mask_id': mask_id828, 'soup_masked': True,
                         'ckpt_dir': rl_dir},
    }
    for exp, cfg in MODELS.items():
        enc, dec = load_model_from(cfg['ckpt_dir'], cfg['vocab'])
        cfg['encoder'], cfg['decoder'] = enc, dec

    seed_info = []
    for sid, row_idx in enumerate(SEED_ROWS):
        row = orig_diet_np_np[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in row]
        seed_info.append((sid, row, decoded[2], row_idx))

    main_rows = []
    dist_cache = {}

    for exp, cfg in MODELS.items():
        enc, dec, num_tokens, fd, mask_id = cfg['encoder'], cfg['decoder'], cfg['vocab'], cfg['fd'], cfg['mask_id']
        for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
            for sid, base_row, seed_soup, row_idx in seed_info:
                np.random.seed(RNG_SEED)
                soup_c, side_c = Counter(), Counter()
                anchor_keeps = []
                nut_all5, gate_pass, cand_zero, gen_fail = 0, 0, 0, 0
                full_pass_counts = []
                for call in range(N_CALLS):
                    best, fail_log, n_invalid = make_meal_generic(
                        core, F, enc, dec, num_tokens, mask_id, fd, base_row, anchor_menu, b, cfg['soup_masked'])
                    full_pass_counts.append(fail_log['n_full_pass'])
                    if fail_log['n_full_pass'] == 0:
                        cand_zero += 1
                    if best is None:
                        gen_fail += 1
                        continue
                    menus, inst, after, ok, nut_ok = best
                    soup_c[menus[1]] += 1
                    side_c[menus[3]] += 1
                    if nut_ok:
                        nut_all5 += 1
                    unreal2 = F.unrealistic_reason(inst)
                    clash2 = core._has_ingredient_clash(menus)
                    p_over2 = core._has_high_p_overload(menus)
                    if unreal2 is None and not clash2 and not p_over2:
                        gate_pass += 1
                    _, det = R.meal_reward(menus, anchor_menu, detail=True)
                    if det:
                        anchor_keeps.append(det['a_keep'])
                n_ok = N_CALLS - gen_fail
                dist_cache[(exp, cond_name, sid)] = {'soup_c': soup_c, 'side_c': side_c, 'seed_soup': seed_soup}
                main_rows.append({
                    'experiment': exp, 'anchor_condition': cond_name, 'seed_id': sid,
                    'nutrition_all_pass_rate': nut_all5 / n_ok if n_ok else None,
                    'reality_gate_pass_rate': gate_pass / n_ok if n_ok else None,
                    'anchor_preservation_rate': float(np.mean(anchor_keeps)) if anchor_keeps else None,
                    'candidate_zero_rate': cand_zero / N_CALLS,
                    'generation_failure_rate': gen_fail / N_CALLS,
                    'mean_full_pass_candidates': float(np.mean(full_pass_counts)),
                    'final_soup_top1_ratio': soup_c.most_common(1)[0][1] / n_ok if n_ok else None,
                    'final_side_top1_ratio': side_c.most_common(1)[0][1] / n_ok if n_ok else None,
                    'unique_soup_count': len(soup_c), 'unique_side_count': len(side_c),
                    'seed_top1_match': (soup_c.most_common(1)[0][0] == seed_soup) if soup_c else None,
                    'dish_hit': core._m2c.get(soup_c.most_common(1)[0][0]) in ('국', '수프(간식)') if soup_c else None,
                })
        print(f'[{exp}] 완료')

    fieldnames = list(main_rows[0].keys())
    out_csv = os.path.join(OUT_DIR, 'final_A_C_RL_comparison.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(main_rows)
    print(f'\n{out_csv} ({len(main_rows)}행)')

    # 조건별/전체 요약
    print('\n===== 요약 =====')
    for exp in MODELS:
        rs_all = [r for r in main_rows if r['experiment'] == exp]
        print(f'\n--- {exp} (3조건 평균) ---')
        nut = np.mean([r['nutrition_all_pass_rate'] for r in rs_all if r['nutrition_all_pass_rate'] is not None])
        gate = np.mean([r['reality_gate_pass_rate'] for r in rs_all if r['reality_gate_pass_rate'] is not None])
        anc = np.mean([r['anchor_preservation_rate'] for r in rs_all if r['anchor_preservation_rate'] is not None])
        czero = np.mean([r['candidate_zero_rate'] for r in rs_all])
        genfail = np.mean([r['generation_failure_rate'] for r in rs_all])
        uniq_s = np.mean([r['unique_soup_count'] for r in rs_all])
        uniq_side = np.mean([r['unique_side_count'] for r in rs_all])
        match = np.mean([r['seed_top1_match'] for r in rs_all if r['seed_top1_match'] is not None])
        dishhit = np.mean([r['dish_hit'] for r in rs_all if r['dish_hit'] is not None])
        print(f'  영양5종 {nut*100:.1f}%  게이트 {gate*100:.1f}%  앵커보존 {anc:.3f}  '
              f'후보0개율 {czero*100:.1f}%  생성실패율 {genfail*100:.2f}%  '
              f'고유국 {uniq_s:.1f}  고유부찬 {uniq_side:.1f}  seed일치율 {match*100:.1f}%  dish_hit {dishhit*100:.1f}%')
        for cond_name, _, _ in ANCHOR_CONDITIONS:
            rs = [r for r in rs_all if r['anchor_condition'] == cond_name]
            nut_c = np.mean([r['nutrition_all_pass_rate'] for r in rs if r['nutrition_all_pass_rate'] is not None])
            czero_c = np.mean([r['candidate_zero_rate'] for r in rs])
            anc_c = np.mean([r['anchor_preservation_rate'] for r in rs if r['anchor_preservation_rate'] is not None])
            print(f'    {cond_name}: 영양5종 {nut_c*100:.1f}%  후보0개율 {czero_c*100:.1f}%  앵커보존 {anc_c:.3f}')

    # anchor sensitivity
    anchor_rows = []
    vocab_max = 828
    for exp in MODELS:
        for sid, *_ in seed_info:
            dists = {}
            for cond_name, _, _ in ANCHOR_CONDITIONS:
                d = dist_cache[(exp, cond_name, sid)]
                vec = np.zeros(vocab_max)
                for menu, cnt in d['soup_c'].items():
                    idx = core.name2idx.get(menu)
                    if idx is not None:
                        vec[idx] = cnt
                if vec.sum() > 0:
                    vec /= vec.sum()
                dists[cond_name] = vec
            for (ca, _, _), (cb, _, _) in itertools.combinations(ANCHOR_CONDITIONS, 2):
                jsd = float(jensenshannon(dists[ca], dists[cb]))
                anchor_rows.append([exp, sid, ca, cb, jsd])
    print('\n===== 앵커 민감도(JSD 평균) =====')
    for exp in MODELS:
        vals = [r[4] for r in anchor_rows if r[0] == exp]
        print(f'  {exp}: {np.mean(vals):.4f}')

    return main_rows


if __name__ == '__main__':
    main()
