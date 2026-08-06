# -*- coding: utf-8 -*-
"""
verify_soupmask_experiments_FOOK.py — A_baseline/B_dropout50/C_mask100 세 체크포인트를
검증실험1(seed 의존성)+검증실험2(앵커 민감도)+검증실험3(품질/제약 성능)로 비교.

기존 서비스 코드(app_core_FOOK.py 등)는 안 건드림 — F.adjust/passes 등 순수 함수만 재사용.
새 체크포인트는 828토큰(SOUP_MASK 포함) 전용 Encoder/Decoder(train_FOOK_soupmask.py와 동일 구조)로
직접 로딩한다 — app_core_FOOK.gen(827토큰)과는 별개 객체.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python verify_soupmask_experiments_FOOK.py
"""
import os, sys, io, csv, copy, json, itertools
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
from train_FOOK_soupmask import build_data_with_mask_token, apply_soup_mask, SOUP_POS

OUT_DIR = os.path.join(CODE, 'verify_soupmask_out')
RESULTS_ROOT = os.path.join(CODE, 'results_soupmask_FOOK')

EXPERIMENTS = [
    ('A_baseline', 'A_원본', 'A_원본'),      # (ascii_dir, train_variant_label, inference_mode)
    ('B_dropout50', 'B_학습시50%', 'C_항상마스킹'),
    ('C_mask100', 'C_항상마스킹', 'C_항상마스킹'),
]
LS_SUFFIX = 'ls0.0'

ANCHOR_CONDITIONS = [('생선구이', '고등어구이', 55), ('육류', '제육불고기', 152), ('두부콩류', '두부양념조림', 36)]
SEED_ROWS = [11, 12, 6, 36, 7, 18, 21, 35, 28, 53]
N_SAMPLE = 100
TEMP = 0.8
SUBSTITUTE_SOUP = '근대된장국'
SUBSTITUTE_SOUP_ALT = '아욱된장국'


def entropy_of(arr_or_counter):
    if isinstance(arr_or_counter, Counter):
        vals = np.array(list(arr_or_counter.values()), dtype=float)
    else:
        vals = np.asarray(arr_or_counter, dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def load_checkpoint(exp_ascii_dir, num_tokens, embed_dim=128, fc_dim=64):
    ckpt_dir = os.path.join(RESULTS_ROOT, f'{exp_ascii_dir}_{LS_SUFFIX}', 'checkpoints')
    kwargs = {'num_tokens': num_tokens, 'embed_dim': embed_dim, 'fc_dim': fc_dim,
              'fully-connected_layer': 'GRU', 'attention': True}
    encoder = Encoder(**kwargs, batch_size=10)
    decoder = Decoder(**kwargs, batch_size=10)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    if not ck:
        raise RuntimeError(f'체크포인트 없음: {ckpt_dir}')
    tf.train.Checkpoint(encoder=encoder, decoder=decoder).restore(ck).expect_partial()
    return encoder, decoder


def soup_logits(encoder, decoder, seeds_batch):
    """국(slot1) 출력의 raw softmax(마스킹 전). seeds_batch: (n,7) — 이미 실험별 규칙대로
    SOUP_POS가 처리된 상태로 들어옴(마스킹 여부는 호출부 책임)."""
    n = seeds_batch.shape[0]
    enc_hidden0 = tf.zeros([n, encoder.units])
    enc_output, enc_hidden = encoder(seeds_batch, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    _, dec_hidden, _ = decoder(seeds_batch[:, 0], dec_hidden, enc_output)     # j=0(밥) 통과
    outputs, dec_hidden, _ = decoder(seeds_batch[:, 1], dec_hidden, enc_output)  # j=1(국)
    probs = np.array(outputs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    return probs[0].copy()


def masked_soup_dist(core, encoder, decoder, seeds_batch, anchor_token, mask_id, num_tokens):
    raw = soup_logits(encoder, decoder, seeds_batch)
    p = raw.copy()
    for t in core.SPECIAL:
        p[t] = 0.0
    p[mask_id] = 0.0                       # 새 토큰은 국 자체로 뽑히면 안 됨
    for t in core.BLOCK_TOK:
        p[t] = 0.0
    p[anchor_token] = 0.0
    slot_ok = np.append(core.SLOT_OK[1], 0.0)     # 828차원으로 패딩(mask 위치=0)
    masked = p * slot_ok
    anchor_grp = core.TOK_GRP.get(anchor_token)
    if anchor_grp is not None:
        grp_idx = np.array(core.GRP_TOK[anchor_grp])
        masked[grp_idx] = 0.0
    if masked.sum() > 0:
        p = masked
    p = np.clip(p, 1e-12, None)
    p = p ** (1.0 / TEMP)
    p /= p.sum()
    return p


def apply_variant(seed_row, inference_mode, mask_id):
    v = seed_row.copy()
    if inference_mode == 'C_항상마스킹':
        v[SOUP_POS] = mask_id
    return v


def make_substitute_variant(seed_row, mask_id, core):
    orig = core.food_dict[int(seed_row[SOUP_POS])]
    sub = SUBSTITUTE_SOUP_ALT if orig == SUBSTITUTE_SOUP else SUBSTITUTE_SOUP
    v = seed_row.copy()
    v[SOUP_POS] = core.name2idx[sub]
    return v


def make_mask_variant(seed_row, mask_id):
    v = seed_row.copy()
    v[SOUP_POS] = mask_id
    return v


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    os.chdir(cwd)
    b = F.meal_bounds(60)

    nutrient_data, food_dict_new, diet_np, incidence, mask_id = build_data_with_mask_token()
    num_tokens = len(food_dict_new)
    orig_diet_np_np = diet_np.numpy()

    models = {}
    for ascii_dir, train_label, inf_mode in EXPERIMENTS:
        print(f'로딩: {ascii_dir}')
        enc, dec = load_checkpoint(ascii_dir, num_tokens)
        models[ascii_dir] = {'encoder': enc, 'decoder': dec, 'inf_mode': inf_mode, 'train_label': train_label}

    # ── 검증실험1+2 통합: seed 10개 x 앵커3개, A/B/C 각각 ──
    seed_info = []
    for sid, row_idx in enumerate(SEED_ROWS):
        row = orig_diet_np_np[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in row]
        seed_info.append((sid, row, decoded[2], '|'.join(decoded[1:6]), row_idx))

    seed_trace = []
    dist_cache = {}   # (exp, cond, sid) -> {'A_원본':p,'C_교체':p,'C_항상마스킹':p, 'sample_top1':..., ...}

    for ascii_dir, m in models.items():
        enc, dec, inf_mode = m['encoder'], m['decoder'], m['inf_mode']
        for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
            anchor_token = core.name2idx[anchor_menu]
            for sid, base_row, seed_soup, seed_tuple, row_idx in seed_info:
                variants = {}
                # 실제 이 실험(A/B/C)의 "실제 디코더 입력" 규칙에 따른 기본 분포
                actual_input_row = apply_variant(base_row, inf_mode, mask_id)
                actual_input_row = actual_input_row.copy()
                actual_input_row[3] = anchor_token   # 주찬 위치(0idx3) 앵커 고정
                seeds_batch = tf.constant(np.tile(actual_input_row, (10, 1)), dtype=tf.int32)
                p_actual = masked_soup_dist(core, enc, dec, seeds_batch, anchor_token, mask_id, num_tokens)

                order = np.argsort(-p_actual)
                top1_menu = core.food_dict[int(order[0])]
                top1_prob = float(p_actual[order[0]])
                ent = entropy_of(p_actual)

                np.random.seed(11)
                samples = np.random.choice(len(p_actual), size=N_SAMPLE, p=p_actual)
                sc = Counter(core.food_dict[int(t)] for t in samples)

                # 추가: seed 국 교체(C_교체) / 마스킹(C_항상마스킹) 강제 분포도 같이 저장(진단용 비교)
                sub_row = make_substitute_variant(base_row, mask_id, core).copy(); sub_row[3] = anchor_token
                mask_row = make_mask_variant(base_row, mask_id).copy(); mask_row[3] = anchor_token
                p_sub = masked_soup_dist(core, enc, dec, tf.constant(np.tile(sub_row, (10, 1)), dtype=tf.int32),
                                          anchor_token, mask_id, num_tokens)
                p_mask = masked_soup_dist(core, enc, dec, tf.constant(np.tile(mask_row, (10, 1)), dtype=tf.int32),
                                           anchor_token, mask_id, num_tokens)

                dist_cache[(ascii_dir, cond_name, sid)] = {
                    'actual': p_actual, 'sub': p_sub, 'mask': p_mask,
                    'seed_soup': seed_soup, 'anchor_menu': anchor_menu,
                }

                seed_trace.append({
                    'experiment_name': ascii_dir, 'model_stage': 'seq2seq_soupmask',
                    'anchor_condition': cond_name, 'seed_id': sid,
                    'original_seed_soup': seed_soup,
                    'actual_decoder_soup_input': '<SOUP_MASK>' if inf_mode == 'C_항상마스킹' else seed_soup,
                    'top1_soup': top1_menu, 'top1_probability': top1_prob, 'entropy': ent,
                    'seed_top1_match': (seed_soup == top1_menu),
                    'input_variant': inf_mode,
                    'sample_top1_soup': sc.most_common(1)[0][0], 'sample_top1_ratio': sc.most_common(1)[0][1] / N_SAMPLE,
                    'sample_unique_count': len(sc),
                })
        print(f'  [{ascii_dir}] seed/anchor 실험 완료')

    trace_csv = os.path.join(OUT_DIR, 'soup_masking_seed_trace.csv')
    fieldnames = ['experiment_name', 'model_stage', 'anchor_condition', 'seed_id', 'original_seed_soup',
                  'actual_decoder_soup_input', 'top1_soup', 'top1_probability', 'entropy', 'seed_top1_match',
                  'input_variant', 'sample_top1_soup', 'sample_top1_ratio', 'sample_unique_count']
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in seed_trace:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f'\n② {trace_csv} ({len(seed_trace)}행)')

    # ── 검증실험2: 앵커 간 비교 (같은 seed, 다른 앵커) ──
    anchor_comp_rows = []
    for ascii_dir in models:
        for sid, *_ in seed_info:
            for (cond_a, anchor_a, _), (cond_b, anchor_b, _) in itertools.combinations(ANCHOR_CONDITIONS, 2):
                da = dist_cache[(ascii_dir, cond_a, sid)]
                db = dist_cache[(ascii_dir, cond_b, sid)]
                jsd = float(jensenshannon(da['actual'], db['actual']))
                top1_a = int(np.argmax(da['actual'])); top1_b = int(np.argmax(db['actual']))
                anchor_comp_rows.append([ascii_dir, sid, anchor_a, anchor_b, jsd,
                                          top1_a != top1_b, core.food_dict[top1_a], core.food_dict[top1_b]])
    comp_csv = os.path.join(OUT_DIR, 'soup_masking_anchor_comparison.csv')
    with open(comp_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['experiment_name', 'context_id', 'anchor_a', 'anchor_b', 'js_divergence',
                    'top1_changed', 'anchor_a_top1', 'anchor_b_top1'])
        w.writerows(anchor_comp_rows)
    print(f'③ {comp_csv} ({len(anchor_comp_rows)}행)')

    # ── 검증실험3: 품질/제약 성능 (단순화: generate 1회 + F.adjust 1회, 48후보 없음 — 알려진 근사) ──
    quality_rows = []
    for ascii_dir, m in models.items():
        enc, dec, inf_mode = m['encoder'], m['decoder'], m['inf_mode']
        for cond_name, anchor_menu, row_idx in ANCHOR_CONDITIONS:
            anchor_token = core.name2idx[anchor_menu]
            base_row = orig_diet_np_np[row_idx].copy()
            input_row = apply_variant(base_row, inf_mode, mask_id).copy()
            input_row[3] = anchor_token
            seeds_batch = tf.constant(np.tile(input_row, (N_SAMPLE, 1)), dtype=tf.int32)

            n = N_SAMPLE
            enc_hidden0 = tf.zeros([n, enc.units])
            enc_output, enc_hidden = enc(seeds_batch, enc_hidden0)
            dec_hidden = copy.deepcopy(enc_hidden)
            res = np.zeros((n, 5), dtype=int)
            np.random.seed(11)
            for j in range(5):
                outputs, dec_hidden, _ = dec(seeds_batch[:, j], dec_hidden, enc_output)
                probs = np.array(outputs, dtype=float)
                if probs.ndim == 1:
                    probs = probs[None, :]
                for i in range(n):
                    if j == 2:
                        res[i, j] = anchor_token
                        continue
                    p = probs[i].copy()
                    for t in core.SPECIAL:
                        p[t] = 0.0
                    p[mask_id] = 0.0
                    p = np.clip(p, 1e-12, None)
                    p /= p.sum()
                    res[i, j] = int(np.random.choice(len(p), p=p))

            dish_hits, anchor_keeps, all5, gate_pass = 0, [], 0, 0
            for i in range(n):
                menus = [core.food_dict[int(t)] for t in res[i]]
                cls = core._m2c.get(menus[1])
                if cls in ('국', '수프(간식)'):
                    dish_hits += 1
                _, after, inst, _ = F.adjust(list(menus), b, anchor=anchor_menu)
                unreal = F.unrealistic_reason(inst)
                clash = core._has_ingredient_clash(menus)
                p_over = core._has_high_p_overload(menus)
                gate_ok = (unreal is None) and (not clash) and (not p_over)
                if gate_ok:
                    gate_pass += 1
                nut_ok = (b['Elo'] <= after['E'] <= b['Ehi'] and b['Plo'] <= after['protein'] <= b['Phi']
                          and after['K'] < b['Kmax'] and after['P'] < b['Pmax'] and after['Na_season'] <= b['Namax'])
                if nut_ok:
                    all5 += 1
                # 앵커 보존율(재료 단위 근사): a_keep via reward_lever 재사용
                import reward_lever_FOOK as R
                if not hasattr(R, '_inited'):
                    R.init(weight=60); R._inited = True
                _, det = R.meal_reward(menus, anchor_menu, detail=True)
                if det:
                    anchor_keeps.append(det['a_keep'])

            quality_rows.append({
                'experiment_name': ascii_dir, 'anchor_condition': cond_name,
                'dish_hit_soup': dish_hits / n, 'anchor_preservation': float(np.mean(anchor_keeps)) if anchor_keeps else None,
                'nutrition_all_pass_rate': all5 / n, 'gate_pass_rate': gate_pass / n, 'n': n,
            })
        print(f'  [{ascii_dir}] 품질 검증 완료')

    # ── ① soup_masking_experiment_summary.csv ──
    summary_rows = []
    for ascii_dir, m in models.items():
        meta_path = os.path.join(RESULTS_ROOT, f'{ascii_dir}_{LS_SUFFIX}', 'meta.json')
        meta = json.load(open(meta_path, encoding='utf-8')) if os.path.exists(meta_path) else {}

        exp_trace = [r for r in seed_trace if r['experiment_name'] == ascii_dir]
        copy_rate = np.mean([r['seed_top1_match'] for r in exp_trace])
        mean_top1 = np.mean([r['top1_probability'] for r in exp_trace])
        mean_ent = np.mean([r['entropy'] for r in exp_trace])

        # seed간 JSD(같은 앵커 내에서 seed 10개 pairwise, 3앵커 평균)
        seed_jsds = []
        for cond_name, _, _ in ANCHOR_CONDITIONS:
            dists = [dist_cache[(ascii_dir, cond_name, sid)]['actual'] for sid, *_ in seed_info]
            for i, j in itertools.combinations(range(len(dists)), 2):
                seed_jsds.append(jensenshannon(dists[i], dists[j]))
        mean_seed_jsd = float(np.mean(seed_jsds))

        anchor_jsds = [r[4] for r in anchor_comp_rows if r[0] == ascii_dir]
        mean_anchor_jsd = float(np.mean(anchor_jsds))

        q = [r for r in quality_rows if r['experiment_name'] == ascii_dir]
        dish_hit = np.mean([r['dish_hit_soup'] for r in q])
        anchor_pres = np.mean([r['anchor_preservation'] for r in q if r['anchor_preservation'] is not None])
        nut_pass = np.mean([r['nutrition_all_pass_rate'] for r in q])
        gate_pass = np.mean([r['gate_pass_rate'] for r in q])

        summary_rows.append({
            'experiment_name': ascii_dir, 'mask_probability': {'A_baseline': 0.0, 'B_dropout50': 0.5,
                                                                 'C_mask100': 1.0}[ascii_dir],
            'label_smoothing': 0.0, 'validation_loss': meta.get('best_val_loss'),
            'seed_soup_copy_rate': copy_rate, 'mean_top1_probability': mean_top1, 'mean_entropy': mean_ent,
            'mean_seed_jsd': mean_seed_jsd, 'anchor_jsd': mean_anchor_jsd,
            'dish_hit_soup': dish_hit, 'anchor_preservation': anchor_pres,
            'nutrition_all_pass_rate': nut_pass, 'full_gate_pass_rate': gate_pass,
        })

    summary_csv = os.path.join(OUT_DIR, 'soup_masking_experiment_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f'① {summary_csv}')

    # ── ④ soup_masking_before_after.csv ──
    ba_csv = os.path.join(OUT_DIR, 'soup_masking_before_after.csv')
    metrics = ['validation_loss', 'seed_soup_copy_rate', 'mean_top1_probability', 'mean_entropy',
               'mean_seed_jsd', 'anchor_jsd', 'dish_hit_soup', 'anchor_preservation',
               'nutrition_all_pass_rate', 'full_gate_pass_rate']
    by_name = {r['experiment_name']: r for r in summary_rows}
    with open(ba_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'A_baseline', 'B_dropout50', 'C_mask100', 'C_minus_A', 'B_minus_A'])
        for met in metrics:
            a, b_, c = by_name['A_baseline'][met], by_name['B_dropout50'][met], by_name['C_mask100'][met]
            w.writerow([met, a, b_, c, (c - a) if (c is not None and a is not None) else '',
                        (b_ - a) if (b_ is not None and a is not None) else ''])
    print(f'④ {ba_csv}')

    return summary_rows


if __name__ == '__main__':
    main()
