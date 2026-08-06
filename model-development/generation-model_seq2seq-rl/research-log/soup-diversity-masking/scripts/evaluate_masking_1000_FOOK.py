# -*- coding: utf-8 -*-
"""
evaluate_masking_1000_FOOK.py — A_baseline/B_mask50/C_mask100(1000epoch, 전체데이터, seed고정)
세 체크포인트를 seed의존성+앵커민감도+슬롯적합성+전체파이프라인 성능으로 비교.

기존 서비스 코드(app_core_FOOK.py)는 안 건드림 — F.adjust/passes/마스킹 배열만 재사용.
A는 827vocab(mask 토큰 없음 — 원본과 완전 동일 구조), B/C는 828vocab.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python evaluate_masking_1000_FOOK.py
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

OUT_DIR = os.path.join(CODE, 'evaluate_masking_1000_out')
CKPT_ROOT = os.path.join(CODE, 'checkpoints_masking_1000')

EXPERIMENTS = ['A_baseline', 'B_mask50', 'C_mask100']
INFER_MASK = {'A_baseline': 'none', 'B_mask50': 'p100', 'C_mask100': 'p100'}  # 추론시 규칙

ANCHOR_CONDITIONS = [('생선구이', '고등어구이', 55), ('육류', '제육불고기', 152), ('두부콩류', '두부양념조림', 36)]
SEED_ROWS = [11, 12, 6, 36, 7, 18, 21, 35, 28, 53]
N_SAMPLE = 200
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


def load_model(exp_name, num_tokens, embed_dim=128, fc_dim=64):
    ckpt_dir = os.path.join(CKPT_ROOT, exp_name, 'checkpoints')
    kwargs = {'num_tokens': num_tokens, 'embed_dim': embed_dim, 'fc_dim': fc_dim,
              'fully-connected_layer': 'GRU', 'attention': True}
    enc = Encoder(**kwargs, batch_size=10)
    dec = Decoder(**kwargs, batch_size=10)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    if not ck:
        raise RuntimeError(f'체크포인트 없음: {ckpt_dir}')
    tf.train.Checkpoint(encoder=enc, decoder=dec).restore(ck).expect_partial()
    return enc, dec


def soup_raw_logits(enc, dec, seeds_batch):
    n = seeds_batch.shape[0]
    enc_hidden0 = tf.zeros([n, enc.units])
    enc_output, enc_hidden = enc(seeds_batch, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    _, dec_hidden, _ = dec(seeds_batch[:, 0], dec_hidden, enc_output)
    outputs, dec_hidden, _ = dec(seeds_batch[:, 1], dec_hidden, enc_output)
    probs = np.array(outputs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    return probs[0].copy()


def masked_soup_dist(core, enc, dec, seeds_batch, anchor_token, mask_id, num_tokens):
    raw = soup_raw_logits(enc, dec, seeds_batch)
    p = raw.copy()
    for t in core.SPECIAL:
        if t < num_tokens:
            p[t] = 0.0
    if mask_id is not None:
        p[mask_id] = 0.0
    for t in core.BLOCK_TOK:
        if t < num_tokens:
            p[t] = 0.0
    p[anchor_token] = 0.0
    slot_ok = core.SLOT_OK[1]
    if num_tokens > len(slot_ok):
        slot_ok = np.append(slot_ok, 0.0)
    masked = p * slot_ok
    anchor_grp = core.TOK_GRP.get(anchor_token)
    if anchor_grp is not None:
        grp_idx = np.array([g for g in core.GRP_TOK[anchor_grp] if g < num_tokens])
        if len(grp_idx):
            masked[grp_idx] = 0.0
    if masked.sum() > 0:
        p = masked
    p = np.clip(p, 1e-12, None)
    p = p ** (1.0 / TEMP)
    p /= p.sum()
    return p


def make_input_row(base_row, anchor_token, variant, mask_id, core):
    """variant: 'original' / 'replaced' / 'masked'. B/C의 실제 encoder soup 위치는
    inf_mask_mode(항상 마스킹)에 의해 이미 결정되므로, 'original'/'replaced' 둘 다
    B/C에서는 실제로 <SOUP_MASK>로 귀결된다(별도 플래그로 기록)."""
    row = base_row.copy()
    row[3] = anchor_token
    if variant == 'replaced':
        orig = core.food_dict[int(row[SOUP_POS])]
        sub = SUBSTITUTE_SOUP_ALT if orig == SUBSTITUTE_SOUP else SUBSTITUTE_SOUP
        row[SOUP_POS] = core.name2idx[sub]
    elif variant == 'masked':
        if mask_id is not None:
            row[SOUP_POS] = mask_id
    return row


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

    models = {}
    for exp in EXPERIMENTS:
        with_mask = exp != 'A_baseline'
        num_tokens = len(food_dict_828) if with_mask else len(food_dict_827)
        fd = food_dict_828 if with_mask else food_dict_827
        enc, dec = load_model(exp, num_tokens)
        models[exp] = {'encoder': enc, 'decoder': dec, 'num_tokens': num_tokens,
                        'food_dict': fd, 'mask_id': mask_id828 if with_mask else None,
                        'infer_mask': INFER_MASK[exp]}
        print(f'로딩: {exp} (vocab={num_tokens})')

    seed_info = []
    for sid, row_idx in enumerate(SEED_ROWS):
        row = orig_diet_np_np[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in row]
        seed_info.append((sid, row, decoded[2], row_idx))

    # ── 검증1+슬롯적합성: seed 10 x 앵커3 x 입력변형(A는 original/replaced만, B/C는 셋다) ──
    seed_rows_out = []
    dist_cache = {}   # (exp, cond, sid, variant) -> dist array

    for exp, m in models.items():
        enc, dec, num_tokens, fd, mask_id = m['encoder'], m['decoder'], m['num_tokens'], m['food_dict'], m['mask_id']
        variants = ['original', 'replaced'] if exp == 'A_baseline' else ['original', 'replaced', 'masked']
        for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
            anchor_token = core.name2idx[anchor_menu]
            for sid, base_row, seed_soup, row_idx in seed_info:
                for variant in variants:
                    row = make_input_row(base_row, anchor_token, variant, mask_id, core)
                    seeds_batch = tf.constant(np.tile(row, (10, 1)), dtype=tf.int32)
                    p = masked_soup_dist(core, enc, dec, seeds_batch, anchor_token, mask_id, num_tokens)
                    order = np.argsort(-p)
                    top1_idx = int(order[0])
                    top1_menu = fd[top1_idx]
                    top1_prob = float(p[top1_idx])
                    ent = entropy_of(p)
                    dist_cache[(exp, cond_name, sid, variant)] = p

                    np.random.seed(11)
                    samples = np.random.choice(len(p), size=N_SAMPLE, p=p)
                    sc = Counter(fd[int(t)] for t in samples)

                    cls = core._m2c.get(top1_menu)
                    dish_hit = cls in ('국', '수프(간식)')
                    top10_menus = [fd[int(i)] for i in order[:10]]
                    top10_soup_ratio = sum(1 for mm in top10_menus if core._m2c.get(mm) in ('국', '수프(간식)')) / 10

                    actual_input = '<SOUP_MASK>' if (exp != 'A_baseline' and m['infer_mask'] == 'p100') else \
                                   (fd[int(row[SOUP_POS])] if row[SOUP_POS] < len(fd) else '<SOUP_MASK>')

                    seed_rows_out.append({
                        'experiment': exp, 'anchor_condition': cond_name, 'seed_id': sid,
                        'input_variant': variant, 'original_seed_soup': seed_soup,
                        'actual_encoder_soup_token': actual_input,
                        'top1_soup': top1_menu, 'top1_probability': top1_prob, 'entropy': ent,
                        'seed_top1_match': (seed_soup == top1_menu),
                        'dish_hit': dish_hit, 'top10_soup_class_ratio': top10_soup_ratio,
                        'sample_top1_soup': sc.most_common(1)[0][0], 'sample_top1_ratio': sc.most_common(1)[0][1] / N_SAMPLE,
                        'sample_unique_count': len(sc),
                    })
        print(f'  [{exp}] seed 실험 완료')

    # replacement_top1_changed / replacement_jsd (original vs replaced, 같은 exp/cond/sid)
    for r in seed_rows_out:
        if r['input_variant'] == 'original':
            key_rep = (r['experiment'], r['anchor_condition'], r['seed_id'], 'replaced')
            key_orig = (r['experiment'], r['anchor_condition'], r['seed_id'], 'original')
            if key_rep in dist_cache:
                p_o, p_r = dist_cache[key_orig], dist_cache[key_rep]
                jsd = float(jensenshannon(p_o, p_r))
                top1_o = np.argmax(p_o); top1_r = np.argmax(p_r)
                r['replacement_jsd'] = jsd
                r['replacement_top1_changed'] = bool(top1_o != top1_r)
        else:
            r['replacement_jsd'] = ''
            r['replacement_top1_changed'] = ''

    trace_csv = os.path.join(OUT_DIR, 'masking_1000_seed_dependency.csv')
    fieldnames = ['experiment', 'anchor_condition', 'seed_id', 'input_variant', 'original_seed_soup',
                  'actual_encoder_soup_token', 'top1_soup', 'top1_probability', 'entropy', 'seed_top1_match',
                  'replacement_top1_changed', 'replacement_jsd', 'dish_hit', 'top10_soup_class_ratio',
                  'sample_top1_soup', 'sample_top1_ratio', 'sample_unique_count']
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in seed_rows_out:
            w.writerow({k: r.get(k, '') for k in fieldnames})
    print(f'\n② {trace_csv} ({len(seed_rows_out)}행)')

    # ── 검증2: 앵커 민감도 (같은 seed, original variant, 앵커 3개 pairwise) ──
    anchor_rows = []
    for exp in EXPERIMENTS:
        for sid, *_ in seed_info:
            for (cond_a, anchor_a, _), (cond_b, anchor_b, _) in itertools.combinations(ANCHOR_CONDITIONS, 2):
                pa = dist_cache.get((exp, cond_a, sid, 'original'))
                pb = dist_cache.get((exp, cond_b, sid, 'original'))
                if pa is None or pb is None:
                    continue
                jsd = float(jensenshannon(pa, pb))
                changed = int(np.argmax(pa)) != int(np.argmax(pb))
                anchor_rows.append([exp, sid, anchor_a, anchor_b, jsd, changed])

    # ── ① summary 집계 준비 ──
    summary = {}
    for exp, m in models.items():
        rs = [r for r in seed_rows_out if r['experiment'] == exp]
        orig_rs = [r for r in rs if r['input_variant'] == 'original']
        copy_rate = np.mean([r['seed_top1_match'] for r in orig_rs])
        mean_top1 = np.mean([r['top1_probability'] for r in orig_rs])
        mean_ent = np.mean([r['entropy'] for r in orig_rs])
        uniq_top1 = len({r['top1_soup'] for r in orig_rs})
        dish_hit = np.mean([r['dish_hit'] for r in orig_rs])
        non_soup_top1 = 1 - dish_hit

        rep_rs = [r for r in rs if r['input_variant'] == 'original' and r.get('replacement_jsd') != '']
        mean_rep_jsd = np.mean([r['replacement_jsd'] for r in rep_rs]) if rep_rs else None

        exp_anchor_rows = [r for r in anchor_rows if r[0] == exp]
        anchor_jsd = np.mean([r[4] for r in exp_anchor_rows])
        anchor_change_rate = np.mean([r[5] for r in exp_anchor_rows])

        top5_conc = []
        for cond_name, _, _ in ANCHOR_CONDITIONS:
            for sid, *_ in seed_info:
                p = dist_cache.get((exp, cond_name, sid, 'original'))
                if p is not None:
                    order = np.argsort(-p)
                    top5_conc.append(float(p[order[:5]].sum()))
        mean_top5 = np.mean(top5_conc)

        summary[exp] = {
            'experiment': exp, 'seed_copy_rate': copy_rate, 'mean_top1_probability': mean_top1,
            'top5_concentration': mean_top5, 'mean_entropy': mean_ent,
            'mean_seed_replacement_jsd': mean_rep_jsd, 'unique_top1_count': uniq_top1,
            'anchor_jsd': anchor_jsd, 'anchor_top1_change_rate': anchor_change_rate,
            'soup_dish_hit': dish_hit, 'non_soup_top1_rate': non_soup_top1,
            'training_distribution_jsd': None,   # 아래서 채움
        }

    # 학습데이터 분포와 JSD (training_main_soup_distribution.csv 재사용)
    train_dist_csv = os.path.join(CODE, 'diagnose_soup_training_bias_out', 'training_main_soup_distribution.csv')
    train_dist = {}
    if os.path.exists(train_dist_csv):
        with open(train_dist_csv, encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                if row['match_type'] == '정확메뉴':
                    train_dist.setdefault(row['anchor_menu_또는_유형군'], Counter())[row['soup_menu']] = int(row['cooccurrence_count'])
    for exp in EXPERIMENTS:
        jsds = []
        for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
            c = train_dist.get(anchor_menu)
            if not c:
                continue
            data_vec = np.zeros(len(core.food_dict))
            for soup, cnt in c.items():
                if soup in core.name2idx:
                    data_vec[core.name2idx[soup]] = cnt
            if data_vec.sum() == 0:
                continue
            data_vec /= data_vec.sum()
            for sid, *_ in seed_info:
                p = dist_cache.get((exp, cond_name, sid, 'original'))
                if p is None:
                    continue
                p827 = p[:len(core.food_dict)]
                if p827.sum() > 0:
                    p827 = p827 / p827.sum()
                    jsds.append(float(jensenshannon(data_vec, p827)))
        summary[exp]['training_distribution_jsd'] = float(np.mean(jsds)) if jsds else None

    summary_csv = os.path.join(OUT_DIR, 'masking_1000_summary.csv')
    fieldnames_s = list(next(iter(summary.values())).keys())
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_s)
        w.writeheader()
        for exp in EXPERIMENTS:
            w.writerow(summary[exp])
    print(f'① {summary_csv}')

    anchor_csv = os.path.join(OUT_DIR, 'masking_1000_anchor_comparison.csv')
    with open(anchor_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'seed_id', 'anchor_a', 'anchor_b', 'js_divergence', 'top1_changed'])
        w.writerows(anchor_rows)
    print(f'   {anchor_csv}')

    # ── 검증5: 전체 파이프라인(make_meal 유사, 48후보) ──
    pipeline_rows = []
    for exp, m in models.items():
        enc, dec, num_tokens, fd, mask_id = m['encoder'], m['decoder'], m['num_tokens'], m['food_dict'], m['mask_id']
        infer_mode = m['infer_mask']
        for cond_name, anchor_menu, row_idx in ANCHOR_CONDITIONS:
            anchor_token = core.name2idx[anchor_menu]
            base_row = orig_diet_np_np[row_idx].copy()
            input_row = base_row.copy(); input_row[3] = anchor_token
            if infer_mode == 'p100' and mask_id is not None:
                input_row[SOUP_POS] = mask_id
            TRIES = 48
            n_calls = N_SAMPLE
            full_pass_counts, anchor_keeps, all5, gate_pass, gen_fail = [], [], 0, 0, 0
            soup_c, side_c = Counter(), Counter()
            np.random.seed(11)
            for call in range(n_calls):
                seeds_batch = tf.constant(np.tile(input_row, (TRIES, 1)), dtype=tf.int32)
                n = TRIES
                enc_hidden0 = tf.zeros([n, enc.units])
                enc_output, enc_hidden = enc(seeds_batch, enc_hidden0)
                dec_hidden = copy.deepcopy(enc_hidden)
                res = np.zeros((n, 5), dtype=int)
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
                            if t < num_tokens:
                                p[t] = 0.0
                        if mask_id is not None:
                            p[mask_id] = 0.0
                        p = np.clip(p, 1e-12, None)
                        p /= p.sum()
                        res[i, j] = int(np.random.choice(len(p), p=p))

                best_ok, best_score, n_pass = None, -1, 0
                for i in range(n):
                    menus = [fd[int(t)] for t in res[i]]
                    if any('SOUP_MASK' in mm or mm is None for mm in menus):
                        continue
                    _, after, inst, _ = F.adjust(list(menus), b, anchor=anchor_menu)
                    unreal = F.unrealistic_reason(inst)
                    clash = core._has_ingredient_clash(menus)
                    p_over = core._has_high_p_overload(menus)
                    nut_ok = (b['Elo'] <= after['E'] <= b['Ehi'] and b['Plo'] <= after['protein'] <= b['Phi']
                              and after['K'] < b['Kmax'] and after['P'] < b['Pmax'] and after['Na_season'] <= b['Namax'])
                    ok = nut_ok and unreal is None and not clash and not p_over
                    if ok:
                        n_pass += 1
                    score = sum([b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                                 after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax']])
                    if unreal is None: score += 0.5
                    if not clash: score += 0.3
                    if not p_over: score += 0.3
                    if ok and best_ok is None:
                        best_ok = (menus, after, inst)
                        if score > best_score:
                            best_score = score
                    elif best_ok is None and score > best_score:
                        best_score = score
                        best_fallback = (menus, after, inst)
                selected = best_ok if best_ok is not None else (best_fallback if n_pass == 0 else None)
                full_pass_counts.append(n_pass)
                if selected is None:
                    gen_fail += 1
                    continue
                menus_sel, after_sel, inst_sel = selected
                soup_c[menus_sel[1]] += 1
                side_c[menus_sel[3]] += 1
                _, det = R.meal_reward(menus_sel, anchor_menu, detail=True)
                if det:
                    anchor_keeps.append(det['a_keep'])
                nut_ok2 = (b['Elo'] <= after_sel['E'] <= b['Ehi'] and b['Plo'] <= after_sel['protein'] <= b['Phi']
                           and after_sel['K'] < b['Kmax'] and after_sel['P'] < b['Pmax'] and after_sel['Na_season'] <= b['Namax'])
                if nut_ok2:
                    all5 += 1
                unreal2 = F.unrealistic_reason(inst_sel)
                clash2 = core._has_ingredient_clash(menus_sel)
                p_over2 = core._has_high_p_overload(menus_sel)
                if unreal2 is None and not clash2 and not p_over2:
                    gate_pass += 1

            n_ok = n_calls - gen_fail
            soup_top1_ratio = soup_c.most_common(1)[0][1] / n_ok if n_ok else 0
            side_top1_ratio = side_c.most_common(1)[0][1] / n_ok if n_ok else 0
            combos = set()
            pipeline_rows.append({
                'experiment': exp, 'anchor_condition': cond_name,
                'anchor_preservation_rate': float(np.mean(anchor_keeps)) if anchor_keeps else None,
                'nutrition_all_pass_rate': all5 / n_ok if n_ok else None,
                'reality_gate_pass_rate': gate_pass / n_ok if n_ok else None,
                'mean_full_pass_candidates': float(np.mean(full_pass_counts)),
                'generation_failure_rate': gen_fail / n_calls,
                'final_soup_top1_ratio': soup_top1_ratio, 'final_side_top1_ratio': side_top1_ratio,
                'unique_soup_count': len(soup_c), 'unique_side_count': len(side_c),
            })
        print(f'  [{exp}] 파이프라인 검증 완료')

    pipeline_csv = os.path.join(OUT_DIR, 'masking_1000_pipeline_metrics.csv')
    fieldnames_p = list(pipeline_rows[0].keys())
    with open(pipeline_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_p)
        w.writeheader()
        w.writerows(pipeline_rows)
    print(f'④ {pipeline_csv}')

    return summary, pipeline_rows


if __name__ == '__main__':
    main()
