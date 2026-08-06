# -*- coding: utf-8 -*-
"""
stage1_full_make_meal_verify_FOOK.py — A_baseline vs C_mask100을 실제 make_meal() 전체
조건(dup_today·clash·seafood_overload·high_p_overload·unreal·passes·첫통과or부분점수최고선택)으로
재검증. app_core_FOOK.make_meal()/gen_batch()를 그대로 복제하되 encoder/decoder/vocab만
바꿔 끼울 수 있게 만들었다(원본 코드는 안 건드림).

C_mask100은 encoder 입력의 국 위치를 항상 SOUP_MASK로 강제한다(추론 시에도 유지 — 학습 때와
동일 조건). A_baseline은 원본 그대로(마스킹 없음, 827vocab).

조건: 기존 진단들과 동일 — 주찬 앵커 3개(고등어구이/제육불고기/두부양념조림) x seed 10개 x
seed당 200회. seed 식단(encoder 컨텍스트)은 diet_np를 그 1개 행으로 바꿔치기해 고정한다
(gen_batch 내부의 랜덤 seed 추첨이 항상 그 행을 뽑게 만듦 — 이전 진단과 동일한 방식,
"seed 10개 x 앵커 3개" 통제 조건을 유지하기 위함).

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python stage1_full_make_meal_verify_FOOK.py
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

OUT_DIR = os.path.join(CODE, 'stage1_full_make_meal_out')
CKPT_ROOT = os.path.join(CODE, 'checkpoints_masking_1000')

ANCHOR_CONDITIONS = [('생선구이', '고등어구이', 55), ('육류', '제육불고기', 152), ('두부콩류', '두부양념조림', 36)]
SEED_ROWS = [11, 12, 6, 36, 7, 18, 21, 35, 28, 53]
N_CALLS = 200
TRIES = 48
TEMP = 0.8
RNG_SEED = 11


def entropy_of(counter):
    vals = np.array(list(counter.values()), dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def load_model(exp_name, num_tokens):
    ckpt_dir = os.path.join(CKPT_ROOT, exp_name, 'checkpoints')
    kwargs = {'num_tokens': num_tokens, 'embed_dim': 128, 'fc_dim': 64,
              'fully-connected_layer': 'GRU', 'attention': True}
    enc = Encoder(**kwargs, batch_size=10)
    dec = Decoder(**kwargs, batch_size=10)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    tf.train.Checkpoint(encoder=enc, decoder=dec).restore(ck).expect_partial()
    return enc, dec


def gen_batch_generic(core, encoder, decoder, num_tokens, mask_id, food_dict,
                       fixed_seed_row_7tok, anchor_token, n, temp, soup_always_masked):
    """app_core_FOOK.gen_batch()를 그대로 복제(seed는 고정 1개 행을 n번 반복 — 통제조건 유지)."""
    seeds = np.tile(fixed_seed_row_7tok, (n, 1)).astype(np.int64)
    if soup_always_masked and mask_id is not None:
        seeds[:, SOUP_POS] = mask_id
    fixed = {2: anchor_token}          # 슬롯2(주찬)=앵커
    seeds[:, 3] = anchor_token          # 위치3 = 슬롯2+1

    seeds_tf = tf.constant(seeds, dtype=tf.int32)
    enc_hidden0 = tf.zeros([n, encoder.units])
    enc_output, enc_hidden = encoder(seeds_tf, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int)
    res[:, 0] = seeds[:, 0]
    res[:, -1] = 826
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
                if t < num_tokens:
                    p[t] = 0.0
            if mask_id is not None:
                p[mask_id] = 0.0
            for t in core.BLOCK_TOK:
                if t < num_tokens:
                    p[t] = 0.0
            for t in used[bi]:
                if t < len(p):
                    p[t] = 0.0
            slot_ok = core.SLOT_OK[j]
            if num_tokens > len(slot_ok):
                slot_ok = np.append(slot_ok, np.zeros(num_tokens - len(slot_ok)))
            masked = p * slot_ok
            for gi in used_grp[bi]:
                idx = np.array([g for g in core.GRP_TOK[gi] if g < num_tokens])
                if len(idx):
                    masked[idx] = 0.0
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
    return [[food_dict[int(t)] for t in r if int(t) not in core.SPECIAL and t != mask_id] for r in res]


def make_meal_generic(core, F, encoder, decoder, num_tokens, mask_id, food_dict,
                       fixed_seed_row_7tok, anchor_menu, b, soup_always_masked, tries=TRIES, temp=TEMP):
    """app_core_FOOK.make_meal() 그대로 복제. used_today는 항상 빈 집합(단일 끼 평가,
    하루 이어짓기 아님) — dup_today는 이 조건에서 항상 False가 정상."""
    anchor_token = core.name2idx[anchor_menu]
    used_today = set()
    best, best_score = None, -1
    fail_log = {'dup_today': 0, 'clash': 0, 'seafood_overload': 0, 'high_p_overload': 0,
                'unrealistic': 0, 'nutrition_fail': 0, 'n_candidates': 0, 'n_full_pass': 0}

    menus_batch = gen_batch_generic(core, encoder, decoder, num_tokens, mask_id, food_dict,
                                     fixed_seed_row_7tok, anchor_token, tries, temp, soup_always_masked)
    for menus in menus_batch:
        if len(menus) != 5:
            continue   # 특수토큰이 섞여 슬롯이 덜 찬 비정상 후보(비정상 메뉴 출력 카운트용, 아래서 별도 집계)
        fail_log['n_candidates'] += 1
        dup_today = any(m in used_today and m != anchor_menu and not core._is_staple_menu(m) for m in menus)
        clash = core._has_ingredient_clash(menus)
        overload = core._has_seafood_overload(menus)
        p_overload = core._has_high_p_overload(menus)
        _, after, inst, _ = F.adjust(list(menus), b, anchor=anchor_menu)
        unreal = F.unrealistic_reason(inst)
        nut_ok = (b['Elo'] <= after['E'] <= b['Ehi'] and b['Plo'] <= after['protein'] <= b['Phi']
                  and after['K'] < b['Kmax'] and after['P'] < b['Pmax'] and after['Na_season'] <= b['Namax'])
        ok = nut_ok and unreal is None and not dup_today and not clash and not overload and not p_overload
        if dup_today: fail_log['dup_today'] += 1
        if clash: fail_log['clash'] += 1
        if overload: fail_log['seafood_overload'] += 1
        if p_overload: fail_log['high_p_overload'] += 1
        if unreal is not None: fail_log['unrealistic'] += 1
        if not nut_ok: fail_log['nutrition_fail'] += 1
        cand = (menus, inst, after, ok, nut_ok)
        if ok:
            fail_log['n_full_pass'] += 1
            if best is None or best[3] is not True:
                best = cand
        score = sum([b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                     after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax']])
        if unreal is None: score += 0.5
        if not dup_today: score += 0.3
        if not clash: score += 0.3
        if not overload: score += 0.3
        if not p_overload: score += 0.3
        if best is None and score > best_score:
            best_score = score
            best = cand
    n_invalid = tries - fail_log['n_candidates']
    return best, fail_log, n_invalid


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

    MODELS = {
        'A_baseline': {'vocab': 827, 'fd': food_dict_827, 'mask_id': None, 'soup_masked': False},
        'C_mask100': {'vocab': 828, 'fd': food_dict_828, 'mask_id': mask_id828, 'soup_masked': True},
    }
    for exp, cfg in MODELS.items():
        enc, dec = load_model(exp, cfg['vocab'])
        cfg['encoder'], cfg['decoder'] = enc, dec
        print(f'로딩: {exp} (vocab={cfg["vocab"]})')

    seed_info = []
    for sid, row_idx in enumerate(SEED_ROWS):
        row = orig_diet_np_np[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in row]
        seed_info.append((sid, row, decoded[2], row_idx))

    main_rows = []       # full_make_meal_A_vs_C.csv (call 단위)
    gate_fail_rows = []  # full_make_meal_gate_failures.csv (조건 단위 집계)
    dist_cache = {}      # (exp, cond, sid) -> soup Counter / side Counter / anchor_keep list 등

    for exp, cfg in MODELS.items():
        enc, dec, num_tokens, fd, mask_id = cfg['encoder'], cfg['decoder'], cfg['vocab'], cfg['fd'], cfg['mask_id']
        for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
            for sid, base_row, seed_soup, row_idx in seed_info:
                np.random.seed(RNG_SEED)
                soup_c, side_c = Counter(), Counter()
                anchor_keeps = []
                nut_all5, gate_pass, cand_zero, gen_fail = 0, 0, 0, 0
                full_pass_counts = []
                agg_fail = {'dup_today': 0, 'clash': 0, 'seafood_overload': 0, 'high_p_overload': 0,
                            'unrealistic': 0, 'nutrition_fail': 0}
                invalid_total = 0
                for call in range(N_CALLS):
                    best, fail_log, n_invalid = make_meal_generic(
                        core, F, enc, dec, num_tokens, mask_id, fd, base_row, anchor_menu, b,
                        cfg['soup_masked'])
                    full_pass_counts.append(fail_log['n_full_pass'])
                    invalid_total += n_invalid
                    for k in agg_fail:
                        agg_fail[k] += fail_log[k]
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
                dist_cache[(exp, cond_name, sid)] = {
                    'soup_c': soup_c, 'side_c': side_c, 'anchor_keeps': anchor_keeps,
                    'seed_soup': seed_soup,
                }
                main_rows.append({
                    'experiment': exp, 'anchor_condition': cond_name, 'anchor_menu': anchor_menu,
                    'seed_id': sid, 'seed_row_idx': row_idx, 'seed_soup': seed_soup,
                    'n_calls': N_CALLS, 'generation_failure_count': gen_fail,
                    'candidate_zero_count': cand_zero,
                    'mean_full_pass_candidates': float(np.mean(full_pass_counts)),
                    'nutrition_all_pass_rate': nut_all5 / n_ok if n_ok else None,
                    'reality_gate_pass_rate': gate_pass / n_ok if n_ok else None,
                    'anchor_preservation_rate': float(np.mean(anchor_keeps)) if anchor_keeps else None,
                    'final_soup_top1_menu': soup_c.most_common(1)[0][0] if soup_c else None,
                    'final_soup_top1_ratio': soup_c.most_common(1)[0][1] / n_ok if n_ok else None,
                    'final_side_top1_ratio': side_c.most_common(1)[0][1] / n_ok if n_ok else None,
                    'unique_soup_count': len(soup_c), 'unique_side_count': len(side_c),
                    'seed_top1_match': (soup_c.most_common(1)[0][0] == seed_soup) if soup_c else None,
                    'invalid_candidate_count': invalid_total,
                })
                gate_fail_rows.append({
                    'experiment': exp, 'anchor_condition': cond_name, 'seed_id': sid,
                    'n_candidates_total': N_CALLS * TRIES - invalid_total,
                    'dup_today_fail': agg_fail['dup_today'], 'clash_fail': agg_fail['clash'],
                    'seafood_overload_fail': agg_fail['seafood_overload'],
                    'high_p_overload_fail': agg_fail['high_p_overload'],
                    'unrealistic_fail': agg_fail['unrealistic'], 'nutrition_fail': agg_fail['nutrition_fail'],
                    'invalid_candidate_count': invalid_total,
                })
        print(f'[{exp}] 완료')

    main_csv = os.path.join(OUT_DIR, 'full_make_meal_A_vs_C.csv')
    with open(main_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(main_rows[0].keys()))
        w.writeheader()
        w.writerows(main_rows)
    print(f'\n{main_csv} ({len(main_rows)}행)')

    gate_csv = os.path.join(OUT_DIR, 'full_make_meal_gate_failures.csv')
    with open(gate_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(gate_fail_rows[0].keys()))
        w.writeheader()
        w.writerows(gate_fail_rows)
    print(gate_csv)

    # ── anchor 민감도(같은 seed, 앵커 3개 pairwise, soup Counter 기반 JSD) ──
    anchor_rows = []
    vocab_max = 828
    for exp in MODELS:
        for sid, *_ in seed_info:
            dists = {}
            for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
                d = dist_cache[(exp, cond_name, sid)]
                vec = np.zeros(vocab_max)
                for menu, cnt in d['soup_c'].items():
                    idx = core.name2idx.get(menu)
                    if idx is not None:
                        vec[idx] = cnt
                if vec.sum() > 0:
                    vec = vec / vec.sum()
                dists[cond_name] = vec
            for (ca, _, _), (cb, _, _) in itertools.combinations(ANCHOR_CONDITIONS, 2):
                jsd = float(jensenshannon(dists[ca], dists[cb]))
                top1_a = dist_cache[(exp, ca, sid)]['soup_c'].most_common(1)
                top1_b = dist_cache[(exp, cb, sid)]['soup_c'].most_common(1)
                changed = (top1_a[0][0] if top1_a else None) != (top1_b[0][0] if top1_b else None)
                anchor_rows.append([exp, sid, ca, cb, jsd, changed])

    anchor_csv = os.path.join(OUT_DIR, 'full_make_meal_anchor_metrics.csv')
    with open(anchor_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'seed_id', 'anchor_a', 'anchor_b', 'js_divergence', 'top1_changed'])
        w.writerows(anchor_rows)
    print(anchor_csv)

    # ── 조건별/전체 요약 콘솔 출력 ──
    print('\n===== 조건별 요약 =====')
    for exp in MODELS:
        print(f'\n--- {exp} ---')
        for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
            rs = [r for r in main_rows if r['experiment'] == exp and r['anchor_condition'] == cond_name]
            nut = np.mean([r['nutrition_all_pass_rate'] for r in rs if r['nutrition_all_pass_rate'] is not None])
            gate = np.mean([r['reality_gate_pass_rate'] for r in rs if r['reality_gate_pass_rate'] is not None])
            anc = np.mean([r['anchor_preservation_rate'] for r in rs if r['anchor_preservation_rate'] is not None])
            fail = np.mean([r['generation_failure_count'] for r in rs]) / N_CALLS
            czero = np.mean([r['candidate_zero_count'] for r in rs]) / N_CALLS
            uniq = np.mean([r['unique_soup_count'] for r in rs])
            match = np.mean([r['seed_top1_match'] for r in rs if r['seed_top1_match'] is not None])
            print(f'  {cond_name}: 영양5종 {nut*100:.1f}%  게이트 {gate*100:.1f}%  앵커보존 {anc:.3f}  '
                  f'생성실패율 {fail*100:.2f}%  후보0개율 {czero*100:.2f}%  고유국 {uniq:.1f}  seed일치율 {match*100:.1f}%')

    return main_rows, gate_fail_rows, anchor_rows


if __name__ == '__main__':
    main()
