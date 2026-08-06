# -*- coding: utf-8 -*-
"""
rl_comparison_v2_rot_fixed_FOOK.py — BASE vs 기존 production RL(i002) 2-way 재검증.
rl_comparison_FOOK.py와 동일 로직이되 core.F.ROT[0]를 각 모델 호출 직전 동일 값(0)으로
리셋한다(김치 로테이션 전역 카운터가 모델 호출 순서에 따라 다르게 누적되던 방법론적 오류
수정). FOOK_adjust_levers.py 전체를 감사해 ROT 외에는 adjust() 호출 간 누적되는 mutable
global state가 없음을 확인했다(SWAP_LOG는 adjust() 시작 시 매번 clear(), KIMCHI_SIDES/
MENU_CLASS/SUBS_P 등은 load_all() 1회 빌드 후 정적, 그 외는 전부 함수-로컬 변수).

--order {base_first, rl_first}: 모델 실행 순서를 바꿔도 결과가 동일한지 검증(요청 5번)
--only {base, rl}: 한 모델만 단독 실행해 2-way 결과와 완전히 일치하는지 검증(요청 6번)

서비스 코드/체크포인트/레버/120개 CSV는 수정하지 않는다.
"""
import os, sys, csv, copy, time, argparse
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CODE = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code'
WORK = r'E:\final\rl_v2_work'
sys.path.insert(0, CODE)
sys.path.insert(0, r'E:\final')

import numpy as np
import tensorflow as tf
import pandas as pd
from Model import Sequence_Generator

ap = argparse.ArgumentParser()
ap.add_argument('--order', choices=['base_first', 'rl_first'], default='base_first')
ap.add_argument('--only', choices=['base', 'rl', 'both'], default='both')
ap.add_argument('--tag', type=str, default=None)
args = ap.parse_args()
tag = args.tag or f'{args.order}_{args.only}'

OUT_DIR = os.path.join(CODE, 'final_service_benchmark_out')

print(f'=== 실행 설정: order={args.order} only={args.only} tag={tag} ===')
print('app_core_FOOK 임포트 (기존 RL i002, production)...')
import app_core_FOOK as core   # noqa: E402

print('BASE 로드 (results_FOOK/checkpoints)...')
BASE_CKPT = os.path.join(core.CODE, 'results_FOOK', 'checkpoints')
kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
          'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
          'num_tokens': len(core.food_dict), 'batch_size': core.diet_np.shape[0], 'imitation_only': True}
base_gen = Sequence_Generator(core.food_dict, core.nutrient_data, core.incidence, **kwargs)
tf.train.Checkpoint(generator=base_gen).restore(tf.train.latest_checkpoint(BASE_CKPT)).expect_partial()


def gen_batch_generic(gen_obj, anchor_menu, n, temp):
    idx = np.random.randint(core.diet_np.shape[0], size=n)
    seeds = core.diet_np.numpy()[idx].copy()
    fixed = {}
    if anchor_menu is not None and anchor_menu in core.name2idx:
        s = core.slot_of.get(anchor_menu, 2)
        seeds[:, s + 1] = core.name2idx[anchor_menu]
        fixed[s] = core.name2idx[anchor_menu]
    enc_hidden = tf.zeros([n, gen_obj.encoder.units])
    enc_output, enc_hidden = gen_obj.encoder(seeds, enc_hidden)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int); res[:, 0] = seeds[:, 0]; res[:, -1] = 826
    used = [set(fixed.values()) for _ in range(n)]
    used_grp = [{core.TOK_GRP[t] for t in fixed.values() if t in core.TOK_GRP} for _ in range(n)]
    for j in range(5):
        outputs, dec_hidden, _ = gen_obj.decoder(seeds[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for b in range(n):
            if j in fixed:
                res[b, j + 1] = fixed[j]; continue
            p = probs[b].copy()
            for t in core.SPECIAL: p[t] = 0.0
            for t in core.BLOCK_TOK: p[t] = 0.0
            for t in used[b]: p[t] = 0.0
            masked = p * core.SLOT_OK[j]
            for gi in used_grp[b]:
                masked[core.GRP_TOK[gi]] = 0.0
            if masked.sum() > 0:
                p = masked
            p = np.clip(p, 1e-12, None); p = p ** (1.0 / temp); p /= p.sum()
            tok = int(np.random.choice(len(p), p=p))
            res[b, j + 1] = tok; used[b].add(tok)
            gi = core.TOK_GRP.get(tok)
            if gi is not None:
                used_grp[b].add(gi)
    return [[core.food_dict[int(t)] for t in r if int(t) not in core.SPECIAL] for r in res]


def make_meal_generic(gen_obj, menu=None, ingredient=None, W=60, tries=48, temp=0.8, bounds=None, used_today=None):
    used_today = used_today or set()
    anchor = menu
    if anchor is None and ingredient:
        cands = [m for m, igs in core.menu_ings.items()
                 if any(ingredient in ig for ig in igs) and (m in core.name2idx or m in core.gun_names)]
        if cands:
            anchor = np.random.choice(cands)
    if anchor in core.F.JAPGOK_RICE:
        anchor = core.F.WHITE_RICE
    gun_s = core.gun_slot.get(anchor) if (anchor and anchor in core.gun_names) else None
    tok_anchor = anchor if (anchor in core.name2idx) else None
    b = bounds if bounds is not None else core.F.meal_bounds(W)
    best = None; best_score = -1
    for menus in gen_batch_generic(gen_obj, tok_anchor, tries, temp):
        if gun_s is not None:
            menus = list(menus); menus[gun_s] = anchor
        dup_today = any(m in used_today and m != anchor and not core._is_staple_menu(m) for m in menus)
        clash = core._has_ingredient_clash(menus)
        overload = core._has_seafood_overload(menus)
        p_overload = core._has_high_p_overload(menus)
        before, after, inst, _ = core.F.adjust(menus, b, anchor=anchor)
        unreal = core.F.unrealistic_reason(inst)
        ok = (core.passes(after, b) and unreal is None and not dup_today and not clash
              and not overload and not p_overload)
        cand = (menus, inst, after, ok)
        if ok:
            return cand, b, anchor
        score = sum([b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                     after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax']])
        if unreal is None: score += 0.5
        if not dup_today: score += 0.3
        if not clash: score += 0.3
        if not overload: score += 0.3
        if not p_overload: score += 0.3
        if score > best_score:
            best, best_score = cand, score
    return best, b, anchor


def nutrient_flags(t, bb):
    return {'열량': bb['Elo'] <= t['E'] <= bb['Ehi'], '단백질': bb['Plo'] <= t['protein'] <= bb['Phi'],
            '칼륨': t['K'] < bb['Kmax'], '인': t['P'] < bb['Pmax'], '나트륨': t['Na_season'] <= bb['Namax']}


def run_one(gen_obj, weight, mode, meals_left, target, day_context):
    """ROT[0]=0을 이 함수 진입 시점에 리셋 - day_context의 필러끼까지 포함해 이 모델의
    이 시나리오 전체가 항상 ROT=0에서 시작하도록 보장한다."""
    core.F.ROT[0] = 0
    try:
        if day_context:
            consumed = {k: 0 for k in ('E', 'protein', 'K', 'P', 'Na', 'Na_season')}
            for mi_ in range(3 - meals_left):
                bounds_i = core.F.meal_bounds(weight, consumed, meals_left=3 - mi_)
                cand_i, _, _ = make_meal_generic(gen_obj, menu=None, W=weight, bounds=bounds_i)
                _, _, after_i, _ = cand_i
                for k in consumed:
                    consumed[k] = round(consumed[k] + after_i[k], 4)
            bounds_test = core.F.meal_bounds(weight, consumed, meals_left=meals_left)
        else:
            bounds_test = core.F.meal_bounds(weight)
        menu_arg = target if mode == 'menu' else None
        ing_arg = target if mode == 'ingredient' else None
        t_start = time.perf_counter()
        cand, b, resolved_anchor = make_meal_generic(gen_obj, menu=menu_arg, ingredient=ing_arg,
                                                       W=weight, bounds=bounds_test)
        elapsed = time.perf_counter() - t_start
        menus, inst, after, ok = cand
        flags = nutrient_flags(after, b)
        all5_pass = all(flags.values())
        unreal = core.F.unrealistic_reason(inst)
        final_menus = list(dict.fromkeys(i['menu'] for i in inst))
        target_menu = resolved_anchor if mode == 'ingredient' else menu_arg
        anchor_preserved = (target_menu is None) or (target_menu in final_menus)
        return {'success_ok': ok, 'all5_nutrient_pass': all5_pass, 'unrealistic': unreal is not None,
                'anchor_preserved': anchor_preserved, '열량_pass': flags['열량'], '단백질_pass': flags['단백질'],
                '칼륨_pass': flags['칼륨'], '인_pass': flags['인'], '나트륨_pass': flags['나트륨'],
                'final_menus': '|'.join(final_menus), 'elapsed_ms': elapsed * 1000, 'error': None}
    except Exception as e:
        return {'success_ok': None, 'all5_nutrient_pass': None, 'unrealistic': None, 'anchor_preserved': None,
                '열량_pass': None, '단백질_pass': None, '칼륨_pass': None, '인_pass': None, '나트륨_pass': None,
                'final_menus': None, 'elapsed_ms': None, 'error': f'{type(e).__name__}: {e}'}


SCEN_CSV = os.path.join(OUT_DIR, 'final_service_benchmark_120_realistic_weight.csv')
with open(SCEN_CSV, encoding='utf-8-sig') as f:
    scenarios = list(csv.DictReader(f))
for s in scenarios:
    s['weight'] = float(s['weight']); s['meals_left'] = int(s['meals_left'])
    s['day_context'] = s['day_context'] == 'True'
    s['anchor_or_ing_input'] = s['anchor_or_ing_input'] or None
assert len(scenarios) == 120

print(f'\n120개 시나리오 실행 (order={args.order}, only={args.only})...')
base_rows, rl_rows = [], []
t0 = time.perf_counter()
for i, s in enumerate(scenarios):
    seed_val = 820000 + i * 977
    common = (s['weight'], s['mode'], s['meals_left'], s['anchor_or_ing_input'], s['day_context'])

    def do_base():
        np.random.seed(seed_val)
        r = run_one(base_gen, *common)
        row = dict(s); row.update(r); row['model'] = 'BASE'; base_rows.append(row)

    def do_rl():
        np.random.seed(seed_val)
        r = run_one(core.gen, *common)
        row = dict(s); row.update(r); row['model'] = 'RL_기존(i002)'; rl_rows.append(row)

    if args.only == 'base':
        do_base()
    elif args.only == 'rl':
        do_rl()
    elif args.order == 'base_first':
        do_base(); do_rl()
    else:
        do_rl(); do_base()

    if (i + 1) % 20 == 0:
        print(f'  {i+1}/120 완료')
print(f'실행 완료, 소요 {time.perf_counter()-t0:.1f}s')

if base_rows:
    bdf = pd.DataFrame(base_rows)
    bdf.to_csv(os.path.join(WORK, f'rot_fixed_base_{tag}.csv'), index=False, encoding='utf-8-sig')
    print(f'BASE 성공: {bdf.success_ok.sum()}/120')
if rl_rows:
    rdf = pd.DataFrame(rl_rows)
    rdf.to_csv(os.path.join(WORK, f'rot_fixed_rl_{tag}.csv'), index=False, encoding='utf-8-sig')
    print(f'RL기존 성공: {rdf.success_ok.sum()}/120')
