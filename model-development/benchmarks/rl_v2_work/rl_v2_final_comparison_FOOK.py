# -*- coding: utf-8 -*-
"""
rl_v2_final_comparison_FOOK.py — STEP4: 선택된 새 체크포인트(R2)를 기존 고정 120개 시나리오
(final_service_benchmark_120_realistic_weight.csv)에서 BASE·기존RL(i002)과 동일 조건으로
최종 비교. rl_comparison_FOOK.py와 완전히 동일한 로직(레버·passes()·후보선택·fallback)을
3-way로 확장했을 뿐, checkpoint 외 코드 경로는 전부 동일.

서비스 코드/체크포인트/레버/기존 120개 CSV는 수정하지 않는다.
"""
import os, sys, csv, copy, time
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

OUT_DIR = os.path.join(CODE, 'final_service_benchmark_out')

print('app_core_FOOK 임포트 (기존 RL i002, production)...')
import app_core_FOOK as core   # noqa: E402

print('BASE 로드 (results_FOOK/checkpoints)...')
BASE_CKPT = os.path.join(core.CODE, 'results_FOOK', 'checkpoints')
kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
          'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
          'num_tokens': len(core.food_dict), 'batch_size': core.diet_np.shape[0], 'imitation_only': True}
base_gen = Sequence_Generator(core.food_dict, core.nutrient_data, core.incidence, **kwargs)
base_status = tf.train.Checkpoint(generator=base_gen).restore(tf.train.latest_checkpoint(BASE_CKPT))

print('R2(새 RL v2, 선택된 체크포인트) 로드 (rl_v2_work/checkpoints_R2)...')
R2_CKPT = os.path.join(WORK, 'checkpoints_R2')
r2_gen = Sequence_Generator(core.food_dict, core.nutrient_data, core.incidence, **kwargs)
r2_status = tf.train.Checkpoint(generator=r2_gen).restore(tf.train.latest_checkpoint(R2_CKPT))
print(f'  R2_CKPT = {tf.train.latest_checkpoint(R2_CKPT)}')


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
    note = ''
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
            return cand, note, b, anchor
        score = sum([b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                     after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax']])
        if unreal is None: score += 0.5
        if not dup_today: score += 0.3
        if not clash: score += 0.3
        if not overload: score += 0.3
        if not p_overload: score += 0.3
        if score > best_score:
            best, best_score = cand, score
    return best, note, b, anchor


NUT_ORDER = ['열량', '단백질', '칼륨', '인', '나트륨']


def nutrient_flags(t, bb):
    return {'열량': bb['Elo'] <= t['E'] <= bb['Ehi'], '단백질': bb['Plo'] <= t['protein'] <= bb['Phi'],
            '칼륨': t['K'] < bb['Kmax'], '인': t['P'] < bb['Pmax'], '나트륨': t['Na_season'] <= bb['Namax']}


def run_one(gen_obj, weight, mode, meals_left, target, day_context):
    error = None
    try:
        if day_context:
            consumed = {k: 0 for k in ('E', 'protein', 'K', 'P', 'Na', 'Na_season')}
            for mi_ in range(3 - meals_left):
                bounds_i = core.F.meal_bounds(weight, consumed, meals_left=3 - mi_)
                cand_i, _, _, _ = make_meal_generic(gen_obj, menu=None, W=weight, bounds=bounds_i)
                _, _, after_i, _ = cand_i
                for k in consumed:
                    consumed[k] = round(consumed[k] + after_i[k], 4)
            bounds_test = core.F.meal_bounds(weight, consumed, meals_left=meals_left)
        else:
            bounds_test = core.F.meal_bounds(weight)
        menu_arg = target if mode == 'menu' else None
        ing_arg = target if mode == 'ingredient' else None
        t_start = time.perf_counter()
        cand, note, b, resolved_anchor = make_meal_generic(gen_obj, menu=menu_arg, ingredient=ing_arg,
                                                             W=weight, bounds=bounds_test)
        elapsed = time.perf_counter() - t_start
        menus, inst, after, ok = cand
        flags = nutrient_flags(after, b)
        all5_pass = all(flags.values())
        unreal = core.F.unrealistic_reason(inst)
        final_menus = list(dict.fromkeys(i['menu'] for i in inst))
        target_menu = resolved_anchor if mode == 'ingredient' else menu_arg
        anchor_preserved = (target_menu is None) or (target_menu in final_menus)
        return {'resolved_anchor': resolved_anchor, 'success_ok': ok, 'all5_nutrient_pass': all5_pass,
                'unrealistic': unreal is not None, 'anchor_preserved': anchor_preserved,
                '열량_pass': flags['열량'], '단백질_pass': flags['단백질'], '칼륨_pass': flags['칼륨'],
                '인_pass': flags['인'], '나트륨_pass': flags['나트륨'],
                'final_menus': '|'.join(final_menus), 'elapsed_ms': elapsed * 1000, 'error': None}
    except Exception as e:
        return {'resolved_anchor': None, 'success_ok': None, 'all5_nutrient_pass': None, 'unrealistic': None,
                'anchor_preserved': None, '열량_pass': None, '단백질_pass': None, '칼륨_pass': None,
                '인_pass': None, '나트륨_pass': None, 'final_menus': None, 'elapsed_ms': None,
                'error': f'{type(e).__name__}: {e}'}


SCEN_CSV = os.path.join(OUT_DIR, 'final_service_benchmark_120_realistic_weight.csv')
with open(SCEN_CSV, encoding='utf-8-sig') as f:
    scenarios = list(csv.DictReader(f))
for s in scenarios:
    s['weight'] = float(s['weight']); s['meals_left'] = int(s['meals_left'])
    s['day_context'] = s['day_context'] == 'True'
    s['anchor_or_ing_input'] = s['anchor_or_ing_input'] or None
assert len(scenarios) == 120

print(f'\n120개 시나리오 x 3모델(BASE/기존RL/R2) 실행 (시나리오당 동일 시드, tries=48)...')
base_rows, rl_rows, r2_rows = [], [], []
t0 = time.perf_counter()
for i, s in enumerate(scenarios):
    seed_val = 820000 + i * 977

    # 주의(중요): core.F.ROT[0]는 lever_kimchi가 호출될 때마다 전역으로 증가하는 카운터다.
    # np.random.seed()만 리셋하고 ROT를 안 건드리면, "한 시나리오당 몇 개 모델을 도는지"에 따라
    # ROT의 절대 위치가 달라져 김치 선택이 갈리고, 드물게는 그게 나트륨 경계값을 넘나들며
    # 완전히 다른 후보가 선택되는 것까지 이어질 수 있다(실제로 재현 검증 중 발견됨). 세 모델이
    # 서로 동일한 조건에서 비교되도록 모델 호출 직전마다 ROT도 0으로 리셋한다
    # (service_rollout_verification_FOOK.py에서 이미 검증된 것과 동일한 패턴).
    core.F.ROT[0] = 0
    np.random.seed(seed_val)
    rb = run_one(base_gen, s['weight'], s['mode'], s['meals_left'], s['anchor_or_ing_input'], s['day_context'])
    row_b = dict(s); row_b.update(rb); row_b['model'] = 'BASE'; base_rows.append(row_b)

    core.F.ROT[0] = 0
    np.random.seed(seed_val)
    rr = run_one(core.gen, s['weight'], s['mode'], s['meals_left'], s['anchor_or_ing_input'], s['day_context'])
    row_r = dict(s); row_r.update(rr); row_r['model'] = 'RL_기존(i002)'; rl_rows.append(row_r)

    core.F.ROT[0] = 0
    np.random.seed(seed_val)
    r2r = run_one(r2_gen, s['weight'], s['mode'], s['meals_left'], s['anchor_or_ing_input'], s['day_context'])
    row_r2 = dict(s); row_r2.update(r2r); row_r2['model'] = 'RL_v2(R2)'; r2_rows.append(row_r2)

    if (i + 1) % 20 == 0:
        print(f'  {i+1}/120 완료')
print(f'360건(120x3) 실행 완료, 소요 {time.perf_counter()-t0:.1f}s')

r2_csv = os.path.join(WORK, 'rl_v2_results.csv')
pd.DataFrame(r2_rows).to_csv(r2_csv, index=False, encoding='utf-8-sig')
print(f'저장: {r2_csv}')

base_df = pd.DataFrame(base_rows); rl_df = pd.DataFrame(rl_rows); r2_df = pd.DataFrame(r2_rows)
print(f'BASE 성공: {base_df.success_ok.sum()}/120')
print(f'기존RL 성공: {rl_df.success_ok.sum()}/120')
print(f'R2(신규RL) 성공: {r2_df.success_ok.sum()}/120')

pair_cols = ['sid', 'tier', 'height', 'sex', 'weight', 'mode', 'cum_state', 'meals_left', 'anchor_or_ing_input']
pw = base_df[pair_cols].copy()
for col in ['success_ok', 'all5_nutrient_pass', 'anchor_preserved', 'unrealistic',
            '열량_pass', '단백질_pass', '칼륨_pass', '인_pass', '나트륨_pass', 'elapsed_ms', 'final_menus']:
    pw[f'{col}_BASE'] = base_df[col]
    pw[f'{col}_RL기존'] = rl_df[col]
    pw[f'{col}_R2'] = r2_df[col]


def trans(b, x):
    if b is True and x is True: return 'both_pass'
    if b is False and x is False: return 'both_fail'
    if b is True and x is False: return 'BASE성공_상대실패'
    if b is False and x is True: return 'BASE실패_상대성공'
    return 'error'


pw['transition_BASE_vs_R2'] = [trans(b, r2) for b, r2 in zip(base_df.success_ok, r2_df.success_ok)]
pw['transition_RL기존_vs_R2'] = [trans(rl, r2) for rl, r2 in zip(rl_df.success_ok, r2_df.success_ok)]

pw_csv = os.path.join(WORK, 'rl_v2_pairwise_results.csv')
pw.to_csv(pw_csv, index=False, encoding='utf-8-sig')
print(f'저장: {pw_csv}')

print('\n=== BASE vs R2 전이 ===')
print(pw['transition_BASE_vs_R2'].value_counts())
print('\n=== 기존RL vs R2 전이 ===')
print(pw['transition_RL기존_vs_R2'].value_counts())
