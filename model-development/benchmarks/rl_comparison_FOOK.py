# -*- coding: utf-8 -*-
"""
rl_comparison_FOOK.py — 기본 Seq2Seq(BASE, results_FOOK/checkpoints) vs RL적용
(results_sweep_FOOK/i002, 실제 production) 공정 비교. Appendix 슬라이드용.

체크포인트만 다르고 나머지(레버·passes()·후보선택·fallback·시나리오·시드·tries)는 완전히
동일하게 고정한다. 기존에 이미 확정·저장된 120개 현실적 표준체중 시나리오
(final_service_benchmark_120_realistic_weight.csv)를 그대로 재사용하고 새로 만들지 않는다.

방법: app_core_FOOK.py의 make_meal()/gen_batch()를 그대로 복제하되 generator 객체를
파라미터로 받도록만 바꾼 함수를 이 스크립트 안에 작성해서 쓴다. name2idx/slot_of/SLOT_OK/
TOK_GRP/GRP_TOK/BLOCK_TOK/menu_ings/F.adjust/F.unrealistic_reason/passes/게이트함수들은
전부 core.*(app_core_FOOK 모듈)의 것을 그대로 재사용 — 체크포인트(encoder/decoder 가중치)
말고는 코드 경로가 완전히 동일하다.

서비스 코드/체크포인트/레버 코드는 수정하지 않는다. 평가·보고만 수행.
"""
import os, sys, csv, copy, time
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CODE = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code'
sys.path.insert(0, CODE)
sys.path.insert(0, r'E:\final')
import numpy as np
import tensorflow as tf
from Model import Sequence_Generator

OUT_DIR = os.path.join(CODE, 'final_service_benchmark_out')
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('[1] app_core_FOOK 임포트 확인 - checkpoint 로딩 방식 확인')
print('=' * 70)
import app_core_FOOK as core   # noqa: E402  RL(i002)이 core.gen으로 자동 로드됨(production)
print(f'  core.CKPT (RL, production) = {core.CKPT}')

print('\n' + '=' * 70)
print('[2] BASE 체크포인트 별도 로드 (results_FOOK/checkpoints) - core.gen과 완전히 동일한 kwargs')
print('=' * 70)
BASE_CKPT = os.path.join(core.CODE, 'results_FOOK', 'checkpoints')
kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
          'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
          'num_tokens': len(core.food_dict), 'batch_size': core.diet_np.shape[0], 'imitation_only': True}
base_gen = Sequence_Generator(core.food_dict, core.nutrient_data, core.incidence, **kwargs)
base_ckpt_status = tf.train.Checkpoint(generator=base_gen).restore(tf.train.latest_checkpoint(BASE_CKPT))
print(f'  BASE_CKPT = {BASE_CKPT}')
print(f'  최종 레버 함수 존재(있어야 함, 두 모델 모두 이 동일한 레버 사용): {hasattr(core.F, "lever_phosphorus_rawP")}')

print('\n' + '=' * 70)
print('[3] 사전확인: BASE 스모크 생성 + 구조 검증 (forward pass 후 assert_consumed)')
print('=' * 70)


def gen_batch_generic(gen_obj, anchor_menu, n, temp):
    """core.gen_batch()의 완전한 클론 - gen_obj만 파라미터화, 나머지 로직 100% 동일."""
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
    """core.make_meal()의 완전한 클론 - gen_obj만 파라미터화, 레버/passes/선택/fallback 로직은
    전부 core.* 그대로 재사용(체크포인트 외 코드 경로 100% 동일)."""
    used_today = used_today or set()
    anchor = menu
    note = ''
    if anchor is None and ingredient:
        cands = [m for m, igs in core.menu_ings.items()
                 if any(ingredient in ig for ig in igs) and (m in core.name2idx or m in core.gun_names)]
        if cands:
            anchor = np.random.choice(cands)
            note = f'(재료 "{ingredient}" -> 메뉴 "{anchor}" 선택)'
        else:
            note = f'(재료 "{ingredient}" 쓰는 메뉴 없음 -> 랜덤)'
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
            return cand, note, b, anchor, core._total_na_warning(after)
        score = sum([b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                     after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax']])
        if unreal is None:
            score += 0.5
        if not dup_today:
            score += 0.3
        if not clash:
            score += 0.3
        if not overload:
            score += 0.3
        if not p_overload:
            score += 0.3
        if score > best_score:
            best, best_score = cand, score
    return best, note + f' [{tries}회 완전통과 실패 -> 최선 {best_score}/5]', b, anchor, ''


# 스모크 + 구조검증
np.random.seed(1)
try:
    smoke_base = make_meal_generic(base_gen, menu='두부양념조림', W=60)
    smoke_rl = make_meal_generic(core.gen, menu='두부양념조림', W=60)
    print(f'  [OK] BASE 스모크: {smoke_base[0][0]}')
    print(f'  [OK] RL 스모크: {smoke_rl[0][0]}')
except Exception as e:
    print(f'  [STOP] 스모크 실패: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)

try:
    base_ckpt_status.assert_consumed()
    print('  [OK] BASE assert_consumed() 완전 통과')
except AssertionError as e:
    msg = str(e)
    unresolved = [ln for ln in msg.splitlines() if 'Unresolved object in checkpoint' in ln]
    non_opt = [ln for ln in unresolved if '.generator.optimizer.' not in ln]
    if unresolved and not non_opt:
        print(f'  [OK] BASE 미매칭 {len(unresolved)}건 전부 optimizer(추론 무관) - 구조 일치로 판정')
    else:
        print(f'  [STOP] BASE 구조 불일치 가능성: {non_opt}')
        sys.exit(1)

print('\n구조적 문제 없음 - 본 비교로 진행.')

# ============================================================
# 기존 확정된 120개 시나리오 로드 (수정하지 않음, 그대로 재사용)
# ============================================================
SCEN_CSV = os.path.join(OUT_DIR, 'final_service_benchmark_120_realistic_weight.csv')
import csv as _csv
with open(SCEN_CSV, encoding='utf-8-sig') as f:
    scenarios = list(_csv.DictReader(f))
for s in scenarios:
    s['weight'] = float(s['weight'])
    s['meals_left'] = int(s['meals_left'])
    s['day_context'] = s['day_context'] == 'True'
    s['anchor_or_ing_input'] = s['anchor_or_ing_input'] or None
assert len(scenarios) == 120, len(scenarios)
print(f'\n기존 시나리오 로드: {len(scenarios)}개 ({SCEN_CSV})')

NUT_ORDER = ['열량', '단백질', '칼륨', '인', '나트륨']
CONSUMED_KEYS = ('E', 'protein', 'K', 'P', 'Na', 'Na_season')


def nutrient_flags(t, bb):
    return {'열량': bb['Elo'] <= t['E'] <= bb['Ehi'],
            '단백질': bb['Plo'] <= t['protein'] <= bb['Phi'],
            '칼륨': t['K'] < bb['Kmax'],
            '인': t['P'] < bb['Pmax'],
            '나트륨': t['Na_season'] <= bb['Namax']}


def run_one(gen_obj, weight, mode, meals_left, target, day_context):
    error = None
    prior_consumed = None
    try:
        if day_context:
            consumed = {k: 0 for k in CONSUMED_KEYS}
            for mi_ in range(3 - meals_left):
                bounds_i = core.F.meal_bounds(weight, consumed, meals_left=3 - mi_)
                cand_i, _, _, _, _ = make_meal_generic(gen_obj, menu=None, W=weight, bounds=bounds_i)
                _, _, after_i, _ = cand_i
                for k in CONSUMED_KEYS:
                    consumed[k] = round(consumed[k] + after_i[k], 4)
            prior_consumed = dict(consumed)
            bounds_test = core.F.meal_bounds(weight, consumed, meals_left=meals_left)
        else:
            bounds_test = core.F.meal_bounds(weight)

        menu_arg = target if mode == 'menu' else None
        ing_arg = target if mode == 'ingredient' else None
        t_start = time.perf_counter()
        cand, note, b, resolved_anchor, warn = make_meal_generic(gen_obj, menu=menu_arg, ingredient=ing_arg,
                                                                   W=weight, bounds=bounds_test)
        elapsed = time.perf_counter() - t_start
        menus, inst, after, ok = cand
        flags = nutrient_flags(after, b)
        all5_pass = all(flags.values())
        unreal = core.F.unrealistic_reason(inst)
        final_menus = list(dict.fromkeys(i['menu'] for i in inst))
        target_menu = resolved_anchor if mode == 'ingredient' else menu_arg
        anchor_preserved = (target_menu is None) or (target_menu in final_menus)
        return {
            'resolved_anchor': resolved_anchor, 'success_ok': ok, 'all5_nutrient_pass': all5_pass,
            'unrealistic': unreal is not None, 'unreal_reason': unreal, 'anchor_preserved': anchor_preserved,
            '열량_pass': flags['열량'], '단백질_pass': flags['단백질'], '칼륨_pass': flags['칼륨'],
            '인_pass': flags['인'], '나트륨_pass': flags['나트륨'],
            'final_menus': '|'.join(final_menus), 'elapsed_ms': elapsed * 1000, 'error': None,
        }
    except Exception as e:
        return {
            'resolved_anchor': None, 'success_ok': None, 'all5_nutrient_pass': None, 'unrealistic': None,
            'unreal_reason': None, 'anchor_preserved': None, '열량_pass': None, '단백질_pass': None,
            '칼륨_pass': None, '인_pass': None, '나트륨_pass': None, 'final_menus': None, 'elapsed_ms': None,
            'error': f'{type(e).__name__}: {e}',
        }


# ============================================================
# 120개 시나리오 x 2모델(BASE/RL), 시나리오별로 두 모델 모두 동일 시드로 실행
# ============================================================
print('\n' + '=' * 70)
print('120개 시나리오 x BASE/RL 실행 (시나리오당 동일 시드, tries=48)')
print('=' * 70)
base_rows = []
rl_rows = []
t0 = time.perf_counter()
for i, s in enumerate(scenarios):
    seed_val = 820000 + i * 977
    np.random.seed(seed_val)
    r_base = run_one(base_gen, s['weight'], s['mode'], s['meals_left'], s['anchor_or_ing_input'], s['day_context'])
    row_b = dict(s); row_b.update(r_base); row_b['model'] = 'BASE'
    base_rows.append(row_b)

    np.random.seed(seed_val)   # RL도 동일 시드로 재설정 - 동일 난수 스트림 소비
    r_rl = run_one(core.gen, s['weight'], s['mode'], s['meals_left'], s['anchor_or_ing_input'], s['day_context'])
    row_r = dict(s); row_r.update(r_rl); row_r['model'] = 'RL'
    rl_rows.append(row_r)

    if (i + 1) % 20 == 0:
        print(f'  {i+1}/120 완료')
print(f'240건(120x2) 실행 완료, 소요 {time.perf_counter()-t0:.1f}s')

base_csv = os.path.join(OUT_DIR, 'rl_comparison_base_results.csv')
rl_csv = os.path.join(OUT_DIR, 'rl_comparison_rl_results.csv')
fieldnames = list(base_rows[0].keys())
with open(base_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(base_rows)
with open(rl_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader(); w.writerows(rl_rows)
print(f'저장: {base_csv}')
print(f'저장: {rl_csv}')

n_base_success = sum(1 for r in base_rows if r['success_ok'])
n_rl_success = sum(1 for r in rl_rows if r['success_ok'])
print(f'BASE 성공: {n_base_success}/120, RL 성공: {n_rl_success}/120')

# ============================================================
# 페어와이즈 전이표 + 전체/조건별 비교 - rl_comparison_pairwise_results.csv
# ============================================================
import pandas as pd
base_df = pd.DataFrame(base_rows)
rl_df = pd.DataFrame(rl_rows)

pair_cols = ['sid', 'tier', 'height', 'sex', 'weight', 'mode', 'cum_state', 'meals_left', 'anchor_or_ing_input']
pw = base_df[pair_cols].copy()
for col in ['success_ok', 'all5_nutrient_pass', 'anchor_preserved', 'unrealistic',
            '열량_pass', '단백질_pass', '칼륨_pass', '인_pass', '나트륨_pass', 'elapsed_ms', 'final_menus', 'error']:
    pw[f'{col}_base'] = base_df[col]
    pw[f'{col}_rl'] = rl_df[col]


def transition(row):
    b, r = row['success_ok_base'], row['success_ok_rl']
    if b is True and r is True:
        return 'both_pass'
    if b is False and r is False:
        return 'both_fail'
    if b is True and r is False:
        return 'BASE성공_RL실패'
    if b is False and r is True:
        return 'BASE실패_RL성공'
    return 'error'


pw['transition'] = pw.apply(transition, axis=1)
pair_csv = os.path.join(OUT_DIR, 'rl_comparison_pairwise_results.csv')
pw.to_csv(pair_csv, index=False, encoding='utf-8-sig')
print(f'저장: {pair_csv}')

trans_counts = pw['transition'].value_counts()
print('\n=== 전이 요약(전체) ===')
print(trans_counts)

print('\n=== 요청방식별 전이 ===')
print(pw.groupby('mode')['transition'].value_counts())
print('\n=== tier별 전이 ===')
print(pw.groupby('tier')['transition'].value_counts())

print(f'\n총 실행 횟수: {len(base_rows)}(BASE) + {len(rl_rows)}(RL) = {len(base_rows)+len(rl_rows)}회')
