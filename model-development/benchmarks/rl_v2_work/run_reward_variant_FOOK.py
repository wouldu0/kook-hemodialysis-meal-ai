# -*- coding: utf-8 -*-
"""
run_reward_variant_FOOK.py — STEP1: 지정된 reward 버전(R0~R4)으로 RL 파인튜닝 1회 실행 +
validation 40건으로 평가. train_rl_FOOK.py와 동일한 학습 설정(on-policy/stochastic/
use_baseline/use_beta, embed_dim=128, fc_dim=64, GRU, imit_weight 등)을 쓰되 reward_fn만
교체한다. 체크포인트는 rl_v2_work/checkpoints_<variant>/에 별도 저장(기존 체크포인트 안 건드림).

사용: python run_reward_variant_FOOK.py --variant R2 --epochs 200 --phosphorus_weight 1.0
"""
import os, sys, time, argparse, csv, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CODE = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code'
WORK = r'E:\final\rl_v2_work'
sys.path.insert(0, CODE)
sys.path.insert(0, r'E:\final')
sys.path.insert(0, WORK)

import numpy as np
import pandas as pd
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument('--variant', required=True, choices=['R0', 'R1', 'R2', 'R3', 'R4'])
ap.add_argument('--epochs', type=int, default=200)
ap.add_argument('--lr', type=float, default=1e-5)
ap.add_argument('--imit', type=float, default=0.3)
ap.add_argument('--seed', type=int, default=12345)
ap.add_argument('--phosphorus_weight', type=float, default=None)
ap.add_argument('--tag', type=str, default=None)   # 결과 라벨(예: R3_pw2.0)
args = ap.parse_args()

label = args.tag or args.variant
print(f'=== 학습: variant={args.variant} label={label} epochs={args.epochs} lr={args.lr} '
      f'imit={args.imit} seed={args.seed} phosphorus_weight={args.phosphorus_weight} ===')

# ---- 재현성: 전역 시드 명시 고정(감사에서 발견된 gap 보완, train_rl_FOOK.py는 이거 없었음) ----
np.random.seed(args.seed)
tf.random.set_seed(args.seed)

from util import sequence_to_sentence
from Model import Sequence_Generator
from train_FOOK import build_data
SPECIAL = {0, 825, 826}

cwd0 = os.getcwd()
os.chdir(r'E:\final')
import FOOK_adjust_levers as F
import reward_lever_FOOK as R0MOD
import reward_lever_v2_FOOK as RV2
print('식약청 DB 로딩...')
F.NUT = F.load_all()
R0MOD.init(weight=60)
os.chdir(cwd0)

cwd1 = os.getcwd()
os.chdir(CODE)   # build_data()가 ../../../data 상대경로를 씀
nutrient_data, food_dict, diet_np, incidence = build_data()
os.chdir(cwd1)
batch_size = int(diet_np.shape[0])


def seqs_to_menus(pred_seqs):
    out = []
    for row in np.asarray(pred_seqs)[:, 1:6]:
        out.append([food_dict[int(t)] if int(t) not in SPECIAL else None for t in row])
    return out


def make_reward_fn(variant, phosphorus_weight):
    func = RV2.REWARD_FUNCS[variant]

    def reward_fn(pred_seqs, anchor_slots):
        menus_list = seqs_to_menus(pred_seqs)
        rs = np.zeros(len(menus_list), dtype=float)
        for i, menus in enumerate(menus_list):
            if any(m is None for m in menus):
                rs[i] = 0.0
                continue
            a = anchor_slots[i] if anchor_slots is not None else 2
            anchor = menus[int(a)]
            if variant in ('R3', 'R4'):
                rs[i] = func(menus, anchor, phosphorus_weight=phosphorus_weight)
            else:
                rs[i] = func(menus, anchor)
        return rs
    return reward_fn


reward_fn = make_reward_fn(args.variant, args.phosphorus_weight)

kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'on-policy', 'policy': 'stochastic', 'use_beta': True,
          'use_buffer': False, 'buffer_size': 5, 'buffer_update': 5,
          'num_epochs': args.epochs, 'lr': args.lr, 'num_tokens': len(food_dict),
          'batch_size': batch_size, 'imitation_only': False, 'reward_fn': reward_fn,
          'use_baseline': True, 'imit_weight': args.imit}
gen = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
x = tf.convert_to_tensor(diet_np)

BASE_CKPT = os.path.join(CODE, 'results_FOOK', 'checkpoints')
ck = tf.train.latest_checkpoint(BASE_CKPT)
tf.train.Checkpoint(generator=gen).restore(ck).expect_partial()
print(f'warm-start: {ck}')

out_dir = os.path.join(WORK, f'checkpoints_{label}')
os.makedirs(out_dir, exist_ok=True)
ckpt = tf.train.Checkpoint(generator=gen)
rng = np.random.default_rng(args.seed)

curve_rows = []
t0 = time.time()
for epoch in range(args.epochs):
    anchor_slots = rng.integers(0, 5, size=batch_size)
    real_seqs, batch_loss, pred_seqs, _, _, _, _ = gen.train(x, x, anchor_slots=anchor_slots)
    r = gen.last_reward
    curve_rows.append({'epoch': epoch + 1, 'loss': float(batch_loss), 'reward_mean': float(r.mean()),
                        'reward_std': float(r.std())})
    if (epoch + 1) % 20 == 0 or epoch == 0:
        uniq = len(set(int(t) for t in np.asarray(pred_seqs)[:, 1:6].reshape(-1)))
        print(f'[epoch {epoch+1:4d}] loss={batch_loss:.4f} reward_mean={r.mean():.3f} std={r.std():.3f} 고유메뉴={uniq}')
print(f'총 학습시간 {time.time()-t0:.1f}s')
ckpt.save(file_prefix=os.path.join(out_dir, 'ckpt'))
print(f'체크포인트 저장: {out_dir}')

curve_csv = os.path.join(WORK, f'training_curve_{label}.csv')
pd.DataFrame(curve_rows).to_csv(curve_csv, index=False, encoding='utf-8-sig')
print(f'저장: {curve_csv}')

# ============================================================
# validation 40건으로 평가 (make_meal_generic 클론 재사용, rl_comparison_FOOK.py와 동일 로직)
# ============================================================
import app_core_FOOK as core   # RL(i002) production도 로드되지만 여기선 core.*(공유함수)만 씀


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


def make_meal_generic(gen_obj, menu=None, ingredient=None, W=60, tries=48, temp=0.8, bounds=None):
    anchor = menu
    if anchor is None and ingredient:
        cands = [m for m, igs in core.menu_ings.items()
                 if any(ingredient in ig for ig in igs) and (m in core.name2idx or m in core.gun_names)]
        anchor = np.random.choice(cands) if cands else None
    if anchor in core.F.JAPGOK_RICE:
        anchor = core.F.WHITE_RICE
    gun_s = core.gun_slot.get(anchor) if (anchor and anchor in core.gun_names) else None
    tok_anchor = anchor if (anchor in core.name2idx) else None
    b = bounds if bounds is not None else core.F.meal_bounds(W)
    best = None; best_score = -1
    for menus in gen_batch_generic(gen_obj, tok_anchor, tries, temp):
        if gun_s is not None:
            menus = list(menus); menus[gun_s] = anchor
        clash = core._has_ingredient_clash(menus)
        overload = core._has_seafood_overload(menus)
        p_overload = core._has_high_p_overload(menus)
        before, after, inst, _ = core.F.adjust(menus, b, anchor=anchor)
        unreal = core.F.unrealistic_reason(inst)
        ok = (core.passes(after, b) and unreal is None and not clash and not overload and not p_overload)
        cand = (menus, inst, after, ok)
        if ok:
            return cand, b, anchor
        score = sum([b['Elo'] <= after['E'] <= b['Ehi'], b['Plo'] <= after['protein'] <= b['Phi'],
                     after['K'] < b['Kmax'], after['P'] < b['Pmax'], after['Na_season'] <= b['Namax']])
        if unreal is None: score += 0.5
        if not clash: score += 0.3
        if not overload: score += 0.3
        if not p_overload: score += 0.3
        if score > best_score:
            best, best_score = cand, score
    return best, b, anchor


def nutrient_flags(t, bb):
    return {'열량': bb['Elo'] <= t['E'] <= bb['Ehi'], '단백질': bb['Plo'] <= t['protein'] <= bb['Phi'],
            '칼륨': t['K'] < bb['Kmax'], '인': t['P'] < bb['Pmax'], '나트륨': t['Na_season'] <= bb['Namax']}


with open(os.path.join(WORK, 'validation_scenarios.csv'), encoding='utf-8-sig') as f:
    val_scenarios = list(csv.DictReader(f))

print(f'\nvalidation {len(val_scenarios)}건 평가 중 (variant={label})...')
results = []
for i, s in enumerate(val_scenarios):
    seed_val = 500000 + i * 977
    np.random.seed(seed_val)
    w = float(s['weight'])
    menu_arg = s['anchor_or_ing_input'] if s['mode'] == 'menu' else None
    ing_arg = s['anchor_or_ing_input'] if s['mode'] == 'ingredient' else None
    try:
        cand, b, anchor = make_meal_generic(gen, menu=menu_arg, ingredient=ing_arg, W=w)
        menus, inst, after, ok = cand
        flags = nutrient_flags(after, b)
        all5 = all(flags.values())
        unreal = core.F.unrealistic_reason(inst)
        final_menus = list(dict.fromkeys(it['menu'] for it in inst))
        target = anchor if s['mode'] in ('menu', 'ingredient') else None
        preserved = (target is None) or (target in final_menus)
        results.append({'sid': s['sid'], 'success_ok': ok, 'all5_pass': all5,
                         'unrealistic': unreal is not None, 'preserved': preserved,
                         '인_pass': flags['인'], '단백질_pass': flags['단백질'],
                         'error': None})
    except Exception as e:
        results.append({'sid': s['sid'], 'success_ok': None, 'all5_pass': None, 'unrealistic': None,
                         'preserved': None, '인_pass': None, '단백질_pass': None, 'error': str(e)})

rdf = pd.DataFrame(results)
val_csv = os.path.join(WORK, f'validation_result_{label}.csv')
rdf.to_csv(val_csv, index=False, encoding='utf-8-sig')

n = len(rdf)
target_rows = [s for s in val_scenarios if s['mode'] in ('menu', 'ingredient')]
target_sids = {s['sid'] for s in target_rows}
preserved_sub = rdf[rdf.sid.isin(target_sids)]

summary = {
    'variant': args.variant, 'label': label, 'phosphorus_weight': args.phosphorus_weight,
    'n_val': n,
    'success_rate': rdf.success_ok.mean(),
    'all5_pass_rate': rdf.all5_pass.mean(),
    'raw_P_pass_rate': rdf['인_pass'].mean(),
    'protein_pass_rate': rdf['단백질_pass'].mean(),
    'preserved_rate': preserved_sub.preserved.mean() if len(preserved_sub) else None,
    'unrealistic_rate': rdf.unrealistic.mean(),
    'error_count': rdf.error.notna().sum(),
    'train_time_sec': time.time() - t0,
}
print('\n=== validation 요약 ===')
for k, v in summary.items():
    print(f'  {k}: {v}')

# 누적 ablation 결과 CSV에 append
abl_csv = os.path.join(WORK, 'reward_ablation_results.csv')
file_exists = os.path.exists(abl_csv)
with open(abl_csv, 'a', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(summary.keys()))
    if not file_exists:
        w.writeheader()
    w.writerow(summary)
print(f'\n누적 저장: {abl_csv}')
