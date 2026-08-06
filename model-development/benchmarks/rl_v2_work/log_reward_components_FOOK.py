# -*- coding: utf-8 -*-
"""
log_reward_components_FOOK.py — STEP1: 가중치를 예시값 그대로 쓰지 않기 위해, BASE 모델(웜스타트
시작점)의 실제 생성 후보들에 대해 보상 구성요소의 실측 분포를 먼저 로그로 남긴다.
"""
import os, sys, csv, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

CODE = r'E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code'
sys.path.insert(0, CODE)
sys.path.insert(0, r'E:\final')
WORK = r'E:\final\rl_v2_work'
sys.path.insert(0, WORK)

import numpy as np
import tensorflow as tf
from Model import Sequence_Generator
import reward_lever_v2_FOOK as RV2
import reward_lever_FOOK as R0MOD
import app_core_FOOK as core

R0MOD.init(weight=60)   # _B 전역 세팅(reward_lever_v2도 이 _B를 공유해서 씀)

BASE_CKPT = os.path.join(core.CODE, 'results_FOOK', 'checkpoints')
kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
          'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
          'num_tokens': len(core.food_dict), 'batch_size': core.diet_np.shape[0], 'imitation_only': True}
base_gen = Sequence_Generator(core.food_dict, core.nutrient_data, core.incidence, **kwargs)
tf.train.Checkpoint(generator=base_gen).restore(tf.train.latest_checkpoint(BASE_CKPT)).expect_partial()
print('BASE 로드 완료.')


def gen_batch_generic(gen_obj, anchor_menu, n, temp=0.8):
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


N = 500
rows = []
np.random.seed(42)
all_menus = []
for _ in range(N // 20):
    all_menus.extend(gen_batch_generic(base_gen, None, 20, temp=0.8))

print(f'생성된 원시 후보: {len(all_menus)}건, 컴포넌트 계산 중...')
for i, menus in enumerate(all_menus):
    if len(menus) != 5:
        continue
    anchor = menus[2]   # 학습 관례상 슬롯2를 앵커로 가정(anchor_slots 랜덤이지만 로깅용으론 대표로 슬롯2 사용)
    c = RV2.eval_meal(menus, anchor)
    if c is None:
        continue
    rows.append({
        'idx': i, 'pass_frac': c['pass_frac'], 'final_pass': c['final_pass'],
        'calorie_violation': c['calorie_violation'], 'protein_violation': c['protein_violation'],
        'potassium_violation': c['potassium_violation'], 'phosphorus_violation': c['phosphorus_violation'],
        'sodium_violation': c['sodium_violation'], 'unrealistic_amount': c['unrealistic_amount'],
        'ingredient_clash': c['ingredient_clash'], 'overload': c['overload'], 'p_overload': c['p_overload'],
        'a_keep': c['a_keep'], 'o_keep': c['o_keep'], 'preserve': c['preserve'],
        'r0_reward': R0MOD.meal_reward(menus, anchor),
    })

out_csv = os.path.join(WORK, 'reward_component_distribution.csv')
with open(out_csv, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f'저장: {out_csv} ({len(rows)}행)')

import pandas as pd
df = pd.DataFrame(rows)
summary = []
for col in ['pass_frac', 'calorie_violation', 'protein_violation', 'potassium_violation',
            'phosphorus_violation', 'sodium_violation', 'a_keep', 'o_keep', 'preserve', 'r0_reward']:
    summary.append({'component': col, 'mean': df[col].mean(), 'std': df[col].std(),
                     'min': df[col].min(), 'max': df[col].max()})
for col in ['final_pass', 'unrealistic_amount', 'ingredient_clash', 'overload', 'p_overload']:
    summary.append({'component': col, 'mean': df[col].mean(), 'std': None, 'min': None, 'max': None})
sdf = pd.DataFrame(summary)
print(sdf.to_string(index=False))

# 상관관계(phosphorus_violation vs final_pass, 등)
corr_p_final = df['phosphorus_violation'].corr(df['final_pass'].astype(float))
corr_protein_final = df['protein_violation'].corr(df['final_pass'].astype(float))
print(f'\nphosphorus_violation vs final_pass 상관: {corr_p_final:.3f}')
print(f'protein_violation vs final_pass 상관: {corr_protein_final:.3f}')
print(f'final_pass율: {df.final_pass.mean():.3f}')
print(f'unrealistic율: {df.unrealistic_amount.mean():.3f}')
