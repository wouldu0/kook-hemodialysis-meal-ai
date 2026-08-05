# -*- coding: utf-8 -*-
"""
track1_rl_final_eval_FOOK.py — Track1(BASE=results_FOOK/checkpoints vs RL=results_sweep_FOOK/i002,
실제 서비스가 로드하는 체크포인트) 최종 성능평가 + RL 채택 여부 검증.

4개 평가 조합:
  A. BASE + 수정 전 레버(OLD, .bak_before_rawP_tofu_path_20260727)
  B. BASE + 최종 레버(NEW, 현재 라이브 FOOK_adjust_levers.py)
  C. RL(i002) + 최종 레버(NEW)   <- 실제 현재 production 조합
  D. RL(i002) + 수정 전 레버(OLD)

같은 (anchor, seed_row, call_id)에 대해 BASE/RL 양쪽 모두 "동일 시드로 초기화한 뒤 생성"하여
난수 소모 스트림을 맞춘다(모델 가중치 차이 외의 변수를 제거). 각 원시 후보(menus)에는
OLD/NEW 레버를 모두 적용해 A/B/C/D 4개 조합을 모두 계산한다 (레버-only 비교는 candidate-level
동일 후보로 완전 paired, 모델 비교는 동일 seed/call 단위 paired).

서비스 코드(app_core_FOOK.py, FOOK_adjust_levers.py, server_FOOK.py)와 체크포인트는 전혀
수정하지 않는다. 읽기 전용 로드 + 별도 스크립트에서의 평가만 수행.
"""
import os, sys, copy, time, json, csv, pickle, importlib.util, importlib.machinery
from collections import Counter, defaultdict
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

OUT_DIR = os.path.join(CODE, 'track1_rl_final_eval_out')
os.makedirs(OUT_DIR, exist_ok=True)

print('app_core_FOOK 로딩 (RL i002 -> core.gen)...')
import app_core_FOOK as core   # noqa: E402

BASE_CKPT = os.path.join(core.CODE, 'results_FOOK', 'checkpoints')
kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
          'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
          'num_tokens': len(core.food_dict), 'batch_size': core.diet_np.shape[0], 'imitation_only': True}
base_gen = Sequence_Generator(core.food_dict, core.nutrient_data, core.incidence, **kwargs)
tf.train.Checkpoint(generator=base_gen).restore(tf.train.latest_checkpoint(BASE_CKPT)).expect_partial()
print('BASE(results_FOOK/checkpoints) 로딩 완료.')

OLD_BAK = r'E:\final\FOOK_adjust_levers.py.bak_before_rawP_tofu_path_20260727'
loader = importlib.machinery.SourceFileLoader("FOOK_adjust_levers_OLD", OLD_BAK)
spec = importlib.util.spec_from_loader(loader.name, loader)
F_old = importlib.util.module_from_spec(spec)
loader.exec_module(F_old)
# 주의(2026-07-27 디버깅으로 확인): F_old.NUT는 반드시 F_old.load_all()로 "독립적으로" 새로
# 로드해야 한다. core.F.NUT(공유 객체)를 그대로 대입하면 OLD 레버가 잘못된 값을 낸다(원인:
# load_all()이 매 호출마다 내부적으로 다르게 구성되는 부분이 있어, NEW용으로 만들어진 NUT
# 객체를 OLD 코드에 그대로 물리면 인덱싱/캐시가 어긋남 - service_rollout_verification_FOOK.py가
# 처음부터 독립 로드 방식을 썼던 이유). 생성된 원시 메뉴는 군메뉴를 포함하지 않으므로(모델
# vocabulary 자체에 군메뉴가 없음) F_old에 군메뉴 주입은 불필요.
F_old.NUT = F_old.load_all()
F_new = core.F
print('OLD(레버 수정 전 백업)/NEW(현재 라이브) 레버 모듈 준비 완료.')
print(f'  F_old에 신규함수 있음(있으면 안됨): {hasattr(F_old, "lever_phosphorus_rawP")}')
print(f'  F_new에 신규함수 있음(있어야 함): {hasattr(F_new, "lever_phosphorus_rawP")}')

diet_np_np = core.diet_np.numpy()

ANCHOR_CONFIGS = [
    ('두부콩류', '두부양념조림', [11, 12, 6, 36, 7], 10),
    ('생선구이', '고등어구이', [11, 12, 6, 36, 7], 10),
    ('육류', '제육불고기', [11, 12, 6, 36, 7], 10),
    ('랜덤', None, [11, 12, 6, 36, 7], 10),
]
TRIES = 24
TEMP = 0.8
W = 60
b = core.F.meal_bounds(W)


def gen_batch_track1(gen_obj, base_row_7tok, anchor_menu, n, temp, seed):
    """base_row_7tok을 n번 타일링(인코더 입력 고정) 후 anchor 슬롯 고정, 디코더만 확률샘플링.
    seed를 호출 직전에 매번 세팅해 BASE/RL이 동일 난수스트림을 소모하도록 한다."""
    np.random.seed(seed)
    seeds = np.tile(base_row_7tok, (n, 1)).astype(np.int64)
    fixed = {}
    if anchor_menu is not None and anchor_menu in core.name2idx:
        s = core.slot_of.get(anchor_menu, 2)
        seeds[:, s + 1] = core.name2idx[anchor_menu]
        fixed[s] = core.name2idx[anchor_menu]
    seeds_tf = tf.constant(seeds, dtype=tf.int32)
    enc_hidden = tf.zeros([n, gen_obj.encoder.units])
    enc_output, enc_hidden = gen_obj.encoder(seeds_tf, enc_hidden)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int); res[:, 0] = seeds[:, 0]; res[:, -1] = 826
    used = [set(fixed.values()) for _ in range(n)]
    used_grp = [{core.TOK_GRP[t] for t in fixed.values() if t in core.TOK_GRP} for _ in range(n)]
    for j in range(5):
        outputs, dec_hidden, _ = gen_obj.decoder(seeds_tf[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for bi in range(n):
            if j in fixed:
                res[bi, j + 1] = fixed[j]; continue
            p = probs[bi].copy()
            for t in core.SPECIAL: p[t] = 0.0
            for t in core.BLOCK_TOK: p[t] = 0.0
            for t in used[bi]: p[t] = 0.0
            masked = p * core.SLOT_OK[j]
            for gi in used_grp[bi]:
                masked[core.GRP_TOK[gi]] = 0.0
            if masked.sum() > 0:
                p = masked
            p = np.clip(p, 1e-12, None); p = p ** (1.0 / temp); p /= p.sum()
            tok = int(np.random.choice(len(p), p=p))
            res[bi, j + 1] = tok; used[bi].add(tok)
            gi = core.TOK_GRP.get(tok)
            if gi is not None:
                used_grp[bi].add(gi)
    return [[core.food_dict[int(t)] for t in r if int(t) not in core.SPECIAL] for r in res]


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


def score_of(flags):
    return sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'], flags['na_pass'], flags['k_pass']])


all_candidates = []
gen_stage_rows = []   # 생성단계(레버 적용 전) 통계용
cid = 0
t0_all = time.perf_counter()

for anchor_name, anchor_menu, seed_rows, n_calls in ANCHOR_CONFIGS:
    print(f'\n[{anchor_name}] 생성 시작 (앵커={anchor_menu})...')
    n_anchor_start = cid
    for sid, row_idx in enumerate(seed_rows):
        base_row = diet_np_np[row_idx].copy()
        for call_id in range(n_calls):
            seed_val = 100000 + hash((anchor_name, sid, call_id)) % 100000
            menus_base_batch = gen_batch_track1(base_gen, base_row, anchor_menu, TRIES, TEMP, seed_val)
            menus_rl_batch = gen_batch_track1(core.gen, base_row, anchor_menu, TRIES, TEMP, seed_val)

            for model_label, menus_batch, gen_obj in [('BASE', menus_base_batch, base_gen), ('RL', menus_rl_batch, core.gen)]:
                for pos_in_batch, menus in enumerate(menus_batch):
                    if len(menus) != 5:
                        continue
                    cid += 1
                    # parse-failure/anchor-preservation(생성단계) 체크용
                    valid_seq = (len(menus) == 5)
                    dup_slot = len(set(menus)) < len(menus)
                    anchor_present = (anchor_menu in menus) if anchor_menu else None

                    F_old.ROT[0] = 0
                    t1 = time.perf_counter()
                    before_o, after_o, inst_o, pok_o = F_old.adjust(list(menus), b, anchor=anchor_menu)
                    el_o = time.perf_counter() - t1

                    F_new.ROT[0] = 0
                    t2 = time.perf_counter()
                    before_n, after_n, inst_n, pok_n = F_new.adjust(list(menus), b, anchor=anchor_menu)
                    el_n = time.perf_counter() - t2

                    unreal_o = F_old.unrealistic_reason(inst_o)
                    unreal_n = F_new.unrealistic_reason(inst_n)
                    flags_o = nutrient_flags(after_o, b)
                    flags_n = nutrient_flags(after_n, b)

                    is_plant = F_new._plant_protein_path_needed(F_new.expand(list(menus)), menus, anchor_menu)

                    all_candidates.append({
                        'candidate_id': cid, 'anchor_type': anchor_name, 'model': model_label,
                        'seed_id': sid, 'call_id': call_id, 'pos_in_batch': pos_in_batch,
                        'menus': menus, 'is_plant_protein': is_plant,
                        'valid_seq': valid_seq, 'dup_slot': dup_slot, 'anchor_present': anchor_present,
                        'after_OLD': after_o, 'after_NEW': after_n,
                        'unreal_OLD': unreal_o, 'unreal_NEW': unreal_n,
                        'flags_OLD': flags_o, 'flags_NEW': flags_n,
                        'elapsed_OLD': el_o, 'elapsed_NEW': el_n,
                        'inst_OLD': inst_o, 'inst_NEW': inst_n,
                        'before_OLD': before_o, 'before_NEW': before_n,
                    })
    print(f'[{anchor_name}] 완료: {cid - n_anchor_start}건 (BASE+RL 합계)')

print(f'\n총 생성: {len(all_candidates)}건, 소요 {time.perf_counter() - t0_all:.1f}s')

with open(os.path.join(OUT_DIR, 'raw_candidates.pkl'), 'wb') as f:
    pickle.dump(all_candidates, f)
print(f"raw 데이터 저장: {os.path.join(OUT_DIR, 'raw_candidates.pkl')}")
