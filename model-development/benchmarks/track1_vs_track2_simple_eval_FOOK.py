# -*- coding: utf-8 -*-
"""
track1_vs_track2_simple_eval_FOOK.py — 실제 production 체크포인트 Track1(RL i002, 827-vocab)과
국 슬롯 마스킹+RL 실험 체크포인트 Track2(C_mask100_rl/checkpoints_best/ckpt-3, 828-vocab)를
같은 최종 FOOK_adjust_levers.py/passes()로 간단 비교한다.

두 트랙은 vocab이 달라 각자의 전용 추론 경로를 쓴다:
  - Track1: app_core_FOOK.gen(Sequence_Generator, 827-vocab) — 이미 로드된 production 객체 재사용
  - Track2: Model.Encoder/Decoder 직접 사용(828-vocab, SOUP_POS 마스킹) — checkpoints_best/ckpt-3

서비스 코드/체크포인트는 수정하지 않는다. 평가·CSV·보고서 생성만 수행.
"""
import os, sys, copy, time, pickle, csv
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
import pandas as pd
import tensorflow as tf
from Model import Encoder, Decoder

OUT_DIR = os.path.join(CODE, 'track1_vs_track2_simple_out')
os.makedirs(OUT_DIR, exist_ok=True)

print('=' * 70)
print('[1] Track1(RL i002, production) 로드 - app_core_FOOK 임포트')
print('=' * 70)
import app_core_FOOK as core   # noqa: E402  — core.gen = Track1 RL(i002), core.F = 최종 레버
print(f'  core.CKPT = {core.CKPT}')

print('\n' + '=' * 70)
print('[2] Track2(국슬롯마스킹+RL, C_mask100_rl/checkpoints_best) 로드')
print('=' * 70)
from train_FOOK_soupmask_1000 import build_data, SOUP_POS
T2_CKPT = os.path.join(CODE, 'results_rl_soupmask_FOOK', 'C_mask100_rl', 'checkpoints_best')
_cwd0 = os.getcwd()
os.chdir(CODE)   # build_data()가 ../../../data 상대경로를 쓰므로 CODE 기준으로 잠시 이동
nutrient_data_t2, food_dict_t2, diet_np_t2, incidence_t2, mask_id_t2 = build_data(with_mask=True)
os.chdir(_cwd0)
num_tokens_t2 = len(food_dict_t2)
diet_np_t2_np = diet_np_t2.numpy()
print(f'  Track2 vocab_size={num_tokens_t2}, mask_id={mask_id_t2}, SOUP_POS={SOUP_POS}')

kwargs_t2 = {'num_tokens': num_tokens_t2, 'embed_dim': 128, 'fc_dim': 64,
             'fully-connected_layer': 'GRU', 'attention': True}
enc_t2 = Encoder(**kwargs_t2, batch_size=10)
dec_t2 = Decoder(**kwargs_t2, batch_size=10)
ck_t2 = tf.train.latest_checkpoint(T2_CKPT)
if not ck_t2:
    print(f'  [STOP] Track2 체크포인트를 찾을 수 없음: {T2_CKPT}')
    sys.exit(1)
tf.train.Checkpoint(encoder=enc_t2, decoder=dec_t2).restore(ck_t2).expect_partial()
print(f'  Track2 체크포인트 로딩: {ck_t2}')

print('\n' + '=' * 70)
print('[3] 사전확인: vocab 정합성(0..826 동일해야 함), mask_id 위치')
print('=' * 70)
mismatch = [i for i in range(len(core.food_dict)) if food_dict_t2.get(i) != core.food_dict.get(i)]
if mismatch:
    print(f'  [STOP] Track1/Track2 기본 vocabulary(0..826)가 불일치함: {len(mismatch)}건 (예: {mismatch[:5]})')
    print('  두 체크포인트에 동일한 앵커 토큰 인덱스를 쓸 수 없으므로 평가를 중단한다.')
    sys.exit(1)
if mask_id_t2 != len(core.food_dict):
    print(f'  [STOP] mask_id({mask_id_t2})가 core.food_dict 크기({len(core.food_dict)})와 다름 - 예상과 다른 구조.')
    sys.exit(1)
print(f'  [OK] Track1/Track2 기본 vocabulary(0..{len(core.food_dict)-1}) 완전 일치, mask_id={mask_id_t2}는 그 다음(추가 토큰)')

print('\n' + '=' * 70)
print('[4] 스모크 테스트: 양쪽 생성 + 동일 core.F.adjust()/core.passes() 적용')
print('=' * 70)
b = core.F.meal_bounds(60)


def gen_batch_track1(base_row_7tok, anchor_menu, n, temp, seed):
    np.random.seed(seed)
    seeds = np.tile(base_row_7tok, (n, 1)).astype(np.int64)
    fixed = {}
    if anchor_menu is not None and anchor_menu in core.name2idx:
        s = core.slot_of.get(anchor_menu, 2)
        seeds[:, s + 1] = core.name2idx[anchor_menu]
        fixed[s] = core.name2idx[anchor_menu]
    seeds_tf = tf.constant(seeds, dtype=tf.int32)
    enc_hidden = tf.zeros([n, core.gen.encoder.units])
    enc_output, enc_hidden = core.gen.encoder(seeds_tf, enc_hidden)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int); res[:, 0] = seeds[:, 0]; res[:, -1] = 826
    used = [set(fixed.values()) for _ in range(n)]
    used_grp = [{core.TOK_GRP[t] for t in fixed.values() if t in core.TOK_GRP} for _ in range(n)]
    for j in range(5):
        outputs, dec_hidden, _ = core.gen.decoder(seeds_tf[:, j], dec_hidden, enc_output)
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


def gen_batch_track2(base_row_7tok, anchor_menu, n, temp, seed):
    np.random.seed(seed)
    seeds = np.tile(base_row_7tok, (n, 1)).astype(np.int64)
    seeds[:, SOUP_POS] = mask_id_t2   # 국 슬롯 마스킹(항상 마스크 - 학습시 p100 모드와 매칭)
    fixed = {}
    if anchor_menu is not None and anchor_menu in core.name2idx:
        s = core.slot_of.get(anchor_menu, 2)
        seeds[:, s + 1] = core.name2idx[anchor_menu]
        fixed[s] = core.name2idx[anchor_menu]
    seeds_tf = tf.constant(seeds, dtype=tf.int32)
    enc_hidden0 = tf.zeros([n, enc_t2.units])
    enc_output, enc_hidden = enc_t2(seeds_tf, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros((n, 7), dtype=int); res[:, 0] = seeds[:, 0]; res[:, -1] = 826
    used = [set(fixed.values()) for _ in range(n)]
    used_grp = [{core.TOK_GRP[t] for t in fixed.values() if t in core.TOK_GRP} for _ in range(n)]
    for j in range(5):
        outputs, dec_hidden, _ = dec_t2(seeds_tf[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        for bi in range(n):
            if j in fixed:
                res[bi, j + 1] = fixed[j]; continue
            p = probs[bi].copy()
            for t in core.SPECIAL:
                if t < num_tokens_t2: p[t] = 0.0
            p[mask_id_t2] = 0.0
            for t in core.BLOCK_TOK:
                if t < num_tokens_t2: p[t] = 0.0
            for t in used[bi]:
                if t < len(p): p[t] = 0.0
            slot_ok = core.SLOT_OK[j]
            if num_tokens_t2 > len(slot_ok):
                slot_ok = np.append(slot_ok, np.zeros(num_tokens_t2 - len(slot_ok)))
            masked = p * slot_ok
            for gi in used_grp[bi]:
                idx = np.array([g for g in core.GRP_TOK[gi] if g < num_tokens_t2])
                if len(idx): masked[idx] = 0.0
            if masked.sum() > 0:
                p = masked
            p = np.clip(p, 1e-12, None); p = p ** (1.0 / temp); p /= p.sum()
            tok = int(np.random.choice(len(p), p=p))
            res[bi, j + 1] = tok; used[bi].add(tok)
            gi = core.TOK_GRP.get(tok)
            if gi is not None:
                used_grp[bi].add(gi)
    return [[food_dict_t2[int(t)] for t in r if int(t) not in core.SPECIAL and t != mask_id_t2] for r in res]


def nutrient_flags(t, bb):
    raw_pass = t['P'] < bb['Pmax']
    protein_low = t['protein'] < bb['Plo']; protein_high = t['protein'] > bb['Phi']
    calorie_low = t['E'] < bb['Elo']; calorie_high = t['E'] > bb['Ehi']
    na_pass = t['Na_season'] <= bb['Namax']; k_pass = t['K'] < bb['Kmax']
    protein_pass = not protein_low and not protein_high
    calorie_pass = not calorie_low and not calorie_high
    all_pass = raw_pass and protein_pass and calorie_pass and na_pass and k_pass
    return {'raw_pass': raw_pass, 'protein_pass': protein_pass, 'calorie_pass': calorie_pass,
            'na_pass': na_pass, 'k_pass': k_pass, 'protein_low': protein_low, 'all_pass': all_pass}


try:
    diet_np_t1_np = core.diet_np.numpy()
    probe_row = diet_np_t1_np[11]
    probe_row_t2 = diet_np_t2_np[11]
    m1 = gen_batch_track1(probe_row, '두부양념조림', 4, 0.8, 777)
    m2 = gen_batch_track2(probe_row_t2, '두부양념조림', 4, 0.8, 777)
    for menus in m1 + m2:
        assert len(menus) == 5, f'생성 길이 이상: {menus}'
        before, after, inst, pok = core.F.adjust(list(menus), b, anchor='두부양념조림')
        ok = core.passes(after, b)
        unreal = core.F.unrealistic_reason(inst)
    print(f'  [OK] Track1 스모크 4건: {m1[0]}')
    print(f'  [OK] Track2 스모크 4건: {m2[0]}')
    print('  [OK] core.F.adjust()/core.passes()/core.F.unrealistic_reason() 양쪽 후보에 정상 적용됨.')
except Exception as e:
    print(f'  [STOP] 스모크 테스트 실패: {e}')
    import traceback; traceback.print_exc()
    sys.exit(1)

print('\n구조적 문제 없음 - 본 평가로 진행.')

# ============================================================
# 본 평가: 4앵커 x 5seed x 10call x TRIES=24, Track1/Track2 동일 (anchor,seed,call) paired
# ============================================================
ANCHOR_CONFIGS = [
    ('두부콩류', '두부양념조림', [11, 12, 6, 36, 7], 10),
    ('생선구이', '고등어구이', [11, 12, 6, 36, 7], 10),
    ('육류', '제육불고기', [11, 12, 6, 36, 7], 10),
    ('랜덤', None, [11, 12, 6, 36, 7], 10),
]
TRIES = 24
TEMP = 0.8

all_rows = []
cid = 0
t0 = time.perf_counter()
for anchor_name, anchor_menu, seed_rows, n_calls in ANCHOR_CONFIGS:
    print(f'\n[{anchor_name}] 생성 시작 (앵커={anchor_menu})...')
    n_start = cid
    for sid, row_idx in enumerate(seed_rows):
        base_row_t1 = diet_np_t1_np[row_idx].copy()
        base_row_t2 = diet_np_t2_np[row_idx].copy()
        for call_id in range(n_calls):
            seed_val = 200000 + hash((anchor_name, sid, call_id)) % 100000
            m1_batch = gen_batch_track1(base_row_t1, anchor_menu, TRIES, TEMP, seed_val)
            m2_batch = gen_batch_track2(base_row_t2, anchor_menu, TRIES, TEMP, seed_val)

            for track_label, menus_batch in [('Track1', m1_batch), ('Track2', m2_batch)]:
                for pos, menus in enumerate(menus_batch):
                    if len(menus) != 5:
                        continue
                    cid += 1
                    before, after, inst, pok = core.F.adjust(list(menus), b, anchor=anchor_menu)
                    flags = nutrient_flags(after, b)
                    unreal = core.F.unrealistic_reason(inst)
                    fm = list(dict.fromkeys(i['menu'] for i in inst))
                    all_rows.append({
                        'candidate_id': cid, 'anchor_type': anchor_name, 'track': track_label,
                        'seed_id': sid, 'call_id': call_id, 'pos_in_batch': pos,
                        'menus_raw': '|'.join(menus), 'final_menus': '|'.join(fm),
                        'all_pass': flags['all_pass'], 'unreal': unreal is not None,
                        'score': sum([flags['raw_pass'], flags['protein_pass'], flags['calorie_pass'],
                                      flags['na_pass'], flags['k_pass']]),
                    })
    print(f'[{anchor_name}] 완료: {cid - n_start}건 (Track1+Track2 합계)')

print(f'\n총 생성: {len(all_rows)}건, 소요 {time.perf_counter() - t0:.1f}s')
df = pd.DataFrame(all_rows)
with open(os.path.join(OUT_DIR, 'raw_candidates_t1t2.pkl'), 'wb') as f:
    pickle.dump(df, f)


# ============================================================
# 배치 선택 시뮬레이션 (첫 통과, 없으면 최고점수 fallback) - Track별
# ============================================================
sel_rows = []
for (anchor_type, track, seed_id, call_id), g in df.groupby(['anchor_type', 'track', 'seed_id', 'call_id']):
    gg = g.sort_values('pos_in_batch')
    passing = gg[gg['all_pass']]
    if len(passing) > 0:
        sel = passing.iloc[0]; rank = int(sel['pos_in_batch']) + 1; success = True
    else:
        idx = gg['score'].idxmax(); sel = gg.loc[idx]; rank = None; success = False
    sel_rows.append({
        'anchor_type': anchor_type, 'track': track, 'seed_id': seed_id, 'call_id': call_id,
        'success': success, 'rank': rank, 'unreal': sel['unreal'], 'final_menus': sel['final_menus'],
    })
sel_df = pd.DataFrame(sel_rows)
with open(os.path.join(OUT_DIR, 'sel_df_t1t2.pkl'), 'wb') as f:
    pickle.dump(sel_df, f)
print(f'배치 선택 시뮬레이션 완료: {sel_df.shape}')

ANCHORS = ['두부콩류', '생선구이', '육류', '랜덤']

# ============================================================
# 핵심 지표 6종 - track1_vs_track2_simple_summary.csv
# ============================================================
def summarize(s):
    n = len(s)
    if n == 0:
        return None
    vc = s['final_menus'].value_counts(normalize=True)
    vc_soup = s['final_menus'].apply(lambda x: x.split('|')[1] if len(x.split('|')) > 1 else None).value_counts(normalize=True)
    return {
        'n_batches': n,
        'success_rate': s['success'].mean(),
        'zero_candidate_rate': 1 - s['success'].mean(),
        'mean_rank_of_first_pass': s['rank'].dropna().mean() if s['rank'].notna().any() else None,
        'unrealistic_rate': s['unreal'].mean(),
        'unique_soup_count': s['final_menus'].apply(lambda x: x.split('|')[1] if len(x.split('|')) > 1 else None).nunique(),
        'top1_soup_share': vc_soup.iloc[0] if len(vc_soup) else None,
        'unique_combo_count': s['final_menus'].nunique(),
    }

summary_rows = []
for track in ('Track1', 'Track2'):
    for a in ANCHORS + ['전체']:
        s = sel_df[sel_df['track'] == track] if a == '전체' else sel_df[(sel_df['track'] == track) & (sel_df['anchor_type'] == a)]
        r = summarize(s)
        if r:
            r.update({'track': track, 'anchor_type': a})
            summary_rows.append(r)
summary_df = pd.DataFrame(summary_rows)[['track', 'anchor_type', 'n_batches', 'success_rate', 'zero_candidate_rate',
                                          'mean_rank_of_first_pass', 'unrealistic_rate', 'unique_soup_count',
                                          'top1_soup_share', 'unique_combo_count']]
summary_csv = os.path.join(OUT_DIR, 'track1_vs_track2_simple_summary.csv')
summary_df.to_csv(summary_csv, index=False, encoding='utf-8-sig')
print(f'\n저장: {summary_csv}')
print(summary_df.to_string(index=False))

# ============================================================
# Paired 전이 (동일 anchor,seed,call 기준: T1만/T2만/둘다/둘다실패)
# ============================================================
pivot = sel_df.pivot_table(index=['anchor_type', 'seed_id', 'call_id'], columns='track', values='success', aggfunc='first')
trans_rows = []
for a in ANCHORS + ['전체']:
    s = pivot if a == '전체' else pivot.loc[pivot.index.get_level_values('anchor_type') == a]
    n = len(s)
    t1_only = ((s['Track1']) & (~s['Track2'])).sum()
    t2_only = ((~s['Track1']) & (s['Track2'])).sum()
    both_pass = ((s['Track1']) & (s['Track2'])).sum()
    both_fail = ((~s['Track1']) & (~s['Track2'])).sum()
    trans_rows.append({'anchor_type': a, 'n': n, 'Track1_only': t1_only, 'Track2_only': t2_only,
                        'both_pass': both_pass, 'both_fail': both_fail})
trans_df = pd.DataFrame(trans_rows)
print('\n=== Paired 전이 (동일 seed·call 기준) ===')
print(trans_df.to_string(index=False))

# ============================================================
# 대표 사례 - track1_vs_track2_simple_cases.csv
# ============================================================
bt = sel_df.pivot(index=['anchor_type', 'seed_id', 'call_id'], columns='track',
                   values=['success', 'final_menus', 'rank'])
bt.columns = [f'{a}_{b}' for a, b in bt.columns]
bt = bt.reset_index()

t2_improved = bt[(~bt['success_Track1']) & (bt['success_Track2'])]
t2_worsened = bt[(bt['success_Track1']) & (~bt['success_Track2'])]


def soup_only_diff(r):
    m1 = r['final_menus_Track1'].split('|'); m2 = r['final_menus_Track2'].split('|')
    if len(m1) != 5 or len(m2) != 5:
        return False
    same_rest = m1[0] == m2[0] and m1[2] == m2[2] and m1[3] == m2[3] and m1[-1] == m2[-1]
    diff_soup = m1[1] != m2[1]
    return same_rest and diff_soup and r['success_Track1'] and r['success_Track2']


soup_only_diff_mask = bt.apply(soup_only_diff, axis=1)
soup_only = bt[soup_only_diff_mask]

case_rows = []
for label, sub, n in [('Track2_개선사례', t2_improved, 3), ('Track2_악화사례', t2_worsened, 3),
                       ('국만다르고_둘다통과', soup_only, 3)]:
    for _, r in sub.head(n).iterrows():
        case_rows.append({
            'category': label, 'anchor_type': r['anchor_type'], 'seed_id': r['seed_id'], 'call_id': r['call_id'],
            'Track1_final_menus': r['final_menus_Track1'], 'Track2_final_menus': r['final_menus_Track2'],
            'Track1_success': r['success_Track1'], 'Track2_success': r['success_Track2'],
            'Track1_rank': r['rank_Track1'], 'Track2_rank': r['rank_Track2'],
        })
cases_df = pd.DataFrame(case_rows)
cases_csv = os.path.join(OUT_DIR, 'track1_vs_track2_simple_cases.csv')
cases_df.to_csv(cases_csv, index=False, encoding='utf-8-sig')
print(f'\n저장: {cases_csv} ({len(cases_df)}건: 개선{min(len(t2_improved),3)}+악화{min(len(t2_worsened),3)}+국만다름{min(len(soup_only),3)})')

trans_csv = os.path.join(OUT_DIR, 'track1_vs_track2_simple_transitions.csv')
trans_df.to_csv(trans_csv, index=False, encoding='utf-8-sig')
print(f'저장(부속): {trans_csv}')

with open(os.path.join(OUT_DIR, 'analysis_state_t1t2.pkl'), 'wb') as f:
    pickle.dump({'df': df, 'sel_df': sel_df, 'summary_df': summary_df, 'trans_df': trans_df, 'cases_df': cases_df}, f)
print('\n분석 상태 저장 완료.')

