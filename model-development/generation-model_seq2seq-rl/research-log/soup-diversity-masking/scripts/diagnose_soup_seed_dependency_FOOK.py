# -*- coding: utf-8 -*-
"""
diagnose_soup_seed_dependency_FOOK.py — 국 슬롯 편중이 (1)주찬 앵커, (2)seed 식단의 국 토큰,
(3)인코더 전체 문맥 중 어디에 기인하는지 진단.

구조적 사실(코드 확인, 중요): 이 디코더는 슬롯 j를 생성할 때 "seed 시퀀스의 위치 j" 토큰을
입력으로 쓴다(app_core_FOOK.gen_batch 주석 "slot s=j -> position j+1" 참고). 5토큰 시퀀스가
[BOS, 밥, 국, 주찬, 부찬, 김치, EOS]이므로:
  - 국(slot1) 출력을 만드는 디코더 스텝의 입력은 seed의 "밥" 값이다(seed의 "국" 값이 아니다!).
  - seed의 "국" 값 자체는 인코더가 전체 시퀀스를 한 번에 읽을 때(enc_output/enc_hidden)만
    영향을 준다 — 직접적인 스텝 입력으로는 전혀 안 쓰인다.
  따라서 "seed 국 토큰 의존"이라는 질문은 정확히는 "인코더가 seed 시퀀스에서 국 위치를 읽은 것이
  국 출력에 얼마나 영향을 주는가"로 이해해야 한다 — 이 스크립트는 그 질문에 답하도록 설계했다:
  encoder에 넣기 전 seed의 국 위치(포지션 2, 0-idx)만 바꿔서(A=원본/B=마스킹/C=교체) 국 출력이
  얼마나 달라지는지 본다.

조건:
  - 주찬 앵커 3개(기존과 동일): 고등어구이/제육불고기/두부양념조림
  - seed 식단 10개(사전 등록 — 결과를 본 뒤 고르지 않음): 학습데이터에서 국 종류별 빈도 상위
    10개를 뽑고, 각 국 종류가 처음 등장한 행을 그대로 썼다(임의선택 아님, 국 종류를 최대한
    다양하게 만드는 재현 가능한 규칙).
      근대된장국(행11) 아욱된장국(행12) 소고기뭇국(행6) 시래기된장국(행36) 청국장찌개(행7)
      북엇국(행18) 감자양파국(행21) 맑은콩나물국(행35) 시금치된장국(행28) 육개장(행53)
  - 입력 변형 3종(모두 진단용 forward만 — 서비스 코드는 안 건드림):
      A_원본  : seed 그대로
      B_마스킹 : seed의 국 위치를 중립 토큰(0='empty')으로 대체
      C_교체  : seed의 국 위치를 다른 실제 국 메뉴로 대체(근대된장국, 이미 근대된장국이면 아욱된장국)
  - Seq2Seq / Seq2Seq+RL 두 체크포인트 각각
  - 레버·게이트 미적용, 샘플링 직전 확률분포 자체를 기록. 표본은 seed당 N=100.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python diagnose_soup_seed_dependency_FOOK.py
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

from Model import Sequence_Generator

OUT_DIR = os.path.join(CODE, 'diagnose_soup_seed_out')
N = 100
SEED = 11
SOUP_SLOT = 1
ANCHOR_SLOT = 2
SOUP_POS = 2      # seed 7토큰 시퀀스 중 "국" 값의 위치(0-idx): [BOS,밥,국,주찬,부찬,김치,EOS]
BAP_POS = 1

ANCHOR_CONDITIONS = [('생선구이', '고등어구이'), ('육류', '제육불고기'), ('두부콩류', '두부양념조림')]
SEED_ROWS = [11, 12, 6, 36, 7, 18, 21, 35, 28, 53]   # 사전 등록: 국 빈도상위10 각각 첫등장 행
SUBSTITUTE_SOUP = '근대된장국'
SUBSTITUTE_SOUP_ALT = '아욱된장국'   # 원래 국이 근대된장국인 경우의 대체값(자기자신 치환 방지)


def build_gen(ckpt_dir, food_dict, nutrient_data, incidence, batch_size):
    kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
              'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
              'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
              'num_tokens': len(food_dict), 'batch_size': batch_size, 'imitation_only': True}
    g = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    tf.train.Checkpoint(generator=g).restore(ck).expect_partial()
    return g


def soup_dist(core, gen_obj, seeds_batch, anchor_token):
    n = seeds_batch.shape[0]
    enc_hidden = tf.zeros([n, gen_obj.encoder.units])
    enc_output, enc_hidden = gen_obj.encoder(seeds_batch, enc_hidden)
    dec_hidden = copy.deepcopy(enc_hidden)
    _, dec_hidden, _ = gen_obj.decoder(seeds_batch[:, 0], dec_hidden, enc_output)   # j=0(밥) 통과
    outputs, dec_hidden, _ = gen_obj.decoder(seeds_batch[:, 1], dec_hidden, enc_output)  # j=1(국)
    probs = np.array(outputs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    p = probs[0].copy()
    for t in core.SPECIAL:
        p[t] = 0.0
    for t in core.BLOCK_TOK:
        p[t] = 0.0
    p[anchor_token] = 0.0
    masked = p * core.SLOT_OK[SOUP_SLOT]
    anchor_grp = core.TOK_GRP.get(anchor_token)
    if anchor_grp is not None:
        masked[core.GRP_TOK[anchor_grp]] = 0.0
    if masked.sum() > 0:
        p = masked
    p = np.clip(p, 1e-12, None)
    p = p ** (1.0 / 0.8)
    p /= p.sum()
    return p


def entropy_of(p):
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def make_variant_seed(base_row, variant, core):
    s = base_row.copy()
    if variant == 'B_마스킹':
        s[SOUP_POS] = 0
    elif variant == 'C_교체':
        orig = core.food_dict[int(s[SOUP_POS])]
        sub = SUBSTITUTE_SOUP_ALT if orig == SUBSTITUTE_SOUP else SUBSTITUTE_SOUP
        s[SOUP_POS] = core.name2idx[sub]
    return s


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    os.chdir(cwd)

    print('Seq2Seq(모방) 체크포인트 로딩...')
    gen_imit = build_gen(os.path.join(CODE, 'results_FOOK', 'checkpoints'),
                          core.food_dict, core.nutrient_data, core.incidence, core.diet_np.shape[0])
    gen_rl = core.gen

    orig_diet_np_np = core.diet_np.numpy()
    seed_info = []   # (seed_id, seed_row, seed_soup, seed_full_tuple)
    for sid, row_idx in enumerate(SEED_ROWS):
        row = orig_diet_np_np[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in row]
        seed_info.append((sid, row, decoded[2], '|'.join(decoded[1:6]), row_idx))
    print('seed 10개(국 종류):', [s[2] for s in seed_info])

    trace = []   # dict rows
    variants = ['A_원본', 'B_마스킹', 'C_교체']

    for cond_name, anchor_menu in ANCHOR_CONDITIONS:
        anchor_token = core.name2idx[anchor_menu]
        for stage_name, gen_obj in [('seq2seq', gen_imit), ('seq2seq_rl', gen_rl)]:
            for sid, base_row, seed_soup, seed_tuple, row_idx in seed_info:
                seed_soup_token = core.name2idx[seed_soup]
                for variant in variants:
                    vrow = make_variant_seed(base_row, variant, core)
                    vrow[ANCHOR_SLOT + 1] = anchor_token   # 앵커 강제(주찬 위치=3)
                    seeds_batch = tf.constant(np.tile(vrow, (N, 1)), dtype=tf.int32)
                    p = soup_dist(core, gen_obj, seeds_batch, anchor_token)
                    order = np.argsort(-p)
                    top5 = [(core.food_dict[int(i)], float(p[i])) for i in order[:5]]
                    ent = entropy_of(p)

                    np.random.seed(SEED)
                    samples = np.random.choice(len(p), size=N, p=p)
                    sc = Counter(core.food_dict[int(t)] for t in samples)
                    sample_top1_menu, sample_top1_cnt = sc.most_common(1)[0]

                    trace.append({
                        'model_stage': stage_name, 'anchor_condition': cond_name, 'anchor_menu': anchor_menu,
                        'seed_id': sid, 'seed_row_idx': row_idx, 'seed_full_menu_tuple': seed_tuple,
                        'seed_soup': seed_soup, 'input_variant': variant,
                        'top1_soup': top5[0][0], 'top1_probability': top5[0][1],
                        'top2_soup': top5[1][0], 'top2_probability': top5[1][1],
                        'top3_soup': top5[2][0], 'top3_probability': top5[2][1],
                        'top4_soup': top5[3][0], 'top4_probability': top5[3][1],
                        'top5_soup': top5[4][0], 'top5_probability': top5[4][1],
                        'entropy': ent,
                        'seed_top1_match': (seed_soup == top5[0][0]),
                        'seed_soup_probability': float(p[seed_soup_token]),
                        'sample_top1_soup': sample_top1_menu,
                        'sample_top1_ratio': sample_top1_cnt / N,
                        'sample_unique_count': len(sc),
                        '_dist': p,   # 내부계산용(CSV엔 안 씀)
                    })
        print(f'[{cond_name}] 완료')

    # ── ① soup_seed_dependency_trace.csv ──
    trace_csv = os.path.join(OUT_DIR, 'soup_seed_dependency_trace.csv')
    fieldnames = ['model_stage', 'anchor_condition', 'anchor_menu', 'seed_id', 'seed_row_idx',
                  'seed_full_menu_tuple', 'seed_soup', 'input_variant',
                  'top1_soup', 'top1_probability', 'top2_soup', 'top2_probability',
                  'top3_soup', 'top3_probability', 'top4_soup', 'top4_probability',
                  'top5_soup', 'top5_probability', 'entropy', 'seed_top1_match',
                  'seed_soup_probability', 'sample_top1_soup', 'sample_top1_ratio', 'sample_unique_count']
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trace:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f'\n① {trace_csv} ({len(trace)}행)')

    # ── ② soup_seed_dependency_summary.csv ──
    summary_csv = os.path.join(OUT_DIR, 'soup_seed_dependency_summary.csv')
    groups = {}
    for r in trace:
        groups.setdefault((r['model_stage'], r['anchor_condition'], r['input_variant']), []).append(r)

    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model_stage', 'anchor_condition', 'input_variant', 'n_seeds',
                    'avg_top1_probability', 'avg_entropy', 'seed_top1_match_rate',
                    'unique_top1_menus_across_seeds', 'top1_menu_change_ratio_across_seeds',
                    'avg_pairwise_js_divergence_across_seeds',
                    'top1_prob_change_vs_A', 'top1_menu_change_rate_vs_A'])
        for (stage, cond, variant), rs in groups.items():
            avg_top1 = np.mean([r['top1_probability'] for r in rs])
            avg_ent = np.mean([r['entropy'] for r in rs])
            match_rate = np.mean([r['seed_top1_match'] for r in rs])
            uniq_top1 = len({r['top1_soup'] for r in rs})
            change_ratio = uniq_top1 / len(rs)
            dists = [r['_dist'] for r in rs]
            pair_jsd = [jensenshannon(dists[i], dists[j]) for i, j in itertools.combinations(range(len(dists)), 2)]
            avg_jsd = float(np.mean(pair_jsd)) if pair_jsd else 0.0

            top1_prob_change_vs_A = ''
            top1_menu_change_rate_vs_A = ''
            if variant != 'A_원본':
                a_rs = {r['seed_id']: r for r in groups[(stage, cond, 'A_원본')]}
                deltas = [r['top1_probability'] - a_rs[r['seed_id']]['top1_probability'] for r in rs]
                menu_changed = [r['top1_soup'] != a_rs[r['seed_id']]['top1_soup'] for r in rs]
                top1_prob_change_vs_A = float(np.mean(deltas))
                top1_menu_change_rate_vs_A = float(np.mean(menu_changed))

            w.writerow([stage, cond, variant, len(rs), avg_top1, avg_ent, match_rate,
                        uniq_top1, change_ratio, avg_jsd,
                        top1_prob_change_vs_A, top1_menu_change_rate_vs_A])
    print(f'② {summary_csv}')

    # ── ③ soup_seed_transition_matrix.csv (A_원본만 대상) ──
    trans_csv = os.path.join(OUT_DIR, 'soup_seed_transition_matrix.csv')
    a_rows = [r for r in trace if r['input_variant'] == 'A_원본']
    model_trans = Counter((r['seed_soup'], r['top1_soup']) for r in a_rows)
    sample_trans = Counter((r['seed_soup'], r['sample_top1_soup']) for r in a_rows)
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['transition_type', 'seed_soup', 'target_soup', 'count'])
        for (s, t), c in model_trans.most_common():
            w.writerow(['seed국->모델Top1국', s, t, c])
        for (s, t), c in sample_trans.most_common():
            w.writerow(['seed국->실제샘플Top1국', s, t, c])
    print(f'③ {trans_csv}')

    return trace


if __name__ == '__main__':
    main()
