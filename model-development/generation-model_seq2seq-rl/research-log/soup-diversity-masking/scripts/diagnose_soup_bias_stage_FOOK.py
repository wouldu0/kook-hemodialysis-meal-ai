# -*- coding: utf-8 -*-
"""
diagnose_soup_bias_stage_FOOK.py — 국 슬롯 편중이 어느 학습 단계(Seq2Seq vs Seq2Seq+RL)부터
있는지 진단. 레버·게이트 없이 "생성 직후"만 본다. 기존 주찬 앵커 다양성 분석과 동일한 3조건·
동일 seed 식단·동일 앵커를 그대로 재사용한다.

핵심 사실(코드 확인): 이 모델의 디코더는 각 슬롯에서 이전에 "샘플링된" 토큰이 아니라 seed
시퀀스의 해당 위치 토큰을 다음 입력으로 쓴다(teacher-forcing 스타일 순회, app_core_FOOK.gen_batch
line 322 `gen.decoder(seeds[:, j], ...)` 참고). 즉 seed 식단·앵커·모델가중치가 고정되면 국 슬롯의
확률분포 자체는 "단 하나"로 결정되고(배치의 어느 행이든 동일), 매 시행마다 달라지는 건 그 고정된
분포에서 뽑는 난수뿐이다. 그래서 이 스크립트는 (a) 그 고정 분포 자체를 1회 계산해 기록하고,
(b) 그 분포에서 N=200번 독립 샘플링해 실제 표본 통계도 함께 낸다 — 두 가지가 서로 다른 원인
(확률분포 자체의 편중 vs 표본추출의 우연)을 가르는 데 필요하다.

마스킹은 실제 서비스(app_core_FOOK.gen_batch)와 동일하게 적용한다: SPECIAL 토큰 제외,
BLOCK_TOK(김치주재료) 제외, 이미 확정된 토큰(앵커) 제외, SLOT_OK[국] 카테고리 마스크,
밥/국/김치 중복방지 그룹 마스크. 레버(F.adjust)와 최종 게이트(passes 등)는 적용하지 않는다
— "생성 직후"만 보는 게 이번 진단의 목적이므로.

조건(기존 주찬 다양성 진단과 동일, 재사용):
  고등어구이(생선구이) 행55 / 제육불고기(육류) 행152 / 두부양념조림(두부콩류) 행36

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python diagnose_soup_bias_stage_FOOK.py
"""
import os, sys, io, csv, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import tensorflow as tf
from collections import Counter

FINAL = r'E:\final'
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, FINAL)

from Model import Sequence_Generator

OUT_DIR = os.path.join(CODE, 'diagnose_soup_bias_out')
N = 200
SEED = 11
SOUP_SLOT = 1     # 국
ANCHOR_SLOT = 2   # 주찬

ANCHOR_CONDITIONS = [
    ('생선구이', '고등어구이', 55),
    ('육류', '제육불고기', 152),
    ('두부콩류', '두부양념조림', 36),
]


def build_gen(ckpt_dir, food_dict, nutrient_data, incidence, batch_size):
    kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
              'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
              'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
              'num_tokens': len(food_dict), 'batch_size': batch_size, 'imitation_only': True}
    g = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
    ck = tf.train.latest_checkpoint(ckpt_dir)
    if not ck:
        raise RuntimeError(f'체크포인트 없음: {ckpt_dir}')
    tf.train.Checkpoint(generator=g).restore(ck).expect_partial()
    return g


def soup_masked_prob(core, gen_obj, seeds_batch, anchor_token):
    """국 슬롯(j=1)의 마스킹 적용 후 확률분포(정규화됨). seeds_batch: (n,7), 전부 동일 행이어야 함
    (배치의 모든 행이 identical -> 분포도 identical: 아래 assert로 확인)."""
    n = seeds_batch.shape[0]
    enc_hidden = tf.zeros([n, gen_obj.encoder.units])
    enc_output, enc_hidden = gen_obj.encoder(seeds_batch, enc_hidden)
    dec_hidden = copy.deepcopy(enc_hidden)
    # j=0(밥): 디코더 상태만 전진(seed가 입력이라 샘플링 결과는 다음 단계에 영향 없음 — teacher-forcing 확인됨)
    _, dec_hidden, _ = gen_obj.decoder(seeds_batch[:, 0], dec_hidden, enc_output)
    # j=1(국)
    outputs, dec_hidden, _ = gen_obj.decoder(seeds_batch[:, 1], dec_hidden, enc_output)
    probs = np.array(outputs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    # 배치 전 행이 동일 입력이므로 분포도 동일해야 함 — 검증
    if not np.allclose(probs[0], probs[-1], atol=1e-5):
        raise AssertionError('배치 행 간 확률분포가 다름 — teacher-forcing 가정이 깨짐, 재검토 필요')

    p = probs[0].copy()
    for t in core.SPECIAL:
        p[t] = 0.0
    for t in core.BLOCK_TOK:
        p[t] = 0.0
    p[anchor_token] = 0.0                          # 이미 확정된(앵커) 토큰 제외 — used[] 반영
    masked = p * core.SLOT_OK[SOUP_SLOT]            # 국 카테고리만 허용
    # 밥/국/김치 중복방지 그룹 마스크(앵커가 이 그룹 소속이면 적용, 국 자체는 아직 안 뽑았으니 해당無)
    anchor_grp = core.TOK_GRP.get(anchor_token)
    if anchor_grp is not None:
        masked[core.GRP_TOK[anchor_grp]] = 0.0
    if masked.sum() > 0:
        p = masked
    p = np.clip(p, 1e-12, None)
    p = p ** (1.0 / 0.8)                            # temp=0.8, gen_batch와 동일
    p /= p.sum()
    return p


def entropy_of(p):
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    os.chdir(cwd)

    print('Seq2Seq(모방) 체크포인트 로딩...')
    gen_imit = build_gen(os.path.join(CODE, 'results_FOOK', 'checkpoints'),
                          core.food_dict, core.nutrient_data, core.incidence, core.diet_np.shape[0])
    gen_rl = core.gen   # 이미 로딩된 실제 배포 RL 체크포인트(results_sweep_FOOK/i002) 재사용

    trace_rows = []
    dist_summary = {}   # (stage, cond) -> dict

    orig_diet_np_np = core.diet_np.numpy()
    for cond_name, anchor_menu, row_idx in ANCHOR_CONDITIONS:
        seed_row = orig_diet_np_np[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in seed_row]
        assert decoded[3] == anchor_menu, f'seed 행 검증 실패: {decoded}'
        anchor_token = core.name2idx[anchor_menu]
        seeds_batch = tf.constant(np.tile(seed_row, (N, 1)), dtype=tf.int32)
        print(f'[{cond_name}] 앵커={anchor_menu} seed행={row_idx}: {decoded}')

        for stage_name, gen_obj in [('seq2seq', gen_imit), ('seq2seq_rl', gen_rl)]:
            p = soup_masked_prob(core, gen_obj, seeds_batch, anchor_token)
            order = np.argsort(-p)
            top5 = [(core.food_dict[int(i)], float(p[i])) for i in order[:5]]

            np.random.seed(SEED)
            samples = np.random.choice(len(p), size=N, p=p)

            for run_id in range(N):
                tok = int(samples[run_id])
                menu = core.food_dict[tok]
                row = {
                    'model_stage': stage_name, 'anchor_condition': cond_name, 'run_id': run_id,
                    'seed_menu_tuple': '|'.join(decoded[1:6]), 'anchor_menu': anchor_menu,
                    'generated_soup': menu, 'selected_probability': float(p[tok]),
                }
                for k in range(5):
                    row[f'top{k+1}_menu'] = top5[k][0]
                    row[f'top{k+1}_probability'] = top5[k][1]
                trace_rows.append(row)

            c = Counter(core.food_dict[int(t)] for t in samples)
            total = sum(c.values())
            top1_menu, top1_cnt = c.most_common(1)[0]
            dist_summary[(stage_name, cond_name)] = {
                'n': total, 'unique_menus': len(c), 'top1_menu': top1_menu,
                'top1_ratio': top1_cnt / total,
                'top5_ratio': sum(v for _, v in c.most_common(5)) / total,
                'sample_entropy': entropy_of(np.array(list(c.values())) / total),
                'avg_selected_probability': float(np.mean([p[int(t)] for t in samples])),
                'distribution_entropy': entropy_of(p),
                'top5_from_dist': top5, 'full_counter': c,
            }
            print(f'  [{stage_name}] top1={top1_menu} 비율={top1_cnt/total*100:.1f}% '
                  f'(확률분포 top1={top5[0][0]} 확률={top5[0][1]*100:.1f}%, 분포엔트로피={entropy_of(p):.2f})')

    # ── ① soup_generation_trace.csv ──
    trace_csv = os.path.join(OUT_DIR, 'soup_generation_trace.csv')
    fieldnames = ['model_stage', 'anchor_condition', 'run_id', 'seed_menu_tuple', 'anchor_menu',
                  'generated_soup', 'selected_probability',
                  'top1_menu', 'top1_probability', 'top2_menu', 'top2_probability',
                  'top3_menu', 'top3_probability', 'top4_menu', 'top4_probability',
                  'top5_menu', 'top5_probability']
    with open(trace_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in trace_rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f'\n① {trace_csv} ({len(trace_rows)}행)')

    # ── ② soup_distribution_summary.csv ──
    dist_csv = os.path.join(OUT_DIR, 'soup_distribution_summary.csv')
    with open(dist_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model_stage', 'anchor_condition', 'n', 'unique_menus', 'top1_menu', 'top1_ratio',
                    'top5_ratio', 'sample_entropy', 'avg_selected_probability', 'distribution_entropy'])
        for (stage, cond), d in dist_summary.items():
            w.writerow([stage, cond, d['n'], d['unique_menus'], d['top1_menu'], d['top1_ratio'],
                        d['top5_ratio'], d['sample_entropy'], d['avg_selected_probability'],
                        d['distribution_entropy']])
    print(f'② {dist_csv}')

    # ── ③ soup_transition_comparison.csv ──
    trans_csv = os.path.join(OUT_DIR, 'soup_transition_comparison.csv')
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['anchor_condition', 'top1_ratio_seq2seq', 'top1_ratio_rl', 'top1_ratio_change',
                    'unique_menus_seq2seq', 'unique_menus_rl', 'unique_menus_change',
                    'top5_ratio_seq2seq', 'top5_ratio_rl', 'top5_ratio_change',
                    'entropy_seq2seq', 'entropy_rl', 'entropy_change',
                    'same_top1_menu', 'seq2seq_top1_menu', 'rl_top1_menu',
                    'seq2seq_top1_prob_under_seq2seq', 'seq2seq_top1_prob_under_rl',
                    'prob_increase_of_seq2seq_top1_menu_after_rl'])
        for cond_name, _, _ in ANCHOR_CONDITIONS:
            di = dist_summary[('seq2seq', cond_name)]
            dr = dist_summary[('seq2seq_rl', cond_name)]
            imit_top1_menu = di['top1_menu']
            # RL 분포에서 "Seq2Seq의 top1 메뉴"가 차지하는 확률(=RL이 그 메뉴를 얼마나 더/덜 미는지)
            rl_dist_top5 = dict(dr['top5_from_dist'])
            imit_dist_top5 = dict(di['top5_from_dist'])
            imit_top1_prob_seq2seq = imit_dist_top5.get(imit_top1_menu, di['top1_ratio'])
            # top5 밖일 수도 있으니 full_counter 기반 표본비율로 근사(정확한 확률값 없으면 표본비율 사용)
            rl_prob_of_imit_top1 = None
            if imit_top1_menu in rl_dist_top5:
                rl_prob_of_imit_top1 = rl_dist_top5[imit_top1_menu]
            else:
                cnt = dr['full_counter'].get(imit_top1_menu, 0)
                rl_prob_of_imit_top1 = cnt / dr['n']  # 표본비율로 근사(정확 확률은 top5 밖이라 미기록)
            w.writerow([
                cond_name,
                di['top1_ratio'], dr['top1_ratio'], dr['top1_ratio'] - di['top1_ratio'],
                di['unique_menus'], dr['unique_menus'], dr['unique_menus'] - di['unique_menus'],
                di['top5_ratio'], dr['top5_ratio'], dr['top5_ratio'] - di['top5_ratio'],
                di['sample_entropy'], dr['sample_entropy'], dr['sample_entropy'] - di['sample_entropy'],
                di['top1_menu'] == dr['top1_menu'], di['top1_menu'], dr['top1_menu'],
                imit_top1_prob_seq2seq, rl_prob_of_imit_top1,
                rl_prob_of_imit_top1 - imit_top1_prob_seq2seq,
            ])
    print(f'③ {trans_csv}')

    return trace_rows, dist_summary


if __name__ == '__main__':
    main()
