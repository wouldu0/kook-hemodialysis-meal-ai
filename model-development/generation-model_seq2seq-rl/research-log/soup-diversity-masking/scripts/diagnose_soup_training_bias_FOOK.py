# -*- coding: utf-8 -*-
"""
diagnose_soup_training_bias_FOOK.py — 국 편중이 학습데이터 자체의 주찬-국 동시등장 편중 때문인지,
모델(Seq2Seq/RL)의 과확신 때문인지 구분.

※ 검증·평가 데이터 표기 관련: FOOK_meals_for_model.csv(=학습에 실제 쓰인 파일)엔 별도로 떼어둔
  train/val/test 분리가 없다(이전 세션에서 train_FOOK.build_data() 코드를 직접 읽어 확인한 사실 —
  build_data()에 split 로직 자체가 없음). 즉 여기서 계산하는 "학습 데이터 분포"는 검증셋이 아니라
  모델이 실제로 본 데이터 전체(1,095끼) 그 자체이며, 그 특성상 "학습 데이터 재현 여부"를 볼 수는
  있어도 "미학습 데이터에 대한 일반화"는 이 분석 범위 밖이다.

분석 대상(기존과 동일 3앵커):
  고등어구이(생선구이형) / 제육불고기(육류형) / 두부양념조림(두부·콩류형)

유형군 분류 규칙(간이 규칙 — 조리형태 Class + 메뉴명 키워드, 사전 정의):
  생선구이류   : Class=='구이' AND 메뉴명에 생선류 키워드 포함
  육류(볶음·불고기)류 : Class in {'볶음','구이','조림'} AND 메뉴명에 육류 키워드 포함
  두부·콩류조림류 : Class=='조림' AND 메뉴명에 두부/콩 키워드 포함
  (메뉴별 정확한 주재료군 DB 매핑은 재료명 표기가 일정치 않아 이번엔 키워드 규칙으로 근사함 —
   한계로 명시)

모델 분포는 이전 진단(diagnose_soup_bias_stage_FOOK.py)과 동일한 단일 seed 행(각 앵커의 "첫
등장" 행: 고등어구이=55, 제육불고기=152, 두부양념조림=36)에서 동일 방식으로 재계산해 사용한다
(soup_distribution_summary.csv와 동일 소스, 재현성 확인 겸 JS divergence용 전체 확률벡터 필요).

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python diagnose_soup_training_bias_FOOK.py
"""
import os, sys, io, csv, copy
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from scipy.spatial.distance import jensenshannon

FINAL = r'E:\final'
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, FINAL)

from Model import Sequence_Generator

OUT_DIR = os.path.join(CODE, 'diagnose_soup_training_bias_out')
DATA_CSV = os.path.join(FINAL, 'data', 'FOOK_meals_for_model.csv')

ANCHOR_CONDITIONS = [('생선구이', '고등어구이', 55), ('육류', '제육불고기', 152), ('두부콩류', '두부양념조림', 36)]

FISH_KW = ('고등어', '갈치', '삼치', '가자미', '조기', '꽁치', '임연수', '병어', '전어', '방어',
           '코다리', '대구', '광어', '연어', '장어', '민어', '보리멸', '동태', '북어', '명태')
MEAT_KW = ('돼지', '소고기', '닭', '오리', '제육', '불고기', '갈비', '삼겹', '차돌', '육개장',
           '돈육', '한우', '안심', '등심')
TOFU_KW = ('두부', '콩', '두류')


def classify_group(menu, cls):
    if cls == '구이' and any(k in menu for k in FISH_KW):
        return '생선구이류'
    if cls in ('볶음', '구이', '조림') and any(k in menu for k in MEAT_KW):
        return '육류(볶음·불고기)류'
    if cls == '조림' and any(k in menu for k in TOFU_KW):
        return '두부·콩류조림류'
    return None


def entropy_of(counter_or_arr):
    if isinstance(counter_or_arr, Counter):
        vals = np.array(list(counter_or_arr.values()), dtype=float)
    else:
        vals = np.asarray(counter_or_arr, dtype=float)
    total = vals.sum()
    if total == 0:
        return 0.0
    p = vals / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


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
    _, dec_hidden, _ = gen_obj.decoder(seeds_batch[:, 0], dec_hidden, enc_output)
    outputs, dec_hidden, _ = gen_obj.decoder(seeds_batch[:, 1], dec_hidden, enc_output)
    probs = np.array(outputs, dtype=float)
    if probs.ndim == 1:
        probs = probs[None, :]
    p = probs[0].copy()
    for t in core.SPECIAL:
        p[t] = 0.0
    for t in core.BLOCK_TOK:
        p[t] = 0.0
    p[anchor_token] = 0.0
    masked = p * core.SLOT_OK[1]
    anchor_grp = core.TOK_GRP.get(anchor_token)
    if anchor_grp is not None:
        masked[core.GRP_TOK[anchor_grp]] = 0.0
    if masked.sum() > 0:
        p = masked
    p = np.clip(p, 1e-12, None)
    p = p ** (1.0 / 0.8)
    p /= p.sum()
    return p


def dist_stats(counter, top_n=5):
    total = sum(counter.values())
    if total == 0:
        return None
    top1_menu, top1_cnt = counter.most_common(1)[0]
    return {
        'training_count': total, 'unique_soup_count': len(counter), 'top1_soup': top1_menu,
        'top1_ratio': top1_cnt / total,
        'top5_concentration': sum(v for _, v in counter.most_common(top_n)) / total,
        'entropy': entropy_of(counter),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    diet_raw = pd.read_csv(DATA_CSV)   # slot1=밥 slot2=국 slot3=주찬 slot4=부찬 slot5=김치

    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    os.chdir(cwd)
    class_of = core._m2c

    print('Seq2Seq(모방) 체크포인트 로딩...')
    gen_imit = build_gen(os.path.join(CODE, 'results_FOOK', 'checkpoints'),
                          core.food_dict, core.nutrient_data, core.incidence, core.diet_np.shape[0])
    gen_rl = core.gen
    orig_diet_np_np = core.diet_np.numpy()

    # ── ① training_main_soup_distribution.csv (정확메뉴 + 유형군, 두 match_type) ──
    dist_rows = []
    exact_stats = {}
    group_stats = {}
    group_members = {}

    for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
        sub = diet_raw[diet_raw['slot3'] == anchor_menu]
        c = Counter(sub['slot2'])
        for soup, cnt in c.most_common():
            dist_rows.append([anchor_menu, cond_name, '정확메뉴', len(sub), soup, cnt, cnt / len(sub)])
        exact_stats[cond_name] = dist_stats(c)
        exact_stats[cond_name]['anchor_menu'] = anchor_menu

    all_mains = diet_raw['slot3'].dropna().unique().tolist()
    for m in all_mains:
        g = classify_group(m, class_of.get(m))
        if g:
            group_members.setdefault(g, []).append(m)

    group_to_cond = {'생선구이류': '생선구이', '육류(볶음·불고기)류': '육류', '두부·콩류조림류': '두부콩류'}
    for grp_name, members in group_members.items():
        cond_name = group_to_cond.get(grp_name, grp_name)
        sub = diet_raw[diet_raw['slot3'].isin(members)]
        c = Counter(sub['slot2'])
        for soup, cnt in c.most_common():
            dist_rows.append([grp_name, cond_name, '유형군', len(sub), soup, cnt, cnt / len(sub)])
        group_stats[cond_name] = dist_stats(c)
        group_stats[cond_name]['anchor_group'] = grp_name
        group_stats[cond_name]['member_count'] = len(members)

    dist_csv = os.path.join(OUT_DIR, 'training_main_soup_distribution.csv')
    with open(dist_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['anchor_menu_또는_유형군', 'anchor_condition', 'match_type', 'training_count',
                    'soup_menu', 'cooccurrence_count', 'cooccurrence_ratio'])
        w.writerows(dist_rows)
    print(f'① {dist_csv} ({len(dist_rows)}행)')
    print('유형군 구성:', {k: len(v) for k, v in group_members.items()}, '\n  ', group_members)

    # ── ② training_main_soup_summary.csv ──
    summary_csv = os.path.join(OUT_DIR, 'training_main_soup_summary.csv')
    with open(summary_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['anchor_menu_또는_유형군', 'anchor_condition', 'match_type', 'training_count',
                    'unique_soup_count', 'top1_soup', 'top1_ratio', 'top5_concentration', 'entropy'])
        for cond_name, anchor_menu, _ in ANCHOR_CONDITIONS:
            s = exact_stats[cond_name]
            w.writerow([anchor_menu, cond_name, '정확메뉴', s['training_count'], s['unique_soup_count'],
                        s['top1_soup'], s['top1_ratio'], s['top5_concentration'], s['entropy']])
        for cond_name, s in group_stats.items():
            w.writerow([s['anchor_group'], cond_name, f"유형군(구성{s['member_count']}종)",
                        s['training_count'], s['unique_soup_count'], s['top1_soup'], s['top1_ratio'],
                        s['top5_concentration'], s['entropy']])
    print(f'② {summary_csv}')

    # ── ③ model_vs_training_soup.csv ──
    model_stats = {}
    for cond_name, anchor_menu, row_idx in ANCHOR_CONDITIONS:
        seed_row = orig_diet_np_np[row_idx].copy()
        decoded = [core.food_dict[int(t)] for t in seed_row]
        assert decoded[3] == anchor_menu
        anchor_token = core.name2idx[anchor_menu]
        seeds_batch = tf.constant(np.tile(seed_row, (10, 1)), dtype=tf.int32)   # batch>1 회피용, 결과 동일

        p_imit = soup_dist(core, gen_imit, seeds_batch, anchor_token)
        p_rl = soup_dist(core, gen_rl, seeds_batch, anchor_token)

        # 학습데이터 경험적 분포를 동일 토큰공간 벡터로(정확메뉴 기준)
        c = Counter(diet_raw[diet_raw['slot3'] == anchor_menu]['slot2'])
        data_vec = np.zeros(len(core.food_dict))
        for soup, cnt in c.items():
            if soup in core.name2idx:
                data_vec[core.name2idx[soup]] = cnt
        data_vec = data_vec / data_vec.sum()

        i_top = int(np.argmax(p_imit)); r_top = int(np.argmax(p_rl))
        model_stats[cond_name] = {
            'anchor_menu': anchor_menu,
            'training_top1_soup': exact_stats[cond_name]['top1_soup'],
            'training_top1_ratio': exact_stats[cond_name]['top1_ratio'],
            'training_entropy': exact_stats[cond_name]['entropy'],
            'seq2seq_top1_soup': core.food_dict[i_top], 'seq2seq_top1_probability': float(p_imit[i_top]),
            'seq2seq_entropy': entropy_of(p_imit),
            'rl_top1_soup': core.food_dict[r_top], 'rl_top1_probability': float(p_rl[r_top]),
            'rl_entropy': entropy_of(p_rl),
            'seq2seq_js_divergence': float(jensenshannon(data_vec, p_imit)),
            'rl_js_divergence': float(jensenshannon(data_vec, p_rl)),
        }

    trans_csv = os.path.join(OUT_DIR, 'model_vs_training_soup.csv')
    with open(trans_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['anchor_condition', 'anchor_menu', 'training_top1_soup', 'training_top1_ratio',
                    'training_entropy', 'seq2seq_top1_soup', 'seq2seq_top1_probability', 'seq2seq_entropy',
                    'rl_top1_soup', 'rl_top1_probability', 'rl_entropy',
                    'seq2seq_overconfidence', 'rl_overconfidence',
                    'seq2seq_entropy_ratio', 'rl_entropy_ratio',
                    'training_top1_eq_seq2seq_top1', 'training_top1_eq_rl_top1',
                    'seq2seq_js_divergence', 'rl_js_divergence'])
        for cond_name, _, _ in ANCHOR_CONDITIONS:
            m = model_stats[cond_name]
            seq_overconf = m['seq2seq_top1_probability'] - m['training_top1_ratio']
            rl_overconf = m['rl_top1_probability'] - m['training_top1_ratio']
            seq_ent_ratio = m['seq2seq_entropy'] / m['training_entropy'] if m['training_entropy'] else float('nan')
            rl_ent_ratio = m['rl_entropy'] / m['training_entropy'] if m['training_entropy'] else float('nan')
            w.writerow([cond_name, m['anchor_menu'], m['training_top1_soup'], m['training_top1_ratio'],
                        m['training_entropy'], m['seq2seq_top1_soup'], m['seq2seq_top1_probability'],
                        m['seq2seq_entropy'], m['rl_top1_soup'], m['rl_top1_probability'], m['rl_entropy'],
                        seq_overconf, rl_overconf, seq_ent_ratio, rl_ent_ratio,
                        m['training_top1_soup'] == m['seq2seq_top1_soup'],
                        m['training_top1_soup'] == m['rl_top1_soup'],
                        m['seq2seq_js_divergence'], m['rl_js_divergence']])
    print(f'③ {trans_csv}')

    return exact_stats, group_stats, model_stats


if __name__ == '__main__':
    main()
