# -*- coding: utf-8 -*-
"""
train_FOOK.py  — 투석 한끼 식단 생성모델 학습 (FOOK 배선, 모방 baseline)

레포의 Preprocessing.py+main.py(argparse) 얽힘을 피한 독립 진입점.
- 데이터: data/FOOK_nutrition.csv (name,weight,Class,E,Prot,K,P,Na) + data/FOOK_meals_for_model.csv (1095x5)
- 토큰: 0=empty, 1..824=메뉴, 825=시작, 826=종료  (nutrition_preprocessor가 구성)
- 학습: off-policy(=teacher forcing) + imitation_only=True -> 순수 모방(MLE). 영양은 이후 레버가 담당.

실행 (foodbert 환경):
  conda activate foodbert
  set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python train_FOOK.py --epochs 300
"""
import os, sys, time, argparse
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
import tensorflow as tf

from util import (nutrition_preprocessor, diet_sequence_preprocessor,
                  food_to_token, diet_to_incidence, sequence_to_sentence,
                  get_score_pandas, get_reward_ver2)
from Model import Sequence_Generator

# 데이터 경로 (Code/ 기준 상대경로 -> E:/final/data)
DATA = os.path.join('..', '..', '..', 'data')
NUT = os.path.join(DATA, 'FOOK_nutrition.csv')
MEALS = os.path.join(DATA, 'FOOK_meals_for_model.csv')


def build_data():
    feature = pd.read_csv(NUT)                       # utf-8-sig 자동
    nutrient_data, food_dict = nutrition_preprocessor(feature_data=feature)()

    diet = pd.read_csv(MEALS)                         # 1095 x 5 (slot1..5)
    dsp = diet_sequence_preprocessor(sequence_data=diet, DB_quality='correct2', integrate=False)
    diet = dsp(nutrient_data)                         # 시작/종료 삽입 + 결측 empty화
    diet_np = food_to_token(diet, nutrient_data, empty_delete=True, num_empty=3)
    incidence = diet_to_incidence(diet_np, food_dict)
    return nutrient_data, food_dict, diet_np, incidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--fc_dim', type=int, default=64)
    ap.add_argument('--network', type=str, default='GRU')       # GRU or LSTM
    ap.add_argument('--attention', action='store_true', default=True)
    args = ap.parse_args()

    nutrient_data, food_dict, diet_np, incidence = build_data()
    batch_size = int(diet_np.shape[0])
    print(f'데이터: {batch_size}끼, 시퀀스길이 {diet_np.shape[1]}, 토큰 {len(food_dict)}')

    kwargs = {
        'fully-connected_layer': args.network,
        'attention': args.attention,
        'embed_dim': args.embed_dim,
        'fc_dim': args.fc_dim,
        'learning': 'off-policy',      # = teacher forcing (실제 시퀀스로 loss)
        'policy': 'greedy',
        'use_beta': False,
        'use_buffer': False,
        'buffer_size': 5,
        'buffer_update': 5,
        'num_epochs': args.epochs,
        'lr': args.lr,
        'num_tokens': len(food_dict),
        'batch_size': batch_size,
        'imitation_only': True,        # ★ 순수 모방 baseline (보상 중화)
    }

    gen = Sequence_Generator(food_dict, nutrient_data, incidence, **kwargs)
    x = tf.convert_to_tensor(diet_np)                 # 전체 풀배치

    os.makedirs('./results_FOOK/checkpoints', exist_ok=True)
    ckpt = tf.train.Checkpoint(generator=gen)

    start = time.time()
    for epoch in range(args.epochs):
        real_seqs, batch_loss, pred_seqs, _, _, _, _ = gen.train(x, x)
        if (epoch + 1) % 20 == 0 or epoch == 0:
            # 모니터링: 생성 한끼 예시 + 영양충족(0~5)
            gen_scores = get_score_pandas(pred_seqs, food_dict, nutrient_data)
            gen_reward = np.apply_along_axis(get_reward_ver2, arr=gen_scores, axis=1, done=0)[:, 0]
            sample = sequence_to_sentence(pred_seqs, food_dict)[0]
            print(f'[epoch {epoch+1:4d}] loss={batch_loss:.4f}  생성영양충족(평균/5)={gen_reward.mean():.2f}  '
                  f'예시={sample}')
    print(f'총 시간 {time.time()-start:.1f}s')
    ckpt.save(file_prefix='./results_FOOK/checkpoints/ckpt')
    print('체크포인트 저장: ./results_FOOK/checkpoints/')


if __name__ == '__main__':
    main()
