# -*- coding: utf-8 -*-
"""
train_FOOK_soupmask_1000.py — A/B/C를 프로덕션과 동일 조건(전체 1095끼·1000epoch·split없음)으로
재학습. 이전 300epoch 버전(train_FOOK_soupmask.py)은 baseline 재현에 실패했음이 확인됐고
(baseline_reproduction_audit.md), 원인은 에폭 수(300이 아니라 1000이 실제 프로덕션 조건 —
project_generation_model_wiring.md에 기록돼 있었음)였다.

A는 원본과 완전히 같은 vocab(827, SOUP_MASK 없음)을 쓴다. B/C만 828(SOUP_MASK 추가).
A/B/C 전부 seed를 명시적으로 고정한다(원본엔 없던 것 — 공정 비교를 위해 이번에만 추가).

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python train_FOOK_soupmask_1000.py --variant A_baseline
  python train_FOOK_soupmask_1000.py --variant B_mask50
  python train_FOOK_soupmask_1000.py --variant C_mask100
"""
import os, sys, time, argparse, copy, json, csv, random, datetime
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import numpy as np
import pandas as pd
import tensorflow as tf

from util import nutrition_preprocessor, diet_sequence_preprocessor, food_to_token, diet_to_incidence
from Model import Encoder, Decoder

DATA = os.path.join('..', '..', '..', 'data')
NUT = os.path.join(DATA, 'FOOK_nutrition.csv')
MEALS = os.path.join(DATA, 'FOOK_meals_for_model.csv')

SOUP_POS = 2
SEED = 42
OUT_ROOT = './checkpoints_masking_1000'


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as e:
        print('deterministic op 설정 실패(무시하고 진행):', e)


def build_data(with_mask):
    feature = pd.read_csv(NUT)
    nutrient_data, food_dict = nutrition_preprocessor(feature_data=feature)()
    diet = pd.read_csv(MEALS)
    dsp = diet_sequence_preprocessor(sequence_data=diet, DB_quality='correct2', integrate=False)
    diet = dsp(nutrient_data)
    diet_np = food_to_token(diet, nutrient_data, empty_delete=True, num_empty=3)
    incidence = diet_to_incidence(diet_np, food_dict)
    mask_id = None
    if with_mask:
        mask_id = len(food_dict)
        food_dict = dict(food_dict)
        food_dict[mask_id] = '<SOUP_MASK>'
    return nutrient_data, food_dict, diet_np, incidence, mask_id


def apply_soup_mask(x_np, mask_id, mode, rng):
    out = x_np.copy()
    if mode == 'none':
        return out
    if mode == 'p50':
        flip = rng.random(out.shape[0]) < 0.5
        out[flip, SOUP_POS] = mask_id
        return out
    if mode == 'p100':
        out[:, SOUP_POS] = mask_id
        return out
    raise ValueError(mode)


VARIANT_CFG = {
    'A_baseline': {'with_mask': False, 'train_mask_mode': 'none', 'infer_mask_mode': 'none', 'mask_prob': 0.0},
    'B_mask50': {'with_mask': True, 'train_mask_mode': 'p50', 'infer_mask_mode': 'p100', 'mask_prob': 0.5},
    'C_mask100': {'with_mask': True, 'train_mask_mode': 'p100', 'infer_mask_mode': 'p100', 'mask_prob': 1.0},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=list(VARIANT_CFG.keys()), required=True)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--fc_dim', type=int, default=64)
    ap.add_argument('--seed', type=int, default=SEED)
    args = ap.parse_args()

    cfg = VARIANT_CFG[args.variant]
    set_all_seeds(args.seed)

    nutrient_data, food_dict, diet_np, incidence, mask_id = build_data(cfg['with_mask'])
    num_tokens = len(food_dict)
    x_np = diet_np.numpy()   # 전체 1095끼, split 없음(원본과 동일)
    print(f'[{args.variant}] n={x_np.shape[0]}(전체, split없음) num_tokens={num_tokens} '
          f'mask_id={mask_id} seed={args.seed}')

    kwargs = {'num_tokens': num_tokens, 'embed_dim': args.embed_dim, 'fc_dim': args.fc_dim,
              'fully-connected_layer': 'GRU', 'attention': True}
    encoder = Encoder(**kwargs, batch_size=x_np.shape[0])
    decoder = Decoder(**kwargs, batch_size=x_np.shape[0])
    optimizer = tf.keras.optimizers.Adam(args.lr)
    sce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

    rng = np.random.default_rng(args.seed + 1)   # 마스킹 난수는 학습 난수와 분리(재현성)
    exp_dir = os.path.join(OUT_ROOT, args.variant)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

    start_time = datetime.datetime.now().isoformat()
    history = []
    t0 = time.time()
    dec_in = tf.constant(x_np[:, :x_np.shape[1] - 1])
    targets = tf.constant(x_np[:, 1:])

    for epoch in range(args.epochs):
        enc_in_np = apply_soup_mask(x_np, mask_id, cfg['train_mask_mode'], rng)
        enc_in = tf.constant(enc_in_np[:, :enc_in_np.shape[1] - 1])

        with tf.GradientTape() as tape:
            enc_hidden0 = tf.zeros([enc_in.shape[0], encoder.units])
            enc_output, enc_hidden = encoder(enc_in, enc_hidden0)
            dec_hidden = copy.deepcopy(enc_hidden)
            total_loss = 0.0
            for t in range(dec_in.shape[1]):
                preds, dec_hidden, _ = decoder(dec_in[:, t], dec_hidden, enc_output)
                if preds.ndim == 1:
                    preds = tf.reshape(preds, [1, -1])
                sl = sce(tf.reshape(targets[:, t], [-1, 1]), preds)
                total_loss += sl
            sum_loss = tf.reduce_sum(total_loss)          # 그래디언트용(원본과 동일한 암묵적 sum)
            mean_loss = tf.reduce_mean(total_loss) / dec_in.shape[1]   # 로깅용

        tv = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(sum_loss, tv)
        optimizer.apply_gradients(zip(grads, tv))

        train_loss = float(mean_loss)
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == args.epochs - 1:
            elapsed = time.time() - t0
            print(f'  [epoch {epoch+1:4d}] train_loss={train_loss:.4f} elapsed={elapsed:.1f}s')
            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'elapsed_sec': elapsed})

    elapsed_total = time.time() - t0
    end_time = datetime.datetime.now().isoformat()
    ckpt_path = ckpt.save(file_prefix=os.path.join(ckpt_dir, 'ckpt'))
    print(f'[{args.variant}] 총 {elapsed_total:.1f}s, 저장: {ckpt_path}')

    meta = {
        'variant': args.variant, 'command': f'python train_FOOK_soupmask_1000.py --variant {args.variant}',
        'epochs': args.epochs, 'random_seed': args.seed, 'vocab_size': num_tokens, 'mask_id': mask_id,
        'mask_probability_train': cfg['mask_prob'], 'infer_mask_mode': cfg['infer_mask_mode'],
        'checkpoint_path': ckpt_path, 'start_time': start_time, 'end_time': end_time,
        'elapsed_sec': elapsed_total, 'final_train_loss': train_loss,
        'embed_dim': args.embed_dim, 'fc_dim': args.fc_dim, 'lr': args.lr,
        'data_n': int(x_np.shape[0]), 'train_val_split': False,
    }
    with open(os.path.join(exp_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log_csv = os.path.join(exp_dir, 'training_log.csv')
    with open(log_csv, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'epoch', 'train_loss', 'checkpoint_path', 'random_seed',
                    'mask_probability', 'vocab_size', 'elapsed_time'])
        for h in history:
            w.writerow([args.variant, h['epoch'], h['train_loss'], ckpt_path, args.seed,
                        cfg['mask_prob'], num_tokens, h['elapsed_sec']])
    print(f'meta: {exp_dir}/meta.json, log: {log_csv}')


if __name__ == '__main__':
    main()
