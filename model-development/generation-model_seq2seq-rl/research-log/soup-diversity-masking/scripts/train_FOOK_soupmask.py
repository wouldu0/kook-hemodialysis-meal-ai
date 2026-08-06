# -*- coding: utf-8 -*-
"""
train_FOOK_soupmask.py — 국 슬롯 seed 복사(shortcut learning) 제거용 target-slot masking 학습.

배경: 지난 진단(diagnose_soup_seed_dependency_FOOK.py)에서 확인한 사실 —
  Seq2Seq는 seed의 국 값을 100% 그대로 복사해서 출력한다(다른 슬롯이 아니라 seed 자체가 정답을
  이미 들고 있는 target leakage). seed 국 위치를 마스킹하면 seed간 분포거리(JSD)가 0.79->0.16으로
  거의 사라짐 — 즉 마스킹이 실제로 복사 경로를 끊는다는 게 이미 확인됨. 이 스크립트는 그 마스킹을
  "진단"이 아니라 "학습 자체"에 적용한다.

구조 확인(Model.py 직접 읽어서 확인):
  Sequence_Generator.train()의 off-policy 루프에서, 슬롯 s를 예측하는 스텝의 디코더 "입력"은
  시퀀스 위치 s(=이전 슬롯의 실제값, teacher-forced)이지 슬롯 s 자신의 값이 아니다. 슬롯 s 자신의
  값이 슬롯 s의 예측에 영향을 주는 유일한 경로는 인코더가 전체 시퀀스를 한 번에 읽을 때
  (enc_output/enc_hidden, 그리고 어텐션이 매 스텝 enc_output을 다시 참조)뿐이다.
  그래서 마스킹은 "인코더에 들어가는 시퀀스"에서만 국 위치(포지션2)를 바꾸고, 디코더의 스텝별
  teacher-forcing 입력(다른 슬롯 예측용)은 그대로 둔다 — 국 슬롯의 shortcut만 정확히 겨냥.

기존 파일은 건드리지 않음: Model.py(Encoder/Decoder/Sequence_Generator)는 그대로 재사용하고,
학습 루프만 이 파일에 새로 작성(Sequence_Generator.train()을 안 씀 — reward 계산 등 imitation_only
에서 안 쓰이는 분기가 새 토큰(SOUP_MASK)에 대해 nutrient_data 조회를 시도하면 깨질 수 있어서,
imitation 전용의 순수 CE 루프를 별도로 짬).

실행 예:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python train_FOOK_soupmask.py --variant A --epochs 300
  python train_FOOK_soupmask.py --variant B --epochs 300
  python train_FOOK_soupmask.py --variant C --epochs 300
  python train_FOOK_soupmask.py --variant C --label_smoothing 0.1 --epochs 300
"""
import os, sys, time, argparse, copy, json
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

SOUP_POS = 2          # 7토큰 시퀀스([BOS,밥,국,주찬,부찬,김치,EOS])에서 "국" 위치
VAL_RATIO = 0.1
SPLIT_SEED = 42


def build_data_with_mask_token():
    feature = pd.read_csv(NUT)
    nutrient_data, food_dict = nutrition_preprocessor(feature_data=feature)()
    diet = pd.read_csv(MEALS)
    dsp = diet_sequence_preprocessor(sequence_data=diet, DB_quality='correct2', integrate=False)
    diet = dsp(nutrient_data)
    diet_np = food_to_token(diet, nutrient_data, empty_delete=True, num_empty=3)
    incidence = diet_to_incidence(diet_np, food_dict)

    mask_id = len(food_dict)                    # 827 (기존 0..826) -> 새 토큰 827
    food_dict = dict(food_dict)
    food_dict[mask_id] = '<SOUP_MASK>'
    return nutrient_data, food_dict, diet_np, incidence, mask_id


def split_train_val(diet_np, val_ratio=VAL_RATIO, seed=SPLIT_SEED):
    n = diet_np.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(n * val_ratio)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    x = diet_np.numpy()
    return tf.constant(x[train_idx]), tf.constant(x[val_idx]), train_idx, val_idx


def apply_soup_mask(x_np, mask_id, mode, rng):
    """x_np: (n,7) numpy. mode: 'A_원본'(변경없음) / 'B_학습시50%'(50% 확률 마스킹, train 전용) /
    'C_항상마스킹'(항상 마스킹, train과 추론 공통)."""
    out = x_np.copy()
    if mode == 'A_원본':
        return out
    if mode == 'B_학습시50%':
        flip = rng.random(out.shape[0]) < 0.5
        out[flip, SOUP_POS] = mask_id
        return out
    if mode == 'C_항상마스킹':
        out[:, SOUP_POS] = mask_id
        return out
    raise ValueError(mode)


def make_ce(label_smoothing):
    if label_smoothing and label_smoothing > 0:
        cce = tf.keras.losses.CategoricalCrossentropy(reduction='none', label_smoothing=label_smoothing)

        def loss_fn(targets_1d, preds, num_classes):
            onehot = tf.one_hot(tf.reshape(targets_1d, [-1]), num_classes)
            return cce(onehot, preds)
        return loss_fn
    else:
        sce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

        def loss_fn(targets_1d, preds, num_classes):
            return sce(tf.reshape(targets_1d, [-1, 1]), preds)
        return loss_fn


def forward_loss(encoder, decoder, enc_inputs, dec_inputs, targets, loss_fn, num_classes, training):
    """imitation(off-policy, teacher-forcing) 순수 CE. enc_inputs=인코더용(마스킹 반영),
    dec_inputs=디코더 스텝별 teacher-forcing 입력(마스킹 미반영, 슬롯 s 예측엔 슬롯 s의 seed값이
    아니라 s-1 위치값이 들어가므로 애초에 국 자기자신 정보가 아님 — 그대로 둬도 leak 아님)."""
    enc_hidden0 = tf.zeros([enc_inputs.shape[0], encoder.units])
    enc_output, enc_hidden = encoder(enc_inputs, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)

    total_loss = 0.0
    per_slot_loss = []
    for t in range(dec_inputs.shape[1]):
        preds, dec_hidden, _ = decoder(dec_inputs[:, t], dec_hidden, enc_output)
        if preds.ndim == 1:
            preds = tf.reshape(preds, [1, -1])
        step_loss = loss_fn(targets[:, t], preds, num_classes)
        total_loss += step_loss
        per_slot_loss.append(tf.reduce_mean(step_loss))
    # 중요: 원본 Sequence_Generator.train()은 total_loss(=배치별 미축약 (n,) 텐서)를 그대로
    # tape.gradient()에 넘긴다 — TF는 비스칼라 타깃에 대해 암묵적으로 sum을 취해 미분한다.
    # 여기서 reduce_mean/steps로 축약한 스칼라를 gradient target으로 쓰면 그래디언트 규모가
    # (배치크기 x 스텝수)배만큼 작아져 Adam이 사실상 거의 안 움직인다(실측: 이 버그로 학습된
    # 모델이 국 슬롯에서 seed 복사를 전혀 재현 못하고 거의 균등분포를 냄 — 원본과 비교 불가능한
    # 상태였음, 2026-07-27 발견). sum_loss=그래디언트용, mean_loss=로깅/얼리스토핑 비교용으로 분리.
    sum_loss = tf.reduce_sum(total_loss)
    mean_loss = tf.reduce_mean(total_loss) / dec_inputs.shape[1]
    return sum_loss, mean_loss, per_slot_loss


def run_experiment(variant, label_smoothing, epochs, lr, embed_dim, fc_dim, patience, out_root, seed=0):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    nutrient_data, food_dict, diet_np, incidence, mask_id = build_data_with_mask_token()
    num_tokens = len(food_dict)
    x_train, x_val, train_idx, val_idx = split_train_val(diet_np)
    print(f'[{variant} ls={label_smoothing}] train={x_train.shape[0]} val={x_val.shape[0]} '
          f'토큰수={num_tokens}(SOUP_MASK id={mask_id})')

    kwargs = {'num_tokens': num_tokens, 'embed_dim': embed_dim, 'fc_dim': fc_dim,
              'fully-connected_layer': 'GRU', 'attention': True}
    encoder = Encoder(**kwargs, batch_size=x_train.shape[0])
    decoder = Decoder(**kwargs, batch_size=x_train.shape[0])
    optimizer = tf.keras.optimizers.Adam(lr)
    loss_fn = make_ce(label_smoothing)

    rng = np.random.default_rng(seed + 1)
    xt_np = x_train.numpy()
    xv_np = x_val.numpy()
    inf_mode = {'A_원본': 'A_원본', 'B_학습시50%': 'C_항상마스킹', 'C_항상마스킹': 'C_항상마스킹'}[variant]

    best_val = float('inf')
    best_epoch = -1
    bad_epochs = 0
    history = []
    # 디렉터리명은 ASCII만 사용(Windows/TF file_io가 비-ASCII 경로에서 간헐적으로 깨짐 확인됨)
    ascii_variant = {'A_원본': 'A_baseline', 'B_학습시50%': 'B_dropout50', 'C_항상마스킹': 'C_mask100'}[variant]
    exp_dir = os.path.join(out_root, f'{ascii_variant}_ls{label_smoothing}')
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

    t0 = time.time()
    for epoch in range(epochs):
        enc_in_np = apply_soup_mask(xt_np, mask_id, variant, rng)
        enc_in = tf.constant(enc_in_np)
        dec_in = tf.constant(xt_np[:, :xt_np.shape[1] - 1])
        targets = tf.constant(xt_np[:, 1:])

        with tf.GradientTape() as tape:
            sum_loss, mean_loss, _ = forward_loss(encoder, decoder, enc_in[:, :enc_in.shape[1] - 1],
                                                   dec_in, targets, loss_fn, num_tokens, training=True)
        tv = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(sum_loss, tv)
        optimizer.apply_gradients(zip(grads, tv))

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == epochs - 1:
            enc_val_np = apply_soup_mask(xv_np, mask_id, inf_mode, rng)
            enc_val = tf.constant(enc_val_np)
            dec_val = tf.constant(xv_np[:, :xv_np.shape[1] - 1])
            targets_val = tf.constant(xv_np[:, 1:])
            _, val_loss, _ = forward_loss(encoder, decoder, enc_val[:, :enc_val.shape[1] - 1],
                                           dec_val, targets_val, loss_fn, num_tokens, training=False)
            val_loss = float(val_loss)
            history.append({'epoch': epoch + 1, 'train_loss': float(mean_loss), 'val_loss': val_loss})
            print(f'  [epoch {epoch+1:4d}] train_loss={float(mean_loss):.4f} val_loss={val_loss:.4f}')
            if val_loss < best_val - 1e-4:
                best_val, best_epoch, bad_epochs = val_loss, epoch + 1, 0
                ckpt.save(file_prefix=os.path.join(ckpt_dir, 'ckpt'))
            else:
                bad_epochs += 10
                if bad_epochs >= patience:
                    print(f'  early stopping @ epoch {epoch+1} (best={best_val:.4f} @ {best_epoch})')
                    break
    elapsed = time.time() - t0
    print(f'[{variant} ls={label_smoothing}] 총 {elapsed:.1f}s, best_val_loss={best_val:.4f} @ epoch {best_epoch}')

    meta = {'variant': variant, 'label_smoothing': label_smoothing, 'epochs_run': epoch + 1,
            'best_val_loss': best_val, 'best_epoch': best_epoch, 'elapsed_sec': elapsed,
            'mask_id': mask_id, 'num_tokens': num_tokens, 'train_idx': train_idx.tolist(),
            'val_idx': val_idx.tolist(), 'history': history,
            'embed_dim': embed_dim, 'fc_dim': fc_dim}
    with open(os.path.join(exp_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['A_원본', 'B_학습시50%', 'C_항상마스킹'], required=True)
    ap.add_argument('--label_smoothing', type=float, default=0.0)
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--embed_dim', type=int, default=128)
    ap.add_argument('--fc_dim', type=int, default=64)
    ap.add_argument('--patience', type=int, default=60)
    ap.add_argument('--out_root', type=str, default='./results_soupmask_FOOK')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    run_experiment(args.variant, args.label_smoothing, args.epochs, args.lr, args.embed_dim,
                    args.fc_dim, args.patience, args.out_root, args.seed)


if __name__ == '__main__':
    main()
