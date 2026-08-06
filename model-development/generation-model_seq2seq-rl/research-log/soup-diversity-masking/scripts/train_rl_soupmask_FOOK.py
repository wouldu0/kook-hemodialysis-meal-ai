# -*- coding: utf-8 -*-
"""
train_rl_soupmask_FOOK.py — C_mask100을 warm-start로 RL(REINFORCE) 재학습.

절대 원칙(코드로 강제):
  - encoder 입력의 국 위치(SOUP_POS)는 매 스텝 항상 SOUP_MASK로 고정한다. anchor_slots가
    무엇이든(국이 앵커로 뽑히더라도) 이 마스킹은 예외 없이 적용한다 — 아래 assert로 매 스텝 검증.
  - 기존 RL 체크포인트(results_sweep_FOOK/i002)는 이 파일 어디서도 import/로드하지 않는다.
  - optimizer는 새로 생성(Adam, 새 state) — 기존 RL optimizer state 재사용 없음.

보상 설계(reward_soupmask 함수):
  기존 reward_lever_FOOK.meal_reward() = pass_frac × (0.5 + 0.5×preserve) 를 베이스로 하되,
  거기 없던 두 항을 추가한다:
    - gate_penalty: 현실성 게이트(비현실적 양·재료겹침·고인군과다) 실패 시 0.6배로 감점
      (완전 배제 대신 완만한 페널티 — 원본 make_meal()의 '부분점수 선택'과 같은 철학)
    - diversity_penalty: 이번 배치 안에서 이 국/부찬이 이미 몇 번 나왔는지에 비례해 소폭 감점
      (mode collapse를 보상 자체에서 억제 — 특정 국 쏠림 재발 방지)
  imit_weight(모방 CE 앵커)는 기존 검증된 값 0.02를 그대로 사용 — 다양성 유지 목적의 안전장치.

실행:
  conda activate foodbert; set TF_USE_LEGACY_KERAS=1
  cd Diet-Generation-As-Sequence-master\\Diet-Generation-As-Sequence-master\\Code
  python train_rl_soupmask_FOOK.py --epochs 300
"""
import os, sys, io, time, copy, json, csv, argparse, random, datetime
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

from Model import Encoder, Decoder
from util import get_action
from train_FOOK_soupmask_1000 import build_data, SOUP_POS

WARM_START_DIR = os.path.join(CODE, 'checkpoints_masking_1000', 'C_mask100', 'checkpoints')
OUT_ROOT = os.path.join(CODE, 'results_rl_soupmask_FOOK')
SEED = 42
N_SLOTS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--imit_weight', type=float, default=0.02)
    ap.add_argument('--diversity_weight', type=float, default=0.15)
    ap.add_argument('--gate_penalty_factor', type=float, default=0.6)
    ap.add_argument('--eval_interval', type=int, default=20)
    ap.add_argument('--temp', type=float, default=1.0)
    args = ap.parse_args()

    print('=== 사전 확인 ===')
    print('warm-start:', WARM_START_DIR)
    print('결과 저장:', OUT_ROOT)
    print('기존 RL 체크포인트(results_sweep_FOOK/i002) 로드 안 함 — 이 스크립트는 그 경로를 import하지 않음')

    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    cwd = os.getcwd()
    os.chdir(FINAL)
    import app_core_FOOK as core
    import FOOK_adjust_levers as F
    import reward_lever_FOOK as R
    R.init(weight=60)
    os.chdir(cwd)
    b = F.meal_bounds(60)

    nutrient_data, food_dict, diet_np, incidence, mask_id = build_data(with_mask=True)
    num_tokens = len(food_dict)
    x_np = diet_np.numpy()
    n = x_np.shape[0]
    print(f'데이터: {n}끼, vocab={num_tokens}, mask_id={mask_id}')

    kwargs = {'num_tokens': num_tokens, 'embed_dim': 128, 'fc_dim': 64,
              'fully-connected_layer': 'GRU', 'attention': True}
    encoder = Encoder(**kwargs, batch_size=n)
    decoder = Decoder(**kwargs, batch_size=n)
    ck = tf.train.latest_checkpoint(WARM_START_DIR)
    assert ck is not None, f'warm-start 체크포인트 없음: {WARM_START_DIR}'
    tf.train.Checkpoint(encoder=encoder, decoder=decoder).restore(ck).expect_partial()
    print('warm-start 로딩 완료(새 optimizer로 시작):', ck)

    optimizer = tf.keras.optimizers.Adam(args.lr)   # 새 optimizer, 기존 state 재사용 없음
    sce = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

    exp_dir = os.path.join(OUT_ROOT, 'C_mask100_rl')
    ckpt_dir_last = os.path.join(exp_dir, 'checkpoints_last')
    ckpt_dir_best = os.path.join(exp_dir, 'checkpoints_best')
    os.makedirs(ckpt_dir_last, exist_ok=True)
    os.makedirs(ckpt_dir_best, exist_ok=True)
    ckpt_last = tf.train.Checkpoint(encoder=encoder, decoder=decoder)
    ckpt_best = tf.train.Checkpoint(encoder=encoder, decoder=decoder)

    x_tf_full = tf.constant(x_np)
    inputs_full = x_tf_full[:, :x_tf_full.shape[1] - 1]     # (n,6)
    targets_full = x_tf_full[:, 1:]                          # (n,6)

    train_log = []
    eval_log = []
    baseline_ev = [None]   # 첫 eval(=RL 시작 직후 C_mask100 근처)을 상대비교 기준선으로 사용
    best_composite, best_epoch = -1e9, -1
    warnings_log = []

    start_time = datetime.datetime.now().isoformat()
    t0 = time.time()
    for epoch in range(args.epochs):
        anchor_slots = np.random.randint(0, N_SLOTS, size=n)

        enc_in_np = x_np.copy()
        enc_in_np[:, SOUP_POS] = mask_id            # ★ 항상 마스킹(예외 없음)
        enc_inputs = tf.constant(enc_in_np[:, :enc_in_np.shape[1] - 1])
        assert int(np.unique(enc_in_np[:, SOUP_POS])[0]) == mask_id and \
               len(np.unique(enc_in_np[:, SOUP_POS])) == 1, 'encoder 국 위치 마스킹 불변식 위반!'

        with tf.GradientTape() as tape:
            enc_hidden0 = tf.zeros([n, encoder.units])
            enc_output, enc_hidden = encoder(enc_inputs, enc_hidden0)
            dec_hidden = copy.deepcopy(enc_hidden)

            pred_seqs = np.zeros((n, N_SLOTS), dtype=int)
            rl_loss = 0.0
            imit_loss = 0.0
            on_tars = None
            for t in range(inputs_full.shape[1]):
                if t == 0:
                    preds, dec_hidden, _ = decoder(inputs_full[:, t], dec_hidden, enc_output)
                else:
                    preds, dec_hidden, _ = decoder(tf.squeeze(on_tars), dec_hidden, enc_output)

                preds_np = np.array(preds, dtype=float)
                if preds_np.ndim == 1:
                    preds_np = preds_np[None, :]
                preds_np = preds_np.copy()
                preds_np[:, mask_id] = 0.0          # 마스크 토큰 자체가 출력으로 나오면 안 됨
                for spec in core.SPECIAL:
                    if spec < num_tokens:
                        preds_np[:, spec] = 0.0

                actions = np.zeros(n, dtype=int)
                for i in range(n):
                    p = preds_np[i]
                    s = p.sum()
                    p = p / s if s > 0 else np.full(len(p), 1.0 / len(p))
                    actions[i] = np.random.choice(len(p), p=p)

                if t < N_SLOTS:
                    force = anchor_slots == t
                    # 앵커 슬롯 값은 원본 데이터의 실제 토큰(디코더 '출력' 자리 강제 — 인코더 입력과는 별개)
                    real_here = np.array(targets_full[:, t])
                    actions[force] = real_here[force]
                    if t == 1:
                        pred_seqs_soup_from_real = real_here  # 참고용(로그만)

                on_tars = tf.reshape(tf.constant(actions, dtype=tf.int32), shape=(-1, 1))
                if t < N_SLOTS:
                    pred_seqs[:, t] = actions

                loss_t = sce(tf.reshape(targets_full[:, t], [-1, 1]), preds)
                if t < N_SLOTS:
                    keep = tf.cast(tf.not_equal(tf.constant(anchor_slots, dtype=tf.int32), t), tf.float32)
                    rl_loss += loss_t * keep
                if args.imit_weight > 0:
                    imit_loss += sce(tf.reshape(targets_full[:, t], [-1, 1]), preds)

            # ── 보상 계산(그래프 밖, numpy) ──
            menus_batch = [[food_dict[int(tok)] for tok in pred_seqs[i]] for i in range(n)]
            soup_counter = Counter(m[1] for m in menus_batch)
            side_counter = Counter(m[3] for m in menus_batch)
            rewards = np.zeros(n, dtype=np.float32)
            for i in range(n):
                menus = menus_batch[i]
                anchor_menu = menus[anchor_slots[i]]
                r, det = R.meal_reward(menus, anchor_menu, detail=True)
                if det:
                    _, _, inst_i, _ = F.adjust(list(menus), b, anchor=anchor_menu)
                    unreal = F.unrealistic_reason(inst_i)
                    clash = core._has_ingredient_clash(menus)
                    p_over = core._has_high_p_overload(menus)
                    if unreal is not None or clash or p_over:
                        r *= args.gate_penalty_factor
                    soup_freq = soup_counter[menus[1]] / n
                    side_freq = side_counter[menus[3]] / n
                    r -= args.diversity_weight * 0.5 * (soup_freq + side_freq)
                    r = max(0.0, r)
                rewards[i] = r

            adv = np.empty_like(rewards)
            for s in np.unique(anchor_slots):
                m = anchor_slots == s
                adv[m] = rewards[m] - (rewards[m].mean() if m.sum() > 1 else rewards.mean())

            final_loss = rl_loss * tf.constant(adv, dtype=tf.float32)
            if args.imit_weight > 0:
                final_loss = final_loss + args.imit_weight * imit_loss

        tv = encoder.trainable_variables + decoder.trainable_variables
        grads = tape.gradient(final_loss, tv)
        optimizer.apply_gradients(zip(grads, tv))

        reward_mean, reward_std = float(rewards.mean()), float(rewards.std())
        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f'[epoch {epoch+1:4d}] reward={reward_mean:.4f}±{reward_std:.4f} '
                  f'고유국={len(soup_counter)} 고유부찬={len(side_counter)} elapsed={elapsed:.1f}s')
            train_log.append({'epoch': epoch + 1, 'reward_mean': reward_mean, 'reward_std': reward_std,
                              'unique_soup': len(soup_counter), 'unique_side': len(side_counter),
                              'elapsed_sec': elapsed})

        if (epoch + 1) % args.eval_interval == 0 or epoch == args.epochs - 1:
            ev = quick_eval(core, F, encoder, decoder, num_tokens, mask_id, food_dict, diet_np.numpy(), b)
            ev['epoch'] = epoch + 1
            print(f'  [eval @ {epoch+1}] seed_copy={ev["seed_copy_rate"]:.2f} top1={ev["mean_top1_probability"]:.3f} '
                  f'anchor_jsd={ev["anchor_jsd"]:.3f} entropy={ev["mean_entropy"]:.2f} '
                  f'dish_hit={ev["dish_hit"]:.2f} nut_pass={ev["nutrition_all_pass_rate"]:.2f} '
                  f'tofu_nut={ev["tofu_nutrition_pass"]:.2f}')

            # quick_eval은 단순화 프록시(48후보 탐색 없이 1회 생성)라 절대수치가 원래 낮다
            # (예: 두부콩류 nut_pass는 Stage1 전체파이프라인 93%대와 달리 여기선 훨씬 낮게 나옴 —
            #  이건 이 프록시의 정상 범위다). 그래서 절대 임계값 대신, C_mask100 warm-start
            # 시점(첫 eval)을 기준선으로 잡고 "그 대비 얼마나 나빠졌는지"로 경고를 판단한다.
            if baseline_ev[0] is None:
                baseline_ev[0] = dict(ev)
                print('  (이 시점을 RL 기준선으로 기록 — 이후 경고는 이 대비 상대적 악화만 감지)')

            base = baseline_ev[0]
            warn = []
            if ev['seed_copy_rate'] > base['seed_copy_rate'] + 0.3:
                warn.append('seed_copy_rate 기준선 대비 급상승')
            if ev['anchor_jsd'] < base['anchor_jsd'] * 0.5:
                warn.append('anchor_jsd 기준선 대비 급락')
            if ev['dish_hit'] < 0.9:
                warn.append('dish_hit 하락')
            if ev['non_soup_top1_rate'] > 0.05:
                warn.append('국아닌메뉴 top1 증가')
            if ev['tofu_nutrition_pass'] < base['tofu_nutrition_pass'] * 0.5:
                warn.append('두부콩류 영양충족률 기준선 대비 급락')
            if ev['mean_entropy'] < base['mean_entropy'] * 0.3:
                warn.append('entropy 급감(mode collapse 의심)')
            composite = (ev['nutrition_all_pass_rate'] + ev['dish_hit'] - ev['seed_copy_rate']
                         + min(ev['anchor_jsd'], 1.0) + ev['tofu_nutrition_pass']
                         - ev['non_soup_top1_rate'])
            ev['composite_score'] = composite
            ev['warnings'] = ';'.join(warn)
            eval_log.append(ev)
            if warn:
                print('  ⚠ 경고:', ev['warnings'], '→ best 후보에서 제외')
                warnings_log.append({'epoch': epoch + 1, 'warnings': ev['warnings']})
            else:
                if composite > best_composite:
                    best_composite = composite
                    best_epoch = epoch + 1
                    ckpt_best.save(file_prefix=os.path.join(ckpt_dir_best, 'ckpt'))
                    print(f'  ★ 새 best (composite={composite:.3f})')

    elapsed_total = time.time() - t0
    end_time = datetime.datetime.now().isoformat()
    ckpt_last.save(file_prefix=os.path.join(ckpt_dir_last, 'ckpt'))
    print(f'\n총 {elapsed_total:.1f}s. best epoch={best_epoch}(composite={best_composite:.3f})')

    with open(os.path.join(exp_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'warm_start': ck, 'epochs': args.epochs, 'lr': args.lr,
                    'imit_weight': args.imit_weight, 'diversity_weight': args.diversity_weight,
                    'gate_penalty_factor': args.gate_penalty_factor, 'seed': SEED,
                    'best_epoch': best_epoch, 'best_composite': best_composite,
                    'start_time': start_time, 'end_time': end_time, 'elapsed_sec': elapsed_total,
                    'vocab_size': num_tokens, 'mask_id': mask_id,
                    'existing_rl_checkpoint_used': False}, f, ensure_ascii=False, indent=2)

    with open(os.path.join(exp_dir, 'rl_C_mask100_training_log.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'reward_mean', 'reward_std', 'unique_soup', 'unique_side', 'elapsed_sec'])
        w.writeheader(); w.writerows(train_log)

    with open(os.path.join(exp_dir, 'rl_C_mask100_eval_history.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        fieldnames = list(eval_log[0].keys()) if eval_log else []
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(eval_log)

    with open(os.path.join(exp_dir, 'warnings_log.csv'), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'warnings'])
        w.writeheader(); w.writerows(warnings_log)

    print('저장 완료:', exp_dir)
    return best_epoch, exp_dir


def quick_eval(core, F, encoder, decoder, num_tokens, mask_id, food_dict, diet_np_np, b,
               anchor_rows=((55, 2), (152, 2), (36, 2)), n_seeds=3, n_sample=60):
    """가벼운 모니터링용 평가(3앵커 x 3seed, n_sample). 최종 정식평가는 별도 스크립트에서 30조건 전체로."""
    seed_idx = [11, 12, 6][:n_seeds]
    results = []
    dists = {}
    dish_hits, non_soup, nut_pass, tofu_nut = [], [], [], []
    for row_idx, slot in anchor_rows:
        anchor_row = diet_np_np[row_idx].copy()
        anchor_token = anchor_row[slot + 1]
        cond_key = row_idx
        for sid in seed_idx:
            base_row = diet_np_np[sid].copy()
            row = base_row.copy()
            row[slot + 1] = anchor_token
            row[SOUP_POS] = mask_id
            seeds_batch = tf.constant(np.tile(row, (10, 1)), dtype=tf.int32)
            nb = 10
            enc_hidden0 = tf.zeros([nb, encoder.units])
            enc_output, enc_hidden = encoder(seeds_batch, enc_hidden0)
            dec_hidden = copy.deepcopy(enc_hidden)
            _, dec_hidden, _ = decoder(seeds_batch[:, 0], dec_hidden, enc_output)
            outputs, dec_hidden, _ = decoder(seeds_batch[:, 1], dec_hidden, enc_output)
            probs = np.array(outputs, dtype=float)
            if probs.ndim == 1:
                probs = probs[None, :]
            p = probs[0].copy()
            for t in core.SPECIAL:
                if t < num_tokens: p[t] = 0.0
            p[mask_id] = 0.0
            p[int(anchor_token)] = 0.0
            p = np.clip(p, 1e-12, None); p /= p.sum()
            top1 = int(np.argmax(p))
            top1_menu = food_dict[top1]
            cls = core._m2c.get(top1_menu)
            dish_hits.append(cls in ('국', '수프(간식)'))
            non_soup.append(cls not in ('국', '수프(간식)'))
            dists[(cond_key, sid)] = p
            results.append({'top1_prob': float(p[top1]), 'entropy': entropy_of(p),
                            'seed_match': food_dict[int(base_row[SOUP_POS])] == top1_menu})

            # 영양(단순 1회 생성+adjust로 근사, 무거운 48후보 생략 — 모니터링용)
            menus = generate_full_meal(core, encoder, decoder, num_tokens, mask_id, food_dict, row, int(anchor_token))
            _, after, inst, _ = F.adjust(menus, b, anchor=food_dict[int(anchor_token)])
            ok = (b['Elo'] <= after['E'] <= b['Ehi'] and b['Plo'] <= after['protein'] <= b['Phi']
                  and after['K'] < b['Kmax'] and after['P'] < b['Pmax'] and after['Na_season'] <= b['Namax'])
            nut_pass.append(ok)
            if row_idx == 36:   # 두부양념조림(두부콩류) 조건
                tofu_nut.append(ok)

    import itertools
    from scipy.spatial.distance import jensenshannon
    jsds = []
    conds = list(set(k[0] for k in dists))
    for sid in seed_idx:
        vecs = [dists[(c, sid)] for c in conds if (c, sid) in dists]
        for i, j in itertools.combinations(range(len(vecs)), 2):
            jsds.append(jensenshannon(vecs[i], vecs[j]))

    return {
        'seed_copy_rate': float(np.mean([r['seed_match'] for r in results])),
        'mean_top1_probability': float(np.mean([r['top1_prob'] for r in results])),
        'mean_entropy': float(np.mean([r['entropy'] for r in results])),
        'anchor_jsd': float(np.mean(jsds)) if jsds else 0.0,
        'dish_hit': float(np.mean(dish_hits)),
        'non_soup_top1_rate': float(np.mean(non_soup)),
        'nutrition_all_pass_rate': float(np.mean(nut_pass)),
        'tofu_nutrition_pass': float(np.mean(tofu_nut)) if tofu_nut else 0.0,
    }


def entropy_of(p):
    nz = p[p > 1e-9]
    return float(-(nz * np.log2(nz)).sum())


def generate_full_meal(core, encoder, decoder, num_tokens, mask_id, food_dict, seed_row, anchor_token, temp=0.8):
    n = 4   # batch=1이면 Decoder.call()의 tf.squeeze가 배치축까지 지워버려서 다음 스텝이 깨짐(코드 내 기존 경고와 동일 이유) -> 배치>1로 우회, row 0만 사용
    seeds_batch = tf.constant(np.tile(seed_row, (n, 1)), dtype=tf.int32)
    enc_hidden0 = tf.zeros([n, encoder.units])
    enc_output, enc_hidden = encoder(seeds_batch, enc_hidden0)
    dec_hidden = copy.deepcopy(enc_hidden)
    res = np.zeros(5, dtype=int)
    for j in range(5):
        outputs, dec_hidden, _ = decoder(seeds_batch[:, j], dec_hidden, enc_output)
        probs = np.array(outputs, dtype=float)
        if probs.ndim == 1:
            probs = probs[None, :]
        if j == 2:
            res[j] = anchor_token
            continue
        p = probs[0].copy()
        for t in core.SPECIAL:
            if t < num_tokens: p[t] = 0.0
        p[mask_id] = 0.0
        p[anchor_token] = 0.0
        p = np.clip(p, 1e-12, None); p /= p.sum()
        res[j] = int(np.random.choice(len(p), p=p))
    return [food_dict[int(t)] for t in res]


if __name__ == '__main__':
    main()
