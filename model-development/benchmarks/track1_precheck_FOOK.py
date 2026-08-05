# -*- coding: utf-8 -*-
"""
track1_precheck_FOOK.py — Track1(BASE vs RL i002) 평가 실행 전 사전확인.

확인 대상 4개 조합:
  A. BASE(results_FOOK/checkpoints) + 수정 전 레버(.bak_before_rawP_tofu_path_20260727)
  B. BASE(results_FOOK/checkpoints) + 최종 레버(현재 FOOK_adjust_levers.py, app_core_FOOK.F)
  C. RL(results_sweep_FOOK/i002, = app_core_FOOK.gen) + 최종 레버
  D. RL(results_sweep_FOOK/i002) + 수정 전 레버

이 스크립트는 서비스 코드/체크포인트/레버 코드를 전혀 수정하지 않는다. 읽기 전용 로드 +
소규모 스모크 생성만 수행한다.
"""
import os, sys, copy, traceback, importlib.util, importlib.machinery
os.environ.setdefault('TF_USE_LEGACY_KERAS', '1')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'E:\final')

print('=' * 70)
print('[1] app_core_FOOK import (RL i002 체크포인트가 gen으로 로드됨)')
print('=' * 70)
import app_core_FOOK as core   # noqa: E402  — 여기서 RL(i002)이 core.gen 으로 로드됨
import numpy as np
import tensorflow as tf
from Model import Sequence_Generator

print(f'  core.CKPT = {core.CKPT}')
print(f'  core.gen 로드 완료 (RL i002)')

print('\n' + '=' * 70)
print('[2] BASE Sequence_Generator 별도 인스턴스 생성 + results_FOOK/checkpoints 복원')
print('=' * 70)
BASE_CKPT = os.path.join(core.CODE, 'results_FOOK', 'checkpoints')
print(f'  BASE_CKPT = {BASE_CKPT}')
kwargs = {'fully-connected_layer': 'GRU', 'attention': True, 'embed_dim': 128, 'fc_dim': 64,
          'learning': 'off-policy', 'policy': 'greedy', 'use_beta': False, 'use_buffer': False,
          'buffer_size': 5, 'buffer_update': 5, 'num_epochs': 1, 'lr': 1e-3,
          'num_tokens': len(core.food_dict), 'batch_size': core.diet_np.shape[0], 'imitation_only': True}
base_gen = Sequence_Generator(core.food_dict, core.nutrient_data, core.incidence, **kwargs)
ckpt_path = tf.train.latest_checkpoint(BASE_CKPT)
print(f'  최신 체크포인트: {ckpt_path}')
# 주의: Keras 서브클래싱 모델은 레이어가 "처음 호출될 때" 지연 생성(lazy build)된다.
# 이 시점(restore 직후, 아직 encoder/decoder를 한 번도 호출 안 함)에는 변수 자체가 없으므로
# assert_existing_objects_matched()는 항상 실패한다 - 이건 진짜 구조 불일치가 아니라 정상적인
# Keras 지연빌드 특성이다(app_core_FOOK.py 실제 서비스 코드도 동일 패턴 + expect_partial() 사용).
# 그래서 구조 검증은 "실제 forward pass 이후 assert_consumed()"로 미룬다(아래 6단계에서 수행).
base_restore_status = tf.train.Checkpoint(generator=base_gen).restore(ckpt_path)
print('  restore() 호출 완료 (지연빌드 특성상 변수 매칭 확인은 실제 forward pass 이후 수행)')

print('')
print('=' * 70)
print('[3] RL(i002) 체크포인트 restore 상태 객체 확보 (검증은 6단계 forward pass 이후)')
print('=' * 70)
rl_restore_status = tf.train.Checkpoint(generator=core.gen).restore(tf.train.latest_checkpoint(core.CKPT))
print('  restore() 호출 완료')

print('\n' + '=' * 70)
print('[4] OLD 레버 모듈 로드 (.bak_before_rawP_tofu_path_20260727, SourceFileLoader)')
print('=' * 70)
OLD_BAK = r'E:\final\FOOK_adjust_levers.py.bak_before_rawP_tofu_path_20260727'
assert os.path.exists(OLD_BAK), f'백업 파일이 없음: {OLD_BAK}'
loader = importlib.machinery.SourceFileLoader("FOOK_adjust_levers_OLD", OLD_BAK)
spec = importlib.util.spec_from_loader(loader.name, loader)
F_old = importlib.util.module_from_spec(spec)
loader.exec_module(F_old)
# 주의: F_old.NUT는 core.F.NUT(공유 객체)를 대입하면 안 되고 F_old.load_all()로 독립적으로
# 새로 로드해야 한다(공유 시 OLD 레버가 잘못된 값을 내는 것을 디버깅으로 확인, 2026-07-27).
F_old.NUT = F_old.load_all()
has_new_funcs_in_old = hasattr(F_old, 'lever_phosphorus_rawP') or hasattr(F_old, '_plant_protein_path_needed')
print(f'  F_old.adjust 존재: {hasattr(F_old, "adjust")}')
print(f'  F_old에 신규(두부콩류 조건부) 함수가 섞여있으면 안 됨 → 존재여부: {has_new_funcs_in_old} (False가 정상)')
if has_new_funcs_in_old:
    print('  [STOP] OLD 백업 파일에 신규 함수가 존재함 — 백업이 오염되었을 가능성. 중단.')
    sys.exit(1)

print('\n' + '=' * 70)
print('[5] NEW(최종) 레버 모듈 확인 (core.F = 현재 라이브 FOOK_adjust_levers.py)')
print('=' * 70)
has_new_funcs = hasattr(core.F, 'lever_phosphorus_rawP') and hasattr(core.F, '_plant_protein_path_needed')
print(f'  core.F.adjust 존재: {hasattr(core.F, "adjust")}')
print(f'  core.F 신규 함수(rawP/조건부 경로) 존재: {has_new_funcs} (True가 정상)')
if not has_new_funcs:
    print('  [STOP] 현재 라이브 레버 파일에 최종 채택된 신규 함수가 없음 — 예상과 다른 상태. 중단.')
    sys.exit(1)


def gen_batch_generic(gen_obj, anchor_menu=None, n=12, temp=0.8, seed_idx=None):
    """core.gen_batch()와 동일 로직, gen 객체를 파라미터로 받아 BASE/RL 어느 쪽이든 사용 가능."""
    if seed_idx is None:
        idx = np.random.randint(core.diet_np.shape[0], size=n)
    else:
        idx = seed_idx
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


print('\n' + '=' * 70)
print('[6] 4개 조합(A~D) 스모크 테스트 (앵커=두부양념조림, n=4)')
print('=' * 70)
ANCHOR = '두부양념조림'
b = core.F.meal_bounds(60)
np.random.seed(12345)
idx = np.random.randint(core.diet_np.shape[0], size=4)

combos = {
    'A (BASE+OLD)':  (base_gen, F_old),
    'B (BASE+NEW)':  (base_gen, core.F),
    'C (RL+NEW)':    (core.gen, core.F),
    'D (RL+OLD)':    (core.gen, F_old),
}

results = {}
for label, (gen_obj, lever_mod) in combos.items():
    try:
        np.random.seed(999)  # decoder 내부 샘플링용 시드도 통일(순수 로드-확인 목적)
        menus_batch = gen_batch_generic(gen_obj, ANCHOR, n=4, temp=0.8, seed_idx=idx)
        out_rows = []
        for menus in menus_batch:
            before, after, inst, p_ok = lever_mod.adjust(menus, b, anchor=ANCHOR)
            ok = core.passes(after, b)   # passes()는 app_core_FOOK 소속 서비스 함수, 레버버전과 무관(불변)
            unreal = core.F.unrealistic_reason(inst)
            out_rows.append((menus, after['P'], ok, unreal))
        results[label] = ('OK', out_rows)
        print(f'  [OK] {label}: {len(out_rows)}건 생성+adjust 성공. 예시 P={out_rows[0][1]:.1f}, pass={out_rows[0][2]}')
    except Exception as e:
        results[label] = ('FAIL', str(e))
        print(f'  [FAIL] {label}: {e}')
        traceback.print_exc()

print('\n' + '=' * 70)
print('[7] forward pass 이후 체크포인트 매칭 재검증 (지연빌드 완료 후)')
print('=' * 70)
def _check_consumed(status, label):
    """assert_consumed() 실패 시, 남은 unresolved 항목이 전부 'generator.optimizer.*'(Adam
    모멘텀/분산 상태, 추론에 안 쓰임)뿐이면 정상으로 간주한다. encoder/decoder/embedding/
    attention 관련 항목이 하나라도 남아있으면 진짜 구조 불일치로 판단해 중단해야 한다."""
    try:
        status.assert_consumed()
        print(f'  [OK] {label} assert_consumed() 완전 통과 - 모든 값(옵티마이저 포함) 매칭됨')
        return True
    except AssertionError as e:
        msg = str(e)
        # AssertionError 메시지에서 unresolved object 경로들을 추출
        unresolved = [ln.strip() for ln in msg.splitlines() if 'Unresolved object in checkpoint' in ln]
        non_optimizer = [ln for ln in unresolved if '.generator.optimizer.' not in ln]
        if unresolved and not non_optimizer:
            print(f'  [OK] {label}: 미매칭 항목 {len(unresolved)}건 전부 generator.optimizer.*(Adam 모멘텀/분산, '
                  f'추론 미사용) - encoder/decoder/embedding/attention은 전부 매칭됨. 구조 일치로 판정.')
            return True
        else:
            print(f'  [STOP] {label} assert_consumed() 실패 - optimizer 외 항목 미매칭(구조 불일치 가능):')
            print(f'    {non_optimizer if non_optimizer else msg}')
            return False


ckpt_ok = True
ckpt_ok &= _check_consumed(base_restore_status, 'BASE')
ckpt_ok &= _check_consumed(rl_restore_status, 'RL(i002)')

print('\n' + '=' * 70)
print('[8] BASE vs RL 가중치 실제 상이 여부 확인 (같은 파일을 잘못 로드한 게 아님을 확인)')
print('=' * 70)
base_emb = base_gen.encoder.embedding_layer.embeddings.numpy()
rl_emb = core.gen.encoder.embedding_layer.embeddings.numpy()
same_weights = np.allclose(base_emb, rl_emb)
print(f'  BASE encoder embedding vs RL(i002) encoder embedding 완전동일: {same_weights} (False가 정상 - 서로 다른 체크포인트)')
if same_weights:
    ckpt_ok = False
    print('  [STOP] BASE와 RL(i002)의 가중치가 동일함 - 같은 체크포인트를 중복 로드했을 가능성. 구조적 문제로 판단, 중단 필요.')

print('\n' + '=' * 70)
print('[9] 결론')
print('=' * 70)
all_ok = all(v[0] == 'OK' for v in results.values()) and ckpt_ok
if all_ok:
    print('  4개 조합(A/B/C/D) 전부 정상 로드 및 스모크 생성+adjust 성공.')
    print('  BASE/RL 체크포인트 모두 forward pass 이후 assert_consumed() 통과(구조 불일치 없음).')
    print('  BASE와 RL(i002) 가중치가 실제로 서로 다름을 확인(체크포인트 오혼입 아님).')
    print('  OLD/NEW 레버 모듈 모두 정상 import 및 함수 확인 완료(OLD에는 신규함수 없음, NEW에는 있음).')
    print('  => 본 평가를 진행해도 구조적 문제 없음. 대규모 실행으로 진행 가능.')
else:
    print('  일부 조합/검증 실패 - 중단하고 원인을 보고해야 함.')
    for k, v in results.items():
        if v[0] == 'FAIL':
            print(f'    {k}: {v[1]}')
