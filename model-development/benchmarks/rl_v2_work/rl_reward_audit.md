# RL 구현 감사 (STEP 0) — 실제 코드 기준

**감사일**: 2026-07-28 | **원칙**: 아래 전부 실제 파일을 열어 확인한 내용이며 추측 없음.

---

## 1. RL 학습 진입점

`E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code\train_rl_FOOK.py`

```
python train_rl_FOOK.py --epochs 200 --lr 1e-5 --imit 0.3 --init ./results_FOOK/checkpoints --out ./results_rl_FOOK/checkpoints
```

실행 명령의 인자 기본값(argparse): `epochs=200, lr=1e-5, imit=0.3, weight=60, init='./results_FOOK/checkpoints', out='./results_rl_FOOK/checkpoints', embed_dim=128, fc_dim=64, network='GRU'`.

**참고**: 동일 이름의 두 번째 사본이 `E:\final\FOOK_재료대체모델링\생성모델_Seq2Seq_RL\train_rl_FOOK.py`에도 존재하나, 실제 production/이전 평가에서 참조된 적이 없어(경로가 다른 프로젝트 폴더) 이번 감사·실험 대상에서 제외했다. `results_sweep_FOOK/i002`를 만든 정확한 스윕 커맨드(예: `--imit 0.02`)는 별도 스윕 스크립트나 로그 파일로 기록돼 있지 않았다 — `app_core_FOOK.py` 31행 위 주석에만 "imit=0.02가 스윕 최적"이라고 남아있을 뿐, 재현 가능한 커맨드 기록은 없다.

---

## 2. Policy gradient / RL loss 계산부

`Model.py`의 `Sequence_Generator.train()` 메서드(208~393행)에서 전부 계산된다. `train_rl_FOOK.py`는 `gen.train(x, x, anchor_slots=anchor_slots)`를 호출만 한다.

핵심 로직(실제 코드 그대로, 300~370행):
```python
# 매 스텝 t(슬롯0~4)마다:
#   on-policy 샘플링: preds = decoder(...), predicted_token = get_action(preds, 'stochastic')
#   앵커 슬롯이면 실제 유저메뉴 토큰으로 강제
#   loss = ce_loss(predicted_token, preds)   # -log π(a_t) 형태
#   total_loss += loss   (전체 누적, 모니터링용 batch_loss 계산에만 씀)
#   if t가 앵커슬롯이 아니면: rl_loss += loss   (정책기울기용, 앵커 스텝 제외)
#   imit_loss += ce_loss(실제데이터토큰, preds)   (모든 스텝 누적, 종료토큰 포함)

# 시퀀스 끝나면:
reward_gen = reward_fn(pred_seqs, anchor_slots)          # = R.meal_reward() 배치호출
final_reward = reward_gen * (beta_score / tar_len)        # use_beta=True: 슬롯적합성(incidence matrix) 곱
adv = _advantage(final_reward, anchor_slots)               # REINFORCE baseline: 앵커슬롯별 평균 빼기
final_loss = rl_loss * adv + imit_weight * imit_loss        # 최종 손실
grads = tape.gradient(final_loss, trainable_variables)
optimizer.apply_gradients(...)
```

- `_advantage()`(195~206행): `reward - baseline`, baseline = 같은 앵커슬롯 그룹 내 평균(표본 1개면 전역평균 폴백).
- **on-policy + stochastic 필수**: off-policy면 CE 타깃이 실제데이터라 정책이 보상을 학습 못 한다(코드 docstring에 명시).
- **현재 KL 항 없음** — `final_loss`에 BASE(reference policy) 대비 분포 규제 항이 전혀 없다. STEP 2에서 추가해야 하는 항은 이 `final_loss` 계산식에 `+ beta * kl_loss`로 들어가야 한다.

---

## 3. Reward 계산 함수 — 실제 수식

`E:\final\reward_lever_FOOK.py`의 `meal_reward()`(66~92행). **주의: 이 파일은 `Code` 서브디렉토리가 아니라 `E:\final` 루트에 있다.**

```text
reward = pass_frac × (0.5 + 0.5 × preserve)

pass_frac = (통과한 영양소 개수) / 5
  통과 조건(pass_flags, 50~56행):
    열량:   Elo <= E <= Ehi
    단백질: Plo <= protein <= Phi
    칼륨:   K <= Kmax          ← 주의: 이하(<=), 실제 서비스 passes()는 미만(<)
    인(P):  P <= Pmax          ← 주의: 이하(<=), 실제 서비스 passes()는 미만(<)
    나트륨: Na_season <= Namax

preserve = 0.7 × anchor_keep + 0.3 × overall_keep
  anchor_keep = 1 - min(1, 앵커메뉴 재료량 변화량 / 앵커메뉴 원래 총량)
  overall_keep = 1 - min(1, 전체 재료량 변화량 / 전체 원래 총량)

최종 학습 reward = final_reward = reward × (beta_score / tar_len)
  beta_score = incidence_mat 기반 슬롯-메뉴 동시출현 점수(use_beta=True일 때, "자연스러움" 사전분포)
```

**핵심 감사 결과 — 실제 서비스 판정과의 불일치(코드로 확인됨)**:

| 항목 | 보상함수(`reward_lever_FOOK.py`) | 실제 서비스(`app_core_FOOK.passes()`) | 정렬 필요 |
|---|---|---|---|
| 칼륨 판정 | `K <= Kmax`(이하) | `K < Kmax`(미만) | 예 |
| 인(P) 판정 | `P <= Pmax`(이하) | `P < Pmax`(미만) | 예 |
| 현실성 검사(`unrealistic_reason`) | **반영 안 됨** | 최종 성공 조건에 포함 | 예 |
| 재료 겹침(`_has_ingredient_clash`) | **반영 안 됨** | 최종 성공 조건에 포함 | 예 |
| 자연나트륨군 과다(`_has_seafood_overload`) | **반영 안 됨** | 최종 성공 조건에 포함 | 예 |
| 고인비율군 과다(`_has_high_p_overload`) | **반영 안 됨** | 최종 성공 조건에 포함 | 예 |
| 위반 정도(연속값) | 없음(이진 pass/fail만, 부분점수는 "몇 개 통과"뿐) | 해당 없음(서비스는 최종 판정만 함) | STEP1에서 도입 검토 |

즉 지금 reward는 "5영양 중 몇 개를 통과했는가 + 재료를 얼마나 덜 바꿨는가"만 보고, **최종
서비스가 실제로 성공/실패를 가르는 나머지 4가지 게이트(현실성·재료겹침·나트륨군과다·
인비율군과다)를 전혀 보지 못한다.** 이게 STEP 1에서 정렬해야 할 핵심 지점이다.

---

## 4. BASE 모델을 불러오는 부분

`train_rl_FOOK.py` 105~111행:
```python
ck = tf.train.latest_checkpoint(args.init)   # 기본값 './results_FOOK/checkpoints'
tf.train.Checkpoint(generator=gen).restore(ck).expect_partial()
```
모방학습 체크포인트에서 **warm-start**(RL 전용 GRU 가중치를 그 위에서 이어 학습). `--init`을 다른 경로(예: 기존 i002)로 주면 그 체크포인트에서 다시 RL을 이어 돌릴 수도 있는 구조.

---

## 5. RL 체크포인트 저장 및 선택 기준

`train_rl_FOOK.py` 113~130행: 학습 루프(`for epoch in range(args.epochs)`)가 전부 끝난 뒤 **딱 한 번**:
```python
ckpt.save(file_prefix=os.path.join(args.out, 'ckpt'))
```
**중간 체크포인트 저장이나 validation 기반 best-checkpoint 선택 로직이 전혀 없다.** 10에폭마다 `loss`/`보상 평균`/`std`/`고유메뉴 수`를 콘솔에 출력만 하고(123~127행), 그 어떤 지표도 파일로 기록되거나 체크포인트 선택에 쓰이지 않는다. `results_sweep_FOOK/i002`가 "최적"으로 채택된 근거는 사람이 여러 `--imit` 값으로 수동 실행해보고 콘솔 출력을 보고 골랐을 가능성이 높으나, 그 과정을 재현할 로그가 남아있지 않다.

---

## 6. 학습 데이터/시나리오 생성부

`train_FOOK.py`의 `build_data()`(35~44행)를 그대로 import해서 씀. `FOOK_meals_for_model.csv`(1,095끼) 전체를 `diet_to_incidence`까지 처리한 뒤, **분할 없이 전량을 `batch_size`로 그대로 한 배치**에 넣는다(`train_rl_FOOK.py` 66행: `batch_size = int(diet_np.shape[0])`). **train/validation 분리가 전혀 없다.**

---

## 7. Validation 방식

**없음.** 10에폭마다 같은 학습배치에 대한 `loss`/`reward` 평균·표준편차만 콘솔에 출력한다(123~127행). 별도 held-out 시나리오로 학습 중간 성능을 점검하는 코드가 `train_rl_FOOK.py`/`Model.py` 어디에도 없다. (참고: `E:\final\Diet-Generation-As-Sequence-master\Diet-Generation-As-Sequence-master\Code\eval_rl_FOOK.py`가 별도로 존재하나, 학습 루프 안에서 자동 호출되지 않고 학습 종료 후 사람이 수동으로 돌리는 스크립트로 보인다 — 이번 감사에서 내용까지 깊이 확인하지는 않았다.)

---

## 8. Seed 설정 방식

`train_rl_FOOK.py` 115행: `rng = np.random.default_rng(0)` — **이건 `anchor_slots`(매 에폭 어느 슬롯을 유저지정으로 볼지)만 결정하는 별도 Generator 인스턴스**다.

실제 토큰 샘플링(`util.py`의 `get_action(preds, option='stochastic')`, 825~830행)은 **전역 `np.random.choice`**를 쓴다 — `train_rl_FOOK.py`에는 이 전역 상태를 고정하는 `np.random.seed()` 호출이 없다. 즉 **anchor_slots 배정은 재현 가능하지만, 정책이 실제로 어떤 토큰을 샘플링하는지는 실행마다 달라질 수 있다** — 현재 RL 학습은 완전히 재현 가능한 상태가 아니다. STEP 1~4에서 새 학습 코드를 작성할 때는 학습 시작 시점에 전역 `np.random.seed()`(+ `tf.random.set_seed()`)를 명시적으로 고정해야 한다.

---

## 9. 다음 단계 진행 판단

STEP 0 감사 완료. `baseline_reproduction_report.md`에서 재현 결과를 확인한 뒤, 문제가 없으면
STEP 1(보상함수 정렬)로 진행한다.
