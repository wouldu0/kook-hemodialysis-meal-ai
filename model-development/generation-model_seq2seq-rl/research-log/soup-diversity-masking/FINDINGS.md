# 국(soup) 슬롯 다양성 붕괴 진단 및 마스킹 실험 (검증 완료, 프로덕션 미배포)

## 상태

**검증 완료, 실서비스 미반영.** `app_core_FOOK.py`는 여전히 이 실험 이전 체크포인트
(`results_sweep_FOOK/i002`)를 로드한다. 아래 결론은 배포 가능하다고 판단됐지만, 이번 세션
범위에서는 실제 반영(체크포인트 교체)까지는 진행하지 않았다.

## 1. 문제 발견

`diagnose_main_anchor_diversity_FOOK.py`(주찬 앵커 고정 시 국·부찬 다양성 저하 진단),
`eval_dishhit_rdi_FOOK.py`(Dish-hit/RDI 정량 평가), `eval_diversity_fixed_anchor_FOOK.py`
(고정 조건 다양성 평가, 실제 프로덕션 `make_meal()` 재사용)로 확인: 같은 환자·같은 seed
식단·같은 앵커 조건에서 국 슬롯이 소수의 메뉴로 심하게 쏠려 반복 생성됨.

## 2. 원인 규명

3개 스크립트로 구조적 원인을 추적:

- `diagnose_soup_bias_stage_FOOK.py`: 디코더가 각 슬롯에서 "이전에 샘플링된 토큰"이 아니라
  **seed 시퀀스의 해당 위치 토큰**을 다음 입력으로 쓰는 teacher-forcing 스타일 구조임을 코드로
  확인(`app_core_FOOK.gen_batch` line 322). 즉 seed·앵커·가중치가 고정되면 국 슬롯의 확률분포
  자체가 하나로 결정되고, 매 시행 달라지는 건 그 분포에서 뽑는 난수뿐.
- `diagnose_soup_seed_dependency_FOOK.py`: 국(slot1) 출력 스텝의 입력은 seed의 "밥" 값이며,
  seed의 "국" 값 자체는 인코더가 전체 시퀀스를 한 번에 읽을 때만 영향을 준다는 것을 코드로 확인
  — "seed 국 토큰 의존"의 실체는 인코더가 seed 시퀀스에서 국 위치를 읽어 학습한 shortcut.
- `diagnose_soup_training_bias_FOOK.py`: 학습데이터(`FOOK_meals_for_model.csv`, 1,095끼,
  train/val/test 분리 없음) 자체의 주찬-국 동시등장 편중 때문인지, 모델의 과확신 때문인지 분리
  검증.

**결론**: 편중의 실체는 디코더가 seed의 국 정보를 "그대로 베끼는" 학습된 shortcut.

## 3. 해결 시도 — 국 슬롯 마스킹 학습 (A/B/C 비교)

`train_FOOK_soupmask.py`(50% 마스킹, B), `train_FOOK_soupmask_1000.py`(100% 마스킹, C)로
재학습 후 `evaluate_masking_1000_FOOK.py`로 A(마스킹 없음)/B(50%)/C(100%) 비교:

| 지표 | A(원본) | B(50%) | C(100%) |
|---|---|---|---|
| seed 복사율 | 100% | 100% | **10%** |
| 앵커 민감도(JSD) | 0.027 | 0.029 | **0.479** |
| 두부·콩류 조건 영양충족(파이프라인) | 10% | 100% | 100% |
| 최종 국 고유종류 | 5~8개 | 18~25개 | **30~35개** |

B는 분포 수준 지표(seed 복사율, 앵커 민감도)에서 A와 사실상 구분되지 않아 **마스킹의 의도된
메커니즘이 작동하지 않은 것**으로 판단해 기각. **C_mask100 채택.**

## 4. 실제 파이프라인 검증 + RL 재학습

`verify_soupmask_experiments_FOOK.py`, `stage1_full_make_meal_verify_FOOK.py`로 실제
`make_meal()` 전체 파이프라인(48후보 탐색 포함) 기준 재검증:

| 지표 | A | C_mask100 |
|---|---|---|
| 영양 5종 충족률 | 68.8% | 97.6% |
| 게이트 통과율 | 87.1% | 98.0% |
| 국 고유종류 | 4.1개 | 30.5개 |
| 두부·콩류 조건 영양충족 | 33.8%(사실상 고장) | 93.2%(정상화) |

이어서 `train_rl_soupmask_FOOK.py`로 C_mask100을 warm-start로 RL 재학습(기존
`results_sweep_FOOK/i002` 체크포인트는 seed 복사 구조 위에서 학습된 것이라 재사용하지 않음).
`stage2_final_comparison_FOOK.py`로 C_mask100 vs C_mask100+RL 비교:

| 지표 | C_mask100 | C_mask100+RL |
|---|---|---|
| 앵커보존률 | 0.859 | **0.865**(개선) |
| 앵커 민감도(JSD) | 0.585 | **0.592**(개선) |
| 영양 5종 충족률 | 97.6% | 97.1%(오차범위) |
| 국 고유종류 | 30.5개 | 28.6개(오차범위, A의 4.1개 대비 여전히 7배 수준) |

300에폭 학습 전체에서 조기경고 조건(mode collapse 등) 발동 없음. **최종 채택 모델:
C_mask100 + RL — "실서비스 적용 가능" 판정.**

## 5. 남은 한계 (원 보고서 기준)

- seed 복사율이 완전히 0%는 아님(RL 10.0%) — 다른 슬롯과의 상관관계로 일부 우연한 일치 잔존.
- 학습분포와의 JSD가 여전히 큼(C 기준 0.747) — 방향성 판단엔 충분하나 절대 수치는 실배포 시
  재검증 필요.
- 이번 평가의 30개 조건(3앵커×10seed)은 이전 진단들과 동일 조건 재사용 — 조건 밖 일반화는
  미검증.
- RL 보상항(diversity_penalty·gate_penalty)은 단일 설정만 시도, 스윕 미실시.

## 산출물 위치 (원본 경로, 참고용)

`Diet-Generation-As-Sequence-master/.../Code/` 하위 `diagnose_main_anchor_out/`,
`eval_diversity_out/`, `diagnose_soup_bias_out/`, `diagnose_soup_seed_out/`,
`diagnose_soup_training_bias_out/`, `evaluate_masking_1000_out/`, `verify_soupmask_out/`,
`stage1_full_make_meal_out/`, `results_rl_soupmask_FOOK/C_mask100_rl/`, `stage2_final_out/`.
원본 결과 CSV(수 MB의 원시 trace 포함)는 로컬(`E:\final`)에만 보존하고, 결론을 요약한
리포트 3건만 `reports/`에 포함했다.
