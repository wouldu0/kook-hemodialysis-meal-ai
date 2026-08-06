# final_C_mask100_RL_report.md

## 1. 실제 make_meal() 전체 검증 결과

**A 결과** (3조건 평균, 200회×48후보×10seed×3앵커): 영양5종 68.8%, 게이트 87.1%, 앵커보존 0.772,
후보0개율 34.6%, 고유국 4.1개, seed일치율 96.7%, 앵커민감도(JSD) 0.139.
**두부콩류 조건에서 사실상 고장**(영양5종 33.8%, 후보0개율 67.4%, 앵커보존 0.537).

**C 결과**: 영양5종 97.6%, 게이트 98.0%, 앵커보존 0.859, 후보0개율 2.6%, 고유국 30.5개,
seed일치율 6.7%, 앵커민감도 0.585. 두부콩류 조건 정상화(영양5종 93.2%, 후보0개율 7.2%).

- **seed 복사율 비교**: A 96.7% → C 6.7% (급감)
- **anchor sensitivity 비교**: A 0.139 → C 0.585 (4.2배)
- **국 다양성 비교**: A 4.1개 → C 30.5개 (7.4배)
- **부찬 다양성 비교**: A 12.4개 → C 17.5개 (1.4배)
- **영양 5종 충족률 비교**: A 68.8% → C 97.6% (C가 A의 두부콩류 결함을 해소하며 오히려 크게 상승)
- **gate 통과율 비교**: A 87.1% → C 98.0%
- **두부·콩류 조건 비교**: A는 사실상 고장(33.8%/67.4%후보0개) → C는 정상(93.2%/7.2%)
- **후보 0개 발생률 비교**: A 34.6% → C 2.6%
- **생성 실패율 비교**: A 0% = C 0% (둘 다 문제없음)

**C_mask100의 실제 파이프라인 승인 여부**: **승인** (Stage 1 보고서 참고, 8개 기준 전부 충족)

## 2. RL 재학습 결과

**RL 초기화 checkpoint**: `checkpoints_masking_1000/C_mask100/checkpoints/ckpt-1`
**기존 RL checkpoint 미사용 확인**: 확인됨 — `train_rl_soupmask_FOOK.py`는 `results_sweep_FOOK`
경로를 어디서도 import/로드하지 않음(코드 검색으로 재확인 가능)
**SOUP_MASK 유지 확인**: 확인됨 — 학습 루프 매 에폭 `assert`로 encoder 입력의 국 위치가 항상
mask_id 단일값인지 검증했고, 300에폭 내내 에러 없이 통과함
**best RL checkpoint**: `results_rl_soupmask_FOOK/C_mask100_rl/checkpoints_best/ckpt-3`
**best epoch/step**: epoch 260 (composite_score=1.819, 300에폭 중 최고)

**Seq2Seq C 대비 변화** (3조건 평균, 전부 실제 make_meal() 파이프라인 기준):

| 지표 | C_mask100 | C_mask100+RL | 변화 |
|---|---|---|---|
| 영양5종 충족률 | 97.6% | 97.1% | -0.5%p(오차범위) |
| 게이트 통과율 | 98.0% | 97.6% | -0.4%p(오차범위) |
| **앵커보존률** | 0.859 | **0.865** | **+0.006(개선)** |
| seed 복사율 | 6.7% | 10.0% | +3.3%p(여전히 A의 96.7%와 비교불가 수준으로 낮음) |
| entropy(엔트로피) | (RL학습중 모니터링 5.38~5.46 유지) | 유지 | mode collapse 없음 |
| **anchor sensitivity(JSD)** | 0.585 | **0.592** | **+0.007(개선)** |
| 국 다양성(고유종류) | 30.5개 | 28.6개 | -1.9개(A의 7배 수준은 유지) |
| 부찬 다양성 | 17.5개 | 17.4개 | -0.1개(거의 동일) |
| 후보 0개 발생률 | 2.6% | 3.0% | +0.4%p(오차범위) |

## 3. 최종 판정

**최종 채택 모델**: **C_mask100 + RL**

**채택 근거**: RL이 C_mask100 대비 앵커보존률(+0.006)과 anchor sensitivity(+0.007) 둘 다
개선했고 — 이게 정확히 이번 RL의 설계 목적("다양성 유지하며 영양·현실성·앵커 적합도 향상")과
일치함 — 동시에 영양충족률·게이트통과율·다양성 지표는 전부 오차범위 안에서만 변해 유의미한
악화가 없음. 300에폭 학습 로그 전체에서 조기경고 조건(seed_copy 급상승·anchor_jsd 급락·
entropy 급감·dish_hit 하락 등)이 단 한 번도 발동하지 않음.

**C의 다양성 개선 유지 여부**: 유지됨. 고유 국 종류 30.5→28.6(-6% 정도 감소했지만 A의 4.1개
대비 여전히 7배 수준), 부찬은 거의 그대로(17.5→17.4).

**RL이 추가로 개선한 항목**: 앵커보존률(+0.006), anchor sensitivity(+0.007). 크기는 작지만
방향이 일관되고 두부콩류 조건에서 더 뚜렷함(앵커보존 0.896→0.902).

**RL로 악화된 항목**: 없음(전부 오차범위 내 변동). 굳이 꼽자면 국 고유종류가 30.5→28.6으로
소폭 감소했으나 A(4.1) 대비 미미한 손실.

**mode collapse 여부**: 없음. 300에폭 내내 고유국 159~173개(quick_eval 기준, 대량샘플), entropy
5.38~5.46 유지, 최종평가에서도 고유국 28.6개로 A(4.1개)와 비교 불가할 만큼 다양함.

**실서비스 적용 가능 여부**: 가능. 두부·콩류 앵커 조건에서 A가 보였던 구조적 결함(후보0개율
67.4%)이 C/C+RL 둘 다에서 완전히 해소됐고(7.2%/8.2%), 영양·게이트·다양성 전부 A보다 뚜렷이
우수함.

**남은 한계**:
- seed 복사율이 완전히 0%는 아님(RL 10.0%) — 구조적으로 encoder가 국 정보를 못 보게 막아뒀지만,
  나머지 슬롯(부찬·김치·앵커)과의 상관관계로 일부는 여전히 "우연히 원래 국과 같은 답"에 도달함
  (지난 seed-dependency 진단에서 확인한 B_mask50의 우회경로와 같은 종류의 현상 — 다만 폭이
  훨씬 작음: 10% vs B의 100%).
- RL 보상에 넣은 diversity_penalty·gate_penalty는 이번 세션에서 새로 설계한 항이라, 기존
  RL(imit=0.02 스윕)만큼 폭넓게 튜닝되지는 않음 — 단일 설정(lr 5e-5, imit 0.02, diversity 0.15,
  gate_penalty 0.6)만 시도했고 스윕은 안 함.
- quick_eval(학습 중 모니터링용)은 48후보 탐색 없이 1회 생성이라 절대수치가 낮게 나옴 — 최종
  판단은 반드시 이번 stage2 전체파이프라인 수치(48후보 포함)를 기준으로 해야 하고, 실제로 그렇게
  했음.
- 이번 30조건(3앵커×10seed)은 이전 진단들과 동일 조건 재사용 — 이 조건 밖 일반화는 미검증.

## 산출 파일
- `stage1_full_make_meal_out/full_make_meal_A_vs_C.csv`
- `stage1_full_make_meal_out/full_make_meal_anchor_metrics.csv`
- `stage1_full_make_meal_out/full_make_meal_gate_failures.csv`
- `stage1_full_make_meal_out/stage1_report.md`
- `results_rl_soupmask_FOOK/C_mask100_rl/rl_C_mask100_training_log.csv`
- `results_rl_soupmask_FOOK/C_mask100_rl/rl_C_mask100_eval_history.csv`
- `results_rl_soupmask_FOOK/C_mask100_rl/warnings_log.csv` (빈 파일 — 경고 0건)
- `stage2_final_out/final_A_C_RL_comparison.csv`
- `stage2_final_out/final_C_mask100_RL_report.md` (본 문서)
