# RL 성능개선 시도 최종 보고서 (STEP0~1 결과 + ROT 방법론 오류 정정 + 최종 의사결정)

**평가일**: 2026-07-28 | **상태**: 평가·감사·보고 완료. **서비스 코드·production 체크포인트
변경 없음.**

---

## 1. 요약(한 줄 결론)

**기존 production RL(i002)을 그대로 유지한다.** 새로 학습한 R2는 기본 Seq2Seq(BASE) 대비는
개선됐지만 현재 production RL(i002) 대비로는 오히려 낮은 성공률을 보여 채택하지 않는다.
추가 KL 규제·hard-case oversampling·reward 탐색은 이번 범위에서 중단한다.

---

## 2. 지금까지 진행 경과

1. **STEP 0**: RL 구현 감사(`train_rl_FOOK.py`/`Model.py`/`reward_lever_FOOK.py`) + 기존
   공정비교 스크립트(`rl_comparison_FOOK.py`) 재현 완료 — 사용자 인용값 12개 전부 일치.
2. **STEP 1**: 서비스 판정과 정렬된 reward v2(R0~R5) 작성, 실측 분포로 가중치 보정(예시값
   맹목 적용 안 함), 별도 validation 40건(120건 최종평가 CSV와 중복 0건)으로 ablation.
   결과: R2(연속형 영양위반 패널티 추가)가 validation에서 가장 우수(성공률 90%). phosphorus_
   weight 스윕은 pw=2.0까지 확인 후(개선 없음) 사용자 지시로 pw=3.0부터 중단. R5(현실성·
   보존·재료겹침 패널티 추가)는 오히려 validation 성공률이 80%로 하락 — R2를 최종 후보로
   선정.
3. **R2를 기존 120개 최종평가 시나리오로 3-way(BASE/기존RL/R2) 비교하던 중, 방법론적 오류를
   발견**(4절) — 이번 보고서의 핵심.
4. **오류를 수정하고 BASE vs 기존RL 공식 2-way를 재검증**(5~6절), R2 위치를 재확인(7절).

---

## 3. STEP 1 Reward Ablation 결과 (validation 40건, 참고용 — 8절에서 신뢰성 재평가)

| 버전 | 성공률 | 5영양 | rawP | 단백질 | 보존 | 비현실 |
|---|---:|---:|---:|---:|---:|---:|
| R0(기존 reward) | 77.5% | 85% | 92.5% | 97.5% | 90% | 5% |
| R1(+final_pass 보너스) | 80% | 85% | 95% | 95% | 90% | 5% |
| **R2(+연속형 영양위반 패널티)** | **90%** | 90% | 92.5% | 97.5% | 90% | **0%** |
| R3 pw=1.0(=R2와 동일 공식) | 90% | 90% | 92.5% | 97.5% | 90% | 0% |
| R3 pw=2.0 | 90% | 90% | 90%(하락) | 100% | 90% | 0% |
| R3 pw=3.0/4.0/6.0/8.0 | **미실시**(사용자 지시로 중단 — pw=2.0에서 성공률 개선 없이 rawP만 하락해 우선순위 낮음) | | | | | |
| R5(=R4공식, +현실성/보존/재료겹침 패널티) | 80%(하락) | 87.5% | 92.5% | 95% | 90% | 7.5% |

**선정: R2** (validation 기준 1순위 지표인 성공률에서 최고, R5 대비 전 지표 우위 또는 동률).

---

## 4. 발견한 방법론적 오류 — `core.F.ROT[0]` 전역 상태 미리셋

R2를 BASE·기존RL과 함께 120개 시나리오로 비교하는 도중, BASE의 raw P 충족률이 이전
`rl_comparison_FOOK.py` 실행 결과(116/120)와 새 3-way 스크립트에서 미묘하게 다르게
나오는 걸 발견해 원인을 추적했다.

**원인**: `FOOK_adjust_levers.py` 111행 `ROT = [0]` — `lever_kimchi`가 저염 김치를 고를 때
`LOWNA_POOL[ROT[0] % len(LOWNA_POOL)]`로 순환선택하고 매번 `ROT[0] += 1`한다. 이 카운터는
**모듈 전역**이며 `adjust()`가 호출될 때마다(즉 후보 하나마다) 증가하고, `adjust()` 자체는
이 카운터를 리셋하지 않는다. `np.random.seed()`만 리셋하고 `ROT`는 그대로 두면, **한
시나리오 안에서 몇 개 모델을 순서대로 호출하는지에 따라 각 모델이 시작하는 ROT 위치가
달라진다** — 2-way(BASE→RL)와 3-way(BASE→RL→R2)에서 BASE가 보는 ROT 시작점부터 다르다.

**영향**: 대부분은 저염김치 종류만 바뀌어 결과에 무관하지만, 드물게 그 차이가 나트륨
경계값을 넘나들며 **완전히 다른 후보가 최종 선택**되는 데까지 이어졌다(15개 표본 검증에서
14개 행의 `final_menus`가 달랐고, 그중 1건은 김치뿐 아니라 밥·국·주찬까지 전부 다른
후보였음).

**전체 mutable global state 감사** (`FOOK_adjust_levers.py`, `app_core_FOOK.py` 전체 검색):

| 이름 | 위치 | 위험도 | 판정 |
|---|---|---|---|
| `ROT` | `FOOK_adjust_levers.py:111` | **높음** | `adjust()` 호출마다 증가, 리셋 안 됨 — **수정 필요(확인됨)** |
| `SWAP_LOG` | `FOOK_adjust_levers.py:43` | 없음 | `adjust()` 시작 시 매번 `.clear()`(1344행) — 안전 |
| `KIMCHI_SIDES` | `FOOK_adjust_levers.py:110` | 없음 | `load_all()` 1회 빌드 후 정적 — 안전 |
| `MENU_CLASS`, `SUBS_P`, `RICE` 등 | 각지 | 없음 | 전부 `load_all()`/import 시 1회 빌드, 이후 읽기 전용 |
| `_SNACK_BUILT`/`SNACK_POOL` | `FOOK_adjust_levers.py:1072,1112` | 없음 | 최초 1회만 빌드(idempotent 플래그), 이후 불변 |
| `inst`/`order`/`room`/`base_fresh` 등 | 각 함수 내부 | 없음 | 전부 함수-로컬 변수(모듈 전역 아님) |
| `menu_ings`(app_core_FOOK) | `app_core_FOOK.py:72` | 없음 | import 시 CSV로 1회 빌드, 이후 읽기 전용 |

**`ROT`가 유일한 문제였다.**

---

## 5. 재검증 방법론

`rl_v2_work/rl_comparison_v2_rot_fixed_FOOK.py` — 각 모델(`run_one()`) 진입 시점마다
`core.F.ROT[0] = 0`을 리셋한다(day_context 필러끼 포함, 그 시나리오·그 모델의 전체 처리가
항상 ROT=0에서 시작). 그 외 시나리오·시드·tries=48·레버·`passes()`·후보선택·fallback은
전부 기존과 동일.

**검증 1 — 실행 순서 무관성**: `base_first`(BASE→RL)와 `rl_first`(RL→BASE) 두 순서로 각각
120건 실행 후 시나리오별 `success_ok`/`final_menus`/개별 영양소 통과 여부를 전부 비교 →
**완전 일치(다른 행 0건, BASE·RL 각각)**.

**검증 2 — 단독실행 일치성**: BASE만 단독으로 120건 실행한 결과, RL만 단독으로 120건 실행한
결과를 각각 2-way(두 순서 모두)의 해당 모델 컬럼과 비교 → **완전 일치(다른 행 0건, 4가지
조합 전부)**.

**결론: 이제 두 모델은 서로 완전히 독립적으로, 실행 순서·동반 모델 유무와 무관하게 항상
동일한 결과를 낸다 — 진짜 "동일 조건" 비교다.**

---

## 6. 정정된 공식 결과 — BASE vs 기존 production RL(i002)

| 지표 | BASE | 기존 RL(i002) |
|---|---:|---:|
| 적합 식단 생성 성공률 | 102/120 (85.0%) | **109/120 (90.8%)** |
| 5대 영양소 완전 충족률 | 109/120 (90.8%) | **113/120 (94.2%)** |
| 열량 충족률 | 119/120 (99.2%) | **120/120 (100%)** |
| 단백질 충족률 | 114/120 (95.0%) | **118/120 (98.3%)** |
| 칼륨 충족률 | 120/120 | 120/120 |
| raw P 충족률 | **116/120 (96.7%)** | 115/120 (95.8%) |
| 나트륨 충족률 | 120/120 | 120/120 |
| 사용자 요청 보존율(n=75) | 71/75 (94.7%) | 71/75 (94.7%) |
| 비현실적 재료량 발생률 | 6/120 (5.0%) | **2/120 (1.7%)** |
| 오류·예외 발생률 | 0/120 | 0/120 |
| 평균 실행시간 | 73.2ms | 61.1ms |

**전이**: 둘다통과 99건, 둘다실패 8건, **BASE만성공 3건, RL만성공 10건**.

**기존 RL(i002)은 raw P만 BASE에 근소하게(1건) 뒤지고, 나머지 전 지표에서 BASE를
앞선다.** 이전 보고서의 "완전 동률"은 ROT 미리셋 버그의 산물이었다 — 실제로는 **기존
RL(i002)이 BASE보다 명확히 우세**하다.

---

## 7. R2의 위치 — BASE보다는 낫지만 기존 RL(i002)보다는 못함

ROT 리셋을 적용한 3-way(BASE/기존RL/R2) 결과(이전 턴에서 이미 산출, 5절 방법론과 동일 적용):

| 지표 | BASE | 기존RL(i002) | R2 |
|---|---:|---:|---:|
| 성공률 | 102/120 | **109/120** | 105/120 |
| 5영양 | 109/120 | **113/120** | 110/120 |
| rawP | 116/120 | 115/120 | **117/120** |
| 단백질 | 114/120 | **118/120** | 114/120 |
| 열량 | 119/120 | **120/120** | 118/120 |
| 비현실 | 6/120 | **2/120** | 4/120 |
| 보존 | 71/75 | 71/75 | 71/75 |

**R2 vs BASE**: 성공률 +3건(102→105), rawP·보존율·비현실 전부 BASE 이상 — R2가 원래
요청하신 "BASE 대비 필수조건"(성공률 최소+3건, rawP·보존율 BASE 이상, 비현실 BASE
이하, 오류 0%)은 **전부 충족**한다.

**R2 vs 기존RL(i002)**: 성공률 105 < 109, 5영양 110 < 113, 단백질 114 < 118, 열량
118 < 120, 비현실 4 > 2 — **거의 모든 지표에서 기존 RL(i002)이 R2보다 낫다.** raw P만
R2가 117로 기존RL의 115보다 근소 우위.

**R2는 "개선"이지만, "이미 있는 것보다 나은 개선"은 아니다.** BASE라는 더 약한 기준선을
넘었을 뿐, 실제로 교체 대상인 기존 production RL(i002)을 능가하지 못했다.

---

## 8. R2 validation(40건)의 신뢰성 재평가

`run_reward_variant_FOOK.py`(R0~R5 학습+validation 스크립트)를 감사한 결과, **`core.F.ROT[0]`
리셋 코드가 어디에도 없다.** 40건의 validation 시나리오를 순회하며 매 시나리오 `F.adjust()`를
호출하지만 ROT를 건드리지 않으므로, (1) 40건 사이에서 ROT가 계속 누적되고, (2) 그보다 더
큰 문제로 **200 에폭 학습 동안 매 에폭 reward 계산(`R0MOD.meal_reward`가 내부에서
`F.adjust()` 호출)이 ROT를 수십만 번 누적시킨 뒤 validation을 시작**하므로, R0/R1/R2/R3/R5
각 변형이 validation을 시작하는 ROT 절대 위치가 서로 다르다(학습 중 실제로 소모한 adjust()
호출 수가 변형마다 미세하게 다를 수 있어서).

**따라서 사용자 지시대로, 3절의 40건 validation 수치(R2 성공률 90% 등)는 공식 의사결정
근거에서 제외하고 참고값으로만 표시한다.** 다행히 **공식 근거인 7절의 120건 비교는 ROT를
올바르게 리셋한 방법론으로 재실행했으므로 그대로 유효하다.** R2 체크포인트 자체(학습된
가중치)는 이 감사와 무관하게 그대로다 — 문제는 어느 신경망을 학습했느냐가 아니라 그
신경망을 "얼마나 잘 골랐는지 검증한 과정"에 있었다.

---

## 9. 최종 의사결정

- **기존 production RL(i002): 유지.** BASE 대비, 그리고 이번에 새로 만든 R2 대비 모두
  가장 우수한 성능을 보였다(raw P 한 항목만 R2에 근소하게 뒤짐).
- **R2: 미채택.** BASE 대비는 개선됐으나 production RL 대비 성공률이 낮다(105/120 <
  109/120) — 교체할 이유가 없다.
- **추가 KL 규제·hard-case oversampling·추가 reward 탐색: 중단.** R2가 이미 기존 RL을
  넘지 못했고, STEP1 validation 자체의 신뢰성 문제(8절)까지 겹쳐 이 방향을 더 파고들
  근거가 약하다. 재개하려면 최소한 (a) validation 코드에도 ROT 리셋 적용, (b) 더 큰
  validation 표본, (c) 학습 자체의 재현성 확보(STEP0 감사에서 지적된 `get_action`의
  전역시드 미고정 문제도 같이 손봐야 함)가 선행되어야 한다.
- **서비스 코드(`app_core_FOOK.py`, `FOOK_adjust_levers.py`)와 production 체크포인트:
  변경 없음.** 이번 세션 전체에서 이 두 파일과 `results_FOOK/checkpoints`,
  `results_sweep_FOOK/i002`, 기존 120개 평가 CSV는 전혀 수정하지 않았다.

---

## 10. 산출 파일

**STEP 0**: `rl_reward_audit.md`, `baseline_reproduction.csv`, `baseline_reproduction_report.md`

**STEP 1**: `reward_lever_v2_FOOK.py`, `log_reward_components_FOOK.py`,
`reward_component_distribution.csv`, `validation_scenarios_FOOK.py`, `validation_scenarios.csv`,
`run_reward_variant_FOOK.py`, `reward_ablation_results.csv`, `training_curve_{R0,R1,R2,
R3_pw1.0,R3_pw2.0,R5}.csv`, `validation_result_{...}.csv`(전부 참고용, 8절 참조),
`checkpoints_{R0,R1,R2,R5}/`(각 변형 체크포인트 — R2가 최종 후보, 나머지는 비교용 보존)

**ROT 재검증**: `rl_comparison_v2_rot_fixed_FOOK.py`, `rot_fixed_base_official.csv`,
`rot_fixed_rl_official.csv`(공식 2-way 결과), `rot_fixed_summary.txt`

**3-way(R2 포함) 재검증**: `rl_v2_final_comparison_FOOK.py`(ROT수정 적용됨),
`rl_v2_results.csv`, `rl_v2_pairwise_results.csv`

**정정된 기존 보고서**: `.../final_service_benchmark_out/rl_comparison_report.md`(상단에
정정 공지 추가)

**본 문서**: `rl_v2_final_report.md`

모든 산출물은 `E:\final\rl_v2_work\`(및 표시된 기존 위치) 아래에 있으며, 서비스 코드·
체크포인트·기존 120개 평가 CSV는 수정하지 않았다.
