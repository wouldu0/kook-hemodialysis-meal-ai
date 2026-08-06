# 서비스 코드 반영 검증 보고서 (service_rollout_verification_report.md)

두부·콩류 앵커 전용 B90+Unified-rawP 경로를 `E:\final\FOOK_adjust_levers.py`에 조건부로
반영하고, OLD(수정 전 백업) vs NEW(수정본) paired 비교로 검증. **비대상 앵커는 100% 동일함을
확인했다.**

---

## 반영 내용 요약

**수정 파일**: `E:\final\FOOK_adjust_levers.py` (백업: `FOOK_adjust_levers.py.bak_before_rawP_
tofu_path_20260727`, 롤백 필요시 이 파일로 교체하면 즉시 원복).

**추가한 함수(전부 신규 추가, 기존 함수는 원본 그대로 보존)**:
1. `_plant_protein_path_needed(inst, menus, anchor)` — 주찬(`menus[2]`) 또는 지정앵커의
   정체성 재료(또는 최대량 고형재료)가 `group=='두류'`인지 판정. **메뉴명 하드코딩 없음**,
   기존 재료 식품군 분류만 사용.
2. `lever_phosphorus_rawP(inst, pmax, anchor=None, plo=0)` — `lever_phosphorus`의 raw P
   통일판(4개 Peff 판정 지점을 raw P로 치환, 그 외 로직 완전 동일).
3. `_cap_scale_menu_rawP(inst, menu, ratio, pmax)` + `lever_protein_capped(...)` +
   `lever_calorie_capped(...)` — 증량(scale_menu 비율>1)에만 남은 raw P 예산의 90%
   (`PROTEIN_CALORIE_CAP_FRAC=0.90`) 상한 적용, 감량은 무변경.

**`adjust()` 수정**: `expand()` 직후 `use_rawP_path = _plant_protein_path_needed(inst, menus,
anchor)`를 1회 계산하고, 패스루프 내부에서 `use_rawP_path`가 True일 때만 위 3개 신규 함수를
쓰고 False면 **기존 `lever_phosphorus`/`lever_protein`/`lever_calorie`를 정확히 그대로**
호출한다. `lever_kimchi`/`lever_sodium`/`lever_sodium_extra`/`lever_potassium`은 두 경로
모두에서 완전히 동일하게(무조건) 호출된다 — 손대지 않았다.

---

## A. 구현 동일성 (두부·콩류)

`_plant_protein_path_needed` 판정: 두부양념조림 → **True**(정확히 걸림). 3,600건(3앵커 합계)
중 두부콩류 1,200건에서 OLD와 NEW가 **199건(16.6%) 동일, 1,001건(83.4%) 차이** — 새 경로가
실제로 작동하고 있음을 확인(199건은 애초에 인/단백/열량 조정이 전혀 필요 없던 후보라 두
경로 모두 무변화인 경우).

---

## B. 비대상 회귀

| 앵커 | 두부콩류 판정 | OLD=NEW 동일 | 차이 |
|---|---|---|---|
| 생선구이 | False | **1,200/1,200 (100%)** | **0건** |
| 육류 | False | **1,200/1,200 (100%)** | **0건** |

**비대상 앵커(생선구이·육류)는 raw 영양값 5종(E/protein/K/P/Na_season) 전부 소수점까지 완전히
동일하다.** sodium/potassium/kimchi 결과도 이 5종 안에 포함되어 함께 확인됨(별도 규명불요 —
동일 판정 자체가 이미 이 값들을 포함).

---

## C. 전체 회귀 테스트

**참고**: 코드 안에 기존 "365일 전체 테스트셋" 하니스가 없어(`pipeline_measure_FOOK.py`는
`adjust()`를 직접 호출하지 않고 레버 순서를 손으로 복제해둔 것이라 이번 수정을 자동 반영하지
않음 — 기존에도 알려진 한계), 이번 검증에서는 **`F.adjust()`를 직접 호출하는 3,600건
규모(5seed×10call×24×3앵커) 자체 회귀셋을 구성**해 OLD/NEW paired로 실행했다.

| 앵커 | variant | 생성성공률 | 후보0개율 | protein_low | 비현실 재료량 | 국다양성 | 부찬다양성 | 김치다양성 | 평균실행시간 |
|---|---|---|---|---|---|---|---|---|---|
| **두부콩류** | OLD | 94.0% | 6.0% | 0.0% | 46.0% | 19 | 13 | 4 | — |
| **두부콩류** | NEW | **100.0%** | **0.0%** | 0.0% | **20.0%** | 20 | 17 | 4 | 1.26ms |
| 생선구이 | OLD | 100.0% | 0.0% | 0.0% | 40.0% | 18 | 13 | 5 | — |
| 생선구이 | NEW | **100.0%(동일)** | **0.0%(동일)** | 0.0%(동일) | **40.0%(동일)** | **18(동일)** | **13(동일)** | **5(동일)** | 0.84ms |
| 육류 | OLD | 100.0% | 0.0% | 0.0% | 52.0% | 26 | 9 | 15 | — |
| 육류 | NEW | **100.0%(동일)** | **0.0%(동일)** | 0.0%(동일) | **52.0%(동일)** | **26(동일)** | **9(동일)** | **15(동일)** | 1.87ms |

**두부콩류만 개선되고(생성성공률 +6%p, 후보0개율 -6%p, 비현실 재료량 46%→20%), 생선구이·
육류는 소수 셋째자리까지 완전히 동일**하다 — 앞서 검증한 교차앵커 실험 결과와 정확히 일치.
실행시간은 두부콩류에서 다소 증가(1.26ms, cap 계산 오버헤드)하나 절대값 자체가 이미 1ms대라
서비스에 영향 없는 수준이다.

산출: [C_full_regression_summary.csv](C_full_regression_summary.csv)

---

## D. 대표 사례

**두부콩류 개선사례 5건**(OLD 실패 → NEW 통과, 전부 raw P가 상한 333.33mg 아래로 내려감):

| candidate_id | 부찬 | OLD phosphorus_raw | NEW phosphorus_raw | OLD 통과 | NEW 통과 |
|---|---|---|---|---|---|
| 7 | 취나물무침 | 333.48(상한 초과) | **332.05(통과)** | False | **True** |
| 24 | 취나물무침 | 340.88 | **332.26** | False | **True** |
| 31 | 취나물무침 | 333.48 | **332.05** | False | **True** |
| 38 | 가자미양념찜 | 346.49 | **330.67** | False | **True** |
| 43 | 콩나물무침 | 334.37 | **330.57** | False | **True** |

candidate 7·31이 완전히 같은 값인 것은 서로 다른 seed·call에서 동일 조합이 재현된 경우다
(정상 — 후보 생성 자체는 확률적이라 같은 조합이 반복 등장할 수 있음).

**생선구이·육류 무변경사례 각 3건**(E/protein/K/P/Na_season **소수점까지 완전 동일**):

| 앵커 | 부찬 | OLD phosphorus_raw | NEW phosphorus_raw | 동일 여부 |
|---|---|---|---|---|
| 생선구이 | 쑥갓두부무침 | 332.31541184406706 | 332.31541184406706 | ✓ |
| 생선구이 | 숙주나물 | 322.5388126493121 | 322.5388126493121 | ✓ |
| 생선구이 | 숙주나물 | 329.55706230268345 | 329.55706230268345 | ✓ |
| 육류 | 취나물무침 | 315.9166108243581 | 315.9166108243581 | ✓ |
| 육류 | 느타리버섯볶음 | 293.31619721099264 | 293.31619721099264 | ✓ |
| 육류 | 숙주나물 | 286.87092251847474 | 286.87092251847474 | ✓ |

산출: [D_representative_cases.csv](D_representative_cases.csv)(11건 전체)

---

## 최종 확인

- **비대상 앵커 회귀**: **위반 0건** — 생선구이·육류 2,400건(합계) 전부 OLD=NEW 완전 일치.
- **두부콩류 개선**: 생성성공률·후보0개율·비현실 재료량 전부 이전 실험(교차앵커 검증)과
  일치하는 방향·크기로 개선됨을 실제 서비스 코드에서 재확인.
- **회귀 중단 조건**("비대상 앵커가 하나라도 달라지거나 전체 성공률·다양성이 유의미하게
  악화되면 중단") **해당 없음** — 중단 사유가 발견되지 않았다.

**이번 세션에서 실제로 `FOOK_adjust_levers.py`를 수정했다.** 백업 파일
(`FOOK_adjust_levers.py.bak_before_rawP_tofu_path_20260727`)이 롤백용으로 남아있다.
**요청대로 통합 테스트 결과 보고까지만 수행했고, 최종 확정(배포·백업 정리 등 후속 조치)
전에는 멈춘다 — 추가 승인 필요.**

---

## 산출 파일

1. [AB_identity_check.csv](AB_identity_check.csv) — 앵커별 OLD/NEW 동일성 판정
2. [C_full_regression_summary.csv](C_full_regression_summary.csv) — 전체 회귀 서비스지표
3. [D_representative_cases.csv](D_representative_cases.csv) — 대표 사례 11건
4. `service_rollout_verification_report.md`(본 문서)

생성 스크립트: `service_rollout_verification_FOOK.py` — OLD는 수정 전 백업파일을
`importlib`로 별도 모듈 로드, NEW는 실제 수정된 `FOOK_adjust_levers.py`를 `app_core_FOOK`
경로 그대로 사용해 동일 후보 3,600건 paired 비교.
