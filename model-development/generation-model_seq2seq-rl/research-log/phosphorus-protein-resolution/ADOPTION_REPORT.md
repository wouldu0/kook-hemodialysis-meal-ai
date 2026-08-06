# FOOK 두부·콩류 phosphorus–protein 충돌 완화 — 최종 채택 확정 보고서

**확정일**: 2026-07-27 | **상태**: 최종 채택 완료(추가 레버 실험·파라미터 탐색 없음)

---

## 1. 최종 채택 로직

**아키텍처 변경 없음 — 규칙 기반 영양 최적화 모듈(`adjust()`) 내부의 조건부 분기 하나만
추가됐다.** 게이트 순서·영양 기준값·후보 생성 모델·checkpoint는 전부 그대로다.

두부·콩류 앵커에서만 다음 두 가지를 적용한다:

1. **raw P 판정 통일**: `lever_phosphorus`의 진입/수렴/최종반환 4개 지점(Peff 기준)을
   실제 서비스 게이트(`passes()`)와 동일한 **raw P** 기준으로 통일(`lever_phosphorus_rawP`).
   Peff는 로그·참고값으로만 남는다.
2. **증량 90% cap**: `lever_protein`/`lever_calorie`가 메뉴 분량을 **늘릴 때만**(감량은
   무변경) 남은 raw P 예산의 90%까지만 쓰도록 제한(`lever_protein_capped`,
   `lever_calorie_capped`, `PROTEIN_CALORIE_CAP_FRAC=0.90`).

**목적**: 두부·콩류 주찬(예: 두부양념조림)은 인이 풍부한 두류 식품 특성상, `lever_phosphorus`가
Peff(두류 0.7배 할인) 기준으로 "충분하다"고 판단해도 뒤이은 `lever_protein`이 그 판단과 무관하게
메뉴를 스케일업해 raw P를 재초과시키는 **phosphorus↔protein 핑퐁**이 다른 앵커보다 뚜렷했다
(전체 레버 audit에서 이 핑퐁이 phosphorus 통과 상태의 70.3%를 재파괴, 그중 90.2%가 최종까지
미복구로 확인됨). 위 두 조치는 이 충돌을 두부·콩류 앵커에 한해서만 완화하기 위한 조건부 로직이다.

---

## 2. 실제 수정 파일

**`E:\final\FOOK_adjust_levers.py`** — 유일하게 수정된 서비스 코드 파일.

- 신규 추가 함수(기존 함수는 전부 원본 그대로 보존): `_plant_protein_path_needed`,
  `lever_phosphorus_rawP`, `_cap_scale_menu_rawP`, `lever_protein_capped`,
  `lever_calorie_capped`, 상수 `PROTEIN_CALORIE_CAP_FRAC=0.90`
- `adjust()` 수정: `expand()` 직후 `use_rawP_path = _plant_protein_path_needed(inst, menus,
  anchor)` 1회 계산 → 패스루프 내 3개 레버 호출부만 조건분기 추가
- 모듈 docstring에 2026-07-27 최종 채택 확정 내역(목적·적용범위·검증·롤백 경로) 기록 완료

`app_core_FOOK.py`, `server_FOOK.py` 등 이 파일을 import하는 다른 서비스 코드는 수정하지
않았다 — `adjust()`의 시그니처와 반환값(`before, after, inst, p_ok`)이 그대로라 별도 반영
불필요.

---

## 3. 적용 대상과 비대상

| 구분 | 조건 | 판정 방법 |
|---|---|---|
| **적용 대상** | 주찬(`menus[2]`) 또는 지정 앵커의 정체성 재료(또는 최대량 고형재료)가 `group=='두류'` | `_plant_protein_path_needed()` — 메뉴명 하드코딩 없이 기존 재료 식품군 분류만 사용 |
| **비대상** | 그 외 전부(생선구이·육류 등 비두류 앵커, 랜덤모드 포함) | 기존 `lever_phosphorus`/`lever_protein`/`lever_calorie` 완전히 그대로 |

**변경 없이 유지된 것**(적용 대상·비대상 공통): `lever_potassium`, `lever_sodium`,
`lever_sodium_extra`, `lever_kimchi`, `add_oil`, `add_snack`, pass1·pass2 나트륨 재검증
구조, 게이트 순서, 영양 기준값, 후보 생성 모델, RL checkpoint. **S1(pre-loop 나트륨 제거)은
이번 확정에 포함하지 않았다**(별도 미채택 사안으로 남김).

---

## 4. 최종 검증 결과

`service_rollout_verification_FOOK.py`로 OLD(수정 전 백업)와 NEW(수정본)를 3,600건
(두부콩류·생선구이·육류 각 1,200건) paired 비교:

| 지표 | OLD | NEW | 비고 |
|---|---|---|---|
| 두부·콩류 생성 성공률 | 94% | **100%** | 개선 |
| 두부·콩류 후보 0개율 | 6% | **0%** | 개선 |
| 두부·콩류 비현실적 재료량 | 46% | **20%** | 개선 |
| 생선구이 1,200건 | — | **OLD=NEW 100% 동일** | 무회귀 확인 |
| 육류 1,200건 | — | **OLD=NEW 100% 동일** | 무회귀 확인 |
| 회귀 중단 조건 위반 | — | **0건** | 반영 유지 근거 |

대표 사례(두부콩류 5건: OLD 실패→NEW 통과, raw P가 상한 333.33mg 아래로 이동)와 무변경
사례(생선구이·육류 각 3건: 5개 영양값 소수점까지 완전 동일) 전부 확인 완료
(`D_representative_cases.csv`).

---

## 5. 변경하지 않은 기능

- `lever_potassium`, `lever_sodium`, `lever_sodium_extra`, `lever_kimchi`, `add_oil`,
  `add_snack` — 전부 원본 그대로
- pass1·pass2 나트륨 재검증 구조 — 그대로 유지(전체 레버 audit에서 `protein→sodium` 핑퐁을
  100% 복구시키는 핵심 메커니즘으로 확인된 바 있어 손대지 않음)
- 영양 기준값(Elo/Ehi/Plo/Phi/Kmax/Pmax/Namax), 게이트 순서, `passes()` 자체
- 대체재 풀, 후보 생성 모델(C_mask100+RL), checkpoint
- S1(pre-loop 나트륨 제거) — 검증은 했으나 이번 확정에 포함하지 않음(생선구이에서 나트륨
  통과율 미세하락 사례가 있어 보류 상태로 유지)
- B90(cap)만 단독 적용하는 안, Unified-rawP만 단독 적용하는 안 — 둘 다 실험 결과 부적합
  판정되어 채택 안 함(B90+Unified 결합만 채택)

---

## 6. 롤백 방법

```
copy E:\final\FOOK_adjust_levers.py.bak_before_rawP_tofu_path_20260727 E:\final\FOOK_adjust_levers.py
```

백업 파일은 삭제하지 않고 그대로 보존한다. 되돌리면 두부·콩류를 포함한 모든 앵커가 즉시
수정 전 원본 로직으로 복귀한다(비대상 앵커는 이번 수정으로도 로직이 안 바뀌었으므로 롤백 시
영향 자체가 없음).

---

## 7. 남은 작업 여부

**이번 세션 기준 신규 레버 실험·파라미터 탐색은 계획하지 않는다**(사용자 지시 반영). 다만
과거 진단에서 식별됐으나 이번 확정 범위 밖으로 남긴 사항은 참고용으로만 기록한다(추가 지시
없이는 진행하지 않음):
- S1(pre-loop 나트륨 제거)의 생선구이 앵커 미세 회귀(0.2%p) 원인 확인
- 두부·콩류 국 다양성 미세 변동의 더 큰 표본 재확인
- 임상 영양사의 최종 검수(cap 90%, raw P 통일 기준 자체에 대한 임상적 승인)

---

## 8. 최종 상태: **완료**

두부·콩류 조건부 raw P 통일 + 90% cap 경로가 `E:\final\FOOK_adjust_levers.py`에 최종
채택·확정되었다. 비대상 앵커(생선구이·육류) 무회귀가 3,600건 규모로 검증되었고, 회귀 중단
조건 위반은 0건이다. 배포·백업 정리·파일 삭제는 수행하지 않았으며, 백업 파일과 모든 실험용
스크립트·산출물은 그대로 보존되어 있다.

---

## 부록: 이번 조사 전체 산출물 목록 (Diet-Generation-As-Sequence-master/…/Code/ 하위)

| 단계 | 출력 폴더 | 핵심 산출물 |
|---|---|---|
| 1. 부찬 다양성 최초 진단 | `side_dish_diagnosis_out/` | side_dish_diagnosis_report.md |
| 2. 영양소별 병목 분해 | `side_dish_nutrition_diagnosis_out/` | side_dish_nutrition_diagnosis_report.md |
| 3. 인 슬롯별 기여 분해 | `side_dish_slot_phosphorus_diagnosis_out/` | side_dish_slot_phosphorus_report.md |
| 4. 패스별 스냅샷(F유형 원인) | `phosphorus_lever_step_diagnosis_out/` | phosphorus_lever_order_diagnosis_report.md |
| 5. 수정안 A/C 검증(기각) | `phosphorus_rawP_fix_out/` | phosphorus_rawP_fix_report.md |
| 6. 수정안 B80/90/100 검증 | `phosphorus_rawP_cap_out/` | phosphorus_rawP_cap_report.md |
| 7. Unified-rawP 단독 검증(기각) | `phosphorus_unified_rawP_out/` | phosphorus_unified_rawP_report.md |
| 8. B90+Unified 최종비교(두부콩류) | `protein_phosphorus_final_compare_out/` | protein_phosphorus_final_compare_report.md |
| 9. 나트륨 이중호출 진단 | `sodium_double_call_diagnosis_out/` | sodium_double_call_diagnosis_report.md |
| 10. S1 교차앵커 검증 | `sodium_preloop_cross_anchor_out/` | sodium_preloop_cross_anchor_report.md |
| 11. 전체 레버 audit | `lever_system_audit_out/` | lever_system_audit_report.md |
| 12. B90+Unified 교차앵커 검증 | `protein_phosphorus_cross_anchor_out/` | protein_phosphorus_cross_anchor_report.md |
| 13. 서비스 반영 통합검증 | `service_rollout_verification_out/` | service_rollout_verification_report.md |

**수정된 서비스 코드**: `E:\final\FOOK_adjust_levers.py`
**롤백 백업**: `E:\final\FOOK_adjust_levers.py.bak_before_rawP_tofu_path_20260727`

모든 실험용 스크립트(`*_FOOK.py`)와 위 13개 산출 폴더는 삭제하지 않고 그대로 보존했다.
