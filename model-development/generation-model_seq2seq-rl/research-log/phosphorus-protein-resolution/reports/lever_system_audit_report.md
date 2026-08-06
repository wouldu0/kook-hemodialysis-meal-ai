# Part 2. 전체 레버 audit (lever_system_audit_report.md)

`adjust()` 내 전체 레버(potassium/phosphorus/protein/sodium/sodium_extra/calorie/kimchi 및
보조함수 add_oil/add_snack)의 구조적 문제(중복호출·순서충돌·핑퐁·기준불일치)를 진단만 수행.
두부·콩류 앵커 2,400건, 매 레버 직후 16개 스냅샷. **코드 수정 없음.**

---

## 1. adjust 실행 순서

| 순번 | 함수 | 위치 | 호출횟수(후보당) | 조정대상 영양소 | 실제 변경대상 | 성공판정 기준 | passes()와 동일기준? | 다음 레버가 다시 바꿀 수 있는가 |
|---|---|---|---|---|---|---|---|---|
| 1 | `lever_kimchi` | 1118행, pre-loop | 1 | 나트륨(간접, 김치 자체 소금) | 김치반찬 통째교체(KIMCHI_SIDES→LOWNA_POOL) | 없음(무조건 1회 시도) | — | 있음(뒤 레버가 김치 재료를 다시 건드리진 않지만 다른 레버가 총량에 영향) |
| 2 | `lever_sodium`(pre) | 1119행 | 1 | 나트륨(첨가염, Na_season) | 조미료류+젓갈 amt 축소 | `season_na≤SALT_MG` | **동일**(Na_season 기준) | **있음**(§6 확인, protein이 60.2% 재붕괴) |
| 3 | `lever_sodium_extra`(pre) | 1120행 | 1 | 나트륨(총량, Na) | 고나트륨 메뉴 통째/부분 축소 | `totals(inst)['Na']≤na_target` | **다름**(총Na 기준, 게이트는 Na_season만 봄) | 있음 |
| 4 | `lever_potassium` | 1129행 | 2(패스당1) | 칼륨(K) | 재료 스왑/양감소 | `totals(inst)['K']<Kmax` | 동일 | 거의 없음(§4, 가장 깨끗) |
| 5 | `lever_phosphorus` | 1130행 | 2(패스당1) | 인(P) | 재료 스왑/양감소 | **`totals(inst)['Peff']<Pmax`** | **다름**(게이트는 raw P) | 있음(§6, calorie가 24.8~48.5% 재붕괴) |
| 6 | `lever_protein` | 1131행 | 2(패스당1) | 단백질 | 메뉴 통째 스케일 | `lo≤totals(inst)['protein']≤hi` | 동일 | (다음패스에서 자기 자신이 재조정) |
| 7 | `lever_sodium`(pass) | 1135행 | 2(패스당1) | 나트륨(첨가염) | 조미료류+젓갈 amt 축소 | 동일 | 동일 | 있음 |
| 8 | `lever_sodium_extra`(pass) | 1138행 | 2(패스당1) | 나트륨(총량) | 고나트륨 메뉴 축소 | 동일 | 다름 | 있음 |
| 9 | `lever_calorie`(+add_oil/add_snack) | 1141행 | 2(패스당1) | 열량 | 밥/기름 증감, 간식추가 | `lo≤totals(inst)['E']≤hi` | 동일 | (다음패스에서 자기 자신이 재조정) |

**핵심 구조적 사실**: 레버 5(`lever_phosphorus`)와 레버 3·8(`lever_sodium_extra`)이 **자기 자신의 성공판정
기준이 최종 `passes()` 게이트와 다르다.** 이 두 곳이 §7의 구조적 불일치 대상이다.

---

## 2. 레버별 호출 횟수

| 레버 | 호출횟수(2,400후보 기준) | 목표영양소 |
|---|---|---|
| kimchi | 2,400(1회) | sodium(간접) |
| sodium | 7,200(3회: pre+pass1+pass2) | sodium |
| sodium_extra | 7,200(3회) | sodium |
| potassium | 4,800(2회: pass1+pass2) | potassium |
| phosphorus | 4,800(2회) | phosphorus |
| protein | 4,800(2회) | protein |
| calorie | 4,800(2회) | calorie |

---

## 3. noop 비율

| 레버 | 전체 noop율 | pre | pass1 | pass2 |
|---|---|---|---|---|
| kimchi | 18.0% | — | — | — |
| sodium | 42.2% | **0.04%**(거의 항상 작동) | 39.6% | 87.0%(대부분 조용) |
| sodium_extra | 83.7% | 77.6% | 88.5% | 85.1% |
| potassium | 97.5% | — | 98.6% | 96.4%(가장 조용) |
| phosphorus | 81.9% | — | 78.5% | 85.2% |
| protein | 30.8% | — | **13.9%**(거의 항상 작동) | 47.7% |
| calorie | 35.9% | — | 30.8% | 40.9% |

**pre-loop `lever_sodium`은 noop율 0.04%로 사실상 매 후보마다 작동한다** — 기본 레시피의 조미료
나트륨이 거의 항상 393mg 상한을 넘는다는 뜻(첨가염 축소가 상시 필요한 조정임을 재확인).
**`potassium`이 가장 조용한 레버**(noop 96~99%, 이미 칼륨은 대부분 통과 상태라 개입할 일이 적음).

---

## 4. 목표 영양소 구제율

| 레버 | 목표영양소 실패였던 경우(모수) | 구제 성공률 |
|---|---|---|
| potassium | 119 | **100.0%**(완벽) |
| sodium | 4,160 | **99.4%** |
| protein | 3,322 | 85.1% |
| calorie | 3,173 | 77.5% |
| phosphorus | 2,650 | **3.4%**(매우 낮음) |
| sodium_extra | 26 | 0.0%(모수 자체가 극히 작음, §7에서 원인 설명) |
| kimchi | 2,399 | 0.0%(자기 단독으론 게이트를 못 뒤집음, §7에서 설명) |

**phosphorus 레버의 구제율이 압도적으로 낮다(3.4%)** — 스스로 활동은 하는데(noop 18~21%)
실제 raw P 게이트를 통과시키는 데는 거의 기여하지 못한다. 이는 §1에서 확인한 **Peff 기준
판정 때문**(내부적으로 Peff만 보고 "성공"이라 여기는 경우가 많아, raw P 기준으로는 구제로
안 잡힘 — 직전 작업들에서 이미 확정된 raw/Peff 불일치의 재확인).

---

## 5. 다른 영양소 신규 실패율

| 레버 | 다른 영양소 신규실패율(호출당) |
|---|---|
| **protein** | **82.6%**(압도적 1위) |
| calorie | 22.6% |
| phosphorus | 15.5% |
| sodium_extra | 12.3% |
| kimchi | 10.4% |
| sodium | 3.3% |
| potassium | **0.46%**(가장 안전) |

**`lever_protein`이 시스템에서 가장 파괴적인 레버**다 — 호출 5번 중 4번 이상 다른 영양소를 새로
깨뜨린다. 반대로 **`lever_potassium`은 가장 안전**(0.46%)해, 이번 audit에서 유일하게 "손댈
필요 없음"이 명확한 레버다.

---

## 6. 레버 간 핑퐁 (12개 지정쌍)

| 쌍 | 앞레버가 맞춘 값 | 뒤레버가 다시 깨뜨림 | 깨진것중 pass2회복 | 최종까지 실패잔존 |
|---|---|---|---|---|
| phosphorus→protein | 102 | 13(12.7%) | 13(100%) | 1 |
| phosphorus→calorie | 24(모수) | 197(**48.5%**) | 188(95.4%) | 9 |
| **protein→phosphorus** | 227(모수) | **1,082(70.3%)** | 187(17.3%) | **976(90.2%, 거의 안 고쳐짐)** |
| calorie→phosphorus | 43(모수) | 195(24.8%) | 59(30.3%) | 136(69.7%) |
| sodium→protein | 0(모수) | 176(9.2%) | 44(25.0%) | 112(63.6%) |
| **protein→sodium** | 4 | **1,428(60.2%)** | **1,428(100%, 완전회복)** | **0** |
| potassium→phosphorus | 3 | 0(0%) | 0 | 0 |
| potassium→protein | 1 | 3(1.2%) | 3(100%) | 0 |
| kimchi→sodium | 0 | 0(0%) | 0 | 0 |
| kimchi→potassium | 119 | 0(0%) | 0 | 0 |
| calorie→sodium | 0 | 0(0%) | 0 | 0 |

**가장 심각한 핑퐁: `protein→phosphorus`.** phosphorus가 통과 상태였던 227건 중 70.3%(1,082건
—단 이 표의 모수는 "phosphorus가 이미 통과였던 스텝 전체 발생횟수"라 후보수와 다름 유의)를
`lever_protein`이 다시 깨뜨리고, 그중 **90.2%가 최종까지 복구되지 않는다.** 이것이 지금까지
여러 차례 진단한 "F_레버후신규실패"의 근본 메커니즘을 이번 전체 audit에서도 그대로 재확인한다.

**가장 성공적인 재검증: `protein→sodium`.** protein이 sodium을 60.2%나 다시 깨뜨리지만
**100%가 pass2까지 완전히 회복된다** — 이전 진단(sodium 이중호출)에서 "pass2 호출은 절대
제거하면 안 된다"고 판정한 근거가 이번 전체 레버 관점에서도 정확히 들어맞는다.

**phosphorus→calorie도 주목할 만하다**(48.5% 깨짐, 95.4% 회복) — phosphorus 레버의 재료
교체가 열량에도 영향을 주지만 대부분 이후 calorie 레버가 무리 없이 복구한다.

---

## 7. passes() 기준 불일치

**확정된 불일치 2건**:
1. **`lever_phosphorus`**: 내부 진입/수렴/최종판정이 전부 **Peff**(흡수보정 인) 기준인데,
   실제 게이트 `passes()`(app_core_FOOK.py:348)는 **raw P**를 본다. (§4의 구제율 3.4% vs
   noop율 18~21%의 괴리로 실측 재확인 — 이전 3개 실험(A/C/Unified-rawP)에서 이미 상세 검증됨)
2. **`lever_sodium_extra`**: 내부 진입/수렴 조건이 **총 나트륨(Na) vs na_target**(남은예산
   기반, 655mg×이월)인데, 게이트가 실제로 보는 것은 **Na_season(첨가염) vs Namax(393mg
   고정)**다. 이번 audit에서 새로 확인됨 — `sodium_extra`의 목표실패모수가 26건뿐인 이유가
   바로 이것: 대부분의 후보가 "총나트륨은 예산 안"이라 sodium_extra 입장에선 이미 할 일이
   없는데, 그것과 "첨가염이 393mg 이내인가"는 별개 질문이라 서로 독립적으로 움직인다.
   (단, `lever_sodium`이 첨가염 쪽을 이미 담당하므로 실무상 치명적 공백은 아님 — 두 레버의
   "역할 분담"이 원래 의도된 설계이나, 이름과 달리 `_extra`가 게이트와 직접 연동되지 않는다는
   점은 코드를 처음 보는 사람에게 오해의 소지가 있음)

---

## 8. 수정이 필요한 레버

수정후보 판정기준(A~E) 대비:

| 레버 | A(제거시 유지+개선) | B(90%+ noop, 재검증無) | C(기준 불일치) | D(핑퐁으로 유의미 저하) | E(원래통과 반복실패) | 판정 |
|---|---|---|---|---|---|---|
| **protein** | — | — | — | **해당(D)**: phosphorus 70.3%·sodium 60.2%·calorie 42.9%(pass1) 동시다발 파괴 | 잠재적 해당(§§이전 B90/C 실험에서 실측) | **수정후보(최우선)** — 단 "제거"가 아니라 스케일업 방식 자체의 재검토(이전 세션의 B90 계열) |
| **phosphorus** | — | — | **해당(C)**: Peff≠raw P | — | — | **수정후보** — 판정기준을 raw P로(단, 이전 Unified-rawP 실험에서 "단독 적용은 핑퐁만 늘림" 확인됨 — B와 결합 필요) |
| **sodium_extra** | — | 부분해당(noop 83.7%, 단 §7에서 근본원인은 "재검증 무의미"가 아니라 "다른 기준을 봄") | **해당(C)**: 총Na≠Na_season | — | — | **명칭/문서 정리 후보**(기능은 유지, 다만 이름이 "게이트 재검증"처럼 오해되지 않게) |
| calorie | — | — | — | 부분해당(phosphorus 24.8%p 깨뜨림, 대부분 미회복) | — | 관찰 필요(수정 시급성은 protein보다 낮음) |
| kimchi | — | — | — | — | — | 해당없음(protein_break 32.7%가 다소 크지만 A~E 조건 명확 미충족) |

**단순 호출횟수가 많다는 이유만으로 수정후보 지정한 레버는 없음**(예: sodium/sodium_extra는
호출이 3회로 가장 많지만, §6에서 확인했듯 그 반복이 실제로 필요한 재검증이라 A~E 어디에도
해당 안 함 — 단 sodium_extra는 C에 해당해 별도 표기).

---

## 9. 건드리면 안 되는 레버

- **`lever_potassium`**: other_nutrient_break_rate 0.46%로 시스템에서 가장 깨끗. 자기 목표
  구제율도 100%. **손댈 이유가 전혀 없음.**
- **`lever_sodium`(pass1·pass2 호출)**: `protein→sodium` 핑퐁을 100% 복구시키는, 이번 audit에서
  확인된 가장 성공적인 재검증 메커니즘. **절대 제거하면 안 됨**(직전 실험에서도 동일 결론).
- **`lever_kimchi`**: 자기 목표(sodium)에는 직접 기여 안 하지만(§4), potassium을 깨뜨리는 사례는
  0건이고 오히려 119건 개선. protein_break 32.7%는 주의 깊게 볼 사안이나, A~E 기준상 "수정후보"로
  단정할 근거는 부족(이번 audit 범위에서는 관찰만 권고).

---

## 10. 다음 최소 검증안 (최대 3개, 구현 안 함)

### 안 1: phosphorus 레버 기준(raw P)과 protein 레버의 스케일업 제한(B90)을 **함께** 적용
- **근거**: §6의 `protein→phosphorus`(70.3% 파괴, 90.2% 미회복)가 시스템 최대 핑퐁이고, 직전
  세션의 Unified-rawP 단독 실험(§Part2 이전 과제)이 "raw P 기준 통일만으로는 핑퐁이 오히려
  25.3%p 악화"됨을 이미 실측했다 — **B90(스케일업 배율 제한)과 병행해야 실효가 있을 가능성이
  높다**는 이전 결론을 이번 전체 audit이 다시 뒷받침한다.
- **검증 방법**: 이전 B90 실험 스크립트에 raw-P 기준 phosphorus 레버를 결합한 3-way(Baseline/
  B90/B90+Unified-rawP) paired 비교(다음 단계 제안, 이번엔 미실행).

### 안 2: `lever_sodium_extra`의 문서·명칭 정리(기능 변경 아님)
- **근거**: §7에서 확인된 "게이트(Na_season)와 다른 기준(총Na)을 본다"는 사실이 코드
  주석에는 있지만 함수명(`_extra`)만 보면 "sodium의 연장/재검증"으로 오인하기 쉽다. 실제로는
  **완전히 다른 목적**(하루 예산 관리)의 레버다.
- **검증 방법**: 코드 변경 없이 주석/문서만 보강(실제 로직·판정 무변경) — 이번 단계에서는
  제안만, 적용 안 함.

### 안 3: `lever_kimchi`의 protein 부수효과 실측
- **근거**: kimchi_lever의 protein_break_rate 32.7%는 이번에 처음 발견된 수치이며, 이전
  어떤 세션에서도 조사되지 않았다. LOWNA_POOL(피클류)이 단백질을 거의 안 가진 재료 위주라
  기존 김치 대비 단백질이 낮아질 가능성이 있다(가설, 미확정).
- **검증 방법**: `LOWNA_POOL` 6종과 원래 `KIMCHI_SIDES`의 평균 단백질 함량을 코드 읽기로
  대조(모델 재실행 불필요, 이번 trace의 delta_protein 컬럼을 kimchi 스텝만 필터링해 재분석
  가능) — 다음 단계 제안, 이번엔 미실행.

**우선순위**: 안1(phosphorus×protein 결합) > 안2(문서화, 비용 거의 없음) > 안3(호기심성 확인).

---

## 산출 파일 (Part 2)

1. [lever_interaction_step_trace.csv](lever_interaction_step_trace.csv) — 2,400후보×16스텝=
   38,400행
2. [lever_call_summary.csv](lever_call_summary.csv) — 레버별 호출횟수·noop·구제율·타영양소파괴율
3. [lever_pair_interaction.csv](lever_pair_interaction.csv) — 12개 지정 상호작용쌍
4. [lever_noop_and_regression.csv](lever_noop_and_regression.csv) — 레버×pass위치×영양소 회귀매트릭스
5. `lever_system_audit_report.md`(본 문서)

생성 스크립트: `lever_system_audit_FOOK.py`(모델 실행, 16스텝 전체 스냅샷, `F.adjust()`와
5/5 완전일치 검증됨) + `lever_system_audit_analysis_FOOK.py`(pandas 후속분석) — 기존
모델/레버/게이트 코드 무수정.

**★ 방법론 노트**: 최초 noop 판정을 (메뉴,재료) 키 기준 분량-diff로 계산했더니 김치(통째교체)·
잠재적으로 potassium/phosphorus(재료 스왑, 같은 리스트원소의 필드만 변경)에서 "키가 바뀌는"
변화를 놓치는 결함이 발견됨(김치 noop=100%로 나왔으나 실제로는 kimchi→potassium 상호작용에서
119건의 실질 변화가 확인돼 모순). **영양 총계(delta_calorie 등 6개 값)의 절대변화로 noop을
재계산**해 이 blind spot을 제거함(본 보고서의 모든 noop 수치는 수정된 방식 기준).
