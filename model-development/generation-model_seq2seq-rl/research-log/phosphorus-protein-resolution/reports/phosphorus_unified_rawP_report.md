# phosphorus_unified_rawP_report.md

`lever_phosphorus()` 내부 판정기준을 Peff→raw P로 통일한 최소수정안("Unified-rawP")만 단독으로
Baseline과 paired 비교. protein/calorie 레버 cap(B90)·사후보정(C)·재호출(A) 전부 미적용.
두부·콩류 앵커 2,400건. **코드 수정 없음, 실험용 진단 스크립트에서만 구현·검증.**

---

## 실험 환경

**checkpoint**: 동일(`results_rl_soupmask_FOOK/C_mask100_rl/checkpoints_best/ckpt-3`)

**후보 수**: 두부콩류 2,400건(seed 5×call 20×24), 배치 100개. paired comparison(생성+공유
프리픽스 후보당 1회, 패스루프만 Baseline/Unified-rawP로 분기).

**Unified-rawP 구현**: `lever_phosphorus()`(770행)를 복제한 `unified_lever_phosphorus_rawP()`에서
Peff/`p_abs()` 사용 4곳을 전부 raw P로 치환:
1. 루프 진입/수렴 조건 (`totals(inst)['Peff']<pmax` → `totals(inst)['P']<pmax`)
2. 대체후보 비교값 (`p_abs(i['P'],...)` → `i['P']`, `p_abs(nd['P'],...)` → `nd['P']`)
3. 양감소 대상 랭킹 키 (`p_abs(x['P'],...)` → `x['P']`)
4. 루프소진 최종반환 (`totals(inst)['Peff']<pmax` → `totals(inst)['P']<pmax`)

그 외 대체재 선정 순서·임계값(`P_SWAP_MIN_GAIN` 등)·`lever_protein`/`lever_calorie`/패스구조는
전부 원본 그대로. **Baseline 경로가 `F.adjust()`와 5/5 완전일치함을 재검증**(직전 작업들과 동일
검증 절차 반복 적용).

산출: [phosphorus_unified_rawP_trace.csv](phosphorus_unified_rawP_trace.csv)(4,800행),
[phosphorus_unified_rawP_summary.csv](phosphorus_unified_rawP_summary.csv),
[phosphorus_unified_rawP_transition.csv](phosphorus_unified_rawP_transition.csv)

---

## 전체 결과

| 지표 | Baseline | Unified-rawP | 변화 |
|---|---|---|---|
| raw P 통과율 | 27.1% | 29.9% | +2.8%p(미미) |
| **5영양 전부 통과율** | 12.0% | **12.7%** | **+0.7%p(사실상 무효)** |
| F_레버후신규실패 발생률 | 35.4% | 34.9% | -0.5%p(거의 변화 없음) |
| protein_low 실패율 | 17.3% | 16.0% | -1.3%p(소폭 개선) |
| calorie_low 실패율 | 11.5% | **17.4%** | **+5.9%p(악화)** |
| 후보(배치) 0개율 | 10.0% | 7.0% | -3.0%p(개선) |
| call 단위 최종생성 성공률 | 90.0% | 93.0% | +3.0%p(개선) |
| **phosphorus↔protein 핑퐁 발생률** | 46.6% | **71.9%** | **+25.3%p(크게 악화)** |
| pass1→pass2 raw P 평균변화 | +4.00mg | +7.15mg | 진동폭 확대 |
| 원래 5영양 통과 후보를 새로 실패시킨 수 | — | **34건**(1.4%) | 신규 부작용 |
| 부찬/국/주찬/김치 다양성 | (Baseline 기준) | 큰 변화 없음(trace 참고) | — |

**raw P 통과율·5영양 전부 통과율은 거의 안 움직인다**(+2.8%p, +0.7%p) — B90(+46.5%p, +7.1%p)과
비교하면 **한 자릿수 이상 작다.** **F발생률도 거의 그대로**(35.4→34.9%)라, 애초 이 실험이
겨냥한 문제(레버후 신규실패)를 거의 해결하지 못한다.

**가장 중요한 발견 — 핑퐁이 오히려 증가**: phosphorus↔protein 핑퐁 발생률이 46.6%→**71.9%**로
25.3%p 악화됐고, `pingpong_resolved=0건`(기존 핑퐁이 해소된 사례가 단 한 건도 없음)인 반면
`pingpong_newly_introduced=606건`(25.2%, 새로 핑퐁이 생긴 사례)이다. pass1→pass2 사이 raw P
변화폭도 4.00mg→7.15mg으로 커져 **진동이 줄기는커녕 늘었다.**

**원인**: `lever_phosphorus`의 진입/수렴 기준을 raw P로 더 엄격하게 만들면 그 레버 자신은
더 적극적으로(더 많이) 재료를 교체·축소하지만, **바로 다음에 실행되는 `lever_protein`은
여전히 phosphorus를 전혀 고려하지 않고 필요하면 그대로 스케일업한다** — phosphorus 레버가
아무리 정확한 기준으로 열심히 낮춰놔도, 그 직후 단백질 레버가 다시 올려버리는 구조 자체는
그대로이므로 "더 정확해진 phosphorus 판정"이 오히려 "더 자주/크게 되돌려지는" 왕복만 늘린
것으로 해석된다.

---

## paired 전환 분석

| 전환 | 건수 | 비율 |
|---|---|---|
| Baseline raw P 실패 → Unified raw P 통과 | 92 | 3.8% |
| **Baseline 5영양 실패 → Unified 5영양 통과(진짜 구제)** | **51** | **2.1%**(B90의 7.8%에 크게 못 미침) |
| Baseline 5영양 통과 → Unified protein_low(신규 부작용) | 5 | 0.2% |
| Baseline 5영양 통과 → Unified calorie_low(신규 부작용) | 11 | 0.5% |
| Baseline F유형 → Unified 5영양 통과 | 18 | 0.8%(F유형 850건 중 2.1%만 완전구제) |
| Baseline F유형 → Unified rawP통과but protein실패 | 3 | 0.1%(B90의 31.6%와 극명히 대비) |
| Baseline·Unified 둘 다 실패 | 2,061 | 85.9% |
| **기존 핑퐁이 해소된 경우** | **0** | **0.0%** |
| **신규로 핑퐁이 생긴 경우** | **606** | **25.2%** |

**B90과의 결정적 차이**: B90은 F유형의 89.3%(759/850)가 "raw P는 고쳤지만 protein이 깨지는"
형태로 실패사유가 이동했다(진짜 개입이 일어났다는 뜻). Unified-rawP는 이 이동이 겨우 0.1%
(3/850)뿐이다 — **즉 Unified-rawP는 F유형에 대해 사실상 아무 실질적 개입도 못 하고 있다**(F발생률
자체가 거의 그대로인 것과 일치). 대신 그 자리에 "핑퐁 증가"라는 새로운, 더 나쁜 부작용이
생겼다.

---

## 최종 판정

**1. raw P 단일 기준만으로 문제가 충분히 해결되는가?**
**아니오, 전혀 충분하지 않다.** raw P 통과율(+2.8%p)·5영양 전부 통과율(+0.7%p)·F발생률(-0.5%p)
전부 개선폭이 미미하다.

**2. 5영양 전체 통과율이 실제로 개선되는가?**
**개선되지만 무의미한 수준(+0.7%p)이다.** B90(+7.1%p)과 비교하면 10분의 1 수준.

**3. 단백질·열량 실패로 단순 이동하는가?**
**부분적으로만, 그리고 예상과 다른 방향으로.** protein_low는 오히려 소폭 개선(-1.3%p)됐지만
calorie_low가 악화(+5.9%p)됐고, 무엇보다 **"실패 사유 이동"보다 "핑퐁(왕복) 증가"가 훨씬 큰
부작용**으로 나타났다(+25.3%p, 기존 핑퐁 해소는 0건). 즉 이 수정은 실패를 다른 영양소로
"이동"시키기보다 **레버 간 왕복 자체를 심화**시켰다.

**4. 단일 기준만으로 부족할 때에만 B90이 추가로 필요한가?**
**예, 정확히 그 경우에 해당한다.** Unified-rawP는 진단적으로 의미 있는 확인(raw/Peff 판정
불일치라는 근본 원인이 실재함을 재확인)이었지만, **그 자체만으로는 실용적 개선이 되지 못하며
오히려 새로운 부작용(핑퐁 25%p 증가)을 만든다.** 문제의 핵심은 "phosphorus 레버가 어떤 기준을
쓰는가"가 아니라 **"phosphorus 레버 다음에 실행되는 protein/calorie 레버가 phosphorus 예산을
전혀 고려하지 않는다"는 순서·상호작용 구조 자체**임이 이번 실험으로 다시 한번 확인됐다 —
따라서 **B90(protein/calorie 증량 자체에 raw P 예산 제약을 미리 거는 방식)이 여전히 필요하며,
이번 Unified-rawP 실험 결과는 이 결론을 약화시키지 않고 오히려 강화한다.**

---

## 결론

Unified-rawP 단독 적용은 **권장하지 않는다.** 진단적 가치(raw/Peff 불일치가 실제 원인이라는
근거 재확인)는 있지만, 실용적 효과는 미미하고 핑퐁이라는 새로운 부작용을 유발한다. 기존
B90(직전 실험, 5영양전부 12.0%→19.1%, F발생 35.4%→0.0%, 부작용은 원래-실패 후보에 집중)이
지금까지 검증된 세 가지 접근(A/C/Unified-rawP) 중 **유일하게 실질적 순이득을 보인 안**이라는
결론은 이번 실험으로도 변하지 않는다.

**다음 단계 제안**: Unified-rawP를 단독으로 서비스에 반영하지 말 것. 대신 이전 결론대로 **B90을
기준으로 검토를 이어가되**, 필요하다면 "B90 + Unified-rawP 동시 적용"(phosphorus 레버 기준도
raw P로 통일하면서 protein/calorie도 cap을 거는 조합)을 후속 소규모 실험으로 확인해볼 수는
있다 — 단, 이번 결과만 보면 Unified-rawP가 추가로 주는 이득이 작아 보이므로 우선순위는 낮다.
**이번 단계에서는 어떤 조합도 실제 서비스 코드에 반영하지 않았다.**

---

## 산출 파일

1. [phosphorus_unified_rawP_trace.csv](phosphorus_unified_rawP_trace.csv) — 2,400후보×2변형=
   4,800행
2. [phosphorus_unified_rawP_summary.csv](phosphorus_unified_rawP_summary.csv) — 변형별 전체지표
3. [phosphorus_unified_rawP_transition.csv](phosphorus_unified_rawP_transition.csv) — paired
   전환 9종
4. `phosphorus_unified_rawP_report.md`(본 문서)

생성 스크립트: `phosphorus_unified_rawP_experiment_FOOK.py` — 기존 모델/레버/게이트 코드
무수정, `unified_lever_phosphorus_rawP`를 진단용 복사본으로 구현.
