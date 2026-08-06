# STEP 0 — 기준선 재현 보고서

**실행일**: 2026-07-28

---

## 실행 명령어

```
conda activate foodbert
set TF_USE_LEGACY_KERAS=1
cd E:\final
python rl_comparison_FOOK.py
```

## 실행 환경

- Python 3.9.25, TensorFlow 2.20.0 (`tf_keras`/legacy Keras 모드)
- OS: Windows, conda 환경 `foodbert`

## 사용 체크포인트

- BASE: `results_FOOK/checkpoints/ckpt-1` (827-vocab, 순수 모방학습)
- RL: `results_sweep_FOOK/i002/ckpt-1` (827-vocab, 실제 production이 로드하는 체크포인트)

수정 없이 기존 `rl_comparison_FOOK.py`를 그대로 재실행했다(코드 변경 없음).

---

## 재현 결과

| 지표 | 사용자 인용값 | 재현값 | 일치 |
|---|---|---|---|
| BASE 성공률 | 104/120 (86.7%) | 104/120 (86.7%) | OK |
| 기존 RL 성공률 | 104/120 (86.7%) | 104/120 (86.7%) | OK |
| BASE 5대영양완전충족률 | 110/120 | 110/120 | OK |
| 기존 RL 5대영양완전충족률 | 108/120 | 108/120 | OK |
| BASE raw P 충족률 | 116/120 | 116/120 | OK |
| 기존 RL raw P 충족률 | 111/120 | 111/120 | OK |
| BASE 성공 → RL 실패 | 7건 | 7건 | OK |
| BASE 실패 → RL 성공 | 7건 | 7건 | OK |
| 사용자 요청 보존율 BASE | 71/75 | 71/75 | OK |
| 사용자 요청 보존율 RL | 71/75 | 71/75 | OK |
| 비현실적 재료량 BASE | 5/120 | 5/120 | OK |
| 비현실적 재료량 RL | 4/120 | 4/120 | OK |

**12개 항목 전부 완전히 일치했다.** `baseline_reproduction.csv` 저장.

---

## 기존 보고서와 다른 수치

없음. 전부 정확히 일치했다.

## 차이가 발생한 원인

해당 없음(차이 없음). 참고로 `rl_comparison_FOOK.py`의 시나리오별 시드는
`seed_val = 820000 + i*977`(정수 연산만 사용, 문자열 `hash()` 미사용)라서 프로세스를 새로
띄워도 100% 결정론적으로 재현된다 — 이번 재실행도 그 특성 덕분에 완전히 일치했다.

## 수정된 파일 유무

**없음.** `rl_comparison_FOOK.py`, `app_core_FOOK.py`, `FOOK_adjust_levers.py`, 체크포인트,
`final_service_benchmark_120_realistic_weight.csv` 전부 그대로다. 이번 STEP 0에서 새로
만든 파일은 전부 `E:\final\rl_v2_work\` 아래에 있다:
- `rl_reward_audit.md`
- `baseline_reproduction.csv`
- `baseline_reproduction_report.md`(본 문서)

---

## 다음 단계 진행 판단

**재현 완료 — STEP 1(보상함수 정렬)로 진행 가능.** `rl_reward_audit.md`에서 확인한 핵심
정렬 필요 지점(칼륨/인 판정의 `<=` vs `<` 불일치, 현실성·재료겹침·과다군 검사가 reward에
전혀 반영 안 됨)을 STEP 1에서 다룬다.
