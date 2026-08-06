# 생성 모델 리서치 로그

`generate_FOOK.py` / `train_FOOK.py` / `train_rl_FOOK.py`의 프로덕션 파이프라인을 두고 진행한
두 갈래의 심화 진단·실험 기록. 원본 조사는 이보다 많은 1회성 디버그 스크립트를 포함하지만,
여기에는 결론까지 도달한 두 건만 스크립트+요약 리포트로 정리해 남긴다.

## [phosphorus-protein-resolution/](phosphorus-protein-resolution/ADOPTION_REPORT.md) — 채택됨 (2026-07-27)

두부·콩류 앵커에서 `lever_phosphorus`(인 레버)가 통과시킨 메뉴를 뒤이은 `lever_protein`이
무관하게 스케일업해 인 기준을 재초과시키는 "phosphorus↔protein 핑퐁"을 13단계로 진단하고,
두부·콩류 앵커에 한정된 조건부 raw P 통일 + 90% cap 로직으로 해결. 비대상 앵커(생선구이·육류)
3,600건 무회귀 검증 완료 후 `FOOK_adjust_levers.py`에 **실제로 반영됨**.

## [soup-diversity-masking/](soup-diversity-masking/FINDINGS.md) — 검증됨, 미배포

같은 앵커·seed 조건에서 국(soup) 슬롯이 소수 메뉴로 쏠리는 문제를 디코더 구조(seed 위치 토큰을
그대로 베끼는 teacher-forcing shortcut) 수준까지 원인 규명하고, 국 슬롯 100% 마스킹 학습
(C_mask100) + RL 재학습으로 국 고유종류를 4.1개 → 28.6개(7배)로 늘리면서 영양·게이트 지표도
개선. "실서비스 적용 가능" 판정까지 났지만, 현재 `app_core_FOOK.py`는 여전히 이전 체크포인트
(`results_sweep_FOOK/i002`)를 쓰고 있어 **아직 배포되지 않았다**.
