# 모델 개발 산출물 (Model Development)

`backend/`가 실제로 서비스에 로딩해서 쓰는 **완성된 결과물**(체크포인트, 레버 로직, RAG
지식베이스)이라면, 이 폴더는 그 결과물이 나오기까지 **AI 엔진을 설계·학습·검증한 과정**입니다.
배포된 서비스는 이 폴더의 코드를 실행 시점에 import하지 않습니다 — 개발 이력·재현·검증 근거를
남기기 위한 스냅샷입니다.

## 구성

### `generation-model_seq2seq-rl/`
식단 한 끼를 순서대로 구성하는 생성 모델. GRU Encoder–Decoder + Bahdanau Attention(Seq2Seq)으로
기본 모델을 학습(`train_FOOK.py`)한 뒤, 영양 기준 위반이 적은 조합을 선호하도록 REINFORCE
정책 경사로 파인튜닝했습니다(`train_rl_FOOK.py`). `backend/Diet-Generation-As-Sequence-master/`에
있는 체크포인트가 이 학습의 결과물입니다.

### `ingredient-substitution_klue-bert-knn/`
메뉴 재료를 저칼륨·저인 대체재로 바꿀 때 쓰는 임베딩 기반 유사도 모델. KLUE-BERT로 재료 문맥
임베딩을 만들고(`foodbert/`), KNN으로 가장 가까운 대체재를 찾습니다(`foodbert_embeddings/`).
`apply_potassium_filter.py` / `apply_phosphorus_filter.py`는 임상영양사 피드백을 반영해 등급
기준(1회 섭취 기준량당 칼륨 저<100mg·중100~200mg·고>200mg 등)을 실제 대체 로직에 적용한
스크립트입니다.

### `benchmarks/`
서비스 성능·회귀 여부를 수치로 검증한 평가 스크립트 모음입니다. 전부 "서비스 코드/체크포인트/
레버 코드는 수정하지 않고 평가만 수행"하는 원칙으로 작성했습니다.

| 스크립트 | 검증 내용 |
|---|---|
| `measure_lever_adjustment_FOOK.py`, `FOOK_lever_adjustment_report.txt` | 레버가 한 끼당 원본을 얼마나 바꾸는지 정량화 — RL 도입 타당성 판단 근거 |
| `reward_lever_FOOK.py`, `bench_lever_reward_FOOK.py` | RL 보상 함수(영양 달성도 + 원본 보존율) 설계·검증 |
| `pipeline_measure_FOOK.py` | 생성모델→레버 전체 파이프라인 통과율 측정 |
| `rl_comparison_FOOK.py`, `track1_precheck_FOOK.py`, `track1_rl_final_eval_FOOK.py`, `track1_rl_final_analysis_FOOK.py` | BASE(RL 미적용) vs RL 체크포인트 공정 비교 — 동일 시드 페어 실행으로 모델 차이만 분리 |
| `track1_vs_track2_simple_eval_FOOK.py` | RL 체크포인트 후보 간(실험용 마스킹 버전 포함) 비교 |
| `final_service_core_eval_FOOK.py` | 실제 production 경로(`app_core_FOOK.make_meal()`) 직접 호출로 핵심 성공률 측정 |
| `final_service_benchmark_120_realistic_weight_FOOK.py`, `external_condition_test_FOOK.py` | 다양한 체중·메뉴·재료 조건에서의 일반화 성능(현실적 표준체중 기준으로 재검증한 버전) |

### `clinical-review/`
임상영양사 검수용으로 정리한 문서 3종 — 영양 기준 산출 방식, 재료 대체 로직, 샘플 식단
결과물을 검토받기 위해 작성했습니다.

## 참고

- 이 폴더의 스크립트는 원래 `E:\final`(작업 디렉터리) 최상단에서 `backend/`의 데이터·모델
  파일들과 같은 위치에 두고 실행하도록 작성되었습니다. 그대로 다시 실행하려면 이 폴더의 내용을
  `backend/`와 같은 위치에 두거나 경로를 맞춰야 합니다 — 여기 있는 것은 실행용이 아니라
  **검증 근거 스냅샷**입니다.
- RAG 챗봇 지식베이스 구축 스크립트(`FOOK_build_rag_kb.py`)는 `backend/`에 있습니다(배포
  서버가 참조하는 `data/FOOK_rag_kb.json`을 만드는 스크립트라 실행 파일 쪽에 뒀습니다).
