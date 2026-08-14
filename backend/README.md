# KOOK 백엔드 API

투석 환자를 위한 한 끼 식단을 생성·조정하는 API 서버입니다.
프론트엔드(React)는 별도로 Vercel에 배포되어 있고, 이 서버(Render)는 API만 제공합니다.

## 주요 엔드포인트

| 경로 | 설명 |
|---|---|
| `GET /health` | 서버·DB 상태 확인 |
| `GET /menus`, `/ingredients` | 메뉴·재료 목록 |
| `POST /generate` | 한 끼 식단 생성 (AI 엔진 + 영양 레버) |
| `POST /generate_day` | 하루 세 끼 연속 생성 |
| `POST /recipe` | 조리법 LLM 편집 |
| `POST /tts` | 조리법 음성 변환 |
| `POST /chat` | 투석·콩팥병 영양 RAG 챗봇 (근거 자료 기반 질의응답 + 멀티턴 음식 후속 질문) |
| `POST /auth/signup`, `/auth/login` | 회원 기능 (Neon Postgres) |
| `POST /auth/find-id`, `/auth/reset-password` | 아이디 찾기·비밀번호 재설정 — 이름+생년월일로 본인 확인(이메일/SMS 발송 없음, 포트폴리오 데모 범위의 확인 수단) |

전체 스펙은 서버 실행 후 `/docs` 에서 확인할 수 있습니다.

## 구성

- **식단 생성**: Seq2Seq(GRU+Attention) 생성 모델 + REINFORCE 강화학습 보정 + 칼륨·인·나트륨 조정 레버
- **재료 대체**: KLUE-BERT 임베딩 + KNN 기반 대체재 추천
- **영양 질의응답**: RAG 챗봇(`FOOK_rag_chatbot.py`) — 대한신장학회 등 자료를 임베딩한
  `data/FOOK_rag_kb.json`에서 관련 근거를 찾아 답변. 특정 재료의 칼륨/인 질문은 RAG 대신
  영양DB(`FOOK_adjust_levers.py`)를 직접 조회한다 — 수치(1회 섭취 기준량 환산·저/중/고칼륨
  등급)는 항상 코드가 확정하고, 여기에 더해 체중+오늘 섭취량(`weight`+`consumed`)이 함께
  오면 "지금 더 먹어도 되는지"까지 코드가 최종 판정해 LLM이 절대 뒤집지 못하게 못박는다.
  나트륨은 첨가염 예산과 범주가 달라 K/P처럼 하드 게이트로 넣지 않지만, 재료 자체의
  1회 섭취량 나트륨이 끼니 나트륨 경고 기준(`NA_TOTAL_WARN`)을 넘으면 K/P가 통과여도
  최종 판정을 "피하는 게 좋음"으로 코드가 강제 하향한다(단순 정보성 안내가 아니라
  판정에 실제로 반영되는 안전장치).
  라우팅은 `find_food()`(특정 재료) → 결정론적 scope gate(도메인 밖 질문은 임베딩/LLM 호출 없이
  즉시 차단) → RAG(Top-10 후보 → 유사도 게이트 → 어휘 재정렬 → Top-5) 순서. `ChatReq.context_food`
  (선택, 하위호환)로 직전 답변의 재료명 하나만 클라이언트가 되돌려 보내면 "그럼 얼마나?" 같은
  한 턴짜리 후속 질문도 처리한다 — 서버는 상태를 갖지 않고 매 요청마다 검증만 한다.
  지식베이스를 다시 만들려면 `FOOK_build_rag_kb.py` 참고 — 원본 출처 문서 4종은
  `data/rag-source/`에 그대로 포함돼 있다.
- **영양 기준**: 대한신장학회 투석환자 영양관리 지침 기반
- **DB**: Neon Postgres (회원·저장 식단·즐겨찾기)

## 환경변수 (Render 대시보드 → Environment, 또는 `render.yaml` Blueprint)

| 이름 | 필수 | 설명 |
|---|---|---|
| `DATABASE_URL` | 선택 | Neon Postgres 접속 문자열. 없으면 회원 관련 API(`/auth/*`, `/me/*`)만 500을 반환하고, `/generate` 등 AI 생성 기능은 정상 동작한다 |
| `OPENAI_API_KEY` | 선택 | `/recipe`, `/tts`, `/chat` 에서만 사용 |
| `CORS_ORIGINS` | 선택 | 허용할 프론트엔드 주소(쉼표 구분). 기본값은 실제 배포 주소(`kook-hemodialysis-meal-ai.vercel.app`) + 로컬 개발 주소(`localhost:5173`) |
| `CORS_ORIGIN_REGEX` | 선택 | 미리보기 배포 등 추가 도메인을 정규식으로 임시 허용할 때만 사용. 기본값 없음(비워두면 `CORS_ORIGINS`만 적용) |

> 메모리 사용량은 TensorFlow 모델 상주 기준 약 410MB입니다(루트 README [실측 성능] 참고).
