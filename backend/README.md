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
| `POST /chat` | 투석·콩팥병 영양 RAG 챗봇 (근거 자료 기반 질의응답) |
| `POST /auth/signup`, `/auth/login` | 회원 기능 (Neon Postgres) |

전체 스펙은 서버 실행 후 `/docs` 에서 확인할 수 있습니다.

## 구성

- **식단 생성**: Seq2Seq(GRU+Attention) 생성 모델 + REINFORCE 강화학습 보정 + 칼륨·인·나트륨 조정 레버
- **재료 대체**: KLUE-BERT 임베딩 + KNN 기반 대체재 추천
- **영양 질의응답**: RAG 챗봇(`FOOK_rag_chatbot.py`) — 대한신장학회 등 자료를 임베딩한
  `data/FOOK_rag_kb.json`에서 관련 근거를 찾아 답변. 특정 재료의 칼륨/인 질문은 RAG 대신
  영양DB(`FOOK_adjust_levers.py`)를 직접 조회해 수치 판정은 코드가, 설명 문장만 LLM이 담당한다.
  지식베이스를 다시 만들려면 `FOOK_build_rag_kb.py` 참고(원본 텍스트는 저장소에 포함하지 않음).
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
