# KOOK 프론트엔드

React + TypeScript + Vite. 백엔드(FastAPI, `../backend`)의 `/generate` 등 API를 호출해
실제 AI 식단 생성·영양 조정 결과를 보여줍니다.

## 실행

```bash
cp .env.example .env   # VITE_API_URL — 기본값은 로컬 백엔드(127.0.0.1:8000)
npm install
npm run dev
```

`http://localhost:5173` 접속. 개발 모드는 기본적으로 로컬 백엔드를 사용합니다.

## 빌드

```bash
npm run build   # tsc -b && vite build
```

## 앱 흐름

온보딩 → 로그인/체험 → 음식·재료 검색 → 식단 생성 → 영양 적합성 판정 →
레시피 재구성(레버 조정) → 최종 식단 확인 → 레시피 상세(음성 안내) / PDF 저장

식단 생성·영양 판정·재조정은 백엔드의 실제 Seq2Seq+RL 생성 모델과 영양소 조정 레버
결과를 그대로 씁니다(`src/App.tsx`의 `apiFetch`). 백엔드가 연결되지 않았을 때만
`src/fookData.ts`의 내장 데이터로 조용히 폴백합니다.
