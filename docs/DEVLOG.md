# KOOK 통합본 — AI 생성 엔진 + Neon DB 회원 기능

투석 환자용 맞춤형 한 끼 식단 생성 서비스. 두 팀원 산출물을 하나로 병합한 버전입니다.

---

## v20 업데이트 — App.tsx 구조 정리 3차-c2: `pages/meal/` 2부 (생성 workflow 전체)

3차-c의 두 번째 묶음. `Home → Generating → MealResult → Analysis → Adjusting →
Comparison → FinalMeal`로 이어지는 핵심 생성 흐름 6개 화면을 한 번에 옮기고,
반드시 한 흐름으로 이어서 브라우저 검증했다(사용자가 명시적으로 요청한 방식 —
이 흐름은 화면 간 상태 의존이 커서 따로따로 검증하면 의미가 없다).

### 새로 생긴 파일
- `pages/meal/GeneratingPage.tsx`, `MealResultPage.tsx`, `AnalysisPage.tsx`,
  `AdjustingPage.tsx`, `ComparisonPage.tsx`, `FinalMealPage.tsx`
- `utils/auth.ts` — `requireUser`(로그인 안 돼 있으면 `/login`으로 보내는
  가드). FinalMealPage와, 아직 App.tsx에 남아있는 Recipe가 둘 다 써서
  공용 유틸로 뺐다.

### 정리
- App.tsx에 남아있던 orphan 주석 2곳을 제자리로 옮겼다: 음성 안내 관련
  주석은 `hooks/useSpeech.ts`로, `/recipe` steps 형식 관련 주석은
  `components/meal/RecipeBody.tsx`로(3a/c1 단계에서 함수만 옮기고 주석은
  놓쳤던 것을 이번에 발견해 바로잡음).

### 검증
- ✅ `npx tsc -b --force`(캐시 무시) 통과, `npm run build` 통과
- ✅ 백엔드 pytest 37개 통과(무변경)
- ✅ 로컬 백엔드+프론트로 전체 흐름을 하나로 이어서 실제 클릭 검증: 홈에서
  메뉴 선택 → 생성 중 화면 → 생성된 식단(가자미구이 포함, 실제 서버 응답) →
  영양 판정(단백질·인 초과 → "레시피 재구성하러 가기") → 재구성 중 화면
  (실제 변경 내역 "배추김치 → 저염오이김치", "양 조절 6건") → 재구성 완료
  (음식별 before/after 재료량) → 최종 식단(모든 영양소 "적절"로 개선,
  611kcal·단백질 24g 등) → 레시피 인라인 펼치기(실제 재료·영양성분 표시,
  OPENAI_API_KEY 미설정으로 조리과정만 예상대로 오류) 확인

App.tsx: 1,121줄 → 412줄

이제 App.tsx엔 `Recipe`, `PdfPreview`만 남았다(3c3). 목표(500~1,000줄)보다
이미 낮아졌지만, 남은 두 화면도 옮기고 나면 `App()` 루트 컴포넌트와
`ErrorBoundary`만 남는 자연스러운 결과라 그대로 진행한다.

---

## v19 업데이트 — App.tsx 구조 정리 3차-c1: `pages/meal/` 1부 (Home, DayPlan)

3차(가장 위험도 높은 단계)를 통째로 옮기지 않고, 사용자 제안대로 더 잘게
나눴다: c1(독립적인 화면) → c2(생성 workflow, 한 흐름으로 검증) →
c3(recipe/PDF). 이번은 c1.

### 새로 생긴 파일
- `pages/meal/HomePage.tsx` — 홈(음식/재료 검색, 랜덤 추천)
- `pages/meal/DayPlanPage.tsx` — 하루 식단
- `components/meal/MealListRow.tsx`, `MealSlotDialog.tsx`, `RecipeBody.tsx`
  (`toSteps` 헬퍼 포함), `RecipeList.tsx` — Home/DayPlan 자체보다는
  FinalMeal · Recipe · DayPlan이 공통으로 쓰는 조각이라 먼저 공용
  컴포넌트로 뺐다 (c2/c3에서 그 페이지들을 옮길 때 재사용)
- `hooks/useSpeech.ts` — 음성 안내 훅. RecipeBody 안에서만 쓰여서 함께 이동

### 검증
- ✅ `npx tsc -b --force`(캐시 무시) 통과, `npm run build` 통과
- ✅ 백엔드 pytest 37개 통과(무변경)
- ✅ 로컬 백엔드+프론트 실제 클릭: 홈(음식 검색 → 메뉴 선택 → 실제 `/generate`
  호출로 식단 생성까지, 아직 App.tsx에 남아있는 Generating/MealResult와의
  연결도 확인) → 하루 식단(`/generate_day` 호출, 아침/점심/저녁 3끼 +
  하루 총 영양 `Nutrients` 정상 표시)
- ⚠️ 검증 중 `/generate_day`가 한 번 TF GRU 레이어 오류(`GRUCell` 객체에
  `kernel` 속성 없음)로 실패하는 걸 발견했다. 재시도하니 정상 동작해서
  로컬 TF 모델 첫 추론 시의 일시적 레이스로 보이며, 프론트 코드(로딩→에러
  처리)는 그대로 옮긴 것이라 이번 리팩터링과는 무관 — 백엔드 쪽 이슈라
  별도로 살펴봐야 한다.

App.tsx: 1,906줄 → 1,121줄

다음은 c2(`Generating` → `MealResult` → `Analysis` → `Adjusting` →
`Comparison` → `FinalMeal`, 한 흐름으로 같이 검증), 그다음 c3(`Recipe`,
`PdfPreview`, TTS 클릭 확인까지).

---

## v18 업데이트 — App.tsx 구조 정리 3차-b: `pages/account/` 분리

3단계 리팩터링 계획 중 3단계, 두 번째 묶음. 마이페이지 쪽 화면을 뺐다.

### 새로 생긴 파일
- `pages/account/AccountPage.tsx` — 내 정보(프로필 요약, 메뉴 목록, 로그아웃)
- `pages/account/LibraryPage.tsx` — 식단 기록/식단 관리/PDF 보관함 공용 목록 화면.
  카드 한 장을 그리는 `SavedCard`는 이 페이지 안에서만 쓰여서 같은 파일에 둠
  (별도 파일로 안 쪼갬 — 2차 때 정한 "너무 잘게 쪼개지 않는다" 원칙)
- `pages/account/TipsPage.tsx` — 칼륨 낮추는 조리 팁

### 검증
- ✅ `npx tsc -b --force`(캐시 무시) 통과, `npm run build` 통과
- ✅ 백엔드 pytest 37개 통과(무변경)
- ✅ 로컬 백엔드+프론트 실제 클릭: (DB 미연결 환경이라 `localStorage`에 가짜
  로그인 세션을 넣어 검증) 내 정보 화면(BottomNav "프로필" 탭 하이라이트,
  프로필 요약·수정 버튼) → 식단 관리(빈 상태 → 저장 항목 1개 넣고 아침/점심/저녁
  구간별 카드 표시 → 삭제 버튼으로 제거까지) → 식단 기록/PDF 보관함(각각 다른
  빈 상태 문구·아이콘) → 칼륨 낮추는 조리 팁(백엔드에서 실제 데이터 정상 로드)
  확인

App.tsx: 2,208줄 → 1,906줄

다음은 3c(`pages/meal/` + `pages/recipe/`) — 마지막이자 가장 위험도 높은 단계.
`Home → Generating → Analysis → Adjusting → Comparison → Final` 흐름을 한 번에
같이 검증해야 한다.

---

## v17 업데이트 — App.tsx 구조 정리 3차-a: `pages/auth/` + `pages/onboarding/` 분리

3단계 리팩터링 계획 중 3단계, 그중 가장 독립적인 auth/onboarding 화면부터 뺐다.

### 새로 생긴 파일
- `pages/onboarding/OnboardingPage.tsx` — 스플래시(`Splash`) + 온보딩 5단계
  (`slides`, `PREVIEW_MENUS`, `ONBOARDING_IMAGE`, `OnboardingVisual`)
- `pages/auth/LoginPage.tsx` — 로그인 + 체험(게스트) 진입
- `pages/auth/FindIdPage.tsx`, `pages/auth/FindPasswordPage.tsx` — 아이디/비밀번호 찾기
- `pages/auth/SignupPage.tsx` — 회원가입 1단계(아이디)
- `pages/auth/ProfileSetupPage.tsx` — 회원가입 2단계(프로필 입력)

각 페이지는 자신이 쓰는 `services/api`, `hooks/useApp`, `components/*`, `utils/*`를
직접 import한다. `App.tsx`는 라우트 등록만 담당.

### 죽은 코드 정리
- `Trust`, `OnboardingShot` — 사용처 0곳 확인 후 이동 없이 삭제.

### 검증
- ✅ `npx tsc -b --force`(캐시 무시) 통과, `npm run build` 통과
- ✅ 백엔드 pytest 37개 통과(무변경)
- ✅ 로컬 백엔드+프론트 실제 클릭: 스플래시 → 온보딩 1~5단계 → 로그인 화면 →
  아이디 찾기 → 비밀번호 찾기 → 회원가입(1단계 제출까지, DB 미연결 환경이라
  서버 오류 메시지까지 정상 표시되는 것 확인) → 체험하기(가상 프로필 모달 →
  체험 진행 → 예시 식단 결과 화면 정상 진입) → 프로필 입력 화면(2/2단계) 확인

App.tsx: 3,185줄 → 2,208줄

다음은 3b(`pages/account/`), 그다음 3c(`pages/meal/` + `pages/recipe/`, 상태 의존이
가장 큰 마지막 단계).

---

## v16 업데이트 — App.tsx 구조 정리 2차: 공용 컴포넌트/유틸 추출

3단계 리팩터링 계획 중 2단계. 여러 화면이 함께 쓰는, 상태·API 의존이 적은 조각부터 뺐다.

### 새로 생긴 파일
- `components/icons.tsx` — 아이콘 15종 (기존에 App.tsx 안에서 여기저기 쓰이던 것)
- `components/Logo.tsx` — 브랜드 로고
- `components/layout/` — `Shell`, `Header`, `BackHeader`, `BottomNav`, `StepHeader`,
  `FlowFooter`, `Button` (거의 모든 화면이 쓰는 레이아웃 뼈대)
- `components/meal/` — `Nutrients`, `NutrientIconRow` (영양소 카드·판정 뱃지)
- `utils/date.ts` — `ageFromBirthdate`, `todayISO`, `MEAL_TIMES`, `defaultMealTime`
- `utils/menu.ts` — `menuMap`, `fallbackPlan`, `roleShort`/`roleLong`, `parseLocalIngredient`
- `utils/nutrition.ts` — `nmeta`, `totalNutrition`, `adjustedNutrition`, `fmt`/`fmt2`,
  `targetOf`/`minTargetOf`, `statusOf`/`STATUS_CLASS`, `parseChange`

이번 단계에선 `App.tsx` 안에서 정의만 옮기고 사용하는 쪽(각 화면 함수)은 import로만
바꿨다 — 화면 자체(Login, Home, Analysis 등)는 아직 App.tsx 안에 있다(3단계에서 이동).

### 검증
- ✅ `npx tsc -b --force`(캐시 무시) 통과, `npm run build` 통과
- ✅ 백엔드 pytest 37개 통과(무변경)
- ✅ 로컬 백엔드+프론트 실제 클릭: 로그인 화면(Shell/Header) → 메뉴 지정 생성
  (BackHeader) → 영양 판정 화면(Nutrients, 실제 서버 값으로 적절/초과 판정 정상 표시,
  서버 미연결 시나리오에서는 예시 뱃지도 정상 표시) 확인

App.tsx: 3,753줄 → 3,185줄

**3차(페이지 컴포넌트를 `pages/`로 분리)는 의도적으로 이번 세션에 포함하지 않았다.**
이 지점을 안정 버전으로 태그해뒀다(`v16-appjs-refactor-stage2`). 3차를 시작할 때는:

1. 가장 독립적인 것부터: `pages/auth/`(LoginPage, FindIdPage, FindPasswordPage,
   SignupPage)
2. 그다음: `pages/account/`(AccountPage, LibraryPage, TipsPage)
3. 마지막(가장 위험도 높음 — 서로 상태 의존이 큼): `pages/meal/`(HomePage,
   GeneratingPage, AnalysisPage, AdjustingPage, ComparisonPage, FinalMealPage,
   DayPlanPage). `Home → Generating → Analysis → Adjusting → Comparison → Final`
   흐름은 한 번에 같이 검증해야 한다.

목표는 `App.tsx`를 억지로 100줄까지 줄이는 게 아니라 500~1,000줄 정도로 자연스럽게
줄이는 것 — 작은 컴포넌트까지 전부 파일로 쪼개면 오히려 찾아다니기 힘들어진다.

---

## v15 업데이트 — App.tsx 구조 정리 1차: API 서비스 함수화 + `?api=` 완전 제거

App.tsx(3,700여 줄)를 pages/components/services/hooks로 나누는 작업의 1단계.
가장 위험도가 낮고 효과가 큰 "API 계층 정리"부터 진행했다.

### 1) `services/api.ts`에 엔드포인트별 함수 추가
- `getMenus`, `getIngredients`, `getMenusByIngredient`, `generateMeal`,
  `generateDayPlan`, `generateRecipe`, `textToSpeech`, `getPotassiumTips`,
  `login`, `signup`, `findId`, `resetPassword`, `updateProfile`, `getMe`,
  `logout`, `warmupBackend` — App.tsx의 15개 `apiFetch()` 직접 호출을 전부
  이름 있는 함수 호출로 교체했다. 화면은 이제 `await generateMeal(body)`처럼
  "뭘 부르는지"만 알면 되고, URL·타임아웃·헤더는 services/api.ts 안에 있다.

### 2) `?api=` 백엔드 우회 기능 자체를 제거
- v13~v14에서 https-only → 도메인 allowlist → confirm() 동의까지 세 겹으로
  막았지만, "그래도 승인하면 결국 인증 토큰이 임의로 지정한 주소로 간다"는
  구조적 위험은 남아있었다. 포트폴리오 서비스에 굳이 필요한 기능이 아니라고
  판단해 기능 자체를 없앴다 — 이제 `API`는 그냥 `VITE_API_URL`(없으면 로컬
  기본값) 고정이다. 백엔드 주소를 바꾸려면 환경변수를 바꾸고 재배포한다.

### 검증
- ✅ `npm run build`(tsc+vite) 통과, 백엔드 pytest 37개 통과(무변경)
- ✅ 로컬 백엔드+프론트로 실제 클릭: 메뉴 검색 → 메뉴 지정 생성(`generateMeal`)
  → 영양 판정까지 실제 서버 응답(예시 데이터 아님, 초과/적절 실측치)으로 확인

---

## v14 업데이트 — v13 리뷰 후속: `?api=` 도메인 allowlist, fetch 전량 일원화

v13 반영 후 같은 리뷰어가 다시 대조 검증한 결과, "부분 수정"으로 남아있던 2건.

### 1) `?api=` 백엔드 우회 — https 검증만으론 부족하다는 지적
- v13에서 https-only + confirm() 동의로 막았지만, "동의만 누르면 결국 임의 주소로 토큰이
  간다"는 구조적 위험은 남아있다는 지적. `isValidApiOverride()`에 도메인 allowlist를
  추가해서 `*.onrender.com`(실제 배포처)·`*.trycloudflare.com`(비상용 예시로 문서에
  적어둔 제공자)이 아닌 호스트는 https여도 confirm() 단계까지 가지 않고 거른다.
  임의의 공격자 도메인은 이제 애초에 후보가 안 된다.

### 2) `App.tsx`가 여전히 `services/api.ts`를 거치지 않고 직접 fetch하던 8곳
- `services/api.ts` 상단 주석은 "화면 컴포넌트들은 이 모듈만 통해 서버와 이야기한다"고
  적혀 있었는데, 실제로는 `/health`, `/menus`, `/ingredients`, `/menus_by_ingredient`,
  `/generate`, `/recipe`, `/generate_day`, `/tts`, `/veg_potassium_tips` — 8개 호출이
  `apiFetch()`를 거치지 않고 직접 `fetch()`를 쓰고 있어서 주석과 실제 코드가 안 맞았다.
- `apiFetch()`에 `timeoutMs`(엔드포인트별로 다른 타임아웃, 예: `/generate_day`는
  90초)와 `responseType: 'blob'`(`/tts`는 오디오 파일이라 JSON 파싱이 안 됨) 옵션을
  추가해서, 8곳 전부를 `apiFetch()` 하나로 통일했다. 이제 인증 헤더·타임아웃·에러 메시지
  파싱이 모든 API 호출에서 동일하게 동작한다.

### 검증
- ✅ 프론트 `npm run build`(tsc+vite) 통과
- ✅ 백엔드 pytest 37개 통과(무변경)
- ✅ 로컬 백엔드+프론트를 띄우고 실제 클릭으로 8개 엔드포인트 전부 확인: 메뉴/재료 검색,
  재료로 메뉴 찾기, 한 끼 생성, 레시피 조회(OPENAI_API_KEY 없을 때 에러 처리까지),
  하루 세 끼 생성(`/generate_day`), 칼륨 낮추는 팁, `/tts`(키 없을 때 400) 응답 확인

---

## v13 업데이트 — 공개 배포 전 보안 점검 반영 (외부 리뷰 7건)

외부 코드 리뷰로 받은 7개 지적사항을 하나씩 실제 코드와 대조 검증한 뒤 반영했다.

### 1) 🔴 `?api=` 백엔드 우회 기능의 토큰 탈취 위험
- 로그인 토큰(`Authorization: Bearer ...`)이 `?api=`로 지정한 주소로 그대로 전송되는데,
  이 값을 URL에서 읽어 검증 없이 localStorage에 저장하고 있었다 — 조작된 링크를 열면
  세션이 공격자 서버로 샐 수 있는 실제 취약점이었다.
- 기능 자체(무료 호스팅 다운 시 비상 백엔드 전환)는 살리되, `https://`만 허용하고
  실제로 전환될 때만 `confirm()`으로 대상 주소를 보여주고 동의를 받아야 저장되도록 했다.
  브라우저가 대화상자를 자동으로 취소해도(자동화 테스트로 확인) 안전한 값으로 남는다.

### 2) 🔴 아이디/비밀번호 찾기의 간소화된 인증
- `이름+생년월일`만으로 본인확인(이메일/SMS 인증 없음)하는 방식은 실제 서비스 기준으로는
  약하다는 지적 — 비밀번호 해싱(PBKDF2-SHA256 310,000회+salt)·세션 토큰 해시저장은
  이미 정상이었으므로, 코드를 바꾸는 대신 README에 "데모 환경의 간소화된 인증"임을
  명시했다.

### 3) 🟠 서버 미연결(오프라인) 시 예시 데이터가 실제 판정처럼 보이던 문제
- 서버 생성이 실패하면 내장 예시 데이터로 화면이 계속 진행되는데, 이후 화면들(영양 판정·
  완성 화면·PDF)이 여전히 '적절/초과' 뱃지를 보여줘서 마치 실제 개인 맞춤 계산처럼 보였다.
- `usingFallback` 상태를 추가해 화면 상단에 "⚠ 예시 데이터를 보고 있어요" 배너를 계속
  띄우고, 뱃지도 중립적인 '예시'로 바꿨다. **PDF는 다운로드되면 앱 화면과 분리된 파일이
  되므로**, PDF 문서 안에도 같은 안내를 별도로 넣었다(화면의 배너만으론 파일엔 안 남음).

### 4) 🟠 `App.tsx` 단일 파일(3,930줄) 정리
- `type ApiResult = any`였던 걸 실제 `/generate` 응답 구조를 반영한 인터페이스로 교체.
- API 통신·로컬↔서버 동기화 로직(`apiFetch`, `saveEverywhere` 등)을
  `src/services/api.ts`로, 공용 타입을 `src/types.ts`로, `AppContext`/`useApp`을
  `src/hooks/useApp.ts`로 분리했다. (화면 컴포넌트 자체의 전면 분할은 이번 범위에
  포함하지 않음 — 회귀 위험 대비 검증 범위를 넘어선다고 판단.)

### 5) 🟡 `standard_weight()` height/sex 검증 누락
- `height`만 주고 `sex`를 안 주거나 오타를 내면 조용히 남성(계수 22) 기준으로 계산되던
  문제. `GenReq`/`DayReq`에 `sex: Literal['남','여','male','female']` 제약과
  "height 있으면 sex 필수" model validator를 추가해 422로 막는다.

### 6) 🟡 문서/설정 불일치
- README의 clone 명령이 옛 저장소 이름(`final_KOOK`)이었던 것을 현재 이름
  (`kook-hemodialysis-meal-ai`)으로 수정. `frontend/package.json`의 이름을
  `fook-complete-prototype` → `kook-frontend`로 정리. 저장소 루트에 남아있던 원인 불명의
  0바이트 `git` 파일 삭제.

### 7) 🟡 자동화 테스트/CI 부재
- `backend/tests/test_adjust_levers.py`에 순수 함수(TensorFlow 모델 로딩 불필요) 37개
  테스트 추가 — `standard_weight`, `day_targets`, `meal_bounds`, `num`, `is_seasoning`,
  `unrealistic_reason`, `p_abs` 등.
- `.github/workflows/ci.yml` 추가: 백엔드는 `requirements-dev.txt`(pytest+openpyxl만,
  TensorFlow 제외)로 pytest만 돌리고, 프론트는 `npm ci && npm run build`로 타입체크+빌드만
  확인한다. 모델 로딩·실제 DB 연결은 CI 범위에 포함하지 않았다.

### 검증
- ✅ 백엔드: `py_compile` 통과, pytest 37개 전부 통과, `DATABASE_URL` 없이 서버 기동 후
  `/generate` 정상, `height`만 보내면 422, `height`+잘못된 `sex`도 422 확인
- ✅ 프론트: `npm run build`(tsc+vite) 통과, `npm audit` 0건
- ✅ 로컬에서 프론트+백엔드 함께 띄우고 브라우저로 실제 클릭 테스트: `?api=` 악성 링크로
  접속해도 confirm 거부 시(자동화 테스트 환경의 기본 동작) localStorage에 저장 안 됨을
  확인, 서버 생성 실패 상황이 실제로 발생해 예시 데이터 배너·PDF 안내·'예시' 뱃지가
  의도대로 전부 표시되는 것까지 우연히 실제 시나리오로 확인됨

---

## v12 업데이트 — DB 없이도 서버가 죽던 버그 수정 · 프론트 의존성 보안 패치

### 1) `DATABASE_URL` 미설정 시 서버 전체가 기동 실패하던 버그
- `server_FOOK.py`의 자체 주석은 "DATABASE_URL이 없으면 회원 관련 API(`/auth/*`,
  `/me/*`)만 500을 반환하고, `/generate` 등 AI 생성 기능은 DB 없이도 정상 동작한다"고
  설명하고 있었지만, 실제로는 그렇지 않았습니다.
- 원인: `database.py`가 모듈 **import 시점**에 `DATABASE_URL`이 비어 있으면
  `RuntimeError`를 던지도록 되어 있었고, `server_FOOK.py`는 파일 최상단에서
  `from database import db`를 실행합니다. 즉 DB 연결 정보가 없으면 AI 엔진(TensorFlow
  모델 로딩 포함)까지 포함한 **서버 프로세스 자체가 뜨지 못하고 죽었습니다.**
- 수정: `database.py`의 `engine` 생성을 지연시켜, `DATABASE_URL`이 없어도 모듈 import는
  성공하고 실제로 `db()`(DB 연결)를 호출하는 회원 관련 라우트에서만 에러가 나도록
  바꿨습니다. 코드가 처음부터 의도했던 동작(주석에 적힌 그대로)과 일치시켰습니다.
- 이 문제는 실제로 로컬에서 `DATABASE_URL` 없이 서버를 띄워보다가 발견했습니다 —
  정적 코드 리뷰만으로는 안 보이고, 실제 기동 시도에서만 드러나는 종류의 버그였습니다.

### 2) 프론트엔드 의존성 보안 취약점 6건 패치
- `npm audit`에서 jsPDF(치명적 등급 1건 포함: 경로 순회·PDF 삽입 공격 등)·
  dompurify·nanoid·postcss·react-router/react-router-dom에서 취약점 6건이 발견됐습니다.
- jsPDF는 3.0 → 4.2.1로 메이저 버전이 올라가는 변경이라, 이 프로젝트가 실제로 쓰는
  API(`new jsPDF()`, `addImage()`, `save()` — `PdfPreview` 컴포넌트)가 영향받는지
  확인 후 적용했습니다. 나머지는 비파괴적(non-breaking) 패치입니다.

### 검증
- ✅ 로컬에서 `DATABASE_URL` 없이 백엔드를 직접 띄워 `/health`·`/generate`·
  `/generate_day`·`/menus`·`/menus_by_ingredient`·`/recipe`·`/tts`·`/chat`이 정상
  동작(또는 문서화된 대로 안전하게 실패)하는지 확인
- ✅ `DATABASE_URL` 없이 `/auth/signup` 호출 시 서버가 죽지 않고 500만 반환하는지 확인
- ✅ 프론트엔드 `npm run build`(TypeScript 검사 + Vite 프로덕션 빌드) 성공
- ✅ 로컬에서 프론트+백엔드를 함께 띄우고 브라우저로 비회원 체험 흐름(한 끼 생성 →
  영양 판정 → 레시피 재구성)을 끝까지 클릭해 콘솔 에러 없음과 PDF 다운로드 정상
  동작(jsPDF 4.2 기준) 확인

---

## v11 업데이트 — 브랜드 KOOK 전환 · 노인 친화 UI · 음성 안내 · 계정 찾기

### 1) 로고를 KOOK으로 교체
- 화면에 보이는 브랜드 표기를 전부 **KOOK**으로 바꿨습니다(헤더 로고, 로그인 화면,
  온보딩, PDF 머리글, PDF 파일명, 브라우저 탭 제목/파비콘).
- 로고는 `frontend/src/App.tsx`의 `<Logo />` 컴포넌트 한 곳에서만 관리합니다.
  `public/assets/kook-logo.png`를 먼저 찾고, 없으면 **같은 디자인을 코드로 그린
  `kook-logo.svg`로 자동 대체**합니다.
- ⚠️ **원본 이미지 파일은 이 작업 환경에서 받을 수 없어서 넣지 못했습니다.**
  보내주신 KOOK 로고 이미지를 `frontend/public/assets/kook-logo.png`에 저장하면
  코드 수정 없이 그 파일이 바로 쓰입니다. 저장 전까지는 SVG 버전이 보입니다
  (뚝배기·초록 김·붓글씨 쿡·낙관·KOOK 워드마크 구성은 같지만, 붓 획의 질감은
  원본 이미지와 다릅니다).
- 내부 식별자(localStorage 키 `fook:*`, 파이썬 파일명 `*_FOOK.py`, API 경로)는
  건드리지 않았습니다. 바꾸면 기존 로그인 세션과 저장 데이터가 날아가기 때문입니다.

### 2) 앱 시작 흐름: 소개 → 로그인
- 앱에 들어오면 **항상** 소개 페이지 6장부터 보여주고, 그 다음 **로그인 화면**으로 갑니다.
  중간에 있던 "시작하기 선택" 화면(`/start`)은 없애고 `/login`으로 넘깁니다.
- 처음엔 "소개를 이미 본 사람은 건너뛰기"를 localStorage 플래그로 처리했는데, 그 기록이
  브라우저에 남아 있으면 **링크를 열자마자 검색 화면이 떠서 소개도 로그인도 안 보이는**
  문제가 있었습니다. 그래서 그 분기를 아예 없앴습니다 — 이제 이전 방문 기록이나 로그인
  기록이 남아 있어도 무조건 소개부터 시작합니다.
- 로그인 화면 구성(요청하신 순서 그대로):
  아이디 입력칸 → 비밀번호 입력칸 → **아이디 찾기 · 비밀번호 찾기 · 회원가입**
  3개 섹션 → 맨 아래 **한 끼 체험해보기**(회원가입 없이 바로 생성).
- 회원가입은 **1단계 아이디 → 2단계 프로필 입력** 2단계입니다. 예전엔 프로필이
  또 2쪽으로 쪼개져 총 3쪽이었는데, 투석 유형 선택을 프로필 쪽으로 합쳐 2단계로 줄였습니다.

### 3) 아이디 찾기 / 비밀번호 찾기 (신규, 백엔드 포함)
- 이 서비스는 이메일·문자 발송 수단이 없어서 "재설정 링크 보내기"를 쓸 수 없습니다.
  대신 **이름 + 생년월일**로 본인 확인합니다.
  - `POST /auth/find-id` — 이름·생년월일이 맞으면 아이디를 알려줍니다.
  - `POST /auth/reset-password` — 아이디·이름·생년월일이 모두 맞으면 새 비밀번호로
    바로 바꾸고, 기존 로그인 세션은 전부 해제합니다.
- 확인 근거가 필요해서 `user_profiles`에 **`birthdate` 컬럼을 추가**했습니다
  (`sql/002_account_recovery.sql`). 예전엔 가입 때 생년월일을 받아 나이만 계산하고
  원본은 버리고 있었습니다.
- 마이그레이션을 깜빡해도 서버가 죽지 않도록, 계정 관련 요청이 처음 들어올 때
  `ADD COLUMN IF NOT EXISTS`를 한 번 자동 실행합니다.
- ⚠️ **기존 가입자**는 프로필에 생년월일이 없으므로 찾기가 안 됩니다. 로그인 후
  내 정보 → 수정에서 생년월일을 한 번 저장하면 그때부터 사용할 수 있습니다.

### 4) 영양 분석 그래프 재작성 (도넛 + 게이지)
- 기존의 얇은 막대 하나를 **원형 도넛 게이지 + 굵은 3구간 막대** 조합으로 바꿨습니다.
  - 도넛: 상한 대비 몇 %인지를 링과 가운데 숫자로 표시.
  - 막대: 미만(파랑) / 적절(초록) / 초과(빨강) 구간과 현재 위치 핀.
  - **초과면 링·핀·판정 배지·카드 테두리까지 전부 빨강**, 미달이면 파랑으로 바뀝니다.
- 구간 이름(미만/적절/초과)이 막대의 실제 구간 폭에 맞춰 그 아래 놓이도록 해서,
  색 구간과 라벨이 어긋나지 않습니다.
- 하단에 "ⓘ 투석환자 여자 65kg 기준"처럼 판정 기준을 명시합니다.

### 5) 음성 안내 — 레시피 조리과정에만, 누르면 나옵니다
- **레시피 상세 화면의 "🔊 조리과정 음성으로 듣기" 버튼 한 곳에만** 있습니다.
  누르면 조리 순서를 번호까지 읽어주고, 다시 누르면 즉시 멈춥니다.
- 처음엔 모든 화면에 음성 버튼을 넣었는데, 화면마다 버튼이 자리를 차지하고 실제로
  필요한 건 조리과정뿐이라 나머지 10개 화면에서는 전부 뺐습니다.
- 재생 방식은 **브라우저 내장 음성합성(무료·즉시·오프라인)** 우선, 안 되는 환경에서만
  서버 `/tts`(OpenAI)로 넘어갑니다. 그래서 `OPENAI_API_KEY` 없이도 음성이 나옵니다.
  Windows/Chrome 기준 한국어 음성(Microsoft Heami)이 잡히는 것을 확인했습니다.

### 6) 이전 페이지 다시 보기
- 생성 흐름의 모든 단계 하단에 **`‹ 이전` · 단계 점 · `다음 ›`** 네비게이션을 넣었습니다
  (한 끼 구성 → 영양 분석 → 재구성 → 완성, 온보딩도 동일). 어느 단계에서든 앞 화면으로
  되돌아가 다시 볼 수 있습니다. 레시피 상세에도 "‹ 이전 화면" 버튼을 넣었습니다.

### 7) 재료 양은 소수점 2자리까지
- 재료·간식의 양(g)을 **항상 소수점 2자리**로 보여줍니다(레시피 상세, 장바구니, PDF,
  조정 내역). 0.5g 차이가 저나트륨·저칼륨 조리에서는 의미가 있는데, 기존엔 반올림해서
  조정 내역이 안 보이는 경우가 있었습니다.
- 백엔드도 맞춰서 바꿨습니다: `dish_ingredients`/`snacks`는 소수점 2자리로,
  변경 내역 문구도 `양 100→70g` → `양 100.00→70.00g`으로 내려줍니다.
- 서버 미연결 시 쓰는 내장 예시 데이터도 같은 형식으로 정규화해서 표시합니다.

### 8) 노인 친화 디자인 — "한 화면에 다 보이게" 기준으로 재조정
- 처음엔 크게 키웠다가(본문 17px, 제목 32px, 버튼 62px) **너무 커서 한 화면에 안 들어온다**는
  피드백을 받아 되돌렸습니다. 지금은 기존 대비 "조금 큰" 수준입니다:
  본문 16px, 제목 26px(모바일 23px), 버튼 54px, 입력칸 50px. 터치 목표는 48px 이상 유지.
- 여백을 줄여 각 단계가 한 화면에 들어오게 했고, 넘치는 부분만 아래로 스크롤됩니다.
  실측(375×812 기준): 소개·로그인·회원가입·홈·한 끼 구성·아이디 찾기는 **넘침 0px**,
  영양 분석은 **영양소 5개가 첫 화면 안에 다 들어오고**(670/675px) 보조 안내만 아래로 넘어갑니다.
- 영양 게이지도 압축했습니다. 줄마다 반복하던 "미만/적절/초과" 구간 이름을 목록 위
  범례 하나로 합치고, 기준값을 수치 옆에 붙여 한 줄로 줄였습니다.
- 화면이 좁아져도 가로 스크롤이 생기지 않도록 430px / 360px 구간을 따로 조정했습니다.

### 검증
- ✅ 백엔드 `server_FOOK.py`, `app_core_FOOK.py` 구문 검사 통과
- ✅ 프론트엔드 `npm run build`(TypeScript 검사 + Vite 프로덕션 빌드) 성공
- ✅ 개발 서버를 띄워 브라우저로 실제 렌더링 확인: 온보딩·로그인·회원가입·프로필·
  아이디 찾기·비밀번호 찾기·홈·한 끼 구성·영양 분석·재구성·완성·레시피 화면 모두
  가로 넘침 0px, 콘솔 에러 없음
- ✅ 소개 화면 우선 진입: 이전 방문 기록(`fook:onboarding-complete`)과 로그인 기록
  (`fook:user`)이 남아 있는 상태에서도 루트(`/`)가 소개 화면으로 가는 것을 실제로 확인
- ✅ 음성 안내: 레시피 화면에만 버튼이 있고(다른 화면 0개), 누르기 전 무음 → 누르면 재생 →
  다시 누르면 정지까지 실제로 확인
- ✅ 재료 양이 `37.50g` / `0.75g`처럼 2자리로 나오는 것 확인
- ⚠️ **새 백엔드 API(`/auth/find-id`, `/auth/reset-password`)는 실제 Neon DB에 붙여
  호출해보지 못했습니다.** 이 환경에 DB 접속 정보와 TensorFlow 런타임이 없어서입니다.
  서버를 띄운 뒤 `/docs`에서 두 엔드포인트를 한 번씩 호출해 확인해 주세요
  (프로필에 생년월일을 저장한 계정으로 테스트해야 합니다).
- ⚠️ 위와 같은 이유로 `/generate` → `/recipe` end-to-end 테스트도 이번에도 못 했습니다.

---

## 이번 통합에서 실제로 한 일

작업 전 두 파일을 각각 열어서 실제 코드를 확인한 결과, 두 프로젝트는 서로 다른 절반씩만
완성되어 있었습니다.

| 기능 | `FOOK_handoff` (AI 백엔드) | `FOOK-total-integrated` (기존 통합본) |
|---|---|---|
| 식단 생성(`/generate`) | ✅ KLUE-BERT 대체 + REINFORCE 강화학습 기반 실제 생성 엔진 | ❌ Neon DB에서 `ORDER BY random()`으로 한 행 뽑아오는 임시 스텁이었음 (`targets`도 하드코딩) |
| 조리법 편집(`/recipe`) | ✅ OpenAI 연동, 원본 레시피 기반 투석 팁 편집 | ❌ 고정 문자열 3줄 |
| 음성 변환(`/tts`) | ✅ 있음 | ❌ 없음 |
| 회원가입/로그인/세션 | ❌ 없음 | ✅ Neon PostgreSQL, PBKDF2 해시, 세션 토큰 |
| 식단기록/즐겨찾기/PDF보관함/장바구니 | ❌ 없음 | ✅ DB 저장 API 있음 (단, 통합 전 프론트는 일부만 연결해 localStorage로 대체 사용 중이었음) |
| 프론트엔드 UX | 초기 프로토타입 수준 | ✅ 온보딩→검색→생성→분석→조정→비교→최종→레시피→PDF까지 완성된 화면 |

**즉 진짜 AI 엔진은 handoff 쪽에만, 진짜 DB/회원 기능은 integrated 쪽에만 있었습니다.**
다행히 두 백엔드가 프론트에 내려주는 API 스키마(`GenReq`, `intake`/`consumed` 키 등)가
동일한 명세서(`BACKEND_HANDOFF.md`)를 기준으로 만들어져 있어 자연스럽게 병합됐습니다.

### 통합 방식

- `backend/`는 integrated의 `server_FOOK.py`(DB·인증·회원 API)를 베이스로 하되,
  `/generate`, `/generate_day`, `/recipe`, `/tts`, `/menus`, `/ingredients`,
  `/veg_potassium_tips`는 handoff의 실제 AI 엔진(`app_core_FOOK.py` 등)을 호출하도록
  전면 교체했습니다. DB 기반 임시 생성 코드는 삭제했습니다.
- handoff의 모델 체크포인트, 데이터셋(CSV/XLSX/JSON), 서브모듈 2개
  (`Diet-Generation-As-Sequence-master`, `Exploiting-Food-Embeddings-for-Ingredient-Substitution-master`)를
  그대로 이식했습니다.
- `frontend/`는 integrated의 완성된 UI를 그대로 유지합니다. 호출하는 엔드포인트
  (`/generate`, `/health`, `/menus`, `/ingredients`, `/auth/*`, `/me/profile`)가
  새 백엔드에 모두 구현되어 있음을 확인했습니다.
- `requirements.txt`는 두 세트(FastAPI+DB / TensorFlow+AI 엔진)를 합쳤습니다.
- **주의**: 통합 과정에서 기존 `backend/.env`(실제 Neon 접속 문자열이 담긴 파일)가 딸려
  들어온 것을 발견해서 삭제했습니다. `.env.example`만 남겼으니 실행 전 각자 값을 채워야 합니다.
- 실행되지 않는 코드와 실제 스키마가 불일치하던 SQL 초안(`002_user_features.sql`)은
  `backend/sql/deprecated/`로 옮기고 이유를 적었습니다. 실제로 쓰이는 스키마는
  `backend/sql/001_full_schema.sql` 하나입니다.

### 검증한 것 / 검증 못한 것

- ✅ 모든 Python 파일 구문 검사 통과
- ✅ `app_core_FOOK.py` / `FOOK_adjust_levers.py` / `recipe_editor_FOOK.py`가 참조하는
  모든 상대경로(모델 체크포인트, 데이터 파일)가 실제로 존재함을 확인
- ✅ 프론트엔드 `npm run build`(TypeScript 검사 + Vite 프로덕션 빌드) 성공
- ✅ 프론트가 호출하는 모든 API 경로가 새 백엔드에 구현되어 있음을 코드 대조로 확인
- ⚠️ **TensorFlow 모델을 실제로 로딩해서 `/generate`를 호출하는 end-to-end 테스트는
  이 작업 환경에서 수행하지 못했습니다** (TF 설치·모델 로딩에 시간이 오래 걸리고,
  이 환경은 PyPI 외 네트워크가 막혀 있습니다). 최초 실행 시 아래 "실행" 항목의
  Swagger UI(`/docs`)에서 `/generate`를 한 번 직접 호출해 확인해 보시길 권합니다.
- ⚠️ Neon DB 연결 자체(`DATABASE_URL`)도 각자의 실제 접속 정보로만 검증 가능하므로,
  `/health` 응답의 `db.ok` 값으로 확인해 주세요.

---

## v10.4 업데이트 — 실제 서비스 목업(스텝 인디케이터·영양 게이지) 기준으로 UI 전면 개편

디자인 목업(온보딩 6장 + 실제 앱 사용 흐름 5단계 화면)을 받아, 그 느낌을 실제 데이터
기반으로 재구성했습니다. 실사 음식 사진 대신 아이콘·카드형 UI로 통일하고, 전체 흐름에
공통 컴포넌트를 적용해 일관성을 맞췄습니다.

### 신규 공용 컴포넌트
- **`StepHeader`**: 원형 단계 인디케이터(①—②—③—④—⑤ + n/N 배지). 온보딩(1~5단계)과
  실제 생성 흐름(홈=1·결과=2·분석=3·조정=4·완성=5) 양쪽에 동일하게 적용해서, 지금
  몇 단계인지 항상 명확히 보이도록 했습니다.
- **`Nutrients`(영양 게이지)**: 기존의 단순 텍스트 목록을, 목업처럼 **"미만/적절/초과"
  3구간 막대 게이지 + 현재 위치 표시 핀 + 판정 배지** 형태로 전면 재작성했습니다. 서버가
  계산한 실제 하한·상한(`targetOf`/`minTargetOf`)을 그대로 반영합니다. 분석 화면의
  "조정 필요" 판정, PDF의 판정 테이블도 모두 이 기준(상한 초과뿐 아니라 하한 미달도 포함)
  으로 통일했습니다.
- **`MealListRow`**: 요청하신 대로 "한 음식당 한 줄"짜리 가로 리스트 카드(역할 태그 +
  음식명 + 화살표). 한 끼 결과, 최종 식단, 하루 식단 화면에 공통으로 사용합니다.

### 실사 음식 사진 전면 제거
- 이전엔 "밥/국/반찬" 역할별로 고정된 사진 3장을 어떤 메뉴가 나오든 그대로 반복
  재사용하고 있어서, 실제로는 의미 없는 장식이었습니다. 이번에 관련 이미지 파일
  (`soup.jpg`, `person.jpg`, `meal.jpg`, `meal-final.jpg`, `uxui-reference.png`)을
  코드와 `public/assets`에서 전부 제거했습니다. 로딩 화면과 레시피 상세 히어로는
  이모지 아이콘으로 대체했습니다. 남은 이미지는 브랜드 로고(`fook-logo.png`) 하나뿐입니다.
- 온보딩(`OnboardingVisual`)도 사진 대신, 실제 화면에서 쓰는 컴포넌트(검색 카드,
  `MealListRow`, 영양 게이지, 조정 카드)를 그대로 미리보기로 보여주도록 다시 만들었습니다.

### 검증
- ✅ 백엔드 구문 검사, 프론트엔드 `npm run build` 모두 통과
- ⚠️ 실제 화면 배치·간격은 이 환경에서 시각적으로 렌더링해 눈으로 확인하지 못했습니다.
  브라우저로 직접 열어서 각 단계별 화면(특히 영양 게이지의 3구간 막대와 핀 위치)이
  의도대로 보이는지 확인해 주세요.

---


### 회원가입/로그인: 이메일 → 아이디로 전환
- 기존엔 프론트(`email.includes("@")`)와 백엔드(`EmailStr`) 둘 다 이메일 형식을 강제하고 있어서,
  이메일이 아닌 값을 넣으면 안내 없이 그냥 안 넘어가는 것처럼 보였습니다.
- 이제 **아이디(영문/숫자/`._-` 4~30자)**로 가입·로그인합니다. DB 컬럼명은 기존 스키마
  그대로 `email`을 재사용하되(마이그레이션 없이 적용하기 위해), 형식 검증만 뺐습니다 —
  실제로는 "아이디"로 취급됩니다.
- **회원가입 1단계(계정)**: 이름 · 아이디 · 비밀번호 · 비밀번호 확인 · 약관 동의.
  조건을 하나씩 실시간으로 검사해서 무엇이 부족한지 바로 보여줍니다(이전엔 "필수 정보를
  입력하세요"라는 뭉뚱그린 문구만 떴습니다).
- **회원가입 2단계(프로필)**: 성별 · **생년월일**(달력 입력, 서버에서 만 나이로 자동 계산) ·
  키 · 체중 · 투석 유형. 왜 이 정보가 필요한지 설명 문구를 추가했고, 범위를 벗어나면
  바로 안내합니다.
- ⚠️ **기존에 이미 가입된 계정이 있다면**: DB의 `email` 컬럼에 이메일 형식으로 저장되어
  있을 것입니다. 로그인 시 그 이메일 전체를 "아이디"란에 그대로 입력하면 동일하게
  로그인됩니다(컬럼과 로직을 그대로 재사용했기 때문). 새로 가입하는 계정부터 원하는
  형식의 아이디를 쓸 수 있습니다.

### 기록/즐겨찾기/PDF보관함/장바구니: 로컬스토리지 → 실제 DB
- 이전엔 이 기능들이 전부 브라우저 `localStorage`에만 저장돼서, 다른 브라우저나 기기로
  로그인하면 아무 기록도 안 보였습니다. 백엔드엔 이미 실제 저장 API
  (`/me/meal-records`, `/me/favorites`, `/me/documents`, `/me/cart`)가 있었는데
  프론트가 호출하지 않고 있었습니다.
- `saveEverywhere`/`loadEverywhere`/`deleteEverywhere` 공용 함수를 추가해서, 로그인
  상태면 서버 DB에 실제로 저장·조회·삭제하고, 서버 요청이 실패할 때만 로컬로 조용히
  폴백하도록 정리했습니다. 즐겨찾기·기록 저장 버튼은 저장 중 상태를 표시합니다.
- 계정 화면(`/account`)은 진입 시 서버 `/me`를 호출해 최신 프로필(나이·키·체중·투석유형)을
  불러와 보여줍니다.

### 백엔드에는 있었지만 화면이 없던 기능 추가
- **하루 식단(`/generate_day`)**: 홈 화면에 "하루 세 끼를 한 번에 만들기" 진입 카드를
  추가했습니다(`/day`). 아침·점심·저녁을 이어서 계산해 하루 총 영양이 기준 안에
  들어오도록 하는 기능으로, 끼니별 카드와 하루 총 영양 요약을 보여줍니다.
- **칼륨 낮추는 조리 팁(`/veg_potassium_tips`)**: 내 정보 화면에 "칼륨 낮추는 조리 팁"
  메뉴를 추가했습니다(`/tips`). 단단한 채소/부드러운 채소별 손질법을 단계별로 보여줍니다.
- 내 정보 화면의 "알림 설정" · "개인정보 및 보안"처럼 실제 기능이 없던 빈 버튼은
  제거하고 실제로 동작하는 메뉴로 교체했습니다.

### 검증
- ✅ 백엔드 `server_FOOK.py` 구문 검사 통과
- ✅ 프론트엔드 `npm run build` 성공
- ⚠️ 실제 서버(DB 포함)를 띄운 채로 회원가입 → 프로필 입력 → 로그인 → 기록/즐겨찾기
  저장 후 다른 세션에서 다시 불러와지는지까지의 end-to-end 테스트는 이 작업 환경에서
  수행하지 못했습니다. 특히 새로 추가된 `/day`, `/tips` 화면은 실제 서버 응답으로
  꼭 한 번씩 확인해 보세요.

---


v10을 실제로 띄워서 써보니, `/generate`는 진짜 AI 엔진을 타는데도 화면 곳곳이 여전히
고정된 예시 값을 보여주는 문제가 있었습니다. 코드를 다시 열어 확인한 결과, 원인은
프론트 화면들이 서버 응답(`apiResult`)을 중간까지만 쓰고 뒷부분은 로컬 하드코딩
데이터(`fookData.ts`)나 목업 문자열로 새는 지점이 여러 곳 있었기 때문입니다.

| 화면/기능 | 문제 | 수정 |
|---|---|---|
| 식단 생성 로딩(`Generating`) | `/generate` 호출에 **8.5초 타임아웃**이 걸려 있어서, 실제 AI 엔진(조건 만족까지 여러 번 재시도하는 구조라 원래 느림)이 정상 동작 중이어도 초과되면 조용히 로컬 데이터로 전환됨 | 타임아웃 60초로 확대, 진행 단계별 안내 메시지 추가 |
| 영양 분석(`Analysis`) | "조정 필요/적정" 판정이 서버가 계산한 회원별 실제 목표치가 아니라 `nmeta`의 고정값(칼로리 550 등)으로만 이뤄짐 | 서버 `targets`(키·체중 기반 실제 계산값) 기준으로 판정하는 `targetOf()` 헬퍼 추가, 전체 화면에 적용 |
| 조정 비교(`Comparison`) | "제공량 100%→70%", "고칼륨 채소→애호박"이 **완전히 하드코딩된 목업**이라 항상 같은 문구가 뜸 | 서버가 실제로 계산해 내려주는 대체 내역(`changes`)을 정규식으로 파싱(`parseChange`)해서 양 조절/재료 대체/메뉴 교체를 구분, 카드가 순차적으로 나타나는 등장 애니메이션(`reveal-in`)으로 표시 |
| 레시피 상세(`Recipe`) | 재료·조리법·영양값이 메뉴명만 같으면 항상 로컬 고정 데이터에서 나옴. **LLM 조리법 편집(`/recipe`)이 아예 호출되지 않았음** | 서버의 실제 재료+양(`dish_ingredients`)을 표시. 화면 진입 시 `/recipe`를 호출해 LLM이 편집한 조리법을 로딩 스피너와 함께 순차 등장(`step-reveal`)시키고, 음성 듣기(`/tts`) 버튼 연결. 재료 교체로 메뉴 이름이 바뀐 경우(`recipe_source`) 안내 문구 표시 |
| 재료 검색(홈 화면) | 재료 자동완성만 보여주고 "이 재료가 들어간 음식"을 보여주지 못함 | 백엔드에 `GET /menus_by_ingredient?q=` 신규 엔드포인트 추가. 재료 선택 시 실제로 그 재료가 들어간 메뉴 목록을 찾아 표시, 그중 하나를 고르면 그 메뉴로 식단 생성 진행 |
| 최종 식단 / 장바구니 / PDF | `PdfPreview`가 `apiResult`를 아예 참조하지 않아 항상 로컬 고정값으로만 렌더링. 장바구니도 실제 재료·양이 아니라 이름만 담김 | `PdfPreview`가 서버의 `nutrition`/`targets`/`dish_ingredients`를 우선 사용하도록 수정. 장바구니는 서버가 계산한 실제 재료명+양(g)을 담도록 수정, 화면에도 양 표시 |
| 조정 중 로딩(`Adjusting`) | 임의의 "제공량→적정량" 애니메이션만 반복 | 서버가 실제로 찾아낸 첫 변경 항목을 로딩 애니메이션에 미리 보여주고, 하단에 실제 변경 목록 미리보기 추가 |

정리하면: **`/generate` API 자체는 처음부터 실제 엔진을 타고 있었지만, 그 이후 화면들
(분석 판정·조정 비교·레시피 상세·PDF·장바구니)이 응답의 일부 필드만 쓰고 나머지는
하드코딩값으로 채워져 있어서 "항상 똑같은 수치"처럼 보였던 것**입니다. 이번 수정으로
`apiResult`(서버 응답)를 화면 전체에서 끝까지 사용하도록 정리했고, 서버 미연결
시(오프라인)에만 기존 로컬 데이터로 자연스럽게 폴백하도록 유지했습니다.

이 수정에 맞춰 백엔드에도 엔드포인트가 하나 늘었습니다(`GET /menus_by_ingredient`).
기존 `/health`, `/generate` 등은 그대로입니다.

### 검증
- ✅ `server_FOOK.py` 구문 검사 통과
- ✅ 프론트엔드 `npm run build`(TypeScript 검사 + Vite 프로덕션 빌드) 성공
- ⚠️ 이전과 마찬가지로, TensorFlow 모델을 실제로 띄운 채로 `/generate`→`/recipe`→`/tts`까지
  이어지는 end-to-end 테스트는 이 작업 환경에서 수행하지 못했습니다. 실제 서버를 띄운 뒤
  화면을 하나씩 눌러보면서 확인해 주세요 — 특히 레시피 상세 화면 진입 시 조리법이
  매번 다르게(회원님이 고른 재료 대체 결과에 맞춰) 나오는지가 이번 수정의 핵심 확인 포인트입니다.

---



1. Neon(PostgreSQL)에서 `backend/sql/001_full_schema.sql` 실행
2. `backend/.env.example`을 `backend/.env`로 복사하고 `DATABASE_URL` 입력
   (조리법 LLM 편집·음성 변환을 쓰려면 `OPENAI_API_KEY`도 함께 설정 — 없어도 나머지 기능은 정상 동작)
3. 백엔드 설치 및 실행
4. 프론트엔드 설치 및 실행

## 백엔드

Python 3.9 기준입니다.

```bash
cd backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# mac/linux
export TF_USE_LEGACY_KERAS=1
# Windows
set TF_USE_LEGACY_KERAS=1

python -m uvicorn server_FOOK:app --reload --port 8000
```

- 첫 실행 시 TF 모델 로딩에 수십 초 걸립니다. 콘솔에 로딩 로그가 끝나면 요청을 받습니다.
- 확인: `http://127.0.0.1:8000/health` (DB 연결 상태 `db.ok`와 AI 엔진 메뉴 개수 `menus`가 함께 나옵니다),
  `http://127.0.0.1:8000/docs` (Swagger UI)

## 프론트엔드

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

확인: `http://localhost:5173`

## 폴더 구성

```
backend/
├─ server_FOOK.py              ← 실행 진입점. DB/인증 + AI 엔진 라우트 전부 여기 있음
├─ app_core_FOOK.py             ← AI 생성 코어 (모델 로딩·한끼 생성) — handoff 원본
├─ FOOK_adjust_levers.py        ← 투석 5대 영양 조정 규칙 — handoff 원본
├─ recipe_editor_FOOK.py        ← /recipe, /tts (LLM/음성) — handoff 원본
├─ database.py / auth_utils.py  ← Neon DB 연결, 비밀번호 해시·세션 — integrated 원본
├─ requirements.txt / .env.example
├─ 식약청_영양성분10.4(수정).xlsx  ← 재료 영양 DB
├─ data/                        ← 메뉴·재료·레시피 데이터 (handoff 원본)
├─ Diet-Generation-.../Code/    ← 생성 모델 정의 + 학습된 체크포인트
├─ Exploiting-Food-Embeddings-.../ ← 재료 대체 사전
└─ sql/
   ├─ 001_full_schema.sql       ← 기본 스키마
   ├─ 002_account_recovery.sql  ← 아이디/비밀번호 찾기용 birthdate 컬럼 (서버가 자동 적용)
   └─ deprecated/               ← 코드와 불일치하는 미사용 초안 (참고용, 실행 금지)

frontend/
├─ src/App.tsx                  ← 전체 화면 라우팅 + API 호출
├─ src/fookData.ts              ← 서버 미연결 시 폴백용 내장 메뉴 데이터
├─ public/assets/kook-logo.svg  ← 코드로 그린 KOOK 로고 (kook-logo.png가 없을 때 대체용)
└─ .env.example                 ← VITE_API_URL
```

## API 개요

- `GET /health` — DB·AI 엔진 상태
- `GET /menus`, `GET /ingredients` — 자동완성용 목록 (AI 엔진 데이터 기준)
- `GET /menus_by_ingredient?q=` — 특정 재료가 들어간 메뉴 목록 (재료 검색 화면용)
- `POST /generate` — 한 끼 생성 (핵심 기능, 느림 — 로딩 UI 필수)
- `POST /generate_day` — 하루 3끼 생성 (매우 느림)
- `POST /recipe` — 조리법 LLM 편집 (`OPENAI_API_KEY` 필요)
- `POST /tts` — 조리법 음성 변환 (`OPENAI_API_KEY` 필요)
- `POST /auth/signup`, `/auth/login`, `/auth/logout` — 회원 인증 (DB)
- `POST /auth/find-id` — 이름+생년월일로 아이디 찾기
- `POST /auth/reset-password` — 아이디+이름+생년월일 확인 후 비밀번호 재설정
- `GET/PUT /me`, `/me/profile` — 프로필
- `GET/POST/DELETE /me/{meal-records|favorites|documents}`, `/me/cart` — 개인 데이터 (DB)

각 엔드포인트의 요청/응답 필드 의미, 함정(나트륨 이중 표기, `targets` 타입 혼재,
`intake`/`consumed` 이어붙이기 방법 등)은 서버 실행 후 `/docs`에서 실제 스키마를 확인하세요.
