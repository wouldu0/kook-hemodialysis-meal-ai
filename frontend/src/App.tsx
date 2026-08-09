import {
  Component,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Navigate,
  Route,
  Routes,
  useNavigate,
  useParams,
} from "react-router-dom";
import type {
  ApiResult,
  MealTime,
  NutrientKey,
  Plan,
  Profile,
} from "./types";
import {
  addSaved,
  currentUser,
  generateMeal,
  saveEverywhere,
  warmupBackend,
} from "./services/api";
import { AppContext, useApp } from "./hooks/useApp";
import { fallbackPlan, labels, menuMap, parseLocalIngredient, roleShort } from "./utils/menu";
import {
  adjustedNutrition,
  fmt,
  fmt2,
  minTargetOf,
  nmeta,
  parseChange,
  STATUS_CLASS,
  statusOf,
  targetOf,
  totalNutrition,
} from "./utils/nutrition";
import type { NStatus, ParsedChange } from "./utils/nutrition";
import { BookmarkIcon, BowlIcon, ChartIcon, CheckIcon, ClipboardIcon, DocIcon, HomeIcon, RefreshIcon } from "./components/icons";
import { Logo } from "./components/Logo";
import { Button } from "./components/layout/Button";
import { Shell } from "./components/layout/Shell";
import { BackHeader } from "./components/layout/BackHeader";
import { BottomNav } from "./components/layout/BottomNav";
import { StepHeader } from "./components/layout/StepHeader";
import { FlowFooter } from "./components/layout/FlowFooter";
import { Nutrients } from "./components/meal/Nutrients";
import { NutrientIconRow } from "./components/meal/NutrientIconRow";
import { OnboardingPage } from "./pages/onboarding/OnboardingPage";
import { LoginPage } from "./pages/auth/LoginPage";
import { FindIdPage } from "./pages/auth/FindIdPage";
import { FindPasswordPage } from "./pages/auth/FindPasswordPage";
import { SignupPage } from "./pages/auth/SignupPage";
import { ProfileSetupPage } from "./pages/auth/ProfileSetupPage";
import { AccountPage } from "./pages/account/AccountPage";
import { LibraryPage } from "./pages/account/LibraryPage";
import { TipsPage } from "./pages/account/TipsPage";
import { HomePage } from "./pages/meal/HomePage";
import { DayPlanPage } from "./pages/meal/DayPlanPage";
import { MealListRow } from "./components/meal/MealListRow";
import { MealSlotDialog } from "./components/meal/MealSlotDialog";
import { RecipeBody } from "./components/meal/RecipeBody";
import { RecipeList } from "./components/meal/RecipeList";

const initialProfile: Profile = {
  gender: "여성",
  birthdate: "",
  age: "60",
  height: "170",
  weight: "65",
  dialysis: "혈액투석",
};
const requireUser = (nav: ReturnType<typeof useNavigate>) => {
  if (!currentUser()) {
    nav("/login");
    return false;
  }
  return true;
};
// ── 음성 안내 ────────────────────────────────────────────────────────────────
// 요구사항: "눌러야 나온다" — 화면에 들어왔다고 저절로 읽지 않고, 버튼을 누른 순간에만 읽는다.
// 브라우저 내장 음성합성(무료·즉시 재생·오프라인)을 우선 쓰고, 그게 없는 환경에서만
// 서버 /tts(OpenAI)를 부른다. 서버 TTS는 OPENAI_API_KEY가 없으면 실패하므로 폴백 순서가 중요하다.
// 어느 화면에서 렌더 오류가 나도 흰 화면이 되지 않도록 막는다.
// (예전에 '빈 화면'이 보이던 상황에서 원인을 바로 알 수 있게 메시지도 띄운다)
class ErrorBoundary extends Component<
  { children: any },
  { error: Error | null }
> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="page">
        <div className="phone">
          <section className="screen">
            <div className="empty-state">
              <div>⚠</div>
              <b>화면을 표시하지 못했어요.</b>
              <p>{String(this.state.error?.message || this.state.error)}</p>
              <button
                className="btn"
                onClick={() => {
                  this.setState({ error: null });
                  window.location.href = "/home";
                }}
              >
                홈으로 돌아가기
              </button>
            </div>
          </section>
        </div>
      </main>
    );
  }
}

function App() {
  // 앱에 들어오면 '항상' 소개 페이지부터 보여주고, 그 다음 로그인 화면으로 간다.
  // 예전엔 한 번 본 사람은 건너뛰도록 localStorage 플래그를 봤는데, 그 기록이 남아 있으면
  // 링크를 열자마자 검색 화면이 떠버려서(소개도 로그인도 안 보임) 그 분기를 없앴다.
  const firstRoute = "/onboarding";
  // 무료 호스팅 백엔드는 한동안 요청이 없으면 잠든다. 사용자가 소개 화면을
  // 넘기는 동안 미리 깨워두면, 회원가입/로그인 차례에는 이미 준비된 상태가 된다.
  useEffect(() => {
    warmupBackend();
  }, []);
  const [profile, setProfile] = useState(initialProfile);
  const [plan, setPlan] = useState<Plan>(fallbackPlan);
  // 검색창은 빈 값으로 시작한다. (비회원 '한 끼 체험'만 예시 메뉴를 채워 넣는다)
  const [query, setQuery] = useState("");
  const [searchMode, setSearchMode] = useState<
    "menu" | "ingredient" | "random"
  >("menu");
  const [apiResult, setApiResult] = useState<ApiResult | null>(null);
  const [usingFallback, setUsingFallback] = useState(false);
  const value = useMemo(
    () => ({
      profile,
      setProfile,
      plan,
      setPlan,
      query,
      setQuery,
      searchMode,
      setSearchMode,
      apiResult,
      setApiResult,
      usingFallback,
      setUsingFallback,
    }),
    [profile, plan, query, searchMode, apiResult, usingFallback],
  );
  return (
    <AppContext.Provider value={value}>
      <ErrorBoundary>
      <Routes>
        <Route path="/" element={<Navigate to={firstRoute} replace />} />
        <Route path="/onboarding" element={<OnboardingPage />} />
        {/* 온보딩 각 단계를 실제 화면으로 분리한다. 하단 '다음'이 화면을 전환하고,
            브라우저 뒤로가기도 단계 단위로 동작한다. */}
        <Route path="/onboarding/:step" element={<OnboardingPage />} />
        {/* 예전 '시작하기 선택' 화면은 없앴다. 소개가 끝나면 곧바로 로그인 화면으로 간다. */}
        <Route path="/start" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/find-id" element={<FindIdPage />} />
        <Route path="/find-password" element={<FindPasswordPage />} />
        <Route path="/signup" element={<SignupPage />} />
        <Route path="/profile" element={<ProfileSetupPage />} />
        <Route path="/home" element={<HomePage />} />
        <Route path="/day" element={<DayPlanPage />} />
        <Route path="/tips" element={<TipsPage />} />
        <Route path="/account" element={<AccountPage />} />
        <Route path="/history" element={<LibraryPage mode="history" />} />
        <Route path="/favorites" element={<LibraryPage mode="favorites" />} />
        <Route path="/documents" element={<LibraryPage mode="documents" />} />
        <Route path="/generating" element={<Generating />} />
        <Route path="/meal" element={<MealResult />} />
        <Route path="/analysis" element={<Analysis />} />
        <Route path="/adjusting" element={<Adjusting />} />
        <Route path="/comparison" element={<Comparison />} />
        <Route path="/final" element={<FinalMeal />} />
        <Route path="/recipe/:name" element={<Recipe />} />
        <Route path="/pdf" element={<PdfPreview />} />
        <Route path="*" element={<Navigate to={firstRoute} replace />} />
      </Routes>
      </ErrorBoundary>
    </AppContext.Provider>
  );
}
function Generating() {
  const nav = useNavigate();
  const { profile, query, searchMode, setApiResult, setPlan, setUsingFallback } =
    useApp();
  const [s, setS] = useState(0);
  const [error, setError] = useState("");
  const msgs = [
    "메뉴와 재료 데이터를 불러오고 있어요",
    "회원님의 키·몸무게 기준으로 영양 목표를 계산하고 있어요",
    "조건에 맞는 조합을 찾을 때까지 반복해서 시도하고 있어요",
    "칼륨·나트륨 등 위험 수치를 낮추기 위해 양과 재료를 조정하고 있어요",
    "최종 식단을 정리하고 있어요",
  ];
  useEffect(() => {
    let live = true;
    const t = setInterval(
      () =>
        setS((v) =>
          v < 2 ? v + 1 : v < 3 ? v + (Math.random() < 0.3 ? 1 : 0) : v,
        ),
      1400,
    );
    const run = async () => {
      const body: any = { weight: Number(profile.weight) || 60, meals_left: 3 };
      // 백엔드가 height와 sex를 항상 같이 요구한다(표준체중 계산에 성별이 필요).
      // 둘을 독립된 조건으로 보내면 gender가 비어있을 때 height만 보내 422가 나므로 묶는다.
      if (profile.height) {
        body.height = Number(profile.height);
        body.sex = profile.gender === "남성" ? "남" : "여";
      }
      if (searchMode === "menu" && query.trim()) body.menu = query.trim();
      if (searchMode === "ingredient" && query.trim())
        body.ingredient = query.trim();
      try {
        const d = await generateMeal(body);
        if (!live) return;
        setS(4);
        setUsingFallback(false);
        setApiResult(d);
        const p: Plan = {
          id: "api",
          day: 1,
          slot: "추천",
          menus: d.meal || [],
        };
        setPlan(p);
        setTimeout(() => live && nav("/meal"), 500);
      } catch (e) {
        if (!live) return;
        setError(
          "서버에 연결하지 못해 내장 예시 데이터로 진행합니다. 백엔드 서버가 켜져 있는지 확인해주세요.",
        );
        // apiResult를 비워둔 채로(=null) 다음 화면들이 로컬 예시 데이터로 렌더되게 하고,
        // 그 화면들이 "예시 데이터입니다" 배너를 계속 보여줄 수 있도록 플래그를 켠다.
        setApiResult(null);
        setUsingFallback(true);
        setTimeout(() => live && nav("/meal"), 1600);
      }
    };
    run();
    return () => {
      live = false;
      clearInterval(t);
    };
  }, [
    nav,
    profile.weight,
    profile.height,
    profile.gender,
    query,
    searchMode,
    setApiResult,
    setPlan,
    setUsingFallback,
  ]);
  return (
    <Shell>
      <div className="loading-page">
        <p className="eyebrow">AI 식단 생성</p>
        <h1>한 끼를 생성하고 있어요.</h1>
        <div className="loader-plate">
          <span className="loader-emoji">🍲</span>
          <div className="orbit" />
        </div>
        <p className={error ? "loading-message warn" : "loading-message"}>
          {error || msgs[s]}
        </p>
        <div className="progress">
          <i style={{ width: `${((s + 1) / msgs.length) * 100}%` }} />
        </div>
        {!error && s >= 2 && (
          <p className="loading-hint">
            영양 기준에 딱 맞는 조합을 찾는 중이라 조금 더 걸릴 수 있어요.
          </p>
        )}
      </div>
    </Shell>
  );
}
function MealResult() {
  const nav = useNavigate();
  const { plan } = useApp();
  return (
    <Shell
      header={false}
      footer={
        <button className="btn" onClick={() => nav("/analysis")}>
          <i className="btn-icon">
            <ChartIcon />
          </i>{" "}
          영양 적합성 판정하기
        </button>
      }
    >
      <BackHeader onBack={() => nav("/home")} />
      <div className="hero-center">
        <span className="hero-mark">
          <BowlIcon />
        </span>
        <h1>
          선택한 음식과
          <br />
          어울리는 <b>식단을 생성했어요</b>
        </h1>
        <span className="compose-pill">
          구성 : <b>밥, 국, 밥찬 3가지</b>
        </span>
      </div>
      <div className="meal-list numbered">
        {plan.menus.map((name, i) => (
          <MealListRow
            key={name}
            name={name}
            role={roleShort(i)}
            onClick={() => nav(`/recipe/${encodeURIComponent(name)}`)}
          />
        ))}
      </div>
    </Shell>
  );
}
function Analysis() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  // 이 화면은 '식단을 생성한 그대로'의 영양을 보여준다. 재구성은 다음 단계에서 하므로
  // 조정 후 수치(apiResult.nutrition)가 아니라 조정 전 수치(nutrition_before)를 쓴다.
  const raw =
    apiResult?.nutrition_before || apiResult?.nutrition || totalNutrition(plan);
  const targets = apiResult?.targets;
  // 게이지(Nutrients)와 동일한 기준: 상한 초과뿐 아니라 하한 미달도 '조정 필요'로 본다.
  // (에너지·단백질은 서버가 [최소,최대] 범위를 주므로 미달도 실제로 의미가 있다.)
  const over = nmeta.filter((n) => {
    const v = raw[n.key];
    const hi = targetOf(targets, n.key);
    const lo = minTargetOf(targets, n.key);
    return v > hi || (lo > 0 && v < lo);
  });
  const changeCount = apiResult?.changes?.length || 0;
  // 이제 이 화면은 '조정 전' 수치를 보여주므로, 초과 항목이 있으면 그대로 '재구성하러 가기'다.
  //  needFix : 기준을 벗어난 항목이 있다 → 재구성 단계로
  //  clean   : 처음부터 전부 기준 안 → 볼 조정 내역이 없으면 최종 식단으로 바로
  const state: "needFix" | "clean" = over.length > 0 ? "needFix" : "clean";
  const goNext = () =>
    nav(state === "needFix" || changeCount > 0 ? "/adjusting" : "/final");
  return (
    <Shell
      header={false}
      footer={
        <button className="btn" onClick={goNext}>
          <i className="btn-icon">
            {state === "needFix" ? <RefreshIcon /> : <DocIcon />}
          </i>{" "}
          {state === "needFix"
            ? "레시피 재구성하러 가기"
            : changeCount > 0
              ? `재구성 내역 보기 (${changeCount}건)`
              : "최종 식단 보기"}
        </button>
      }
    >
      <BackHeader onBack={() => nav("/meal")} dot />
      <h1 className="analysis-title">
        영양 적합성 <b>판정 결과</b>
      </h1>
      <p className="sub">
        개인 프로필을 기반으로
        <br />
        영양 섭취 적합 여부를 판정했습니다.
      </p>
      <NutrientIconRow />
      <Nutrients values={raw} targets={targets} isFallback={!apiResult} />
      {/* 하단 안내 카드 — 상태에 따라 색과 문구가 달라진다 */}
      <div className={"notice-card " + state}>
        <i className="notice-mark">{state === "needFix" ? "!" : "✓"}</i>
        <div>
          <b>
            {state === "needFix"
              ? over.map((x) => x.label).join(" · ") + " — 레시피 재구성 필요"
              : "레시피 재구성 필요 없음"}
          </b>
          <span>
            {state === "needFix"
              ? "지금은 식단을 생성한 그대로의 영양이에요. 다음 단계에서 메뉴의 양과 식재료를 조정해 기준에 맞춰드릴게요."
              : "생성한 그대로 모든 항목이 기준 안에 있어요."}
          </span>
        </div>
      </div>
    </Shell>
  );
}
function Adjusting() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  const changes: string[] = apiResult?.changes || [];
  const parsed = changes.map(parseChange);
  // 1단계에서 보여줄 것: '음식명 자체가 바뀐' 변경 (예: 배추김치 → 저염물김치 교체)
  const renamed = parsed.filter((c) => c.kind === "swap").slice(0, 3);
  // 2단계 뱃지. 재료를 바꿔 메뉴명까지 달라진 교체(recipe_source에 원본명이 남는 경우)도
  // '재료 대체'로 센다. 저염김치 풀에서 통째로 바뀐 것만 '저염 메뉴 교체'로 따로 센다.
  const isIngredientSwap = (c: ParsedChange) =>
    c.kind === "ingredient" ||
    (c.kind === "swap" && !!apiResult?.recipe_source?.[c.after || ""]);
  const amountCount = parsed.filter((c) => c.kind === "amount").length;
  const ingCount = parsed.filter(isIngredientSwap).length;
  const swapCount = parsed.filter(
    (c) => c.kind === "swap" && !isIngredientSwap(c),
  ).length;
  const badges: [string, number][] = [];
  if (amountCount) badges.push(["양 조절", amountCount]);
  if (ingCount) badges.push(["재료 대체", ingCount]);
  if (swapCount) badges.push(["저염 메뉴 교체", swapCount]);
  const [p, setP] = useState(0);
  useEffect(() => {
    // 0.9초 간격으로 단계가 올라간다. p>=1 음식명 교체, p>=2부터 뱃지가 하나씩.
    const t = setInterval(() => setP((x) => Math.min(x + 1, 4)), 900);
    const e = setTimeout(() => nav("/comparison"), 5400);
    return () => {
      clearInterval(t);
      clearTimeout(e);
    };
  }, [nav]);
  return (
    <Shell
      footer={
        <FlowFooter step={4} total={5} onPrev={() => nav("/analysis")} />
      }
    >
      <StepHeader step={4} total={5} />
      <h1 className="onboarding-title">
        <b>4.</b> 판정에 따른 레시피 재구성
      </h1>
      <p className="sub center">
        초과되는 영양소에 대하여
        <br />
        재료의 <b>양을 조절</b>하거나 <b>대체</b>하여
        <br />
        혈액투석 환자 맞춤형으로 레시피를 재구성해줍니다
      </p>

      {/* ① 음식명이 바뀌는 장면이 먼저 나온다 */}
      <div className="rename-stage">
        {(renamed.length ? renamed : plan.menus.slice(0, 3).map((m) => ({
          raw: m,
          menu: m,
          before: m,
          after: m,
        }))).map((c: any, i: number) => (
          <div
            className={p >= 1 ? "rename-row on" : "rename-row"}
            key={c.raw}
            style={{ transitionDelay: `${i * 160}ms` }}
          >
            <b className="rename-before">{c.before}</b>
            <i className="rename-arrow">→</i>
            <b className="rename-after">{c.after}</b>
          </div>
        ))}
        {!renamed.length && (
          <p className="rename-note">
            {p >= 1 ? "메뉴는 그대로 두고 재료만 조정하고 있어요." : " "}
          </p>
        )}
      </div>

      {/* ② 그 다음 양 조절 / 재료 대체 뱃지가 붉게 찍힌다 */}
      <div className="adjust-badges">
        {(badges.length ? badges : ([["양 조절", 0], ["재료 대체", 0]] as [string, number][])).map(
          ([label, count], i) => (
            <span
              className={p >= i + 2 ? "adjust-badge on" : "adjust-badge"}
              key={label}
            >
              {label}
              {count > 0 && <em>{count}건</em>}
            </span>
          ),
        )}
      </div>

      <div className="adjust-loader">
        <span className={p >= 1 ? "adjust-ring spin" : "adjust-ring"}>
          <ClipboardIcon />
        </span>
        <b>레시피 재구성 중입니다</b>
        <small>
          KOOK AI가 영양 밸런스를 맞추고 있어요...
          <br />
          잠시만 기다려주세요
        </small>
      </div>
    </Shell>
  );
}
function Comparison() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  // after = 재구성 후(서버 최종 수치), before = 재구성 전(서버가 nutrition_before로 내려줌).
  // 서버 미연결이면 로컬 메뉴 영양 합계를 '전', 보정값을 '후'로 써서 화면 구조를 유지한다.
  const after = apiResult?.nutrition || adjustedNutrition(totalNutrition(plan));
  const before = apiResult?.nutrition_before || totalNutrition(plan);
  const changes: string[] = apiResult?.changes || [];
  const parsed = changes.map(parseChange);
  const hasReal = changes.length > 0;
  const speech = hasReal
    ? `판정에 따라 레시피를 재구성했습니다. ${parsed
        .slice(0, 5)
        .map((c) =>
          c.kind === "removed"
            ? `${c.menu} 제외`
            : `${c.menu}, ${c.before}에서 ${c.after}로 변경`,
        )
        .join(". ")}.`
    : "이번 식단은 별도 조정이 없었습니다. 생성된 조합이 처음부터 영양 기준을 만족했습니다.";
  // 음식별로 묶는다. 서버 changes의 메뉴명은 '교체 전' 이름일 수 있어서(예: 배추김치 →
  // 저염물김치 교체), 교체 후 이름(c.after)으로도 매칭해야 카드가 비지 않는다.
  const perMenu = plan.menus.map((menu) => {
    const mine = parsed.filter(
      (c) =>
        c.menu === menu ||
        (c.kind === "swap" && c.after === menu) ||
        c.raw.startsWith(`${menu}:`),
    );
    const tags: string[] = [];
    const lines: string[] = [];
    for (const c of mine) {
      if (c.kind === "amount") {
        if (!tags.includes("양 조절")) tags.push("양 조절");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "ingredient") {
        if (!tags.includes("재료 대체")) tags.push("재료 대체");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "swap" && apiResult?.recipe_source?.[c.after || ""]) {
        // 서버는 '재료를 바꿔서 메뉴 이름까지 달라진 경우'(시금치나물 → 양배추나물)도
        // "교체"라는 한 문장으로 내려준다. 이때만 recipe_source에 원래 이름이 남으므로,
        // 그걸로 '재료 대체'와 '저염 메뉴로 통째 교체'를 구분한다.
        if (!tags.includes("재료 대체")) tags.push("재료 대체");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "swap") {
        if (!tags.includes("저염 메뉴 교체")) tags.push("저염 메뉴 교체");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "removed") {
        if (!tags.includes("메뉴 제외")) tags.push("메뉴 제외");
        lines.push("기준을 넘겨 이번 식단에서 제외했어요.");
      }
    }
    return { menu, tags, lines, details: mine };
  });
  // 어떤 메뉴에도 붙지 않은 변경(메뉴 교체·제외 등)은 따로 모아 보여준다.
  const matched = new Set(perMenu.flatMap((m) => m.details.map((c) => c.raw)));
  const unmatched = parsed.filter((c) => !matched.has(c.raw));
  // 카드가 위에서부터 하나씩 나타나는 등장 애니메이션
  const [revealed, setRevealed] = useState(0);
  useEffect(() => {
    const t = setInterval(
      () => setRevealed((v) => (v >= perMenu.length ? v : v + 1)),
      220,
    );
    return () => clearInterval(t);
  }, [perMenu.length]);
  return (
    <Shell
      header={false}
      footer={
        <button className="btn" onClick={() => nav("/final")}>
          <i className="btn-icon">
            <DocIcon />
          </i>{" "}
          재료와 조리과정 보러가기 <i className="btn-arrow">›</i>
        </button>
      }
    >
      <BackHeader onBack={() => nav("/analysis")} dot />
      <div className="hero-center">
        <span className="hero-check">
          <CheckIcon />
        </span>
        <h1>
          레시피 <b>재구성 완료</b>
        </h1>
        <p className="sub center">
          개인 프로필 영양 섭취 기준에 맞게
          <br />
          식단을 재구성했어요.
        </p>
      </div>
      <h2 className="section-title">음식별 재구성 내용</h2>
      <div className="permenu-list">
        {perMenu.map((m, i) => (
          <article
            className={i < revealed ? "permenu-card show" : "permenu-card"}
            key={m.menu}
          >
            <div className="permenu-head">
              <b>{m.menu}</b>
              <span className="permenu-tags">
                {m.tags.length ? (
                  m.tags.map((t) => (
                    <em className="tag-adjust" key={t}>
                      {t}
                    </em>
                  ))
                ) : (
                  <em className="tag-keep">조정 내용 없음</em>
                )}
              </span>
            </div>
            {m.lines.length ? (
              <ul className="permenu-changes">
                {m.lines.map((l, k) => {
                  const [b, a] = l.split(" → ");
                  return (
                    <li key={`${l}-${k}`}>
                      {a ? (
                        <>
                          <span className="pc-before">{b}</span>
                          <i>→</i>
                          <span className="pc-after">{a}</span>
                        </>
                      ) : (
                        <span className="pc-plain">{l}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p>변경 없이 그대로 유지했어요.</p>
            )}
          </article>
        ))}
      </div>
      {!hasReal && (
        <div className="info-box">
          <b>이번 식단은 별도 조정이 없었어요</b>
          <span>생성된 조합이 처음부터 영양 기준을 만족했어요.</span>
        </div>
      )}
      {unmatched.length > 0 && (
        <section className="change-section">
          <div className="change-heading">
            <span>+</span>
            <div>
              <b>그 밖의 조정</b>
              <small>메뉴 교체·제외처럼 개별 음식에 묶이지 않는 변경이에요.</small>
            </div>
          </div>
          {unmatched.map((c) => (
            <article className="change-card reveal-in" key={c.raw}>
              <div className="change-card-title">
                <b>{c.menu || "식단"}</b>
                <span>
                  {c.kind === "swap"
                    ? "메뉴 교체"
                    : c.kind === "removed"
                      ? "메뉴 제외"
                      : "재료 교체"}
                </span>
              </div>
              <div className="amount-flow mini">
                {c.after ? (
                  <>
                    <strong>{c.before}</strong>
                    <i>→</i>
                    <strong className="highlight">{c.after}</strong>
                  </>
                ) : (
                  <strong>{c.raw}</strong>
                )}
              </div>
            </article>
          ))}
        </section>
      )}
      {apiResult?.note && (
        <div className="info-box">
          <b>AI 안내</b>
          <span>{apiResult.note}</span>
        </div>
      )}
      <section className="change-section">
        <div className="change-heading">
          <span>✓</span>
          <div>
            <b>최종 영양소</b>
            <small>레시피 재구성 전과 후를 나란히 비교했어요.</small>
          </div>
        </div>
        <div className="ba-list">
          {nmeta.map((n) => {
            const b = Number(before?.[n.key] ?? 0);
            const a = Number(after?.[n.key] ?? 0);
            const hi = targetOf(apiResult?.targets, n.key);
            const lo = minTargetOf(apiResult?.targets, n.key);
            const aStatus = statusOf(a, lo, hi);
            const diff = a - b;
            const same = Math.round(Math.abs(diff)) === 0;
            const dir = same ? "same" : diff < 0 ? "down" : "up";
            return (
              <div className="ba-item" key={n.key}>
                <span className="ba-item-name">
                  {n.icon} {n.label}
                </span>
                {/* 증감 뱃지는 화살표 안에 쌓지 않고 아래 행으로 뺀다.
                    그래야 전/후 숫자가 화살표와 같은 줄에 정렬된다. */}
                <div className="ba-flow">
                  <span className="ba-from">
                    {fmt(b)}
                    <small>{n.unit}</small>
                  </span>
                  <span className={`ba-arrow ${dir}`}>
                    <svg viewBox="0 0 40 16" aria-hidden="true">
                      <path d="M2 8h30" />
                      <path d="M27 3l6 5-6 5" />
                    </svg>
                  </span>
                  <span className={`ba-to ${STATUS_CLASS[aStatus]}`}>
                    {fmt(a)}
                    <small>{n.unit}</small>
                  </span>
                  {!same && (
                    <em className={`ba-delta ${dir}`}>
                      {diff < 0 ? "−" : "+"}
                      {fmt(Math.abs(diff))}
                    </em>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        {!apiResult?.nutrition_before && (
          <p className="ba-note">
            ⓘ 서버가 재구성 전 수치를 주지 않아, 원본 메뉴 기준 예시값으로 비교했어요.
          </p>
        )}
      </section>
    </Shell>
  );
}
// 식단을 기록할 때 날짜와 끼니를 고르는 팝업.
// 여기서 고른 값에 따라 '식단 관리'의 아침/점심/저녁 섹션으로 들어간다.
function FinalMeal() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  const item = {
    id: `meal-${Date.now()}`,
    title: plan.menus[1] || "맞춤 한 끼",
    subtitle: plan.menus.join(" · "),
    createdAt: new Date().toISOString(),
    menus: plan.menus,
  };
  const [savingKey, setSavingKey] = useState<string | null>(null);
  const [openRecipe, setOpenRecipe] = useState(""); // 인라인으로 펼친 레시피
  const [askJoin, setAskJoin] = useState(false); // 비회원 체험 종료 시 회원가입 안내
  const [askSlot, setAskSlot] = useState<{ key: string; msg: string } | null>(
    null,
  ); // 날짜·끼니 선택 팝업
  // 식단 기록은 날짜·끼니를 먼저 고르게 하고, 그 외(PDF 등)는 바로 저장한다.
  const save = async (key: string, msg: string) => {
    if (!requireUser(nav)) return;
    if (key === "fook:favorites") return setAskSlot({ key, msg });
    setSavingKey(key);
    try {
      await saveEverywhere(key, item);
      alert(msg);
    } finally {
      setSavingKey(null);
    }
  };
  const saveWithSlot = async (date: string, time: MealTime) => {
    if (!askSlot) return;
    const { key, msg } = askSlot;
    setAskSlot(null);
    setSavingKey(key);
    try {
      await saveEverywhere(key, { ...item, mealDate: date, mealTime: time });
      alert(`${date} ${time} 식단으로 ${msg}`);
    } finally {
      setSavingKey(null);
    }
  };
  const finalValues =
    apiResult?.nutrition || adjustedNutrition(totalNutrition(plan));
  return (
    <Shell
      header={false}
      footer={
        <FlowFooter
          step={5}
          total={5}
          onPrev={() => nav("/comparison")}
          // 비회원 체험이면 여기서 끝내지 않고 회원가입 안내를 먼저 띄운다.
          onNext={() => (currentUser() ? nav("/home") : setAskJoin(true))}
          nextLabel="완료"
        />
      }
    >
      <BackHeader onBack={() => nav("/comparison")} dot />
      <div className="hero-center">
        <h1>
          오늘의 한 끼, <b>완성!</b>
        </h1>
        <p className="sub center">
          오늘의 한 끼가 완성되었어요.
          <br />
          선택한 식단의 <b>재료와 조리과정</b>을 확인해보세요.
        </p>
      </div>
      <Nutrients
        values={finalValues}
        targets={apiResult?.targets}
        isFallback={!apiResult}
      />
      {/* 하단 탭: 기록 저장 · 홈 · PDF 다운로드 (장바구니 탭은 뺐다) */}
      <div className="final-tabs">
        <button
          disabled={savingKey === "fook:favorites"}
          onClick={() => save("fook:favorites", "기록된 식단에 추가했어요.")}
        >
          <BookmarkIcon />
          <span>{savingKey === "fook:favorites" ? "기록 중..." : "기록하기"}</span>
        </button>
        <button onClick={() => nav("/home")}>
          <HomeIcon />
          <span>홈</span>
        </button>
        <button onClick={() => nav("/pdf")}>
          <DocIcon />
          <span>PDF 다운로드</span>
        </button>
      </div>
      <h2 className="section-title with-icon">📖 레시피 보러가기</h2>
      {/* 새 화면으로 넘어가지 않고 고른 음식의 레시피를 이 자리에서 펼친다 */}
      <RecipeList selected={openRecipe} onSelect={setOpenRecipe} />
      {/* 비회원 체험을 마쳤을 때만 뜨는 회원가입 안내 */}
      {askSlot && (
        <MealSlotDialog
          onCancel={() => setAskSlot(null)}
          onConfirm={saveWithSlot}
        />
      )}
      {askJoin && (
        <div className="modal-bg" onClick={() => setAskJoin(false)}>
          <div className="modal ask" onClick={(e) => e.stopPropagation()}>
            <span className="modal-mark">🍚</span>
            <h2>
              회원가입하면 개인 맞춤형 식단을
              <br />
              제공받을 수 있어요
            </h2>
            <p>
              키 체중, 투석 유형을 등록하면
              <br />
              내 기준에 맞춘 영양 계산과 식단 관리 등을 쓸 수 있어요
            </p>
            <div className="ask-actions">
              <button className="ask-no" onClick={() => nav("/login")}>
                나중에
              </button>
              <button className="ask-yes" onClick={() => nav("/login")}>
                회원가입 하러가기
              </button>
            </div>
          </div>
        </div>
      )}
    </Shell>
  );
}
// 서버 /recipe의 steps는 배열이 아니라 '여러 줄 문자열'로 올 수 있다
// (recipe_editor_FOOK.edit_recipe가 LLM 응답 텍스트를 그대로 반환한다).
// 어떤 형태로 와도 화면이 깨지지 않게 문자열 배열로 맞춰준다.
function Recipe() {
  const nav = useNavigate();
  const { name = "" } = useParams();
  const { plan } = useApp();
  const decoded = decodeURIComponent(name);
  const [selected, setSelected] = useState(decoded);
  const [askSlot, setAskSlot] = useState(false); // 날짜·끼니 선택 팝업
  const saveMeal = () => {
    if (!requireUser(nav)) return;
    setAskSlot(true);
  };
  const saveWithSlot = async (date: string, time: MealTime) => {
    setAskSlot(false);
    await saveEverywhere("fook:favorites", {
      id: `meal-${Date.now()}`,
      title: plan.menus[1] || decoded,
      subtitle: plan.menus.join(" · "),
      createdAt: new Date().toISOString(),
      menus: plan.menus,
      mealDate: date,
      mealTime: time,
    });
    alert(`${date} ${time} 식단으로 기록했어요.`);
  };
  return (
    <Shell
      header={false}
      footer={
        <div className="footer-actions three">
          <button onClick={() => nav("/home")}>
            <HomeIcon /> 홈으로
          </button>
          <button onClick={saveMeal}>
            <BookmarkIcon /> 기록하기
          </button>
          <button onClick={() => nav("/pdf")}>
            <DocIcon /> PDF 다운로드
          </button>
        </div>
      }
    >
      <BackHeader title="레시피 보러가기" dot />
      {/* 이번 식단 목록에 없는 메뉴로 직접 들어온 경우에도 그 메뉴를 보여준다 */}
      {decoded && !plan.menus.includes(decoded) && (
        <>
          <div className="recipe-detail-head">
            <span className="rm-num">1</span>
            <h1 className="recipe-title">{decoded}</h1>
          </div>
          <RecipeBody menuName={decoded} />
          <h2 className="section-title">이번 한 끼의 다른 메뉴</h2>
        </>
      )}
      <RecipeList selected={selected} onSelect={setSelected} />
      {askSlot && (
        <MealSlotDialog
          onCancel={() => setAskSlot(false)}
          onConfirm={saveWithSlot}
        />
      )}
    </Shell>
  );
}
function PdfPreview() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  const ref = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  // 서버가 실제로 계산한 값이 있으면 그걸 쓰고, 없을 때만(오프라인) 폴백 계산을 쓴다.
  const values =
    apiResult?.nutrition || adjustedNutrition(totalNutrition(plan));
  const targets = apiResult?.targets;
  const download = async () => {
    if (!ref.current) return;
    setLoading(true);
    try {
      const [{ default: html2canvas }, { jsPDF }] = await Promise.all([
        import("html2canvas"),
        import("jspdf"),
      ]);
      const canvas = await html2canvas(ref.current, {
        scale: 2,
        backgroundColor: "#fff",
      });
      const pdf = new jsPDF("p", "mm", "a4");
      // A4 한 장(210×297mm) 안에 반드시 들어가게 한다.
      // 가로를 먼저 맞춰보고, 그래도 세로가 넘치면 세로 기준으로 다시 줄인다.
      const PAGE_W = 210;
      const PAGE_H = 297;
      const MARGIN = 8;
      const maxW = PAGE_W - MARGIN * 2;
      const maxH = PAGE_H - MARGIN * 2;
      let w = maxW;
      let h = (canvas.height * w) / canvas.width;
      if (h > maxH) {
        h = maxH;
        w = (canvas.width * h) / canvas.height;
      }
      pdf.addImage(
        canvas.toDataURL("image/jpeg", 0.96),
        "JPEG",
        (PAGE_W - w) / 2,
        MARGIN,
        w,
        h,
      );
      pdf.save(`KOOK_${plan.menus[1]}_맞춤한끼.pdf`);
      if (currentUser())
        await saveEverywhere("fook:documents", {
          id: `pdf-${Date.now()}`,
          title: `${plan.menus[1]} 맞춤 한 끼 PDF`,
          subtitle: "레시피 · 영양정보",
          createdAt: new Date().toISOString(),
          menus: plan.menus,
        });
    } finally {
      setLoading(false);
    }
  };
  return (
    <Shell
      header={false}
      footer={
        <div className="pdf-footer-actions">
          <Button secondary onClick={() => nav(-1)}>
            ‹ 이전으로
          </Button>
          <Button onClick={download}>
            {loading ? "PDF 생성 중..." : "PDF 다운로드"}
          </Button>
        </div>
      }
    >
      <BackHeader title="PDF 미리보기" />
      <div className="pdf-page" ref={ref}>
        <div className="pdf-brand">
          <Logo />
          <div>
            <b>KOOK 맞춤 한 끼 레시피</b>
            <span>혈액투석 환자용 식단 참고 자료</span>
          </div>
        </div>
        <h2>{plan.menus[1]} 한 끼</h2>
        {/* PDF는 다운로드되면 앱 화면(Shell의 예시 데이터 배너)과 분리된 별도 파일이 되므로,
            서버 미연결로 예시 데이터를 쓴 경우 이 안내를 파일 안에도 반드시 함께 남긴다. */}
        {!apiResult && (
          <p className="pdf-fallback-notice">
            ⚠ 서버 연결에 실패해 내장 예시 데이터로 생성된 문서입니다. 실제 개인
            맞춤 계산 결과가 아닙니다.
          </p>
        )}
        {/* 사진 자리(pdf-hero-space)는 뺐다. A4 한 장에 담기지 않아서 표를 위로 당긴다. */}
        <h3>한 끼 전체 레시피</h3>
        <table className="meal-recipe-table">
          <thead>
            <tr>
              <th>구분</th>
              <th>음식</th>
              <th>재료</th>
              <th>조리 과정</th>
            </tr>
          </thead>
          <tbody>
            {plan.menus.map((name, i) => {
              const m = menuMap.get(name); // 오프라인 폴백용
              const serverIngs: [string, number][] | undefined =
                apiResult?.dish_ingredients?.[name];
              const ingList = (
                serverIngs || (m?.ingredients || []).map(parseLocalIngredient)
              ).map(([ing, amt]) => (amt > 0 ? `${ing} ${fmt2(amt)}g` : ing));
              return (
                <tr key={name}>
                  <th>{labels[i]}</th>
                  <td>{name}</td>
                  <td>
                    <ul>
                      {ingList.map((x) => (
                        <li key={x}>{x}</li>
                      ))}
                    </ul>
                  </td>
                  <td>
                    <ol>
                      {(
                        m?.steps || [
                          "레시피 상세 화면에서 AI 조리법을 확인하세요.",
                        ]
                      ).map((x, j) => (
                        <li key={j}>{x}</li>
                      ))}
                    </ol>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <h3>한 끼 영양 정보</h3>
        <table className="nutrition-pdf-table">
          <thead>
            <tr>
              <th>영양소</th>
              <th>섭취량</th>
              <th>판정</th>
            </tr>
          </thead>
          <tbody>
            {nmeta.map((n) => {
              const t = targetOf(targets, n.key);
              const lo = minTargetOf(targets, n.key);
              const v = values[n.key];
              const ok = v <= t && (lo === 0 || v >= lo);
              return (
                <tr key={n.key}>
                  <th>{n.label}</th>
                  <td>
                    {fmt(v)}
                    {n.unit} (기준 {lo > 0 ? `${fmt(lo)}~${fmt(t)}` : `${fmt(t)} 이하`}
                    {n.unit})
                  </td>
                  <td>{!apiResult ? "예시" : ok ? "적정" : "주의"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}
export default App;
