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
  Plan,
  Profile,
} from "./types";
import {
  currentUser,
  saveEverywhere,
  warmupBackend,
} from "./services/api";
import { AppContext, useApp } from "./hooks/useApp";
import { fallbackPlan, labels, menuMap, parseLocalIngredient } from "./utils/menu";
import {
  adjustedNutrition,
  fmt,
  fmt2,
  minTargetOf,
  nmeta,
  targetOf,
  totalNutrition,
} from "./utils/nutrition";
import { BookmarkIcon, DocIcon, HomeIcon } from "./components/icons";
import { Logo } from "./components/Logo";
import { Button } from "./components/layout/Button";
import { Shell } from "./components/layout/Shell";
import { BackHeader } from "./components/layout/BackHeader";
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
import { MealSlotDialog } from "./components/meal/MealSlotDialog";
import { RecipeBody } from "./components/meal/RecipeBody";
import { RecipeList } from "./components/meal/RecipeList";
import { GeneratingPage } from "./pages/meal/GeneratingPage";
import { MealResultPage } from "./pages/meal/MealResultPage";
import { AnalysisPage } from "./pages/meal/AnalysisPage";
import { AdjustingPage } from "./pages/meal/AdjustingPage";
import { ComparisonPage } from "./pages/meal/ComparisonPage";
import { FinalMealPage } from "./pages/meal/FinalMealPage";
import { requireUser } from "./utils/auth";

const initialProfile: Profile = {
  gender: "여성",
  birthdate: "",
  age: "60",
  height: "170",
  weight: "65",
  dialysis: "혈액투석",
};
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
        <Route path="/generating" element={<GeneratingPage />} />
        <Route path="/meal" element={<MealResultPage />} />
        <Route path="/analysis" element={<AnalysisPage />} />
        <Route path="/adjusting" element={<AdjustingPage />} />
        <Route path="/comparison" element={<ComparisonPage />} />
        <Route path="/final" element={<FinalMealPage />} />
        <Route path="/recipe/:name" element={<Recipe />} />
        <Route path="/pdf" element={<PdfPreview />} />
        <Route path="*" element={<Navigate to={firstRoute} replace />} />
      </Routes>
      </ErrorBoundary>
    </AppContext.Provider>
  );
}
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
