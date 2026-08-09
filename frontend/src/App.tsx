import {
  Component,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import type { ApiResult, Plan, Profile } from "./types";
import { warmupBackend } from "./services/api";
import { AppContext } from "./hooks/useApp";
import { fallbackPlan } from "./utils/menu";
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
import { GeneratingPage } from "./pages/meal/GeneratingPage";
import { MealResultPage } from "./pages/meal/MealResultPage";
import { AnalysisPage } from "./pages/meal/AnalysisPage";
import { AdjustingPage } from "./pages/meal/AdjustingPage";
import { ComparisonPage } from "./pages/meal/ComparisonPage";
import { FinalMealPage } from "./pages/meal/FinalMealPage";
import { RecipePage } from "./pages/recipe/RecipePage";
import { PdfPreviewPage } from "./pages/recipe/PdfPreviewPage";

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
  const [dishSteps, setDishStepsState] = useState<Record<string, string[]>>({});
  const setDishSteps = (menu: string, steps: string[]) =>
    setDishStepsState((prev) => ({ ...prev, [menu]: steps }));
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
      dishSteps,
      setDishSteps,
    }),
    [profile, plan, query, searchMode, apiResult, usingFallback, dishSteps],
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
        <Route path="/recipe/:name" element={<RecipePage />} />
        <Route path="/pdf" element={<PdfPreviewPage />} />
        <Route path="*" element={<Navigate to={firstRoute} replace />} />
      </Routes>
      </ErrorBoundary>
    </AppContext.Provider>
  );
}
export default App;
