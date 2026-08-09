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
import { dietPlans, ingredientData, menuData } from "./fookData";
import type {
  ApiResult,
  MealTime,
  MenuRecord,
  NutrientKey,
  Plan,
  Profile,
  SavedItem,
} from "./types";
import {
  addSaved,
  apiFetch,
  authToken,
  currentUser,
  deleteEverywhere,
  loadEverywhere,
  saveEverywhere,
  saveSession,
  storage,
} from "./services/api";
import { AppContext, useApp } from "./hooks/useApp";

const initialProfile: Profile = {
  gender: "여성",
  birthdate: "",
  age: "60",
  height: "170",
  weight: "65",
  dialysis: "혈액투석",
};
// YYYY-MM-DD 생년월일로 만 나이를 계산한다. 형식이 이상하면 null.
function ageFromBirthdate(birthdate: string): number | null {
  const m = birthdate.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return null;
  const [, y, mo, d] = m.map(Number) as unknown as number[];
  const b = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  if (isNaN(b.getTime())) return null;
  const today = new Date();
  let age = today.getFullYear() - b.getFullYear();
  const beforeBirthday =
    today.getMonth() < b.getMonth() ||
    (today.getMonth() === b.getMonth() && today.getDate() < b.getDate());
  if (beforeBirthday) age -= 1;
  return age >= 1 && age <= 120 ? age : null;
}
const menuMap = new Map(menuData.map((m) => [m.name, m as MenuRecord]));
const fallbackPlan =
  (dietPlans.find((p) => p.menus.includes("시금치된장국")) as Plan) ||
  (dietPlans[0] as Plan);
// 끼니 구분 — 식단 관리 화면에서 아침/점심/저녁 섹션으로 나누는 기준
const MEAL_TIMES: MealTime[] = ["아침", "점심", "저녁"];
// 지금 시각으로 기본 끼니를 고른다 (10시 전 아침, 15시 전 점심, 그 뒤 저녁)
function defaultMealTime(): MealTime {
  const h = new Date().getHours();
  return h < 10 ? "아침" : h < 15 ? "점심" : "저녁";
}
// <input type="date">에 넣을 오늘 날짜 (YYYY-MM-DD)
function todayISO(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate(),
  ).padStart(2, "0")}`;
}
const requireUser = (nav: ReturnType<typeof useNavigate>) => {
  if (!currentUser()) {
    nav("/login");
    return false;
  }
  return true;
};
const labels = ["밥", "국", "어육류", "밑반찬", "김치류"];
// 목업 기준 표기.
//  · 식단 생성 화면(2/5): 밥 / 국 / 반찬1 / 반찬2 / 반찬3
//  · 레시피 화면: 밥 / 국 / 찬 1 (어육류) / 찬 2 (밑반찬) / 찬 3 (김치류)
function roleShort(i: number) {
  if (i === 0) return "밥";
  if (i === 1) return "국";
  return `반찬${i - 1}`;
}
function roleLong(i: number) {
  if (i === 0) return "밥";
  if (i === 1) return "국";
  return `찬 ${i - 1} (${labels[i] || "반찬"})`;
}
const nmeta: {
  key: NutrientKey;
  label: string;
  unit: string;
  target: number;
  icon: string;
}[] = [
  { key: "energy", label: "열량", unit: "kcal", target: 550, icon: "🔥" },
  { key: "protein", label: "단백질", unit: "g", target: 24, icon: "💪" },
  { key: "phosphorus", label: "인", unit: "mg", target: 550, icon: "🦴" },
  { key: "potassium", label: "칼륨", unit: "mg", target: 1200, icon: "🌿" },
  { key: "sodium", label: "나트륨", unit: "mg", target: 400, icon: "🧂" },
];
function totalNutrition(plan: Plan) {
  return plan.menus.reduce(
    (a, name) => {
      const n = menuMap.get(name)?.nutrition;
      for (const k of [
        "energy",
        "protein",
        "phosphorus",
        "potassium",
        "sodium",
      ] as NutrientKey[])
        a[k] += n?.[k] || 0;
      return a;
    },
    { energy: 0, protein: 0, phosphorus: 0, potassium: 0, sodium: 0 },
  );
}
function adjustedNutrition(raw: ReturnType<typeof totalNutrition>) {
  return {
    energy: Math.min(raw.energy, 520),
    protein: Math.min(raw.protein, 24),
    phosphorus: Math.min(raw.phosphorus, 520),
    potassium: Math.min(raw.potassium, 1150),
    sodium: Math.min(raw.sodium, 390),
  };
}
function fmt(v: number) {
  return Math.round(v).toLocaleString("ko-KR");
}
// 재료·간식의 '양(g)'은 반올림하지 않고 항상 소수점 2자리까지 보여준다.
// (0.5g 차이가 저나트륨/저칼륨 조리에서는 의미가 있어서, 반올림하면 조정 내역이 안 보인다.)
function fmt2(v: number) {
  return Number(v || 0).toLocaleString("ko-KR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}
// 서버 미연결(오프라인) 폴백 데이터의 재료는 "시금치, 생것 37.5g"처럼 양이 문자열에 붙어 있다.
// 서버 응답과 같은 [이름, 양] 형태로 쪼개서, 표시할 때 동일하게 소수점 2자리로 맞춘다.
function parseLocalIngredient(raw: string): [string, number] {
  const m = raw.match(/^(.*?)\s+([\d.]+)\s*g$/);
  return m ? [m[1], Number(m[2])] : [raw, 0];
}
// 브랜드 로고. 사용자가 직접 준비한 kook-logo.png가 있으면 그걸 쓰고,
// 없으면 같은 폴더의 kook-logo.svg(코드로 그린 동일 디자인)로 자동 대체한다.
function Logo({ className = "" }: { className?: string }) {
  return (
    <img
      className={className}
      src="/assets/kook-logo.png"
      alt="KOOK"
      onError={(e) => {
        const img = e.currentTarget;
        if (!img.src.endsWith(".svg")) img.src = "/assets/kook-logo.svg";
      }}
    />
  );
}

// ── 음성 안내 ────────────────────────────────────────────────────────────────
// 요구사항: "눌러야 나온다" — 화면에 들어왔다고 저절로 읽지 않고, 버튼을 누른 순간에만 읽는다.
// 브라우저 내장 음성합성(무료·즉시 재생·오프라인)을 우선 쓰고, 그게 없는 환경에서만
// 서버 /tts(OpenAI)를 부른다. 서버 TTS는 OPENAI_API_KEY가 없으면 실패하므로 폴백 순서가 중요하다.
// ── 목업 아이콘 (선화 스타일, currentColor로 색을 상속받는다) ─────────────────
const svg = (d: any, extra: any = {}) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...extra}
  >
    {d}
  </svg>
);
const BellIcon = () =>
  svg(
    <>
      <path d="M18 8a6 6 0 1 0-12 0c0 7-2 8-2 8h16s-2-1-2-8" />
      <path d="M13.7 20a2 2 0 0 1-3.4 0" />
    </>,
  );
const UserIcon = () =>
  svg(
    <>
      <path d="M19 21v-1a7 7 0 0 0-14 0v1" />
      <circle cx="12" cy="7.5" r="3.8" />
    </>,
  );
const SearchIcon = () =>
  svg(
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.6-3.6" />
    </>,
  );
const LeafIcon = () =>
  svg(
    <>
      <path
        d="M11 20A7 7 0 0 1 4 13c0-6 7-9 16-9 0 9-3 16-9 16Z"
        fill="currentColor"
        stroke="none"
      />
      <path d="M4.5 20c3.5-6 8.5-9.5 13.5-11.5" stroke="#fff" />
    </>,
  );
const DiceIcon = () =>
  svg(
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="4" fill="currentColor" stroke="none" />
      <g fill="#fff">
        <circle cx="8.2" cy="8.2" r="1.5" />
        <circle cx="15.8" cy="8.2" r="1.5" />
        <circle cx="12" cy="12" r="1.5" />
        <circle cx="8.2" cy="15.8" r="1.5" />
        <circle cx="15.8" cy="15.8" r="1.5" />
      </g>
    </>,
  );
const SlidersIcon = () =>
  svg(
    <>
      <path d="M4 7h11M18.5 7H20M4 17h5M12.5 17H20" />
      <circle cx="16" cy="7" r="1.9" />
      <circle cx="10.5" cy="17" r="1.9" />
    </>,
  );
const CheckIcon = () => svg(<path d="m5 12.5 4.5 4.5L19 7" />, { strokeWidth: 2.4 });
const BowlIcon = () =>
  svg(
    <>
      <path d="M3.5 11h17a8.5 8.5 0 0 1-17 0Z" />
      <path d="M9 6c0-1.2 1-1.6 1-2.8M12.5 6c0-1.6 1.2-2 1.2-3.4M16 6c0-1.2 1-1.6 1-2.6" />
    </>,
  );
const HomeIcon = () =>
  svg(<path d="M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1Z" />);
const ClipboardIcon = () =>
  svg(
    <>
      <rect x="5" y="4" width="14" height="17" rx="2.5" />
      <path d="M9 4.5A1.5 1.5 0 0 1 10.5 3h3A1.5 1.5 0 0 1 15 4.5V6H9Z" />
      <path d="M9 11h6M9 15h4" />
    </>,
  );
const DocIcon = () =>
  svg(
    <>
      <path d="M6 3h7l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" />
      <path d="M13 3v5h5M8.5 13h7M8.5 17h5" />
    </>,
  );
const BookmarkIcon = () =>
  svg(<path d="M7 3h10a1 1 0 0 1 1 1v17l-6-4-6 4V4a1 1 0 0 1 1-1Z" />);
const SpeakerIcon = () =>
  svg(
    <>
      <path d="M4 9.5h3.5L12 5.5v13L7.5 14.5H4Z" />
      <path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11" />
    </>,
  );
const RefreshIcon = () =>
  svg(
    <>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <path d="M20 4v4h-4" />
    </>,
  );
const ChartIcon = () =>
  svg(
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="3" />
      <path d="M8 16v-4M12 16V8M16 16v-2.5" />
    </>,
  );

function useSpeech() {
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stop = () => {
    if (typeof window !== "undefined" && window.speechSynthesis)
      window.speechSynthesis.cancel();
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setSpeaking(false);
  };
  useEffect(() => stop, []);
  const speak = async (text: string) => {
    stop();
    const clean = text.replace(/\s+/g, " ").trim();
    if (!clean) return;
    const synth = typeof window !== "undefined" ? window.speechSynthesis : null;
    if (synth) {
      // 음성 목록은 비동기로 채워지는 브라우저가 있어서, 비어 있으면 한 번 기다렸다 다시 본다.
      let voices = synth.getVoices();
      if (!voices.length) {
        await new Promise<void>((res) => {
          const t = setTimeout(res, 800);
          synth.onvoiceschanged = () => {
            clearTimeout(t);
            res();
          };
        });
        voices = synth.getVoices();
      }
      const u = new SpeechSynthesisUtterance(clean);
      const ko = voices.find((v) => /^ko/i.test(v.lang));
      if (ko) u.voice = ko; // lang만 지정하면 무시하는 브라우저가 있어 음성을 직접 지정한다
      u.lang = "ko-KR";
      u.rate = 0.9; // 어르신이 따라오기 쉽도록 조금 느리게
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      setSpeaking(true);
      synth.speak(u);
      return;
    }
    setSpeaking(true);
    try {
      const blob = await apiFetch("/tts", {
        method: "POST",
        body: JSON.stringify({ text: clean }),
        responseType: "blob",
      });
      const audio = new Audio(URL.createObjectURL(blob));
      audioRef.current = audio;
      audio.onended = () => setSpeaking(false);
      await audio.play();
    } catch {
      setSpeaking(false);
    }
  };
  return { speak, stop, speaking };
}
// 화면 설명을 읽어주는 버튼. 누르기 전에는 아무 소리도 나지 않는다.
// 흐름 화면 하단의 '이전 / 단계 점 / 다음' 네비게이션.
// 어느 단계에서든 앞 화면으로 다시 돌아가 볼 수 있어야 한다는 요구사항을 이걸로 처리한다.
function FlowFooter({
  step,
  total,
  onPrev,
  onNext,
  nextLabel = "다음",
  prevLabel = "이전",
}: {
  step: number;
  total: number;
  onPrev?: () => void;
  onNext?: () => void;
  nextLabel?: string;
  prevLabel?: string;
}) {
  return (
    <div className="flow-footer">
      {onPrev ? (
        <button className="flow-btn ghost" onClick={onPrev}>
          <i>‹</i> {prevLabel}
        </button>
      ) : (
        <span />
      )}
      <div className="flow-dots">
        {Array.from({ length: total }, (_, i) => (
          <span key={i} className={i + 1 === step ? "fdot on" : "fdot"} />
        ))}
      </div>
      {onNext ? (
        <button className="flow-btn solid" onClick={onNext}>
          {nextLabel} <i>›</i>
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}
// 서버 targets는 키마다 형태가 다르다: energy/protein은 [최소,최대] 배열,
// potassium/phosphorus/sodium은 숫자 하나. 화면에서 '상한선' 하나만 필요할 때 쓰는 공용 함수.
// targets가 아직 없으면(서버 미연결) nmeta의 대표값으로 대체한다.
function targetOf(targets: any, key: NutrientKey): number {
  const raw = targets?.[key];
  if (raw != null) return Array.isArray(raw) ? Number(raw[1]) : Number(raw);
  return nmeta.find((n) => n.key === key)?.target ?? 0;
}
// energy/protein은 서버가 [최소,최대] 범위로 내려준다. 그 최소값(하한)을 뽑는다.
// potassium/phosphorus/sodium처럼 상한만 있는 항목은 하한이 없다는 뜻으로 0을 반환.
function minTargetOf(targets: any, key: NutrientKey): number {
  const raw = targets?.[key];
  if (Array.isArray(raw)) return Number(raw[0]);
  return 0;
}
// changes 문자열 한 줄을 화면에 쓸 형태로 분류한다. app_core_FOOK._changes()가
// 만드는 3가지 패턴을 그대로 파싱: 교체/양조절/재료대체/메뉴제외.
type ParsedChange = {
  kind: "amount" | "ingredient" | "swap" | "removed" | "other";
  menu: string;
  before?: string;
  after?: string;
  raw: string;
};
function parseChange(line: string): ParsedChange {
  let m = line.match(/^(.+?): (.+) 양 ([\d.]+)→([\d.]+)g$/);
  if (m)
    return {
      kind: "amount",
      menu: m[1],
      before: `${m[2]} ${m[3]}g`,
      after: `${m[2]} ${m[4]}g`,
      raw: line,
    };
  m = line.match(/^(.+?): (.+) → (.+)$/);
  if (m)
    return {
      kind: "ingredient",
      menu: m[1],
      before: m[2],
      after: m[3],
      raw: line,
    };
  m = line.match(/^(.+?) → (.+?) 교체$/);
  if (m)
    return { kind: "swap", menu: m[1], before: m[1], after: m[2], raw: line };
  m = line.match(/^(.+?) 뺌$/);
  if (m) return { kind: "removed", menu: m[1], raw: line };
  return { kind: "other", menu: "", raw: line };
}

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
    apiFetch("/health").catch(() => {});
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
        <Route path="/onboarding" element={<Onboarding />} />
        {/* 온보딩 각 단계를 실제 화면으로 분리한다. 하단 '다음'이 화면을 전환하고,
            브라우저 뒤로가기도 단계 단위로 동작한다. */}
        <Route path="/onboarding/:step" element={<Onboarding />} />
        {/* 예전 '시작하기 선택' 화면은 없앴다. 소개가 끝나면 곧바로 로그인 화면으로 간다. */}
        <Route path="/start" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<Login />} />
        <Route path="/find-id" element={<FindId />} />
        <Route path="/find-password" element={<FindPassword />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/profile" element={<ProfileSetup />} />
        <Route path="/home" element={<Home />} />
        <Route path="/day" element={<DayPlan />} />
        <Route path="/tips" element={<Tips />} />
        <Route path="/account" element={<Account />} />
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
function StepHeader({ step, total }: { step: number; total: number }) {
  return (
    <div className="step-header">
      <div className="step-dots">
        {Array.from({ length: total }, (_, i) => i + 1).map((n, i) => (
          <div className="step-dot-wrap" key={n}>
            <span className={n <= step ? "step-circle done" : "step-circle"}>
              {n}
            </span>
            {i < total - 1 && (
              <span className={n < step ? "step-line done" : "step-line"} />
            )}
          </div>
        ))}
      </div>
      <span className="step-count">
        {step} / {total}
      </span>
    </div>
  );
}
function Shell({
  children,
  footer,
  header = true,
  full = false,
}: {
  children: any;
  footer?: any;
  header?: boolean;
  full?: boolean;      // 이미지 한 장이 화면 전체를 채울 때 본문 여백을 없앤다
}) {
  const { usingFallback } = useApp();
  return (
    <main className="page">
      <div className="phone">
        {header && <Header />}
        <section className={full ? "screen screen-full" : "screen"}>
          {usingFallback && (
            <div className="warning-box fallback-banner">
              <b>⚠ 예시 데이터를 보고 있어요</b>
              <span>
                서버 연결에 실패해서 실제 개인 맞춤 계산이 아닌 내장 예시 값으로
                화면을 보여주고 있습니다. 아래 영양 판정·재구성 내역은 참고용이
                아니라 그저 화면 구성을 보여주기 위한 예시입니다.
              </span>
            </div>
          )}
          {children}
        </section>
        {footer && <footer className="footer">{footer}</footer>}
      </div>
    </main>
  );
}
function Header() {
  const nav = useNavigate();
  return (
    <header className="header">
      <button className="logo-button" onClick={() => nav("/home")}>
        <Logo />
      </button>
      <div className="header-actions">
        <button className="icon-ghost" aria-label="알림" onClick={() => nav("/history")}>
          <BellIcon />
        </button>
        <button
          className="icon-avatar"
          aria-label="내 정보"
          onClick={() => nav("/account")}
        >
          <UserIcon />
        </button>
      </div>
    </header>
  );
}
function BackHeader({
  title,
  onBack,
  dot = false,
}: {
  title?: string;
  onBack?: () => void;
  dot?: boolean;
}) {
  const nav = useNavigate();
  return (
    <header className="header detail-header">
      <div className="header-lead">
        <button
          className="icon-back"
          aria-label="뒤로"
          onClick={() => (onBack ? onBack() : nav(-1))}
        >
          ‹
        </button>
        {/* 어느 화면에 있어도 브랜드가 보이도록 왼쪽 위에 로고를 둔다. 누르면 홈으로. */}
        <button
          className="header-brand"
          aria-label="KOOK 홈으로"
          onClick={() => nav("/home")}
        >
          <Logo />
        </button>
        {title && <b className="header-lead-title">{title}</b>}
      </div>
      <div className="header-actions">
        <button
          className={dot ? "icon-ghost has-dot" : "icon-ghost"}
          aria-label="알림"
          onClick={() => nav("/history")}
        >
          <BellIcon />
        </button>
        <button
          className="icon-avatar"
          aria-label="내 정보"
          onClick={() => nav("/account")}
        >
          <UserIcon />
        </button>
      </div>
    </header>
  );
}
function Button({
  children,
  onClick,
  secondary = false,
  disabled = false,
}: any) {
  return (
    <button
      className={secondary ? "btn btn-secondary" : "btn"}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
function Trust() {
  return (
    <div className="trust">
      대한신장학회의 혈액투석 환자 영양관리 자료를 참고하여 설계했어요.
    </div>
  );
}

// 온보딩 5단계. [번호, 제목, 설명, 미리보기 종류]
const slides = [
  [
    "1.",
    "원하는 음식 및 재료 검색",
    "먹고 싶은 음식이나 재료를 선택하세요",
    "search",
  ],
  [
    "2.",
    "한 끼 식단 생성",
    "선택한 음식과 어울리는 식단을 생성해드려요",
    "meal",
  ],
  [
    "3.",
    "영양 적합성 판정",
    "혈액투석 환자 개인 프로필 기준으로\n영양 적합성을 판정합니다.",
    "nutrition",
  ],
  [
    "4.",
    "판정에 따른 레시피 재구성",
    "초과되는 영양소에 대하여\n재료의 양을 조절하거나 대체하여\n혈액투석 환자 맞춤형으로 레시피를 재구성해줍니다",
    "adjust",
  ],
  [
    "5.",
    "오늘의 한 끼, 완성!",
    "오늘의 한 끼가 완성되었어요.\n선택한 식단의 재료와 조리과정을 확인해보세요.",
    "final",
  ],
] as const;
// 첫 화면: "오늘 뭐 해 먹지?" 스플래시. 시작하기를 누르면 온보딩 5단계로 들어간다.
function Splash({ onStart }: { onStart: () => void }) {
  // 준비된 홈 이미지가 있으면 그 한 장을 화면 전체로 쓴다.
  // 그림 속 '시작하기' 버튼은 일러스트 위에 겹쳐 그려져 있어 잘라낼 수 없으므로,
  // 그 자리에 투명한 진짜 버튼을 겹쳐 둔다 (위치는 이미지에서 실측한 비율).
  const [shotFailed, setShotFailed] = useState(false);
  if (!shotFailed)
    return (
      <Shell header={false} full>
        <div className="splash-shot">
          <div className="splash-frame">
            <img
              src="/assets/onboarding-home.png"
              alt="오늘 뭐 해 먹지? 고민에 푹 빠질 땐 KOOK이 도와드립니다"
              onError={() => setShotFailed(true)}
            />
            {/* 그림 속 '시작하기' 버튼 위에 겹치는 투명 버튼 */}
            <button
              className="splash-hit"
              onClick={onStart}
              aria-label="시작하기"
            />
          </div>
        </div>
      </Shell>
    );
  return (
    <Shell
      header={false}
      footer={
        <button className="btn btn-pill" onClick={onStart}>
          시작하기 <i className="btn-arrow">→</i>
        </button>
      }
    >
      <div className="splash">
        <Logo className="splash-logo" />
        <h1 className="splash-title">오늘 뭐 먹지?</h1>
        <p className="splash-sub">
          고민에 <b>푹</b> 빠질 땐,
          <br />
          <b>KOOK(쿡)</b>이 도와드립니다
        </p>
        <div className="thought-bubble">
          {[
            ["🥣", "건강한 식단"],
            ["📋", "영양 균형"],
            ["🍲", "간편한 관리"],
          ].map(([icon, text]) => (
            <div key={text}>
              <span>{icon}</span>
              <small>{text}</small>
            </div>
          ))}
        </div>
        <p className="splash-caption">혈액투석 환자를 위한 AI 기반 맞춤형 식단 관리 및 레시피 재구성 솔루션</p>
      </div>
    </Shell>
  );
}
function Onboarding() {
  // URL로 단계를 구분한다. /onboarding = 스플래시, /onboarding/1~5 = 각 단계 화면.
  const { step } = useParams();
  const nav = useNavigate();
  const [shotFailed, setShotFailed] = useState(false);
  useEffect(() => setShotFailed(false), [step]);
  const finish = () => nav("/login");
  if (!step) return <Splash onStart={() => nav("/onboarding/1")} />;
  const n = Number(step);
  // 범위를 벗어난 주소(/onboarding/9 등)는 스플래시로 되돌린다.
  if (!Number.isInteger(n) || n < 1 || n > slides.length)
    return <Navigate to="/onboarding" replace />;
  const i = n - 1;
  const s = slides[i];
  const last = i === slides.length - 1;
  const footer = (
    <FlowFooter
      step={n}
      total={slides.length}
      prevLabel="이전"
      onPrev={() => nav(i > 0 ? `/onboarding/${i}` : "/onboarding")}
      onNext={last ? finish : () => nav(`/onboarding/${n + 1}`)}
      nextLabel={last ? "시작" : "다음"}
    />
  );
  // 단계 이미지가 준비돼 있으면 그 이미지 한 장이 곧 화면이다.
  // (이미지 안에 제목과 1/5 배지가 이미 들어 있어서 화면 텍스트와 겹치지 않게 감춘다)
  const shot = ONBOARDING_IMAGE[s[3]];
  return (
    <Shell header={false} footer={footer} full={!!shot && !shotFailed}>
      {shot && !shotFailed ? (
        <div className="onboarding-page">
          <img
            src={shot}
            alt={`${s[0]} ${s[1]} — ${s[2]}`}
            onError={() => setShotFailed(true)}
          />
          {/* 이미지 화면에도 왼쪽 위에 로고를 얹고, 건너뛰기는 반대쪽으로 보낸다 */}
          <span className="shot-brand" aria-hidden="true">
            <Logo />
          </span>
          <button className="skip float right" onClick={finish}>
            건너뛰기
          </button>
        </div>
      ) : (
        <>
          <div className="onboarding-top">
            <button className="skip" onClick={finish}>
              건너뛰기
            </button>
            <span className="step-count">
              {i + 1} / {slides.length}
            </span>
          </div>
          <h1 className="onboarding-title">
            <b>{s[0]}</b> {s[1]}
          </h1>
          <p className="sub center">{s[2]}</p>
          <OnboardingVisual type={s[3]} />
        </>
      )}
    </Shell>
  );
}
// 온보딩 각 단계의 미리보기. 실제 화면을 축소해 보여주는 용도라 동작하지 않는다.
const PREVIEW_MENUS = [
  ["백미밥", "밥"],
  ["시금치 된장국", "국"],
  ["닭가슴살 채소볶음", "반찬"],
  ["애호박전", "반찬"],
  ["저염 배추김치", "반찬"],
] as const;
// 온보딩 단계별 이미지. public/assets 에 파일이 있으면 그 이미지를 쓰고,
// 없으면(파일 미준비) 아래의 CSS 미리보기로 자동 대체된다.
const ONBOARDING_IMAGE: Record<string, string> = {
  search: "/assets/onboarding-1.png",
  meal: "/assets/onboarding-2.png",
  nutrition: "/assets/onboarding-3.png",
  adjust: "/assets/onboarding-4.png",
  final: "/assets/onboarding-5.png",
};
function OnboardingShot({ type, children }: { type: string; children: any }) {
  const src = ONBOARDING_IMAGE[type];
  const [failed, setFailed] = useState(false);
  if (!src || failed) return children;
  return (
    <figure className="onboarding-shot">
      <img src={src} alt="" onError={() => setFailed(true)} />
    </figure>
  );
}
function OnboardingVisual({ type }: { type: string }) {
  if (type === "search")
    return (
      <div className="visual-card preview-card">
        <div className="tabs three-tabs preview-tabs">
          <span className="tab active">⌕ 음식 검색</span>
          <span className="tab">🥕 재료 검색</span>
          <span className="tab">⇄ 랜덤 추천</span>
        </div>
        <div className="fake-search">
          <SearchIcon />
          <span>음식 이름을 입력하세요</span>
          <b>🎙</b>
        </div>
        <p className="preview-label">추천 검색어</p>
        <div className="chips">
          {["된장찌개", "닭가슴살", "연어구이", "비빔밥", "두부조림", "미역국"].map(
            (x) => (
              <button key={x} disabled>
                {x}
              </button>
            ),
          )}
        </div>
        <p className="preview-label">최근 검색</p>
        <div className="chips recent">
          {["된장찌개", "연어구이", "두부조림"].map((x) => (
            <button key={x} disabled>
              ◷ {x} ✕
            </button>
          ))}
        </div>
        <span className="preview-cta">✨ 선택하기</span>
      </div>
    );
  if (type === "meal")
    return (
      <div className="visual-card preview-card">
        <span className="compose-pill">
          구성 : <b>밥, 국, 반찬 3가지</b>
        </span>
        <div className="meal-list">
          {PREVIEW_MENUS.map(([name, role]) => (
            <div className="meal-row" key={name}>
              <b className="meal-row-name">{name}</b>
              <span className="meal-row-sub">{role}</span>
            </div>
          ))}
        </div>
      </div>
    );
  if (type === "nutrition")
    return (
      <div className="visual-card preview-card">
        <NutrientIconRow caption="KOOK AI가 영양 기준 적합 여부를 판정합니다." />
        <Nutrients
          values={{
            energy: 620,
            protein: 22,
            phosphorus: 210,
            potassium: 1200,
            sodium: 1800,
          }}
          targets={{
            energy: [600, 700],
            protein: [22, 24],
            phosphorus: 333,
            potassium: 1000,
            sodium: 1000,
          }}
        />
        <p className="gauge-note">
          ⓘ 혈액투석 환자 남자 65세 키 170cm, 몸무게 60kg 기준
        </p>
      </div>
    );
  if (type === "adjust")
    return (
      <div className="visual-card preview-card">
        <div className="adjust-loader">
          <span className="adjust-ring">
            <ClipboardIcon />
          </span>
          <b>레시피 재구성 중입니다</b>
          <small>
            KOOK AI가 영양 밸런스를 맞추고 있어요...
            <br />
            잠시만 기다려주세요
          </small>
        </div>
        <p className="preview-label with-icon">
          <ClipboardIcon /> 레시피 재구성 내용
        </p>
        <div className="lever-list">
          {[
            [<LeafIcon key="l" />, "재료 대체", "칼륨, 인 함량이 높은 재료를 저함량 재료로 대체했어요."],
            [<span key="s">🧂</span>, "조미료량 조정", "나트륨 섭취를 줄이기 위해 소금, 간장 등 조미료의 양을 조절했어요."],
            [<span key="c">🥛</span>, "재료 양 조절", "초과되는 영양소의 섭취를 줄이기 위해 재료의 양을 조절했어요."],
          ].map(([icon, title, desc]: any) => (
            <div className="lever-row" key={title}>
              <span className="lever-icon">{icon}</span>
              <div>
                <b>{title}</b>
                <small>{desc}</small>
              </div>
              <i>›</i>
            </div>
          ))}
        </div>
      </div>
    );
  return (
    <div className="visual-card preview-card">
      <div className="final-plate">
        <span>🍚 🍲 🥘 🥗 🥬</span>
      </div>
      <div className="preview-actions">
        <span>
          <BookmarkIcon /> 기록하기
        </span>
        <span>
          <DocIcon /> PDF 다운로드
        </span>
      </div>
      <p className="preview-label with-icon">📖 레시피 보러가기</p>
      <div className="meal-list">
        {PREVIEW_MENUS.map(([name]) => (
          <div className="meal-row" key={name}>
            <b className="meal-row-name">{name}</b>
            <i className="meal-row-arrow">›</i>
          </div>
        ))}
      </div>
    </div>
  );
}
// 영양소 5종 아이콘 요약 줄 (목업 3/5 · 영양 적합성 판정 화면 공통)
function NutrientIconRow({ caption }: { caption?: string }) {
  return (
    <div className="nutrient-icon-row">
      <div className="nutrient-icons">
        {nmeta.map((n) => (
          <div key={n.key}>
            <span>{n.icon}</span>
            <small>{n.label}</small>
          </div>
        ))}
      </div>
      {caption && <p className="nutrient-icon-caption">{caption}</p>}
    </div>
  );
}
function Login() {
  const nav = useNavigate();
  const { setPlan, setQuery, setApiResult } = useApp();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [show, setShow] = useState(false);
  const [askTry, setAskTry] = useState(false); // 체험 전 가상 프로필 안내
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const submit = async () => {
    if (!username.trim() || !password) {
      setError("아이디와 비밀번호를 입력해주세요.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const d = await apiFetch("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username: username.trim(), password }),
      });
      saveSession(d);
      nav("/home");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const tryGuest = () => {
    setApiResult(null);
    setPlan(fallbackPlan);
    setQuery("시금치된장국");
    nav("/generating");
  };
  return (
    <Shell
      header={false}
      footer={
        <div className="auth-links">
          <button onClick={() => nav("/find-id")}>아이디 찾기</button>
          <span />
          <button onClick={() => nav("/find-password")}>비밀번호 찾기</button>
          <span />
          <button onClick={() => nav("/signup")}>회원가입</button>
        </div>
      }
    >
      <div className="login-brand">
        <Logo />
        <p className="login-tagline">
          혈액투석 환자 맞춤형 <b>AI 식단 관리 솔루션</b>
        </p>
        <div className="login-divider">
          <span />♥<span />
        </div>
        <p className="login-sub">
          건강한 한 끼, <b>KOOK</b>이 함께합니다.
        </p>
      </div>
      <div className="form login-form">
        <div className="field icon-field">
          <UserIcon />
          <input
            value={username}
            placeholder="아이디를 입력해주세요"
            autoComplete="username"
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
        </div>
        <div className="field icon-field">
          <span className="lock">🔒</span>
          <input
            type={show ? "text" : "password"}
            value={password}
            placeholder="비밀번호를 입력해주세요"
            autoComplete="current-password"
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
          />
          <button
            type="button"
            className="reveal"
            aria-label={show ? "비밀번호 숨기기" : "비밀번호 보기"}
            onClick={() => setShow(!show)}
          >
            {show ? "🙈" : "👁"}
          </button>
        </div>
        {error && <p className="form-error">{error}</p>}
        <button className="btn login-btn" disabled={busy} onClick={submit}>
          {busy ? "로그인 중..." : "로그인"}
        </button>
      </div>
      <div className="guest-try">
        <span className="guest-thumb">🥣</span>
        <div>
          <b>한 끼 식단 체험해보기</b>
          <small>
            예시 프로필을 기반으로
            <br />
            AI 식단 추천과 레시피 재구성 과정을
            <br />
            직접 체험해보세요.
          </small>
        </div>
        <button className="guest-cta" onClick={() => setAskTry(true)}>
          체험하기 →
        </button>
      </div>
      <p className="guest-note">※ 체험은 예시 프로필을 기반으로 진행됩니다.</p>
      {/* 체험 시작 전에 어떤 기준으로 계산되는지 먼저 알린다 */}
      {askTry && (
        <div className="modal-bg" onClick={() => setAskTry(false)}>
          <div className="modal ask" onClick={(e) => e.stopPropagation()}>
            <span className="modal-mark">👤</span>
            <h2>가상의 프로필로 진행됩니다</h2>
            <p>
              회원님의 정보가 아직 없어서
              <br />
              아래 예시 프로필 기준으로 영양을 계산해요.
            </p>
            <div className="guest-profile">
              <div>
                <small>성별 · 나이</small>
                <b>남성 · 65세</b>
              </div>
              <div>
                <small>키 · 체중</small>
                <b>170cm · 60kg</b>
              </div>
              <div>
                <small>투석 유형</small>
                <b>혈액투석</b>
              </div>
            </div>
            <div className="ask-actions">
              <button className="ask-no" onClick={() => setAskTry(false)}>
                취소
              </button>
              <button className="ask-yes" onClick={tryGuest}>
                체험 진행하기
              </button>
            </div>
          </div>
        </div>
      )}
    </Shell>
  );
}
// 아이디 찾기 — 이름 + 생년월일로 본인 확인 (이메일 발송 수단이 없는 서비스라서)
function FindId() {
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [birthdate, setBirthdate] = useState("");
  const [found, setFound] = useState<string[] | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const ok = name.trim().length > 0 && ageFromBirthdate(birthdate) != null;
  const submit = async () => {
    setBusy(true);
    setError("");
    setFound(null);
    try {
      const d = await apiFetch("/auth/find-id", {
        method: "POST",
        body: JSON.stringify({ name: name.trim(), birthdate }),
      });
      setFound(d.usernames || []);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Shell
      header={false}
      footer={
        found ? (
          <Button onClick={() => nav("/login")}>로그인하러 가기</Button>
        ) : (
          <Button disabled={!ok || busy} onClick={submit}>
            {busy ? "찾는 중..." : "아이디 찾기"}
          </Button>
        )
      }
    >
      <BackHeader title="아이디 찾기" onBack={() => nav("/login")} />
      <h1>
        가입할 때 입력한
        <br />
        이름과 생년월일을 알려주세요.
      </h1>
      {found ? (
        <div className="result-box">
          <b>회원님의 아이디예요</b>
          {found.map((u) => (
            <strong key={u}>{u}</strong>
          ))}
        </div>
      ) : (
        <div className="form">
          <label>
            이름
            <div className="field">
              <input
                value={name}
                placeholder="가입할 때 입력한 이름"
                onChange={(e) => setName(e.target.value)}
              />
            </div>
          </label>
          <label>
            생년월일
            <div className="field">
              <input
                type="date"
                value={birthdate}
                max={new Date().toISOString().slice(0, 10)}
                onChange={(e) => setBirthdate(e.target.value)}
              />
            </div>
          </label>
          {error && <p className="form-error">{error}</p>}
          <p className="field-hint">
            프로필에 생년월일을 입력하지 않은 계정은 확인할 방법이 없어
            찾을 수 없어요.
          </p>
        </div>
      )}
    </Shell>
  );
}
// 비밀번호 찾기 — 아이디+이름+생년월일이 모두 맞으면 새 비밀번호로 바로 재설정한다.
function FindPassword() {
  const nav = useNavigate();
  const [username, setUsername] = useState("");
  const [name, setName] = useState("");
  const [birthdate, setBirthdate] = useState("");
  const [pw, setPw] = useState("");
  const [pw2, setPw2] = useState("");
  const [done, setDone] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const ok =
    username.trim().length > 0 &&
    name.trim().length > 0 &&
    ageFromBirthdate(birthdate) != null &&
    pw.length >= 8 &&
    pw === pw2;
  const submit = async () => {
    if (!ok) {
      setError(
        pw.length > 0 && pw.length < 8
          ? "새 비밀번호는 8자 이상이어야 해요."
          : pw !== pw2
            ? "새 비밀번호가 서로 일치하지 않아요."
            : "아이디 · 이름 · 생년월일을 모두 입력해주세요.",
      );
      return;
    }
    setBusy(true);
    setError("");
    try {
      await apiFetch("/auth/reset-password", {
        method: "POST",
        body: JSON.stringify({
          username: username.trim(),
          name: name.trim(),
          birthdate,
          new_password: pw,
        }),
      });
      setDone(true);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Shell
      header={false}
      footer={
        done ? (
          <Button onClick={() => nav("/login")}>로그인하러 가기</Button>
        ) : (
          <Button disabled={busy} onClick={submit}>
            {busy ? "변경 중..." : "새 비밀번호로 바꾸기"}
          </Button>
        )
      }
    >
      <BackHeader title="비밀번호 찾기" onBack={() => nav("/login")} />
      {done ? (
        <>
          <h1>비밀번호를 바꿨어요.</h1>
          <div className="result-box">
            <b>새 비밀번호로 로그인해주세요</b>
            <span>
              보안을 위해 다른 기기에 로그인돼 있던 기록은 모두 해제했어요.
            </span>
          </div>
        </>
      ) : (
        <>
          <h1>
            본인 확인 후
            <br />
            새 비밀번호를 정해주세요.
          </h1>
          <div className="form">
            <label>
              아이디
              <div className="field">
                <input
                  value={username}
                  placeholder="아이디"
                  onChange={(e) => setUsername(e.target.value)}
                />
              </div>
            </label>
            <label>
              이름
              <div className="field">
                <input
                  value={name}
                  placeholder="가입할 때 입력한 이름"
                  onChange={(e) => setName(e.target.value)}
                />
              </div>
            </label>
            <label>
              생년월일
              <div className="field">
                <input
                  type="date"
                  value={birthdate}
                  max={new Date().toISOString().slice(0, 10)}
                  onChange={(e) => setBirthdate(e.target.value)}
                />
              </div>
            </label>
            <label>
              새 비밀번호
              <div className="field">
                <input
                  type="password"
                  value={pw}
                  placeholder="8자 이상"
                  onChange={(e) => setPw(e.target.value)}
                />
              </div>
            </label>
            <label>
              새 비밀번호 확인
              <div className="field">
                <input
                  type="password"
                  value={pw2}
                  placeholder="한 번 더 입력하세요"
                  onChange={(e) => setPw2(e.target.value)}
                />
              </div>
            </label>
            {error && <p className="form-error">{error}</p>}
          </div>
        </>
      )}
    </Shell>
  );
}
function Signup() {
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [passwordCheck, setPasswordCheck] = useState("");
  const [agree, setAgree] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  // 각 조건을 개별로 검사해서, 왜 버튼이 막혀 있는지 사용자가 바로 알 수 있게 한다.
  const usernameOk = /^[a-zA-Z0-9_.-]{4,30}$/.test(username.trim());
  const passwordOk = password.length >= 8;
  const passwordMatchOk = password.length > 0 && password === passwordCheck;
  const nameOk = name.trim().length > 0;
  const canSubmit = usernameOk && passwordOk && passwordMatchOk && nameOk && agree;
  const submit = async () => {
    if (!canSubmit) {
      const missing: string[] = [];
      if (!nameOk) missing.push("이름");
      if (!usernameOk) missing.push("아이디(영문/숫자 4자 이상)");
      if (!passwordOk) missing.push("비밀번호(8자 이상)");
      if (!passwordMatchOk) missing.push("비밀번호 확인 일치");
      if (!agree) missing.push("약관 동의");
      setError(`다음을 확인해주세요: ${missing.join(" · ")}`);
      return;
    }
    setBusy(true);
    setError("");
    try {
      const d = await apiFetch("/auth/signup", {
        method: "POST",
        body: JSON.stringify({
          name: name.trim(),
          username: username.trim(),
          password,
        }),
      });
      saveSession(d);
      // 회원가입 직후엔 반드시 프로필(신체정보) 입력 화면으로 넘어간다.
      nav("/profile");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Shell
      header={false}
      footer={
        <Button disabled={busy} onClick={submit}>
          {busy ? "계정 생성 중..." : "다음: 프로필 입력"}
        </Button>
      }
    >
      <BackHeader title="회원가입" onBack={() => nav("/login")} />
      <StepHeader step={1} total={2} />
      <p className="eyebrow">1단계 · 아이디</p>
      <h1>
        사용하실 아이디를
        <br />
        만들어주세요.
      </h1>
      <div className="form">
        <label>
          이름
          <div className="field">
            <input
              value={name}
              placeholder="이름"
              onChange={(e) => setName(e.target.value)}
            />
          </div>
        </label>
        <label>
          아이디
          <div className="field">
            <input
              value={username}
              placeholder="영문/숫자/._- 4~30자"
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          {username.length > 0 && !usernameOk && (
            <small className="field-hint warn">
              영문, 숫자, ._- 만 사용해 4~30자로 입력해주세요.
            </small>
          )}
        </label>
        <label>
          비밀번호
          <div className="field">
            <input
              type="password"
              value={password}
              placeholder="8자 이상"
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          {password.length > 0 && !passwordOk && (
            <small className="field-hint warn">8자 이상 입력해주세요.</small>
          )}
        </label>
        <label>
          비밀번호 확인
          <div className="field">
            <input
              type="password"
              value={passwordCheck}
              placeholder="비밀번호를 한 번 더 입력하세요"
              onChange={(e) => setPasswordCheck(e.target.value)}
            />
          </div>
          {passwordCheck.length > 0 && !passwordMatchOk && (
            <small className="field-hint warn">
              비밀번호가 일치하지 않아요.
            </small>
          )}
        </label>
        <label className="check-row">
          <input
            type="checkbox"
            checked={agree}
            onChange={(e) => setAgree(e.target.checked)}
          />
          <span>
            <b>필수 약관에 동의합니다.</b>
            <small>
              서비스 이용약관 및 개인정보 처리방침(키·몸무게 등 신체정보는
              맞춤형 식단 계산에만 사용돼요)
            </small>
          </span>
        </label>
        {error && <p className="form-error">{error}</p>}
      </div>
      <p className="auth-switch">
        이미 계정이 있나요? <button onClick={() => nav("/login")}>로그인</button>
      </p>
    </Shell>
  );
}
function ProfileSetup() {
  const nav = useNavigate();
  const { profile, setProfile } = useApp();
  const [draft, setDraft] = useState(profile);
  const [pd, setPd] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const computedAge = ageFromBirthdate(draft.birthdate);
  const heightOk = Number(draft.height) >= 100 && Number(draft.height) <= 250;
  const weightOk = Number(draft.weight) >= 20 && Number(draft.weight) <= 300;
  const valid = !!draft.gender && computedAge != null && heightOk && weightOk;
  const finish = async () => {
    setBusy(true);
    setError("");
    try {
      if (authToken())
        await apiFetch("/me/profile", {
          method: "PUT",
          body: JSON.stringify({
            gender: draft.gender,
            birthdate: draft.birthdate,
            height: Number(draft.height),
            weight: Number(draft.weight),
            dialysis: draft.dialysis,
          }),
        });
      const withAge = { ...draft, age: String(computedAge ?? draft.age) };
      setProfile(withAge);
      storage.set("fook:profile", withAge);
      nav("/home");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <Shell
      header={false}
      footer={
        <Button disabled={!valid || busy} onClick={finish}>
          {busy ? "저장 중..." : "프로필 입력 완료"}
        </Button>
      }
    >
      <BackHeader title="프로필 입력" />
      <StepHeader step={2} total={2} />
      <p className="eyebrow">2단계 · 프로필 입력</p>
      <h1>
        맞춤형 식단을 만들려면
        <br />키·몸무게가 필요해요.
      </h1>
      <p className="sub">
        섭취 가능한 열량·칼륨·나트륨 기준이 체격에 따라 달라지기 때문에,
        정확하게 입력할수록 식단이 더 안전하게 맞춰져요.
      </p>
      <div className="form">
        <label>
          성별
          <div className="segments">
            {["여성", "남성"].map((g) => (
              <button
                key={g}
                className={draft.gender === g ? "segment active" : "segment"}
                onClick={() => setDraft({ ...draft, gender: g })}
              >
                {g}
              </button>
            ))}
          </div>
        </label>
        <label>
          생년월일
          <div className="field">
            <input
              type="date"
              value={draft.birthdate}
              max={new Date().toISOString().slice(0, 10)}
              onChange={(e) =>
                setDraft({ ...draft, birthdate: e.target.value })
              }
            />
          </div>
          {draft.birthdate && computedAge == null && (
            <small className="field-hint warn">
              올바른 생년월일을 입력해주세요.
            </small>
          )}
          {computedAge != null && (
            <small className="field-hint">만 {computedAge}세</small>
          )}
          <small className="field-hint">
            아이디·비밀번호를 잊었을 때 본인 확인에도 쓰여요.
          </small>
        </label>
        <label>
          키
          <div className="field">
            <input
              inputMode="decimal"
              value={draft.height}
              onChange={(e) => setDraft({ ...draft, height: e.target.value })}
            />
            <span>cm</span>
          </div>
          {draft.height && !heightOk && (
            <small className="field-hint warn">
              100~250cm 범위로 입력해주세요.
            </small>
          )}
        </label>
        <label>
          체중
          <div className="field">
            <input
              inputMode="decimal"
              value={draft.weight}
              onChange={(e) => setDraft({ ...draft, weight: e.target.value })}
            />
            <span>kg</span>
          </div>
          {draft.weight && !weightOk && (
            <small className="field-hint warn">
              20~300kg 범위로 입력해주세요.
            </small>
          )}
        </label>
        {error && <p className="form-error">{error}</p>}
      </div>
      <h2 className="section-title">투석 유형</h2>
      <div className="dialysis-cards">
        <button className="dialysis-card selected">
          <div className="medical-icon">HD</div>
          <b>혈액투석</b>
          <span>현재 이용 가능</span>
          <i>✓</i>
        </button>
        <button className="dialysis-card" onClick={() => setPd(true)}>
          <div className="medical-icon">PD</div>
          <b>복막투석</b>
          <span>개발 중</span>
        </button>
      </div>
      {pd && (
        <div className="modal-bg">
          <div className="modal">
            <h2>복막투석은 현재 개발 중입니다.</h2>
            <p>현재는 혈액투석 환자용 식단만 지원합니다.</p>
            <Button onClick={() => setPd(false)}>확인</Button>
          </div>
        </div>
      )}
    </Shell>
  );
}
function Home() {
  const nav = useNavigate();
  const { query, setQuery, setPlan, searchMode, setSearchMode, setApiResult } =
    useApp();
  const [menuList, setMenuList] = useState<string[]>([]);
  const [ingList, setIngList] = useState<string[]>([]);
  // 데이터 출처 배지는 요청에 따라 화면에서 뺐다. 목록 로딩만 그대로 둔다.
  // 재료 검색 모드에서 '이 재료가 들어간 메뉴' 목록. 재료를 고르면 채워진다.
  const [ingQuery, setIngQuery] = useState("");
  const [ingMenus, setIngMenus] = useState<string[] | null>(null);
  const [ingLoading, setIngLoading] = useState(false);
  const [randomAsk, setRandomAsk] = useState(false);
  const user = currentUser();
  useEffect(() => {
    Promise.all([apiFetch("/menus"), apiFetch("/ingredients")])
      .then(([m, i]) => {
        setMenuList(m.menus || []);
        setIngList(i.ingredients || []);
      })
      .catch(() => {
        // 서버 미연결이면 로컬 예시 데이터로 자동 폴백된다(아래 names/ingredients).
      });
  }, []);
  const localMenus = menuData.map((m) => m.name);
  const names = menuList.length ? menuList : localMenus;
  const suggestions = (
    query.trim()
      ? names.filter((n) => n.includes(query.trim()))
      : names.filter((n) => /된장국|구이|볶음/.test(n))
  ).slice(0, 8);
  const ingredients = ingList.length
    ? ingList
    : ingredientData.map((x) => x.name);
  const ingSuggestions = ingredients
    .filter((x) => x.includes(ingQuery.trim()))
    .slice(0, 8);
  const choose = (name: string) => {
    setQuery(name);
    const p = dietPlans.find((x) => x.menus.includes(name)) as Plan | undefined;
    if (p) setPlan(p);
  };
  // 재료를 하나 고르면, 그 재료가 실제로 들어간 메뉴를 서버에서 찾아 보여준다.
  const pickIngredient = (ing: string) => {
    setIngQuery(ing);
    setIngMenus(null);
    setIngLoading(true);
    apiFetch(`/menus_by_ingredient?q=${encodeURIComponent(ing)}`)
      .then((d) => setIngMenus(d.menus || []))
      .catch(() => setIngMenus([]))
      .finally(() => setIngLoading(false));
  };
  const chooseFromIngredient = (menuName: string) => {
    setQuery(menuName);
    setSearchMode("menu"); // /generate에는 결국 menu 조건으로 넘어간다
    const p = dietPlans.find((x) => x.menus.includes(menuName)) as
      Plan | undefined;
    if (p) setPlan(p);
  };
  const canGenerate =
    searchMode === "random" ||
    (searchMode === "menu" && query.trim().length > 0) ||
    (searchMode === "ingredient" && query.trim().length > 0);
  return (
    <Shell
      footer={
        // 하단 탭바는 화면 맨 아래에 고정하고, 그 바로 위에 선택/생성 버튼을 둔다.
        <>
          <Button
            disabled={!canGenerate}
            onClick={() => {
              setApiResult(null);
              nav("/generating");
            }}
          >
            {canGenerate
              ? "선택한 조건으로 식단 생성하기"
              : searchMode === "ingredient"
                ? "재료가 들어간 메뉴를 선택해주세요"
                : "메뉴를 검색하거나 선택해주세요"}
          </Button>
          <BottomNav active="home" />
        </>
      }
    >
      <h1 className="greeting">
        안녕하세요, <b>{user?.name || "회원"}님!</b>
      </h1>
      <p className="sub">
        건강한 한 끼, <b className="brand-word">KOOK</b>이 함께합니다.
      </p>
      {/* 목업 1: 음식 검색 / 재료 검색 / 랜덤 추천 3분할 카드 */}
      <div className="quick-cards">
        <button
          className={searchMode === "menu" ? "quick-card active" : "quick-card"}
          onClick={() => setSearchMode("menu")}
        >
          <span className="quick-icon">
            <SearchIcon />
          </span>
          <b>음식 검색</b>
          <small>메뉴명을 검색해보세요</small>
        </button>
        <button
          className={
            searchMode === "ingredient" ? "quick-card active" : "quick-card"
          }
          onClick={() => {
            setSearchMode("ingredient");
            setIngMenus(null);
            setIngQuery("");
          }}
        >
          <span className="quick-icon">
            <LeafIcon />
          </span>
          <b>재료 검색</b>
          <small>재료로 메뉴를 찾아보세요</small>
        </button>
        <button
          className={
            searchMode === "random" ? "quick-card active" : "quick-card"
          }
          onClick={() => {
            setSearchMode("random");
            setQuery("");
            setRandomAsk(true); // 랜덤 탭은 안내문 대신 확인 팝업을 띄운다
          }}
        >
          <span className="quick-icon">
            <DiceIcon />
          </span>
          <b>랜덤 추천</b>
          <small>AI가 랜덤으로 추천해드려요</small>
        </button>
      </div>
      {searchMode === "menu" && (
        <>
          <div className="search">
            <SearchIcon />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="메뉴명을 검색해보세요 (예: 두부, 계란찜, 소고기)"
            />
            <span className="search-filter">
              <SlidersIcon />
            </span>
          </div>
          <div className="result-head">
            <h2 className="section-title">음식 검색 결과</h2>
            <span className="sort-pill">최신순 ⌄</span>
          </div>
          <div className="pick-list">
            {suggestions.map((n) => (
              <article
                key={n}
                className={query === n ? "pick-row active" : "pick-row"}
              >
                <b>{n}</b>
                <button onClick={() => choose(n)}>
                  {query === n ? "선택됨" : "선택하기"}
                </button>
              </article>
            ))}
            {suggestions.length === 0 && (
              <div className="info-box">
                <b>검색 결과가 없어요</b>
                <span>다른 메뉴명으로 다시 검색해보세요.</span>
              </div>
            )}
          </div>
        </>
      )}
      {searchMode === "ingredient" && (
        <>
          <div className="search">
            <SearchIcon />
            <input
              value={ingQuery}
              onChange={(e) => {
                setIngQuery(e.target.value);
                setIngMenus(null);
                setQuery("");
              }}
              placeholder="사용하고 싶은 재료를 검색하세요"
            />
          </div>
          {ingMenus === null ? (
            <>
              <div className="result-head">
                <h2 className="section-title">재료 검색 결과</h2>
              </div>
              <div className="pick-list">
                {ingSuggestions.map((n) => (
                  <article className="pick-row" key={n}>
                    <b>{n}</b>
                    <button onClick={() => pickIngredient(n)}>메뉴 보기</button>
                  </article>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="ingredient-picked-row">
                <span>
                  '<b>{ingQuery}</b>'가 들어간 메뉴
                </span>
                <button
                  className="link-button"
                  onClick={() => {
                    setIngMenus(null);
                    setQuery("");
                  }}
                >
                  다시 검색
                </button>
              </div>
              {ingLoading && (
                <div className="recipe-loading">
                  <div className="spinner" />
                  <span>메뉴를 찾고 있어요...</span>
                </div>
              )}
              {!ingLoading && ingMenus.length === 0 && (
                <div className="info-box">
                  <b>일치하는 메뉴를 찾지 못했어요</b>
                  <span>다른 재료로 다시 검색해보세요.</span>
                </div>
              )}
              <div className="pick-list">
                {ingMenus.slice(0, 20).map((n) => (
                  <article
                    key={n}
                    className={query === n ? "pick-row active" : "pick-row"}
                  >
                    <span className="pick-main">
                      <b>{n}</b>
                      <small>{ingQuery} 포함</small>
                    </span>
                    <button onClick={() => chooseFromIngredient(n)}>
                      {query === n ? "선택됨" : "선택하기"}
                    </button>
                  </article>
                ))}
              </div>
            </>
          )}
        </>
      )}
      {searchMode === "random" && !randomAsk && (
        <button className="random-again" onClick={() => setRandomAsk(true)}>
          🎲 랜덤 추천 다시 받기
        </button>
      )}
      {/* 랜덤 추천 확인 팝업: 왼쪽 아니오 / 오른쪽 네 */}
      {randomAsk && (
        <div className="modal-bg" onClick={() => setRandomAsk(false)}>
          <div className="modal ask" onClick={(e) => e.stopPropagation()}>
            <span className="modal-mark">🎲</span>
            <h2>랜덤으로 식단 추천 해드릴까요?</h2>
            <p>
              먹고 싶은 음식을 고르지 않아도
              <br />
              KOOK AI가 영양 기준에 맞는 한 끼를 만들어드려요.
            </p>
            <div className="ask-actions">
              <button
                className="ask-no"
                onClick={() => {
                  setRandomAsk(false);
                  setSearchMode("menu");
                }}
              >
                아니오
              </button>
              <button
                className="ask-yes"
                onClick={() => {
                  setRandomAsk(false);
                  setApiResult(null);
                  nav("/generating");
                }}
              >
                네
              </button>
            </div>
          </div>
        </div>
      )}
    </Shell>
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
        const d = await apiFetch("/generate", {
          method: "POST",
          body: JSON.stringify(body),
          timeoutMs: 60000,
        });
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
function MealListRow({
  name,
  role,
  onClick,
}: {
  name: string;
  role: string;
  onClick: () => void;
}) {
  return (
    <button className="meal-row" onClick={onClick}>
      <span className="meal-row-role">{role}</span>
      <b className="meal-row-name">{name}</b>
      <i className="meal-row-arrow">›</i>
    </button>
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
// 영양소 하나의 상태를 판정한다. lo가 0이면(칼륨·인·나트륨) 상한만 본다.
type NStatus = "미만" | "적절" | "초과";
function statusOf(v: number, lo: number, hi: number): NStatus {
  if (lo > 0 && v < lo) return "미만";
  return v > hi ? "초과" : "적절";
}
const STATUS_CLASS: Record<NStatus, string> = {
  미만: "under",
  적절: "ok",
  초과: "over",
};
// 영양소 목록. 목업 요청 반영:
//  · 원 그래프(도넛) 없음
//  · 한 줄에 "아이콘 이름 / 수치 (기준 얼마) / 판정 뱃지"
//  · 기준을 넘으면 카드 전체가 빨간 배경, 적절하면 초록 배경
function Nutrients({
  values,
  targets,
  isFallback = false,
}: {
  values: ReturnType<typeof totalNutrition>;
  targets?: any;
  // true면 실제 서버 판정이 아니라 내장 예시 데이터라는 뜻 — 뱃지를 '적절/초과' 대신
  // 중립적인 '예시'로 표시해서, 개인 맞춤 판정처럼 보이지 않게 한다.
  isFallback?: boolean;
}) {
  return (
    <div className="nutri-list">
      {nmeta.map((n) => {
        const v = Number(values?.[n.key] || 0);
        const hi = targetOf(targets, n.key);
        const lo = minTargetOf(targets, n.key);
        const status = statusOf(v, lo, hi);
        const cls = isFallback ? "demo" : STATUS_CLASS[status];
        return (
          <div className={`nutri-card ${cls}`} key={n.key}>
            <span className="nutri-icon">{n.icon}</span>
            <span className="nutri-body">
              <b className="nutri-name">{n.label}</b>
              <span className="nutri-value">
                {fmt(v)}
                <small> {n.unit}</small>
                <em className="nutri-target">
                  {lo > 0
                    ? `(기준 ${fmt(lo)}~${fmt(hi)} ${n.unit})`
                    : `(기준 ${fmt(hi)} ${n.unit} 이하)`}
                </em>
              </span>
            </span>
            <i className={`nutri-badge ${cls}`}>{isFallback ? "예시" : status}</i>
          </div>
        );
      })}
    </div>
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
function MealSlotDialog({
  onCancel,
  onConfirm,
}: {
  onCancel: () => void;
  onConfirm: (date: string, time: MealTime) => void;
}) {
  const [date, setDate] = useState(todayISO());
  const [time, setTime] = useState<MealTime>(defaultMealTime());
  return (
    <div className="modal-bg" onClick={onCancel}>
      <div className="modal ask slot" onClick={(e) => e.stopPropagation()}>
        <span className="modal-mark">🍽</span>
        <h2>언제 먹은 식단인가요?</h2>
        <p>
          날짜와 끼니를 고르면
          <br />
          식단 관리의 해당 섹션에 기록돼요.
        </p>
        <label className="slot-date">
          <span>날짜</span>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </label>
        <div className="slot-times">
          {MEAL_TIMES.map((t) => (
            <button
              key={t}
              type="button"
              className={`slot-chip${t === time ? " on" : ""}`}
              aria-pressed={t === time}
              onClick={() => setTime(t)}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="ask-actions">
          <button className="ask-no" onClick={onCancel}>
            취소
          </button>
          <button className="ask-yes" onClick={() => onConfirm(date, time)}>
            기록하기
          </button>
        </div>
      </div>
    </div>
  );
}
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
function toSteps(v: unknown): string[] {
  if (Array.isArray(v)) return v.map((x) => String(x).trim()).filter(Boolean);
  if (typeof v === "string")
    return v
      .split(/\r?\n/)
      // 화면이 번호를 따로 붙이므로 원문 앞의 번호(1. / 1) / ① / - 등)는 떼어낸다
      .map((line) =>
        line.replace(/^\s*(\d+\s*[.)]|[①-⑳]|[-*·])\s*/, "").trim(),
      )
      .filter(Boolean);
  return [];
}
// 레시피 본문(재료 · 조리과정 · 영양성분 · 음성안내).
// 목록에서 음식을 고르면 화면을 옮기지 않고 이 컴포넌트만 그 자리에서 펼친다.
function RecipeBody({ menuName }: { menuName: string }) {
  const { apiResult } = useApp();
  // 서버가 이 끼를 생성하며 실제로 계산한 재료(dish_ingredients)를 최우선으로 쓴다.
  // 재료 교체로 이름이 바뀐 메뉴는 recipe_source에 원래 이름이 있어 조리법은 원본 기준 조회.
  const serverIngredients: [string, number][] | undefined =
    apiResult?.dish_ingredients?.[menuName];
  const recipeSourceName: string | undefined =
    apiResult?.recipe_source?.[menuName];
  const m = menuMap.get(menuName); // 서버 데이터가 없을 때 쓰는 로컬 폴백
  const ingredients: [string, number][] =
    serverIngredients || (m?.ingredients || []).map(parseLocalIngredient);
  const nutrition = apiResult?.dish_nutrition?.[menuName] || m?.nutrition;
  const isServerNutrition = !!apiResult?.dish_nutrition?.[menuName];

  const [steps, setSteps] = useState<string[] | null>(null);
  const [loadingRecipe, setLoadingRecipe] = useState(false);
  const [recipeError, setRecipeError] = useState("");
  // 음성 안내 팝업에서 지금 읽고 있는 단계 (-1 = 팝업 닫힘)
  const [voiceStep, setVoiceStep] = useState(-1);
  const speech = useSpeech();

  useEffect(() => {
    setSteps(null);
    setRecipeError("");
    if (!serverIngredients || !serverIngredients.length) return;
    let live = true;
    setLoadingRecipe(true);
    apiFetch("/recipe", {
      method: "POST",
      body: JSON.stringify({
        menu: menuName,
        ingredients: serverIngredients,
        source: recipeSourceName,
      }),
    })
      .then((d) => {
        if (!live) return;
        if (d.error) setRecipeError(d.error);
        else setSteps(toSteps(d.steps));
      })
      .catch(() => {
        if (live) setRecipeError("조리법을 불러오지 못했어요.");
      })
      .finally(() => live && setLoadingRecipe(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [menuName]);

  const displaySteps: string[] =
    steps ?? (m?.steps?.length ? toSteps(m.steps) : []);
  const recipeSpeech = [
    `${menuName} 조리 순서입니다.`,
    ...displaySteps.map((s, i) => `${i + 1}번. ${s}`),
  ].join(" ");
  // 서버 데이터도 로컬 데이터도 없는 메뉴(주소로 직접 들어온 경우 등)
  const noData =
    ingredients.length === 0 && displaySteps.length === 0 && !loadingRecipe;

  return (
    <>
      {recipeSourceName && recipeSourceName !== menuName && (
        <p className="recipe-note">
          재료 대체로 '{recipeSourceName}' 레시피를 기준으로 조리법을 조정했어요.
        </p>
      )}
      {noData ? (
        <div className="info-box">
          <b>이 메뉴의 레시피 정보가 없어요</b>
          <span>
            식단을 새로 만들면 이번 끼에 쓰인 재료와 조리과정을 볼 수 있어요.
          </span>
        </div>
      ) : (
        <>
          <h2 className="section-title">재료</h2>
          <div className="kv-table">
            {ingredients.map(([ing, amt], i) => (
              <div className="kv-row" key={`${ing}-${i}`}>
                <span>{ing}</span>
                <b>{amt > 0 ? `${fmt2(amt)} g` : "적당량"}</b>
              </div>
            ))}
            {ingredients.length === 0 && (
              <div className="kv-row">
                <span>등록된 재료가 없습니다.</span>
                <b>-</b>
              </div>
            )}
          </div>
          <h2 className="section-title">
            조리과정
            {steps && <span className="tag-ai">AI 편집</span>}
          </h2>
          {loadingRecipe && (
            <div className="recipe-loading">
              <div className="spinner" />
              <span>AI가 회원님 식단에 맞춰 조리법을 다듬고 있어요...</span>
            </div>
          )}
          {!loadingRecipe && recipeError && (
            <p className="form-error">{recipeError}</p>
          )}
          {!loadingRecipe && (
            <ol className="steps table-steps">
              {(displaySteps.length
                ? displaySteps
                : ["등록된 조리과정이 없습니다."]
              ).map((x, i) => (
                <li key={i} className="step-reveal">
                  <span>{i + 1}</span>
                  <p>{x}</p>
                </li>
              ))}
            </ol>
          )}
          <h2 className="section-title">영양 성분</h2>
          <div className="dish-nutri">
            {nmeta.map((n) => (
              <div key={n.key}>
                <span>{n.icon}</span>
                <small>{n.label}</small>
                <b>
                  {fmt(Number((nutrition as any)?.[n.key] || 0))}
                  <i>{n.unit}</i>
                </b>
              </div>
            ))}
          </div>
          <p className="dish-nutri-note">
            {isServerNutrition
              ? "이번 식단에 실제로 쓰인 재료·양 기준으로 계산했어요."
              : "표준 1인분 기준 예시값이에요."}
          </p>
          {displaySteps.length > 0 && (
            <div className="tts-row right">
              <button
                className="tts-button"
                onClick={() => {
                  setVoiceStep(0);
                  speech.speak(`${menuName} 조리 순서입니다. 1번. ${displaySteps[0]}`);
                }}
              >
                <SpeakerIcon />
                {" 음성 안내 받기"}
              </button>
            </div>
          )}
          {/* 음성 안내 팝업 — 조리 단계를 하나씩 읽어준다 */}
          {voiceStep >= 0 && (
            <div
              className="modal-bg"
              onClick={() => {
                speech.stop();
                setVoiceStep(-1);
              }}
            >
              <div className="modal voice" onClick={(e) => e.stopPropagation()}>
                <div className="voice-head">
                  <span className="voice-count">
                    {voiceStep + 1} / {displaySteps.length}
                  </span>
                  <button
                    className="voice-close"
                    aria-label="닫기"
                    onClick={() => {
                      speech.stop();
                      setVoiceStep(-1);
                    }}
                  >
                    ✕
                  </button>
                </div>
                <p className="voice-menu">{menuName}</p>
                <div className="voice-body">
                  <span className="voice-num">{voiceStep + 1}</span>
                  <p>{displaySteps[voiceStep]}</p>
                </div>
                <div className="ask-actions">
                  <button
                    className="ask-no"
                    onClick={() =>
                      speech.speak(
                        `${voiceStep + 1}번. ${displaySteps[voiceStep]}`,
                      )
                    }
                  >
                    {speech.speaking ? "다시 듣기" : "다시 듣기"}
                  </button>
                  {voiceStep < displaySteps.length - 1 ? (
                    <button
                      className="ask-yes"
                      onClick={() => {
                        const next = voiceStep + 1;
                        setVoiceStep(next);
                        speech.speak(`${next + 1}번. ${displaySteps[next]}`);
                      }}
                    >
                      다음 →
                    </button>
                  ) : (
                    <button
                      className="ask-yes"
                      onClick={() => {
                        speech.stop();
                        setVoiceStep(-1);
                      }}
                    >
                      완료
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
// 메뉴 목록 + 고른 메뉴의 레시피를 '같은 화면에서' 펼쳐 보여준다 (화면 이동 없음).
function RecipeList({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (n: string) => void;
}) {
  const { plan } = useApp();
  return (
    <>
      <h2 className="section-title tight">식단 메뉴</h2>
      <p className="sub small">
        메뉴를 선택하면 재료와 조리과정을 확인할 수 있어요.
      </p>
      <div className="recipe-menu-list">
        {plan.menus.map((n, i) => {
          const open = n === selected;
          return (
            <div key={n}>
              <button
                className={open ? "recipe-menu-row active" : "recipe-menu-row"}
                onClick={() => onSelect(open ? "" : n)}
                aria-expanded={open}
              >
                <span className="rm-num">{i + 1}</span>
                <span className="rm-txt">
                  <b>{n}</b>
                  <small>{roleLong(i)}</small>
                </span>
                <i className={open ? "rm-caret open" : "rm-caret"}>›</i>
              </button>
              {open && (
                <section className="recipe-inline">
                  <RecipeBody menuName={n} />
                </section>
              )}
            </div>
          );
        })}
      </div>
    </>
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
function DayPlan() {
  const nav = useNavigate();
  const { profile } = useApp();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [data, setData] = useState<any>(null);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    const body: any = { weight: Number(profile.weight) || 60 };
    if (profile.height) {
      body.height = Number(profile.height);
      body.sex = profile.gender === "남성" ? "남" : "여";
    }
    apiFetch("/generate_day", {
      method: "POST",
      body: JSON.stringify(body),
      timeoutMs: 90000,
    })
      .then((d) => live && setData(d))
      .catch(() => live && setError("서버에 연결하지 못했어요. 백엔드가 켜져 있는지 확인해주세요."))
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <Shell header={false}>
      <BackHeader title="하루 식단" />
      <p className="eyebrow">아침 · 점심 · 저녁</p>
      <h1>
        하루치 식단을
        <br />
        한 번에 만들었어요.
      </h1>
      <p className="sub">
        끼니마다 앞서 먹은 양을 반영해서, 하루 전체 영양 총량이 기준 안에
        들어오도록 이어서 계산해요.
      </p>
      {loading && (
        <div className="loading-page">
          <div className="loader-plate">
            <span className="loader-emoji">🍲</span>
            <div className="orbit" />
          </div>
          <p className="loading-message">
            아침 · 점심 · 저녁을 순서대로 계산하고 있어요
          </p>
          <p className="loading-hint">세 끼를 이어서 계산하느라 평소보다 오래 걸려요.</p>
        </div>
      )}
      {!loading && error && (
        <div className="warning-box">
          <b>{error}</b>
        </div>
      )}
      {!loading && data && (
        <>
          {data.meals.map((m: any, i: number) => (
            <section key={i} className="change-section">
              <div className="change-heading">
                <span>{m.label?.slice(0, 1) || i + 1}</span>
                <div>
                  <b>{m.label}</b>
                  <small>{m.meal?.join(" · ")}</small>
                </div>
              </div>
              <div className="meal-list">
                {(m.meal || []).map((name: string, j: number) => (
                  <MealListRow
                    key={name}
                    name={name}
                    role={labels[j] || "구성"}
                    onClick={() => nav(`/recipe/${encodeURIComponent(name)}`)}
                  />
                ))}
              </div>
            </section>
          ))}
          <h2 className="section-title">하루 총 영양</h2>
          <Nutrients
            values={data.day_nutrition}
            targets={data.day_targets}
          />
        </>
      )}
      <Button onClick={() => nav("/home")}>홈으로</Button>
    </Shell>
  );
}
function Tips() {
  const nav = useNavigate();
  const [tips, setTips] = useState<
    { category: string; steps: { title: string; detail: string }[] }[] | null
  >(null);
  const [error, setError] = useState("");
  useEffect(() => {
    apiFetch("/veg_potassium_tips")
      .then((d) => setTips(d.tips || []))
      .catch(() => setError("팁을 불러오지 못했어요."));
  }, []);
  return (
    <Shell header={false}>
      <BackHeader title="칼륨 낮추는 조리 팁" />
      <p className="eyebrow">채소 손질 가이드</p>
      <h1>
        채소의 칼륨을
        <br />
        줄이는 방법이에요.
      </h1>
      {error && <p className="form-error">{error}</p>}
      {!tips && !error && (
        <div className="recipe-loading">
          <div className="spinner" />
          <span>불러오는 중...</span>
        </div>
      )}
      {tips?.map((cat) => (
        <section key={cat.category} className="change-section">
          <div className="change-heading">
            <div>
              <b>{cat.category}</b>
            </div>
          </div>
          <ol className="steps">
            {cat.steps.map((s, i) => (
              <li key={i} className="step-reveal">
                <span>{i + 1}</span>
                <p>
                  <b>{s.title}</b> — {s.detail}
                </p>
              </li>
            ))}
          </ol>
        </section>
      ))}
      <Button onClick={() => nav("/account")}>내 정보로 돌아가기</Button>
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
function BottomNav({
  active,
}: {
  active: "home" | "history" | "favorites" | "account";
}) {
  const nav = useNavigate();
  // 목업 하단 탭: 홈 / 식단 관리 / 프로필 3개
  const items = [
    ["home", <HomeIcon key="h" />, "홈", "/home"],
    ["favorites", <ClipboardIcon key="c" />, "식단 관리", "/favorites"],
    ["account", <UserIcon key="u" />, "프로필", "/account"],
  ] as const;
  return (
    <nav className="bottom-nav three">
      {items.map(([k, icon, label, path]) => (
        <button
          key={k}
          className={active === k ? "active" : ""}
          onClick={() => nav(path)}
        >
          <i>{icon}</i>
          <span>{label}</span>
        </button>
      ))}
    </nav>
  );
}
function Account() {
  const nav = useNavigate();
  const { profile, setProfile } = useApp();
  const user = currentUser();
  if (!user) return <Navigate to="/login" replace />;
  // 계정 화면에 들어올 때마다 서버의 최신 프로필을 한 번 불러와 화면에 반영한다.
  // (다른 기기에서 수정했거나, 세션이 오래된 경우에도 항상 최신 값을 보여주기 위함)
  useEffect(() => {
    apiFetch("/me")
      .then((d) => {
        if (!d?.profile) return;
        const p = d.profile;
        setProfile({
          ...profile,
          gender: p.gender || profile.gender,
          // 생년월일까지 받아둬야 '수정'으로 프로필 화면에 들어갔을 때 값이 채워져 있다.
          birthdate: p.birthdate || profile.birthdate,
          age: p.age != null ? String(p.age) : profile.age,
          height: p.height_cm != null ? String(p.height_cm) : profile.height,
          weight: p.weight_kg != null ? String(p.weight_kg) : profile.weight,
          dialysis: p.dialysis_type || profile.dialysis,
        });
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const logout = async () => {
    try {
      await apiFetch("/auth/logout", { method: "POST" });
    } catch {}
    localStorage.removeItem("fook:user");
    localStorage.removeItem("fook:token");
    nav("/start");
  };
  return (
    <Shell header={false} footer={<BottomNav active="account" />}>
      <BackHeader title="내 정보" />
      <div className="profile-summary">
        <div className="avatar">{user.name.slice(0, 1)}</div>
        <div>
          <h2>{user.name}님</h2>
          <p>@{user.username}</p>
        </div>
        <button onClick={() => nav("/profile")}>수정</button>
      </div>
      <div className="metric-grid">
        <div>
          <b>{profile.age}세</b>
          <span>나이</span>
        </div>
        <div>
          <b>{profile.height}cm</b>
          <span>신장</span>
        </div>
        <div>
          <b>{profile.weight}kg</b>
          <span>체중</span>
        </div>
        <div>
          <b>{profile.dialysis}</b>
          <span>투석 유형</span>
        </div>
      </div>
      <div className="menu-list">
        <button onClick={() => nav("/history")}>
          <span>식단 기록</span>
          <i>›</i>
        </button>
        <button onClick={() => nav("/favorites")}>
          <span>식단 관리</span>
          <i>›</i>
        </button>
        <button onClick={() => nav("/documents")}>
          <span>PDF 보관함</span>
          <i>›</i>
        </button>
        <button onClick={() => nav("/tips")}>
          <span>칼륨 낮추는 조리 팁</span>
          <i>›</i>
        </button>
      </div>
      <button className="logout" onClick={logout}>
        로그아웃
      </button>
    </Shell>
  );
}
// 저장 목록의 카드 한 장 (식단 기록 / 식단 관리 / PDF 보관함 공통)
function SavedCard({
  item,
  mode,
  onOpen,
  onRemove,
}: {
  item: SavedItem;
  mode: "history" | "favorites" | "documents";
  onOpen: (x: SavedItem) => void;
  onRemove: (id: string) => void;
}) {
  return (
    <article>
      <div className="saved-thumb">{mode === "documents" ? "PDF" : "KOOK"}</div>
      <button className="saved-main" onClick={() => onOpen(item)}>
        <b>{item.title}</b>
        <span>{item.subtitle}</span>
        <small>
          {item.mealDate
            ? new Date(item.mealDate).toLocaleDateString("ko-KR")
            : new Date(item.createdAt).toLocaleDateString("ko-KR")}
        </small>
      </button>
      <button className="delete-mini" onClick={() => onRemove(item.id)}>
        ×
      </button>
    </article>
  );
}
function LibraryPage({
  mode,
}: {
  mode: "history" | "favorites" | "documents";
}) {
  const nav = useNavigate();
  if (!currentUser()) return <Navigate to="/login" replace />;
  const key = `fook:${mode}`;
  const [items, setItems] = useState<SavedItem[]>(storage.get(key, []));
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    setLoading(true);
    loadEverywhere(key).then((list) => {
      if (live) {
        setItems(list);
        setLoading(false);
      }
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);
  const meta = {
    history: ["식단 기록", "최근 생성하고 기록한 식단"],
    favorites: ["식단 관리", "기록한 식단을 모아두고 다시 불러올 수 있어요"],
    documents: ["PDF 보관함", "생성한 레시피 문서 기록"],
  }[mode];
  const remove = async (id: string) => {
    setItems((prev) => prev.filter((x) => x.id !== id));
    await deleteEverywhere(key, id);
  };
  const openItem = (x: SavedItem) => {
    if (x.menus) {
      storage.set("fook:restore", x);
      nav("/home");
    }
  };
  return (
    <Shell
      header={false}
      footer={<BottomNav active={mode === "favorites" ? "favorites" : "history"} />}
    >
      <BackHeader title={meta[0]} />
      <p className="eyebrow">MY KOOK</p>
      <h1>{meta[0]}</h1>
      <p className="sub">{meta[1]}</p>
      {loading && (
        <div className="recipe-loading">
          <div className="spinner" />
          <span>불러오는 중...</span>
        </div>
      )}
      {!loading && items.length > 0 && mode === "favorites" && (
        // 식단 관리는 아침 / 점심 / 저녁 섹션으로 한 줄씩 나눠서 보여준다
        <>
          {MEAL_TIMES.map((t) => (
            <section className="meal-slot-section" key={t}>
              <h2 className="section-title">
                {t}
                <span className="slot-count">
                  {items.filter((x) => x.mealTime === t).length}
                </span>
              </h2>
              {items.some((x) => x.mealTime === t) ? (
                <div className="saved-list">
                  {items
                    .filter((x) => x.mealTime === t)
                    .map((x) => (
                      <SavedCard
                        key={x.id}
                        item={x}
                        mode={mode}
                        onOpen={openItem}
                        onRemove={remove}
                      />
                    ))}
                </div>
              ) : (
                <p className="slot-empty">아직 {t} 기록이 없어요.</p>
              )}
            </section>
          ))}
          {/* 끼니를 고르기 전에 저장한 예전 기록 */}
          {items.some((x) => !x.mealTime) && (
            <section className="meal-slot-section">
              <h2 className="section-title">끼니 미지정</h2>
              <div className="saved-list">
                {items
                  .filter((x) => !x.mealTime)
                  .map((x) => (
                    <SavedCard
                      key={x.id}
                      item={x}
                      mode={mode}
                      onOpen={openItem}
                      onRemove={remove}
                    />
                  ))}
              </div>
            </section>
          )}
        </>
      )}
      {!loading && items.length > 0 && mode !== "favorites" && (
        <div className="saved-list">
          {items.map((x) => (
            <SavedCard
              key={x.id}
              item={x}
              mode={mode}
              onOpen={openItem}
              onRemove={remove}
            />
          ))}
        </div>
      )}
      {!loading && items.length === 0 && (
        <div className="empty-state">
          <div>
            {mode === "favorites" ? "♡" : mode === "documents" ? "PDF" : "◷"}
          </div>
          <b>아직 기록된 항목이 없어요.</b>
          <p>맞춤 식단을 생성한 뒤 기록해보세요.</p>
          <Button onClick={() => nav("/home")}>식단 만들러 가기</Button>
        </div>
      )}
    </Shell>
  );
}
export default App;
