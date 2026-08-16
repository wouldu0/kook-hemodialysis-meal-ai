import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Logo } from "../../components/Logo";
import { UserIcon } from "../../components/icons";
import { Shell } from "../../components/layout/Shell";
import { useApp } from "../../hooks/useApp";
import { getMe, login, saveSession } from "../../services/api";
import { fallbackPlan } from "../../utils/menu";

export function LoginPage() {
  const nav = useNavigate();
  const {
    setProfile,
    setPlan,
    setQuery,
    setSearchMode,
    setApiResult,
    setUsingFallback,
  } = useApp();
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
      const d = await login(username.trim(), password);
      saveSession(d);

      // 로그인 직후 서버의 실제 회원 프로필을 다시 읽는다. 비회원 체험을 먼저 했거나
      // 페이지를 새로 연 경우에도 가상/기본 프로필이 다음 식단 계산에 남지 않게 한다.
      try {
        const me = await getMe();
        const p = me?.profile;
        if (p) {
          setProfile({
            gender: p.gender || "",
            birthdate: p.birthdate || "",
            age: p.age != null ? String(p.age) : "",
            height: p.height_cm != null ? String(p.height_cm) : "",
            weight: p.weight_kg != null ? String(p.weight_kg) : "",
            dialysis: p.dialysis_type || "혈액투석",
          });
        }
      } catch {
        // 인증은 됐지만 개인 프로필을 못 읽으면 맞춤 식단을 잘못 계산할 수 있으므로
        // 불완전한 세션을 남기지 않고 로그인부터 다시 시도하게 한다.
        localStorage.removeItem("fook:user");
        localStorage.removeItem("fook:token");
        throw new Error("회원 정보를 불러오지 못했습니다. 잠시 후 다시 로그인해주세요.");
      }

      // 비회원 체험/이전 화면에서 남은 메뉴·결과 상태를 회원 홈으로 가져오지 않는다.
      sessionStorage.removeItem("fook:guest");
      setQuery("");
      setSearchMode("menu");
      setPlan(fallbackPlan);
      setApiResult(null);
      setUsingFallback(false);
      nav("/home");
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };
  const tryGuest = () => {
    // 이전에 이 브라우저에서 로그인했던 정보가 localStorage에 남아 있어도 비회원 체험에는
    // 절대 섞이지 않게 한다. 서버의 계정/기록 자체를 지우는 것이 아니라 이 브라우저가
    // 들고 있던 인증 토큰만 내려놓는다.
    localStorage.removeItem("fook:user");
    localStorage.removeItem("fook:token");
    sessionStorage.setItem("fook:guest", "1");

    // 모달에서 안내한 가상 프로필과 실제 생성 요청에 쓰는 값이 정확히 같아야 한다.
    setProfile({
      gender: "남성",
      birthdate: "",
      age: "65",
      height: "170",
      weight: "60",
      dialysis: "혈액투석",
    });
    setApiResult(null);
    setPlan(fallbackPlan);
    setSearchMode("menu");
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
