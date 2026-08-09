import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { StepHeader } from "../../components/layout/StepHeader";
import { saveSession, signup } from "../../services/api";

export function SignupPage() {
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
      const d = await signup(name.trim(), username.trim(), password);
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
