import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { resetPassword } from "../../services/api";
import { ageFromBirthdate } from "../../utils/date";

// 비밀번호 찾기 — 아이디+이름+생년월일이 모두 맞으면 새 비밀번호로 바로 재설정한다.
export function FindPasswordPage() {
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
      await resetPassword(username.trim(), name.trim(), birthdate, pw);
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
