import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { findId } from "../../services/api";
import { ageFromBirthdate } from "../../utils/date";

// 아이디 찾기 — 이름 + 생년월일로 본인 확인 (이메일 발송 수단이 없는 서비스라서)
export function FindIdPage() {
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
      const d = await findId(name.trim(), birthdate);
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
