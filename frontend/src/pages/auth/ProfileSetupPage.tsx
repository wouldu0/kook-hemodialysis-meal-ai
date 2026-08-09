import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { StepHeader } from "../../components/layout/StepHeader";
import { useApp } from "../../hooks/useApp";
import { authToken, storage, updateProfile } from "../../services/api";
import { ageFromBirthdate } from "../../utils/date";

export function ProfileSetupPage() {
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
        await updateProfile({
          gender: draft.gender,
          birthdate: draft.birthdate,
          height: Number(draft.height),
          weight: Number(draft.weight),
          dialysis: draft.dialysis,
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
