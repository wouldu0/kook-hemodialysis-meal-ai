import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { getPotassiumTips } from "../../services/api";

export function TipsPage() {
  const nav = useNavigate();
  const [tips, setTips] = useState<
    { category: string; steps: { title: string; detail: string }[] }[] | null
  >(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getPotassiumTips()
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
