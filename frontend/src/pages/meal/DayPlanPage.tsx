import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "../../components/layout/Button";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { MealListRow } from "../../components/meal/MealListRow";
import { Nutrients } from "../../components/meal/Nutrients";
import { useApp } from "../../hooks/useApp";
import { generateDayPlan } from "../../services/api";
import type { DayPlanResult } from "../../types";
import { labels } from "../../utils/menu";

export function DayPlanPage() {
  const nav = useNavigate();
  const { profile } = useApp();
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [data, setData] = useState<DayPlanResult | null>(null);
  useEffect(() => {
    let live = true;
    setLoading(true);
    setError("");
    const body: any = { weight: Number(profile.weight) || 60 };
    if (profile.height) {
      body.height = Number(profile.height);
      body.sex = profile.gender === "남성" ? "남" : "여";
    }
    generateDayPlan(body)
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
          {data.meals.map((m, i) => (
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
              {m.warning && (
                <div className="warning-box">
                  <b>⚠ 나트륨 안내</b>
                  <span>{m.warning}</span>
                </div>
              )}
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
