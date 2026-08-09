import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Shell } from "../../components/layout/Shell";
import { useApp } from "../../hooks/useApp";
import { generateMeal } from "../../services/api";
import type { Plan } from "../../types";

export function GeneratingPage() {
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
        const d = await generateMeal(body);
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
