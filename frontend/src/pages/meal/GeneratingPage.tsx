import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Shell } from "../../components/layout/Shell";
import { useApp } from "../../hooks/useApp";
import {
  authToken,
  buildTodayMealContext,
  generateMeal,
  isBackendReady,
  loadEverywhere,
  warmupBackend,
} from "../../services/api";
import type { Plan } from "../../types";

export function GeneratingPage() {
  const nav = useNavigate();
  const { profile, query, searchMode, setApiResult, setPlan, setUsingFallback } =
    useApp();
  const [s, setS] = useState(0);
  const [error, setError] = useState("");
  const [warming, setWarming] = useState(!isBackendReady());
  const [warmupRetry, setWarmupRetry] = useState(false);
  // 오늘 기록한 식사가 이번 요청에 실제로 반영됐을 때만 보여줄 한 줄 힌트.
  const [todayHint, setTodayHint] = useState("");
  const msgs = [
    "메뉴와 재료 데이터를 불러오고 있어요",
    "회원님의 키·몸무게 기준으로 영양 목표를 계산하고 있어요",
    "조건에 맞는 조합을 찾을 때까지 반복해서 시도하고 있어요",
    "칼륨·나트륨 등 위험 수치를 낮추기 위해 양과 재료를 조정하고 있어요",
    "최종 식단을 정리하고 있어요",
  ];

  useEffect(() => {
    let live = true;
    const controller = new AbortController();
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
        // Render Free가 잠든 상태라면 /generate부터 보내지 않고, 앱 진입 때 시작된 동일
        // /health warm-up Promise가 끝날 때까지 기다린다. 첫 120초 안에 준비되지 않아도
        // 바로 예시 데이터로 포기하지 않고 한 번 더 /health를 시도한다. warmupBackend()는
        // 실패 시 Promise 캐시를 비우므로 두 번째 호출은 실제 재시도가 된다.
        if (!isBackendReady()) setWarming(true);
        try {
          await warmupBackend();
        } catch (firstWarmupError) {
          if (!live) return;
          console.warn("백엔드 첫 준비 확인이 시간 초과되어 한 번 더 확인합니다.", firstWarmupError);
          setWarmupRetry(true);
          await warmupBackend();
        }
        if (!live) return;
        setWarmupRetry(false);
        setWarming(false);
        setS(0);

        // 로그인한 회원만 서버/로컬 식사 기록을 다음 추천에 반영한다. 비회원 체험에서는
        // 브라우저에 예전 회원의 로컬 기록이 남아 있더라도 읽지 않아 계정 데이터가 섞이지 않는다.
        if (authToken()) {
          try {
            const history = await loadEverywhere("fook:history");
            const ctx = buildTodayMealContext(history);
            if (ctx.consumed) body.consumed = ctx.consumed;
            if (ctx.used_today) body.used_today = ctx.used_today;
            body.meals_left = ctx.mealsLeft;
            if (ctx.consumed || ctx.used_today) {
              setTodayHint(
                `오늘 기록한 식사 ${ctx.usedSlotCount}끼를 반영해 남은 영양 기준으로 추천하고 있어요.`,
              );
            }
          } catch (histErr) {
            console.warn("오늘 식사 기록을 불러오지 못해 기본값으로 진행합니다.", histErr);
          }
        }

        if (!live) return;
        const d = await generateMeal(body, { signal: controller.signal });
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
        if (!live || controller.signal.aborted) return;
        console.error("식단 생성 준비/요청 실패", e);
        setError(
          "서버 준비가 오래 걸리거나 연결에 실패해 내장 예시 데이터로 진행합니다. 잠시 후 다시 시도해주세요.",
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
      controller.abort();
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
        <h1>{warming ? "AI 서버를 준비하고 있어요." : "한 끼를 생성하고 있어요."}</h1>
        <div className="loader-plate">
          <span className="loader-emoji">🍲</span>
          <div className="orbit" />
        </div>
        <p className={error ? "loading-message warn" : "loading-message"}>
          {error ||
            (warming
              ? warmupRetry
                ? "서버가 거의 준비됐는지 한 번 더 확인하고 있어요. 조금만 더 기다려주세요."
                : "무료 데모 서버는 첫 실행 시 1~2분 정도 걸릴 수 있어요."
              : msgs[s])}
        </p>
        <div className="progress">
          <i
            style={{
              width: warming ? (warmupRetry ? "35%" : "20%") : `${((s + 1) / msgs.length) * 100}%`,
            }}
          />
        </div>
        {!error && !warming && todayHint && (
          <p className="loading-hint">{todayHint}</p>
        )}
        {!error && !warming && !todayHint && s >= 2 && (
          <p className="loading-hint">
            영양 기준에 딱 맞는 조합을 찾는 중이라 조금 더 걸릴 수 있어요.
          </p>
        )}
      </div>
    </Shell>
  );
}
