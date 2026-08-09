import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { FlowFooter } from "../../components/layout/FlowFooter";
import { Shell } from "../../components/layout/Shell";
import { StepHeader } from "../../components/layout/StepHeader";
import { ClipboardIcon } from "../../components/icons";
import { useApp } from "../../hooks/useApp";
import { parseChange } from "../../utils/nutrition";
import type { ParsedChange } from "../../utils/nutrition";

export function AdjustingPage() {
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
            {p >= 1 ? "메뉴는 그대로 두고 재료만 조정하고 있어요." : " "}
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
