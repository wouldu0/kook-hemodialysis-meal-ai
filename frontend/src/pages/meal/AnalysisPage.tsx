import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { DocIcon, RefreshIcon } from "../../components/icons";
import { Nutrients } from "../../components/meal/Nutrients";
import { NutrientIconRow } from "../../components/meal/NutrientIconRow";
import { useApp } from "../../hooks/useApp";
import { displayTarget, displayValue, minTargetOf, nmeta, totalNutrition } from "../../utils/nutrition";

export function AnalysisPage() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  // 이 화면은 '식단을 생성한 그대로'의 영양을 보여준다. 재구성은 다음 단계에서 하므로
  // 조정 후 수치(apiResult.nutrition)가 아니라 조정 전 수치(nutrition_before)를 쓴다.
  const raw =
    apiResult?.nutrition_before || apiResult?.nutrition || totalNutrition(plan);
  const targets = apiResult?.targets;
  // 게이지(Nutrients)와 동일한 기준: 상한 초과뿐 아니라 하한 미달도 '조정 필요'로 본다.
  // (에너지·단백질은 서버가 [최소,최대] 범위를 주므로 미달도 실제로 의미가 있다.)
  const over = nmeta.filter((n) => {
    const v = displayValue(raw, n.key);
    const hi = displayTarget(targets, n.key);
    const lo = minTargetOf(targets, n.key);
    return v > hi || (lo > 0 && v < lo);
  });
  const changeCount = apiResult?.changes?.length || 0;
  // 이제 이 화면은 '조정 전' 수치를 보여주므로, 초과 항목이 있으면 그대로 '재구성하러 가기'다.
  //  needFix : 기준을 벗어난 항목이 있다 → 재구성 단계로
  //  clean   : 처음부터 전부 기준 안 → 볼 조정 내역이 없으면 최종 식단으로 바로
  const state: "needFix" | "clean" = over.length > 0 ? "needFix" : "clean";
  const goNext = () =>
    nav(state === "needFix" || changeCount > 0 ? "/adjusting" : "/final");
  return (
    <Shell
      header={false}
      footer={
        <button className="btn" onClick={goNext}>
          <i className="btn-icon">
            {state === "needFix" ? <RefreshIcon /> : <DocIcon />}
          </i>{" "}
          {state === "needFix"
            ? "레시피 재구성하러 가기"
            : changeCount > 0
              ? `재구성 내역 보기 (${changeCount}건)`
              : "최종 식단 보기"}
        </button>
      }
    >
      <BackHeader onBack={() => nav("/meal")} dot />
      <h1 className="analysis-title">
        영양 적합성 <b>판정 결과</b>
      </h1>
      <p className="sub">
        개인 프로필을 기반으로
        <br />
        영양 섭취 적합 여부를 판정했습니다.
      </p>
      <NutrientIconRow />
      <Nutrients values={raw} targets={targets} isFallback={!apiResult} />
      {/* 하단 안내 카드 — 상태에 따라 색과 문구가 달라진다 */}
      <div className={"notice-card " + state}>
        <i className="notice-mark">{state === "needFix" ? "!" : "✓"}</i>
        <div>
          <b>
            {state === "needFix"
              ? over.map((x) => x.label).join(" · ") + " — 레시피 재구성 필요"
              : "레시피 재구성 필요 없음"}
          </b>
          <span>
            {state === "needFix"
              ? "지금은 식단을 생성한 그대로의 영양이에요. 다음 단계에서 메뉴의 양과 식재료를 조정해 기준에 맞춰드릴게요."
              : "생성한 그대로 모든 항목이 기준 안에 있어요."}
          </span>
        </div>
      </div>
    </Shell>
  );
}
