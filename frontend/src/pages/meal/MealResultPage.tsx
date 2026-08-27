import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { BowlIcon, ChartIcon } from "../../components/icons";
import { MealListRow } from "../../components/meal/MealListRow";
import { useApp } from "../../hooks/useApp";
import { roleShort } from "../../utils/menu";

export function MealResultPage() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  // 이 화면은 "방금 생성된 그대로"의 식단을 보여준다 — 영양 판정(AnalysisPage)이 조정 전
  // 수치(nutrition_before)를 쓰는 것과 같은 원칙. plan.menus는 레버 재구성이 끝난 최종
  // 메뉴라 여기서 그대로 쓰면 재구성 안내(ComparisonPage)보다 먼저 바뀐 이름이 보이므로,
  // 레버가 손대기 전 슬롯별 메뉴명(raw_menus_display)이 있으면 그걸 우선 쓴다.
  const displayMenus =
    apiResult?.raw_menus_display && apiResult.raw_menus_display.length > 0
      ? apiResult.raw_menus_display
      : plan.menus;
  return (
    <Shell
      header={false}
      footer={
        <button className="btn" onClick={() => nav("/analysis")}>
          <i className="btn-icon">
            <ChartIcon />
          </i>{" "}
          영양 적합성 판정하기
        </button>
      }
    >
      <BackHeader onBack={() => nav("/home")} />
      <div className="hero-center">
        <span className="hero-mark">
          <BowlIcon />
        </span>
        <h1>
          선택한 음식과
          <br />
          어울리는 <b>식단을 생성했어요</b>
        </h1>
        <span className="compose-pill">
          구성 : <b>밥, 국, 밥찬 3가지</b>
        </span>
      </div>
      <div className="meal-list numbered">
        {displayMenus.map((name, i) => (
          <MealListRow
            key={name}
            name={name}
            role={roleShort(i)}
            onClick={() => nav(`/recipe/${encodeURIComponent(name)}`)}
          />
        ))}
      </div>
    </Shell>
  );
}
