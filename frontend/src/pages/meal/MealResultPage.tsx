import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { BowlIcon, ChartIcon } from "../../components/icons";
import { MealListRow } from "../../components/meal/MealListRow";
import { useApp } from "../../hooks/useApp";
import { roleShort } from "../../utils/menu";

export function MealResultPage() {
  const nav = useNavigate();
  const { plan } = useApp();
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
        {plan.menus.map((name, i) => (
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
