import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { BookmarkIcon, DocIcon, HomeIcon } from "../../components/icons";
import { MealSlotDialog } from "../../components/meal/MealSlotDialog";
import { RecipeBody } from "../../components/meal/RecipeBody";
import { RecipeList } from "../../components/meal/RecipeList";
import { useApp } from "../../hooks/useApp";
import { saveEverywhere } from "../../services/api";
import type { MealTime } from "../../types";
import { requireUser } from "../../utils/auth";

export function RecipePage() {
  const nav = useNavigate();
  const { name = "" } = useParams();
  const { plan } = useApp();
  const decoded = decodeURIComponent(name);
  const [selected, setSelected] = useState(decoded);
  const [askSlot, setAskSlot] = useState(false); // 날짜·끼니 선택 팝업
  const saveMeal = () => {
    if (!requireUser(nav)) return;
    setAskSlot(true);
  };
  const saveWithSlot = async (date: string, time: MealTime) => {
    setAskSlot(false);
    await saveEverywhere("fook:favorites", {
      id: `meal-${Date.now()}`,
      title: plan.menus[1] || decoded,
      subtitle: plan.menus.join(" · "),
      createdAt: new Date().toISOString(),
      menus: plan.menus,
      mealDate: date,
      mealTime: time,
    });
    alert(`${date} ${time} 식단으로 기록했어요.`);
  };
  return (
    <Shell
      header={false}
      footer={
        <div className="footer-actions three">
          <button onClick={() => nav("/home")}>
            <HomeIcon /> 홈으로
          </button>
          <button onClick={saveMeal}>
            <BookmarkIcon /> 기록하기
          </button>
          <button onClick={() => nav("/pdf")}>
            <DocIcon /> PDF 다운로드
          </button>
        </div>
      }
    >
      <BackHeader title="레시피 보러가기" dot />
      {/* 이번 식단 목록에 없는 메뉴로 직접 들어온 경우에도 그 메뉴를 보여준다 */}
      {decoded && !plan.menus.includes(decoded) && (
        <>
          <div className="recipe-detail-head">
            <span className="rm-num">1</span>
            <h1 className="recipe-title">{decoded}</h1>
          </div>
          <RecipeBody menuName={decoded} />
          <h2 className="section-title">이번 한 끼의 다른 메뉴</h2>
        </>
      )}
      <RecipeList selected={selected} onSelect={setSelected} />
      {askSlot && (
        <MealSlotDialog
          onCancel={() => setAskSlot(false)}
          onConfirm={saveWithSlot}
        />
      )}
    </Shell>
  );
}
