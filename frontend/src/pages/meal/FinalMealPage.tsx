import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { FlowFooter } from "../../components/layout/FlowFooter";
import { Shell } from "../../components/layout/Shell";
import { BookmarkIcon, DocIcon, HomeIcon } from "../../components/icons";
import { MealSlotDialog } from "../../components/meal/MealSlotDialog";
import { Nutrients } from "../../components/meal/Nutrients";
import { RecipeList } from "../../components/meal/RecipeList";
import { useApp } from "../../hooks/useApp";
import { currentUser, saveEverywhere } from "../../services/api";
import type { MealTime } from "../../types";
import { requireUser } from "../../utils/auth";
import { adjustedNutrition, totalNutrition } from "../../utils/nutrition";

// 식단을 기록할 때 날짜와 끼니를 고르는 팝업.
// 여기서 고른 값에 따라 '식단 관리'의 아침/점심/저녁 섹션으로 들어간다.
export function FinalMealPage() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  const item = {
    id: `meal-${Date.now()}`,
    title: plan.menus[1] || "맞춤 한 끼",
    subtitle: plan.menus.join(" · "),
    createdAt: new Date().toISOString(),
    menus: plan.menus,
  };
  const [savingKey, setSavingKey] = useState<string | null>(null);
  const [openRecipe, setOpenRecipe] = useState(""); // 인라인으로 펼친 레시피
  const [askJoin, setAskJoin] = useState(false); // 비회원 체험 종료 시 회원가입 안내
  const [askSlot, setAskSlot] = useState<{ key: string; msg: string } | null>(
    null,
  ); // 날짜·끼니 선택 팝업
  // 식단 기록은 날짜·끼니를 먼저 고르게 하고, 그 외(PDF 등)는 바로 저장한다.
  const save = async (key: string, msg: string) => {
    if (!requireUser(nav)) return;
    if (key === "fook:favorites") return setAskSlot({ key, msg });
    setSavingKey(key);
    try {
      await saveEverywhere(key, item);
      alert(msg);
    } finally {
      setSavingKey(null);
    }
  };
  const saveWithSlot = async (date: string, time: MealTime) => {
    if (!askSlot) return;
    const { key, msg } = askSlot;
    setAskSlot(null);
    setSavingKey(key);
    try {
      await saveEverywhere(key, { ...item, mealDate: date, mealTime: time });
      alert(`${date} ${time} 식단으로 ${msg}`);
    } finally {
      setSavingKey(null);
    }
  };
  const finalValues =
    apiResult?.nutrition || adjustedNutrition(totalNutrition(plan));
  return (
    <Shell
      header={false}
      footer={
        <FlowFooter
          step={5}
          total={5}
          onPrev={() => nav("/comparison")}
          // 비회원 체험이면 여기서 끝내지 않고 회원가입 안내를 먼저 띄운다.
          onNext={() => (currentUser() ? nav("/home") : setAskJoin(true))}
          nextLabel="완료"
        />
      }
    >
      <BackHeader onBack={() => nav("/comparison")} dot />
      <div className="hero-center">
        <h1>
          오늘의 한 끼, <b>완성!</b>
        </h1>
        <p className="sub center">
          오늘의 한 끼가 완성되었어요.
          <br />
          선택한 식단의 <b>재료와 조리과정</b>을 확인해보세요.
        </p>
      </div>
      <Nutrients
        values={finalValues}
        targets={apiResult?.targets}
        isFallback={!apiResult}
      />
      {/* 하단 탭: 기록 저장 · 홈 · PDF 다운로드 (장바구니 탭은 뺐다) */}
      <div className="final-tabs">
        <button
          disabled={savingKey === "fook:favorites"}
          onClick={() => save("fook:favorites", "기록된 식단에 추가했어요.")}
        >
          <BookmarkIcon />
          <span>{savingKey === "fook:favorites" ? "기록 중..." : "기록하기"}</span>
        </button>
        <button onClick={() => nav("/home")}>
          <HomeIcon />
          <span>홈</span>
        </button>
        <button onClick={() => nav("/pdf")}>
          <DocIcon />
          <span>PDF 다운로드</span>
        </button>
      </div>
      <h2 className="section-title with-icon">📖 레시피 보러가기</h2>
      {/* 새 화면으로 넘어가지 않고 고른 음식의 레시피를 이 자리에서 펼친다 */}
      <RecipeList selected={openRecipe} onSelect={setOpenRecipe} />
      {/* 비회원 체험을 마쳤을 때만 뜨는 회원가입 안내 */}
      {askSlot && (
        <MealSlotDialog
          onCancel={() => setAskSlot(null)}
          onConfirm={saveWithSlot}
        />
      )}
      {askJoin && (
        <div className="modal-bg" onClick={() => setAskJoin(false)}>
          <div className="modal ask" onClick={(e) => e.stopPropagation()}>
            <span className="modal-mark">🍚</span>
            <h2>
              회원가입하면 개인 맞춤형 식단을
              <br />
              제공받을 수 있어요
            </h2>
            <p>
              키 체중, 투석 유형을 등록하면
              <br />
              내 기준에 맞춘 영양 계산과 식단 관리 등을 쓸 수 있어요
            </p>
            <div className="ask-actions">
              <button className="ask-no" onClick={() => nav("/login")}>
                나중에
              </button>
              <button className="ask-yes" onClick={() => nav("/login")}>
                회원가입 하러가기
              </button>
            </div>
          </div>
        </div>
      )}
    </Shell>
  );
}
