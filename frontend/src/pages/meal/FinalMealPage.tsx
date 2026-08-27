import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { FlowFooter } from "../../components/layout/FlowFooter";
import { Shell } from "../../components/layout/Shell";
import { ClipboardIcon, DocIcon } from "../../components/icons";
import { MealSlotDialog } from "../../components/meal/MealSlotDialog";
import { Nutrients } from "../../components/meal/Nutrients";
import { RecipeList } from "../../components/meal/RecipeList";
import { useApp } from "../../hooks/useApp";
import { currentUser, saveEverywhere } from "../../services/api";
import type { MealTime } from "../../types";
import { requireUser } from "../../utils/auth";
import { adjustedNutrition, totalNutrition } from "../../utils/nutrition";

// 식사를 기록할 때 날짜와 끼니를 고르는 팝업.
// 식단 전체(콤보) 찜은 없다 — 찜은 메뉴 단위로만 한다(RecipeList의 하트 버튼).
// 여기서는 실제 먹은 식사를 날짜·끼니와 함께 기록(fook:history)만 한다.
export function FinalMealPage() {
  const nav = useNavigate();
  const { plan, apiResult, setQuery, setSearchMode } = useApp();
  const item = {
    id: `meal-${Date.now()}`,
    title: plan.menus[1] || "맞춤 한 끼",
    subtitle: plan.menus.join(" · "),
    createdAt: new Date().toISOString(),
    menus: plan.menus,
    // /generate 응답을 그대로 함께 저장해둔다 — 나중에 consumed/used_today를
    // 연결할 때 재계산 없이 바로 쓸 수 있게 하기 위함 (재계산·Peff 추가 없음).
    raw_menus: apiResult?.raw_menus,
    intake: apiResult?.intake,
    dish_ingredients: apiResult?.dish_ingredients,
  };
  const [saving, setSaving] = useState(false);
  const [openRecipe, setOpenRecipe] = useState(""); // 인라인으로 펼친 레시피
  const [askJoin, setAskJoin] = useState(false); // 비회원 체험 종료 시 회원가입 안내
  const [askSlot, setAskSlot] = useState(false); // 날짜·끼니 선택 팝업

  const save = () => {
    if (!requireUser(nav)) return;
    setAskSlot(true);
  };

  const saveWithSlot = async (date: string, time: MealTime) => {
    setAskSlot(false);
    setSaving(true);
    try {
      await saveEverywhere("fook:history", { ...item, mealDate: date, mealTime: time });
      alert(`${date} ${time} 식사로 식사 기록에 추가했어요.`);
    } finally {
      setSaving(false);
    }
  };

  const finalValues =
    apiResult?.nutrition || adjustedNutrition(totalNutrition(plan));

  // "완료"로 홈에 돌아갈 땐 이번에 검색/선택했던 메뉴·재료가 다음 방문에도 남아있지
  // 않게 비워준다 — 로그인 직후(submit())·비회원 체험 시작(tryGuest())과 같은 처리다.
  const finish = () => {
    setQuery("");
    setSearchMode("menu");
    nav("/home");
  };

  return (
    <Shell
      header={false}
      footer={
        <FlowFooter
          step={5}
          total={5}
          onPrev={() => nav("/comparison")}
          // 비회원 체험이면 여기서 끝내지 않고 회원가입 안내를 먼저 띄운다.
          onNext={() => (currentUser() ? finish() : setAskJoin(true))}
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

      <div className="final-tabs">
        <button disabled={saving} onClick={save}>
          <ClipboardIcon />
          <span>{saving ? "기록 중..." : "기록하기"}</span>
        </button>
        <button onClick={() => nav("/pdf")}>
          <DocIcon />
          <span>PDF 다운로드</span>
        </button>
      </div>

      <h2 className="section-title with-icon">📖 레시피 보러가기</h2>
      {/* 새 화면으로 넘어가지 않고 고른 음식의 레시피를 이 자리에서 펼친다 */}
      <RecipeList selected={openRecipe} onSelect={setOpenRecipe} />

      {askSlot && (
        <MealSlotDialog
          onCancel={() => setAskSlot(false)}
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
