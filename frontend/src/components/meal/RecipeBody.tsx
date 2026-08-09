import { useEffect, useState } from "react";
import { useApp } from "../../hooks/useApp";
import { useSpeech } from "../../hooks/useSpeech";
import { generateRecipe } from "../../services/api";
import { menuMap, parseLocalIngredient } from "../../utils/menu";
import { fmt, fmt2, nmeta } from "../../utils/nutrition";
import { SpeakerIcon } from "../icons";

function toSteps(v: unknown): string[] {
  if (Array.isArray(v)) return v.map((x) => String(x).trim()).filter(Boolean);
  if (typeof v === "string")
    return v
      .split(/\r?\n/)
      // 화면이 번호를 따로 붙이므로 원문 앞의 번호(1. / 1) / ① / - 등)는 떼어낸다
      .map((line) =>
        line.replace(/^\s*(\d+\s*[.)]|[①-⑳]|[-*·])\s*/, "").trim(),
      )
      .filter(Boolean);
  return [];
}

// 레시피 본문(재료 · 조리과정 · 영양성분 · 음성안내).
// 목록에서 음식을 고르면 화면을 옮기지 않고 이 컴포넌트만 그 자리에서 펼친다.
export function RecipeBody({ menuName }: { menuName: string }) {
  const { apiResult } = useApp();
  // 서버가 이 끼를 생성하며 실제로 계산한 재료(dish_ingredients)를 최우선으로 쓴다.
  // 재료 교체로 이름이 바뀐 메뉴는 recipe_source에 원래 이름이 있어 조리법은 원본 기준 조회.
  const serverIngredients: [string, number][] | undefined =
    apiResult?.dish_ingredients?.[menuName];
  const recipeSourceName: string | undefined =
    apiResult?.recipe_source?.[menuName];
  const m = menuMap.get(menuName); // 서버 데이터가 없을 때 쓰는 로컬 폴백
  const ingredients: [string, number][] =
    serverIngredients || (m?.ingredients || []).map(parseLocalIngredient);
  const nutrition = apiResult?.dish_nutrition?.[menuName] || m?.nutrition;
  const isServerNutrition = !!apiResult?.dish_nutrition?.[menuName];

  const [steps, setSteps] = useState<string[] | null>(null);
  const [loadingRecipe, setLoadingRecipe] = useState(false);
  const [recipeError, setRecipeError] = useState("");
  // 음성 안내 팝업에서 지금 읽고 있는 단계 (-1 = 팝업 닫힘)
  const [voiceStep, setVoiceStep] = useState(-1);
  const speech = useSpeech();

  useEffect(() => {
    setSteps(null);
    setRecipeError("");
    if (!serverIngredients || !serverIngredients.length) return;
    let live = true;
    setLoadingRecipe(true);
    generateRecipe({
      menu: menuName,
      ingredients: serverIngredients,
      source: recipeSourceName,
    })
      .then((d) => {
        if (!live) return;
        if (d.error) setRecipeError(d.error);
        else setSteps(toSteps(d.steps));
      })
      .catch(() => {
        if (live) setRecipeError("조리법을 불러오지 못했어요.");
      })
      .finally(() => live && setLoadingRecipe(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [menuName]);

  const displaySteps: string[] =
    steps ?? (m?.steps?.length ? toSteps(m.steps) : []);
  const recipeSpeech = [
    `${menuName} 조리 순서입니다.`,
    ...displaySteps.map((s, i) => `${i + 1}번. ${s}`),
  ].join(" ");
  // 서버 데이터도 로컬 데이터도 없는 메뉴(주소로 직접 들어온 경우 등)
  const noData =
    ingredients.length === 0 && displaySteps.length === 0 && !loadingRecipe;

  return (
    <>
      {recipeSourceName && recipeSourceName !== menuName && (
        <p className="recipe-note">
          재료 대체로 '{recipeSourceName}' 레시피를 기준으로 조리법을 조정했어요.
        </p>
      )}
      {noData ? (
        <div className="info-box">
          <b>이 메뉴의 레시피 정보가 없어요</b>
          <span>
            식단을 새로 만들면 이번 끼에 쓰인 재료와 조리과정을 볼 수 있어요.
          </span>
        </div>
      ) : (
        <>
          <h2 className="section-title">재료</h2>
          <div className="kv-table">
            {ingredients.map(([ing, amt], i) => (
              <div className="kv-row" key={`${ing}-${i}`}>
                <span>{ing}</span>
                <b>{amt > 0 ? `${fmt2(amt)} g` : "적당량"}</b>
              </div>
            ))}
            {ingredients.length === 0 && (
              <div className="kv-row">
                <span>등록된 재료가 없습니다.</span>
                <b>-</b>
              </div>
            )}
          </div>
          <h2 className="section-title">
            조리과정
            {steps && <span className="tag-ai">AI 편집</span>}
          </h2>
          {loadingRecipe && (
            <div className="recipe-loading">
              <div className="spinner" />
              <span>AI가 회원님 식단에 맞춰 조리법을 다듬고 있어요...</span>
            </div>
          )}
          {!loadingRecipe && recipeError && (
            <p className="form-error">{recipeError}</p>
          )}
          {!loadingRecipe && (
            <ol className="steps table-steps">
              {(displaySteps.length
                ? displaySteps
                : ["등록된 조리과정이 없습니다."]
              ).map((x, i) => (
                <li key={i} className="step-reveal">
                  <span>{i + 1}</span>
                  <p>{x}</p>
                </li>
              ))}
            </ol>
          )}
          <h2 className="section-title">영양 성분</h2>
          <div className="dish-nutri">
            {nmeta.map((n) => (
              <div key={n.key}>
                <span>{n.icon}</span>
                <small>{n.label}</small>
                <b>
                  {fmt(Number((nutrition as any)?.[n.key] || 0))}
                  <i>{n.unit}</i>
                </b>
              </div>
            ))}
          </div>
          <p className="dish-nutri-note">
            {isServerNutrition
              ? "이번 식단에 실제로 쓰인 재료·양 기준으로 계산했어요."
              : "표준 1인분 기준 예시값이에요."}
          </p>
          {displaySteps.length > 0 && (
            <div className="tts-row right">
              <button
                className="tts-button"
                onClick={() => {
                  setVoiceStep(0);
                  speech.speak(`${menuName} 조리 순서입니다. 1번. ${displaySteps[0]}`);
                }}
              >
                <SpeakerIcon />
                {" 음성 안내 받기"}
              </button>
            </div>
          )}
          {/* 음성 안내 팝업 — 조리 단계를 하나씩 읽어준다 */}
          {voiceStep >= 0 && (
            <div
              className="modal-bg"
              onClick={() => {
                speech.stop();
                setVoiceStep(-1);
              }}
            >
              <div className="modal voice" onClick={(e) => e.stopPropagation()}>
                <div className="voice-head">
                  <span className="voice-count">
                    {voiceStep + 1} / {displaySteps.length}
                  </span>
                  <button
                    className="voice-close"
                    aria-label="닫기"
                    onClick={() => {
                      speech.stop();
                      setVoiceStep(-1);
                    }}
                  >
                    ✕
                  </button>
                </div>
                <p className="voice-menu">{menuName}</p>
                <div className="voice-body">
                  <span className="voice-num">{voiceStep + 1}</span>
                  <p>{displaySteps[voiceStep]}</p>
                </div>
                <div className="ask-actions">
                  <button
                    className="ask-no"
                    onClick={() =>
                      speech.speak(
                        `${voiceStep + 1}번. ${displaySteps[voiceStep]}`,
                      )
                    }
                  >
                    {speech.speaking ? "다시 듣기" : "다시 듣기"}
                  </button>
                  {voiceStep < displaySteps.length - 1 ? (
                    <button
                      className="ask-yes"
                      onClick={() => {
                        const next = voiceStep + 1;
                        setVoiceStep(next);
                        speech.speak(`${next + 1}번. ${displaySteps[next]}`);
                      }}
                    >
                      다음 →
                    </button>
                  ) : (
                    <button
                      className="ask-yes"
                      onClick={() => {
                        speech.stop();
                        setVoiceStep(-1);
                      }}
                    >
                      완료
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </>
  );
}
