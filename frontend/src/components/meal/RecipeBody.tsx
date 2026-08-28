import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../../hooks/useApp";
import { useSpeech } from "../../hooks/useSpeech";
import { generateRecipe } from "../../services/api";
import { menuMap, parseLocalIngredient } from "../../utils/menu";
import { speakOrdinal } from "../../utils/speech";
import { displayValue, fmt, fmt2, nmeta } from "../../utils/nutrition";
import { Button } from "../layout/Button";
import { SpeakerIcon } from "../icons";

// 서버 /recipe의 steps는 배열이 아니라 '여러 줄 문자열'로 올 수 있다
// (recipe_editor_FOOK.edit_recipe가 LLM 응답 텍스트를 그대로 반환한다).
// 어떤 형태로 와도 화면이 깨지지 않게 문자열 배열로 맞춰준다.
// PdfPreviewPage도 같은 응답 형태를 다뤄야 해서 이 파서를 그대로 가져다 쓴다.
export function toSteps(v: unknown): string[] {
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
// showMakeMealButton: 찜한 메뉴 상세(MenuDetailPage)에서만 켠다 — 방금 생성한 식단의
// 메뉴 목록(FinalMealPage/RecipePage)에서는 "이미 그 식단을 만든 상태"라 이 버튼이
// 어울리지 않는다.
export function RecipeBody({
  menuName,
  showMakeMealButton = false,
  snapshot,
  snapshotLoading = false,
  snapshotOnly = false,
  onStepsReady,
}: {
  menuName: string;
  showMakeMealButton?: boolean;
  // 찜한 메뉴(MenuDetailPage)처럼 "그 시점에 저장해둔" 재료·조리과정·영양을 전역 apiResult보다
  // 우선 써야 할 때 넘긴다. 없으면(방금 생성한 식단 화면 등) 기존처럼 apiResult/정적 데이터를 쓴다.
  snapshot?: {
    ingredients?: [string, number][];
    steps?: string[];
    nutrition?: Record<string, number>;
    recipeSource?: string;
  };
  // true면 snapshot을 아직 조회 중이라는 뜻(예: MenuDetailPage가 찜 목록을 불러오는 동안) —
  // 이때는 snapshot이 아직 undefined라고 해서 "스냅샷 없음"으로 단정하면 안 되므로, 전역
  // apiResult로 성급하게 조리법을 새로 생성하지 않고 snapshot이 확정될 때까지 기다린다.
  snapshotLoading?: boolean;
  // true면(MenuDetailPage 전용) snapshot에 없는 값을 전역 apiResult로 채우지 않는다 — apiResult는
  // "지금 세션에서 마지막으로 생성한 식단"일 뿐이라, 그 시점이 이 찜과 무관할 수 있다(예: 재료
  // 교체 전 이름이 우연히 같은 다른 메뉴가 최근에 생성됐다면 recipeSource가 엉뚱하게 섞여
  // 들어갈 수 있음). snapshot에도 없으면 정적 폴백(menuMap)까지만 내려간다.
  snapshotOnly?: boolean;
  // snapshot에 조리과정이 아직 없어서 이 컴포넌트가 새로 생성했을 때, 그 결과를 호출한 쪽(찜
  // 저장소)에 되돌려준다 — 다음에 봐도 똑같은 내용이 뜨도록 한 번만 저장해두기 위함.
  onStepsReady?: (steps: string[]) => void;
}) {
  const nav = useNavigate();
  const { apiResult, setDishSteps, setSearchMode, setQuery } = useApp();
  const makeMealWithThisMenu = () => {
    setSearchMode("menu");
    setQuery(menuName);
    nav("/generating");
  };
  // 우선순위: 찜 스냅샷 > (snapshotOnly가 아니면) 이번 세션에 실제로 계산한 재료(dish_ingredients)
  // > 정적 폴백. 재료 교체로 이름이 바뀐 메뉴는 recipe_source에 원래 이름이 있어 조리법은
  // 원본 기준 조회 — 이것도 snapshotOnly면 전역 apiResult를 보지 않는다(위 snapshotOnly 설명 참고).
  const serverIngredients: [string, number][] | undefined =
    snapshot?.ingredients ||
    (snapshotOnly ? undefined : apiResult?.dish_ingredients?.[menuName]);
  const recipeSourceName: string | undefined =
    snapshot?.recipeSource ||
    (snapshotOnly ? undefined : apiResult?.recipe_source?.[menuName]);
  const m = menuMap.get(menuName); // 서버 데이터가 없을 때 쓰는 로컬 폴백
  const ingredients: [string, number][] =
    serverIngredients || (m?.ingredients || []).map(parseLocalIngredient);
  const apiNutrition = snapshotOnly ? undefined : apiResult?.dish_nutrition?.[menuName];
  const nutrition = snapshot?.nutrition || apiNutrition || m?.nutrition;
  const isServerNutrition = !!(snapshot?.nutrition || apiNutrition);

  const [steps, setSteps] = useState<string[] | null>(null);
  const [loadingRecipe, setLoadingRecipe] = useState(false);
  const [recipeError, setRecipeError] = useState("");
  // 음성 안내 팝업에서 지금 읽고 있는 단계 (-1 = 팝업 닫힘)
  const [voiceStep, setVoiceStep] = useState(-1);
  const speech = useSpeech();

  useEffect(() => {
    // 스냅샷을 아직 불러오는 중이면 아무것도 하지 않고 기다린다 — 여기서 apiResult로 성급하게
    // 조리법을 생성해버리면, 스냅샷이 도착했을 때 그 결과를 버리고 또 한 번 새로 생성하게 된다.
    if (snapshotLoading) return;
    setRecipeError("");
    // 찜 스냅샷에 이미 그 시점의 조리과정이 있으면 그대로 쓴다 — 다시 생성(=새 LLM 응답)하지
    // 않아야 나중에 봐도 항상 같은 내용이 뜬다.
    if (snapshot?.steps && snapshot.steps.length) {
      setSteps(snapshot.steps);
      return;
    }
    setSteps(null);
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
        else {
          const parsed = toSteps(d.steps);
          setSteps(parsed);
          // PDF 미리보기가 이 메뉴를 다시 /recipe로 불러오지 않고 재사용하도록 캐싱.
          setDishSteps(menuName, parsed);
          // 찜한 메뉴를 열어봤는데 스냅샷에 조리과정이 없어 방금 새로 생성한 경우, 호출한
          // 쪽(MenuDetailPage)이 이걸 찜 기록에 채워 넣어 다음부터는 고정되게 한다.
          onStepsReady?.(parsed);
        }
      })
      .catch(() => {
        if (live) setRecipeError("조리법을 불러오지 못했어요.");
      })
      .finally(() => live && setLoadingRecipe(false));
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [menuName, snapshot, snapshotLoading]);

  const displaySteps: string[] =
    steps ?? (m?.steps?.length ? toSteps(m.steps) : []);
  const recipeSpeech = [
    `${menuName} 조리 순서입니다.`,
    ...displaySteps.map((s, i) => `${speakOrdinal(i + 1)}. ${s}`),
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
                  {fmt(displayValue(nutrition, n.key))}
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
          {showMakeMealButton && (
            <Button secondary onClick={makeMealWithThisMenu}>
              이 메뉴로 새 식단 만들기
            </Button>
          )}
          {displaySteps.length > 0 && (
            <div className="tts-row right">
              <button
                className="tts-button"
                onClick={() => {
                  setVoiceStep(0);
                  speech.speak(
                    `${menuName} 조리 순서입니다. ${speakOrdinal(1)}. ${displaySteps[0]}`,
                  );
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
                        `${speakOrdinal(voiceStep + 1)}. ${displaySteps[voiceStep]}`,
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
                        speech.speak(
                          `${speakOrdinal(next + 1)}. ${displaySteps[next]}`,
                        );
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
