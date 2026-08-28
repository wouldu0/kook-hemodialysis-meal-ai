import { useEffect, useMemo, useState } from "react";
import { useParams } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { RecipeBody } from "../../components/meal/RecipeBody";
import { deleteEverywhere, loadEverywhere, saveEverywhere } from "../../services/api";
import type { SavedItem } from "../../types";

// 찜한 메뉴 하나만 보는 화면. RecipePage(방금 생성한 식단의 레시피 보기)와 달리
// "이번 한 끼의 다른 메뉴" 같은 관련 없는 목록을 같이 보여주지 않고, 딱 그 메뉴의
// 재료·조리과정·영양성분과 "이 메뉴로 새 식단 만들기" 버튼만 보여준다.
//
// 찜 당시(또는 그 뒤 처음 열어봤을 때) 저장해둔 재료·조리과정 스냅샷이 있으면 그걸 최우선으로
// 보여준다 — 그래야 다시 볼 때마다 재료 양이나 AI 조리법이 바뀌어 보이지 않는다(2026-08).
export function MenuDetailPage() {
  const { name = "" } = useParams();
  const decoded = decodeURIComponent(name);
  const [fav, setFav] = useState<SavedItem | null | undefined>(undefined); // undefined = 로딩 중

  useEffect(() => {
    let live = true;
    setFav(undefined);
    loadEverywhere("fook:favorites").then((list) => {
      if (!live) return;
      setFav(list.find((x) => x.kind === "menu" && x.menuName === decoded) || null);
    });
    return () => {
      live = false;
    };
  }, [decoded]);

  // 찜 스냅샷에 조리과정이 아직 없어(펼쳐본 적 없는 메뉴를 바로 찜한 경우) RecipeBody가 지금
  // 막 새로 생성했다면, 다음에 봐도 똑같이 뜨도록 찜 기록에 그 조리과정을 한 번만 채워 넣는다.
  // saveEverywhere()는 서버에 항상 새 행으로 insert하는 방식이라(update API가 없음) 기존 id로
  // 그냥 다시 저장하면 같은 메뉴가 두 번 찜한 것처럼 중복 저장된다 — 그래서 기존 찜을 지우고
  // 조리과정만 채운 새 찜으로 다시 저장한다(id가 바뀌어도 화면엔 같은 메뉴 하나로만 보인다).
  const persistSteps = async (steps: string[]) => {
    if (!fav || (fav.recipeSteps && fav.recipeSteps.length)) return;
    const updated: SavedItem = { ...fav, id: `menu-${Date.now()}`, recipeSteps: steps };
    await deleteEverywhere("fook:favorites", fav.id);
    await saveEverywhere("fook:favorites", updated);
    setFav(updated);
  };

  // RecipeBody의 조리과정 effect가 [menuName, snapshot]을 의존성으로 본다. 이 객체를 매 렌더마다
  // 새로 만들면(리터럴) 내용이 같아도 참조가 달라져 effect가 불필요하게 다시 돌고, 조리과정이
  // 아직 없는 동안에는 그때마다 /recipe를 또 호출하는 문제가 있었다 — fav가 실제로 바뀔 때만
  // 새로 만들어지도록 useMemo로 참조를 고정한다.
  const snapshot = useMemo(
    () =>
      fav
        ? {
            ingredients: fav.recipeIngredients,
            steps: fav.recipeSteps,
            nutrition: fav.nutrition,
            recipeSource: fav.recipeSource,
          }
        : undefined,
    [fav],
  );

  return (
    <Shell header={false}>
      <BackHeader title={decoded} dot />
      <RecipeBody
        menuName={decoded}
        showMakeMealButton
        snapshot={snapshot}
        snapshotLoading={fav === undefined}
        // 이 화면은 항상 "찜한(또는 찜하려는) 그 메뉴 자체"만 보여준다 — 전역 apiResult는
        // 세션에서 마지막으로 생성한 식단일 뿐이라, 우연히 이름이 같은 메뉴가 있어도 그
        // 데이터가 섞여 들어가면 안 된다(재료 중량·recipeSource 등).
        snapshotOnly
        onStepsReady={persistSteps}
      />
    </Shell>
  );
}
