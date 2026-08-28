import { useEffect, useState } from "react";
import { useApp } from "../../hooks/useApp";
import { deleteEverywhere, loadEverywhere, saveEverywhere } from "../../services/api";
import type { SavedItem } from "../../types";
import { menuMap, roleLong } from "../../utils/menu";
import { RecipeBody } from "./RecipeBody";

// 메뉴 목록 + 고른 메뉴의 레시피를 '같은 화면에서' 펼쳐 보여준다 (화면 이동 없음).
export function RecipeList({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (n: string) => void;
}) {
  const { plan, apiResult, dishSteps } = useApp();
  // 메뉴 단위 찜(kind:"menu") 목록만 따로 들고 있는다 — 식단 콤보 찜(fook:favorites의
  // 기존 항목)과 저장 위치(같은 키)는 같지만 화면에서 다루는 대상은 다르다.
  const [menuFavs, setMenuFavs] = useState<SavedItem[]>([]);

  useEffect(() => {
    let live = true;
    loadEverywhere("fook:favorites").then((list) => {
      if (live) setMenuFavs(list.filter((x) => x.kind === "menu"));
    });
    return () => {
      live = false;
    };
  }, []);

  const toggleFavorite = async (name: string) => {
    const existing = menuFavs.find((x) => x.menuName === name);
    if (existing) {
      setMenuFavs((prev) => prev.filter((x) => x.id !== existing.id));
      await deleteEverywhere("fook:favorites", existing.id);
      return;
    }
    const nutrition = apiResult?.dish_nutrition?.[name] || menuMap.get(name)?.nutrition;
    // 지금 이 식단 생성 결과에 이 메뉴의 실제 재료가 있으면 같이 스냅샷 저장한다. 조리과정은
    // 세션 중 이미 펼쳐봐서 dishSteps에 캐시돼 있을 때만 같이 저장되고, 없으면 나중에 찜한
    // 메뉴를 처음 열 때 한 번 생성해서 채워 넣는다(MenuDetailPage 참고).
    const item: SavedItem = {
      id: `menu-${Date.now()}`,
      title: name,
      subtitle: "메뉴",
      createdAt: new Date().toISOString(),
      kind: "menu",
      menuName: name,
      nutrition,
      recipeIngredients: apiResult?.dish_ingredients?.[name],
      recipeSteps: dishSteps[name],
      recipeSource: apiResult?.recipe_source?.[name],
    };
    setMenuFavs((prev) => [item, ...prev]);
    await saveEverywhere("fook:favorites", item);
  };

  return (
    <>
      <h2 className="section-title tight">식단 메뉴</h2>
      <p className="sub small">
        메뉴를 선택하면 재료와 조리과정을 확인할 수 있어요.
      </p>
      <div className="recipe-menu-list">
        {plan.menus.map((n, i) => {
          const open = n === selected;
          const fav = menuFavs.some((x) => x.menuName === n);
          return (
            <div key={n}>
              <div className="recipe-menu-row-wrap">
                <button
                  className={open ? "recipe-menu-row active" : "recipe-menu-row"}
                  onClick={() => onSelect(open ? "" : n)}
                  aria-expanded={open}
                >
                  <span className="rm-num">{i + 1}</span>
                  <span className="rm-txt">
                    <b>{n}</b>
                    <small>{roleLong(i)}</small>
                  </span>
                  <i className={open ? "rm-caret open" : "rm-caret"}>›</i>
                </button>
                <button
                  className={fav ? "menu-fav-btn on" : "menu-fav-btn"}
                  aria-label={fav ? "메뉴 찜 해제" : "메뉴 찜하기"}
                  onClick={() => toggleFavorite(n)}
                >
                  {fav ? "♥" : "♡"}
                </button>
              </div>
              {open && (
                <section className="recipe-inline">
                  <RecipeBody menuName={n} />
                </section>
              )}
            </div>
          );
        })}
      </div>
    </>
  );
}
