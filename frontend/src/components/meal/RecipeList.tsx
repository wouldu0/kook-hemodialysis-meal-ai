import { useApp } from "../../hooks/useApp";
import { roleLong } from "../../utils/menu";
import { RecipeBody } from "./RecipeBody";

// 메뉴 목록 + 고른 메뉴의 레시피를 '같은 화면에서' 펼쳐 보여준다 (화면 이동 없음).
export function RecipeList({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (n: string) => void;
}) {
  const { plan } = useApp();
  return (
    <>
      <h2 className="section-title tight">식단 메뉴</h2>
      <p className="sub small">
        메뉴를 선택하면 재료와 조리과정을 확인할 수 있어요.
      </p>
      <div className="recipe-menu-list">
        {plan.menus.map((n, i) => {
          const open = n === selected;
          return (
            <div key={n}>
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
