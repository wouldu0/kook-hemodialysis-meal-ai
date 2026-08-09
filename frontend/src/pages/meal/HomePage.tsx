import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { dietPlans, ingredientData, menuData } from "../../fookData";
import type { Plan } from "../../types";
import { currentUser, getIngredients, getMenus, getMenusByIngredient } from "../../services/api";
import { useApp } from "../../hooks/useApp";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { BottomNav } from "../../components/layout/BottomNav";
import { DiceIcon, LeafIcon, SearchIcon, SlidersIcon } from "../../components/icons";

export function HomePage() {
  const nav = useNavigate();
  const { query, setQuery, setPlan, searchMode, setSearchMode, setApiResult } =
    useApp();
  const [menuList, setMenuList] = useState<string[]>([]);
  const [ingList, setIngList] = useState<string[]>([]);
  // 데이터 출처 배지는 요청에 따라 화면에서 뺐다. 목록 로딩만 그대로 둔다.
  // 재료 검색 모드에서 '이 재료가 들어간 메뉴' 목록. 재료를 고르면 채워진다.
  const [ingQuery, setIngQuery] = useState("");
  const [ingMenus, setIngMenus] = useState<string[] | null>(null);
  const [ingLoading, setIngLoading] = useState(false);
  const [randomAsk, setRandomAsk] = useState(false);
  const user = currentUser();
  useEffect(() => {
    Promise.all([getMenus(), getIngredients()])
      .then(([m, i]) => {
        setMenuList(m.menus || []);
        setIngList(i.ingredients || []);
      })
      .catch(() => {
        // 서버 미연결이면 로컬 예시 데이터로 자동 폴백된다(아래 names/ingredients).
      });
  }, []);
  const localMenus = menuData.map((m) => m.name);
  const names = menuList.length ? menuList : localMenus;
  const suggestions = (
    query.trim()
      ? names.filter((n) => n.includes(query.trim()))
      : names.filter((n) => /된장국|구이|볶음/.test(n))
  ).slice(0, 8);
  const ingredients = ingList.length
    ? ingList
    : ingredientData.map((x) => x.name);
  const ingSuggestions = ingredients
    .filter((x) => x.includes(ingQuery.trim()))
    .slice(0, 8);
  const choose = (name: string) => {
    setQuery(name);
    const p = dietPlans.find((x) => x.menus.includes(name)) as Plan | undefined;
    if (p) setPlan(p);
  };
  // 재료를 하나 고르면, 그 재료가 실제로 들어간 메뉴를 서버에서 찾아 보여준다.
  const pickIngredient = (ing: string) => {
    setIngQuery(ing);
    setIngMenus(null);
    setIngLoading(true);
    getMenusByIngredient(ing)
      .then((d) => setIngMenus(d.menus || []))
      .catch(() => setIngMenus([]))
      .finally(() => setIngLoading(false));
  };
  const chooseFromIngredient = (menuName: string) => {
    setQuery(menuName);
    setSearchMode("menu"); // /generate에는 결국 menu 조건으로 넘어간다
    const p = dietPlans.find((x) => x.menus.includes(menuName)) as
      Plan | undefined;
    if (p) setPlan(p);
  };
  const canGenerate =
    searchMode === "random" ||
    (searchMode === "menu" && query.trim().length > 0) ||
    (searchMode === "ingredient" && query.trim().length > 0);
  return (
    <Shell
      footer={
        // 하단 탭바는 화면 맨 아래에 고정하고, 그 바로 위에 선택/생성 버튼을 둔다.
        <>
          <Button
            disabled={!canGenerate}
            onClick={() => {
              setApiResult(null);
              nav("/generating");
            }}
          >
            {canGenerate
              ? "선택한 조건으로 식단 생성하기"
              : searchMode === "ingredient"
                ? "재료가 들어간 메뉴를 선택해주세요"
                : "메뉴를 검색하거나 선택해주세요"}
          </Button>
          <BottomNav active="home" />
        </>
      }
    >
      <h1 className="greeting">
        안녕하세요, <b>{user?.name || "회원"}님!</b>
      </h1>
      <p className="sub">
        건강한 한 끼, <b className="brand-word">KOOK</b>이 함께합니다.
      </p>
      {/* 목업 1: 음식 검색 / 재료 검색 / 랜덤 추천 3분할 카드 */}
      <div className="quick-cards">
        <button
          className={searchMode === "menu" ? "quick-card active" : "quick-card"}
          onClick={() => setSearchMode("menu")}
        >
          <span className="quick-icon">
            <SearchIcon />
          </span>
          <b>음식 검색</b>
          <small>메뉴명을 검색해보세요</small>
        </button>
        <button
          className={
            searchMode === "ingredient" ? "quick-card active" : "quick-card"
          }
          onClick={() => {
            setSearchMode("ingredient");
            setIngMenus(null);
            setIngQuery("");
          }}
        >
          <span className="quick-icon">
            <LeafIcon />
          </span>
          <b>재료 검색</b>
          <small>재료로 메뉴를 찾아보세요</small>
        </button>
        <button
          className={
            searchMode === "random" ? "quick-card active" : "quick-card"
          }
          onClick={() => {
            setSearchMode("random");
            setQuery("");
            setRandomAsk(true); // 랜덤 탭은 안내문 대신 확인 팝업을 띄운다
          }}
        >
          <span className="quick-icon">
            <DiceIcon />
          </span>
          <b>랜덤 추천</b>
          <small>AI가 랜덤으로 추천해드려요</small>
        </button>
      </div>
      {searchMode === "menu" && (
        <>
          <div className="search">
            <SearchIcon />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="메뉴명을 검색해보세요 (예: 두부, 계란찜, 소고기)"
            />
            <span className="search-filter">
              <SlidersIcon />
            </span>
          </div>
          <div className="result-head">
            <h2 className="section-title">음식 검색 결과</h2>
            <span className="sort-pill">최신순 ⌄</span>
          </div>
          <div className="pick-list">
            {suggestions.map((n) => (
              <article
                key={n}
                className={query === n ? "pick-row active" : "pick-row"}
              >
                <b>{n}</b>
                <button onClick={() => choose(n)}>
                  {query === n ? "선택됨" : "선택하기"}
                </button>
              </article>
            ))}
            {suggestions.length === 0 && (
              <div className="info-box">
                <b>검색 결과가 없어요</b>
                <span>다른 메뉴명으로 다시 검색해보세요.</span>
              </div>
            )}
          </div>
        </>
      )}
      {searchMode === "ingredient" && (
        <>
          <div className="search">
            <SearchIcon />
            <input
              value={ingQuery}
              onChange={(e) => {
                setIngQuery(e.target.value);
                setIngMenus(null);
                setQuery("");
              }}
              placeholder="사용하고 싶은 재료를 검색하세요"
            />
          </div>
          {ingMenus === null ? (
            <>
              <div className="result-head">
                <h2 className="section-title">재료 검색 결과</h2>
              </div>
              <div className="pick-list">
                {ingSuggestions.map((n) => (
                  <article className="pick-row" key={n}>
                    <b>{n}</b>
                    <button onClick={() => pickIngredient(n)}>메뉴 보기</button>
                  </article>
                ))}
              </div>
            </>
          ) : (
            <>
              <div className="ingredient-picked-row">
                <span>
                  '<b>{ingQuery}</b>'가 들어간 메뉴
                </span>
                <button
                  className="link-button"
                  onClick={() => {
                    setIngMenus(null);
                    setQuery("");
                  }}
                >
                  다시 검색
                </button>
              </div>
              {ingLoading && (
                <div className="recipe-loading">
                  <div className="spinner" />
                  <span>메뉴를 찾고 있어요...</span>
                </div>
              )}
              {!ingLoading && ingMenus.length === 0 && (
                <div className="info-box">
                  <b>일치하는 메뉴를 찾지 못했어요</b>
                  <span>다른 재료로 다시 검색해보세요.</span>
                </div>
              )}
              <div className="pick-list">
                {ingMenus.slice(0, 20).map((n) => (
                  <article
                    key={n}
                    className={query === n ? "pick-row active" : "pick-row"}
                  >
                    <span className="pick-main">
                      <b>{n}</b>
                      <small>{ingQuery} 포함</small>
                    </span>
                    <button onClick={() => chooseFromIngredient(n)}>
                      {query === n ? "선택됨" : "선택하기"}
                    </button>
                  </article>
                ))}
              </div>
            </>
          )}
        </>
      )}
      {searchMode === "random" && !randomAsk && (
        <button className="random-again" onClick={() => setRandomAsk(true)}>
          🎲 랜덤 추천 다시 받기
        </button>
      )}
      {/* 랜덤 추천 확인 팝업: 왼쪽 아니오 / 오른쪽 네 */}
      {randomAsk && (
        <div className="modal-bg" onClick={() => setRandomAsk(false)}>
          <div className="modal ask" onClick={(e) => e.stopPropagation()}>
            <span className="modal-mark">🎲</span>
            <h2>랜덤으로 식단 추천 해드릴까요?</h2>
            <p>
              먹고 싶은 음식을 고르지 않아도
              <br />
              KOOK AI가 영양 기준에 맞는 한 끼를 만들어드려요.
            </p>
            <div className="ask-actions">
              <button
                className="ask-no"
                onClick={() => {
                  setRandomAsk(false);
                  setSearchMode("menu");
                }}
              >
                아니오
              </button>
              <button
                className="ask-yes"
                onClick={() => {
                  setRandomAsk(false);
                  setApiResult(null);
                  nav("/generating");
                }}
              >
                네
              </button>
            </div>
          </div>
        </div>
      )}
    </Shell>
  );
}
