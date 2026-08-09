import { useEffect, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { BottomNav } from "../../components/layout/BottomNav";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { currentUser, deleteEverywhere, loadEverywhere, storage } from "../../services/api";
import type { SavedItem } from "../../types";
import { MEAL_TIMES } from "../../utils/date";

// 저장 목록의 카드 한 장 (식단 기록 / 식단 관리 / PDF 보관함 공통)
function SavedCard({
  item,
  mode,
  onOpen,
  onRemove,
}: {
  item: SavedItem;
  mode: "history" | "favorites" | "documents";
  onOpen: (x: SavedItem) => void;
  onRemove: (id: string) => void;
}) {
  return (
    <article>
      <div className="saved-thumb">{mode === "documents" ? "PDF" : "KOOK"}</div>
      <button className="saved-main" onClick={() => onOpen(item)}>
        <b>{item.title}</b>
        <span>{item.subtitle}</span>
        <small>
          {item.mealDate
            ? new Date(item.mealDate).toLocaleDateString("ko-KR")
            : new Date(item.createdAt).toLocaleDateString("ko-KR")}
        </small>
      </button>
      <button className="delete-mini" onClick={() => onRemove(item.id)}>
        ×
      </button>
    </article>
  );
}

export function LibraryPage({
  mode,
}: {
  mode: "history" | "favorites" | "documents";
}) {
  const nav = useNavigate();
  if (!currentUser()) return <Navigate to="/login" replace />;
  const key = `fook:${mode}`;
  const [items, setItems] = useState<SavedItem[]>(storage.get(key, []));
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let live = true;
    setLoading(true);
    loadEverywhere(key).then((list) => {
      if (live) {
        setItems(list);
        setLoading(false);
      }
    });
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);
  const meta = {
    history: ["식단 기록", "최근 생성하고 기록한 식단"],
    favorites: ["식단 관리", "기록한 식단을 모아두고 다시 불러올 수 있어요"],
    documents: ["PDF 보관함", "생성한 레시피 문서 기록"],
  }[mode];
  const remove = async (id: string) => {
    setItems((prev) => prev.filter((x) => x.id !== id));
    await deleteEverywhere(key, id);
  };
  const openItem = (x: SavedItem) => {
    if (x.menus) {
      storage.set("fook:restore", x);
      nav("/home");
    }
  };
  return (
    <Shell
      header={false}
      footer={<BottomNav active={mode === "favorites" ? "favorites" : "history"} />}
    >
      <BackHeader title={meta[0]} />
      <p className="eyebrow">MY KOOK</p>
      <h1>{meta[0]}</h1>
      <p className="sub">{meta[1]}</p>
      {loading && (
        <div className="recipe-loading">
          <div className="spinner" />
          <span>불러오는 중...</span>
        </div>
      )}
      {!loading && items.length > 0 && mode === "favorites" && (
        // 식단 관리는 아침 / 점심 / 저녁 섹션으로 한 줄씩 나눠서 보여준다
        <>
          {MEAL_TIMES.map((t) => (
            <section className="meal-slot-section" key={t}>
              <h2 className="section-title">
                {t}
                <span className="slot-count">
                  {items.filter((x) => x.mealTime === t).length}
                </span>
              </h2>
              {items.some((x) => x.mealTime === t) ? (
                <div className="saved-list">
                  {items
                    .filter((x) => x.mealTime === t)
                    .map((x) => (
                      <SavedCard
                        key={x.id}
                        item={x}
                        mode={mode}
                        onOpen={openItem}
                        onRemove={remove}
                      />
                    ))}
                </div>
              ) : (
                <p className="slot-empty">아직 {t} 기록이 없어요.</p>
              )}
            </section>
          ))}
          {/* 끼니를 고르기 전에 저장한 예전 기록 */}
          {items.some((x) => !x.mealTime) && (
            <section className="meal-slot-section">
              <h2 className="section-title">끼니 미지정</h2>
              <div className="saved-list">
                {items
                  .filter((x) => !x.mealTime)
                  .map((x) => (
                    <SavedCard
                      key={x.id}
                      item={x}
                      mode={mode}
                      onOpen={openItem}
                      onRemove={remove}
                    />
                  ))}
              </div>
            </section>
          )}
        </>
      )}
      {!loading && items.length > 0 && mode !== "favorites" && (
        <div className="saved-list">
          {items.map((x) => (
            <SavedCard
              key={x.id}
              item={x}
              mode={mode}
              onOpen={openItem}
              onRemove={remove}
            />
          ))}
        </div>
      )}
      {!loading && items.length === 0 && (
        <div className="empty-state">
          <div>
            {mode === "favorites" ? "♡" : mode === "documents" ? "PDF" : "◷"}
          </div>
          <b>아직 기록된 항목이 없어요.</b>
          <p>맞춤 식단을 생성한 뒤 기록해보세요.</p>
          <Button onClick={() => nav("/home")}>식단 만들러 가기</Button>
        </div>
      )}
    </Shell>
  );
}
