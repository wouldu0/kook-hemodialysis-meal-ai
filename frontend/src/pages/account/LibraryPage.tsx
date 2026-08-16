import { useEffect, useMemo, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { BottomNav } from "../../components/layout/BottomNav";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { currentUser, deleteEverywhere, loadEverywhere, storage } from "../../services/api";
import type { SavedItem } from "../../types";
import { MEAL_TIMES } from "../../utils/date";

function itemDate(item: SavedItem) {
  return item.mealDate || item.createdAt.slice(0, 10);
}

function SavedCard({
  item,
  mode,
  onOpen,
  onRemove,
  showMealTime = false,
}: {
  item: SavedItem;
  mode: "history" | "favorites" | "documents";
  onOpen: (x: SavedItem) => void;
  onRemove: (id: string) => void;
  showMealTime?: boolean;
}) {
  return (
    <article>
      <div className="saved-thumb">{mode === "documents" ? "PDF" : "KOOK"}</div>
      <button className="saved-main" onClick={() => onOpen(item)}>
        <b>{item.title}</b>
        <span>{item.subtitle}</span>
        <small>
          {showMealTime && item.mealTime ? `${item.mealTime} · ` : ""}
          {new Date(itemDate(item)).toLocaleDateString("ko-KR")}
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
    history: ["식사 기록", "날짜별로 먹은 식사를 확인하고 다음 추천에 반영해요"],
    favorites: ["찜한 식단", "마음에 든 식단을 모아두고 다시 볼 수 있어요"],
    documents: ["PDF 보관함", "생성한 레시피 문서 기록"],
  }[mode];

  const sortedItems = useMemo(
    () =>
      [...items].sort(
        (a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime(),
      ),
    [items],
  );

  const historyDates = useMemo(() => {
    if (mode !== "history") return [] as [string, SavedItem[]][];
    const groups = new Map<string, SavedItem[]>();
    for (const item of items) {
      const date = itemDate(item);
      groups.set(date, [...(groups.get(date) || []), item]);
    }
    return [...groups.entries()]
      .sort(([a], [b]) => b.localeCompare(a))
      .map(([date, dayItems]) => [
        date,
        [...dayItems].sort((a, b) => {
          const ai = a.mealTime ? MEAL_TIMES.indexOf(a.mealTime) : MEAL_TIMES.length;
          const bi = b.mealTime ? MEAL_TIMES.indexOf(b.mealTime) : MEAL_TIMES.length;
          if (ai !== bi) return ai - bi;
          return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
        }),
      ] as [string, SavedItem[]]);
  }, [items, mode]);

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
      footer={<BottomNav active={mode === "history" ? "history" : "account"} />}
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

      {!loading && items.length > 0 && mode === "history" && (
        <>
          {historyDates.map(([date, dayItems]) => (
            <section className="meal-slot-section" key={date}>
              <h2 className="section-title">
                {new Date(date).toLocaleDateString("ko-KR", {
                  year: "numeric",
                  month: "long",
                  day: "numeric",
                })}
                <span className="slot-count">{dayItems.length}</span>
              </h2>
              <div className="saved-list">
                {dayItems.map((x) => (
                  <SavedCard
                    key={x.id}
                    item={x}
                    mode={mode}
                    showMealTime
                    onOpen={openItem}
                    onRemove={remove}
                  />
                ))}
              </div>
            </section>
          ))}
        </>
      )}

      {!loading && items.length > 0 && mode === "favorites" && (
        <div className="saved-list">
          {sortedItems.map((x) => (
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

      {!loading && items.length > 0 && mode === "documents" && (
        <div className="saved-list">
          {sortedItems.map((x) => (
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
          <b>
            {mode === "favorites"
              ? "아직 찜한 식단이 없어요."
              : mode === "history"
                ? "아직 기록한 식사가 없어요."
                : "아직 기록된 항목이 없어요."}
          </b>
          <p>
            {mode === "favorites"
              ? "마음에 드는 식단을 찜해두면 여기에서 다시 볼 수 있어요."
              : mode === "history"
                ? "식단을 만든 뒤 실제 식사로 기록해보세요."
                : "맞춤 식단을 생성한 뒤 기록해보세요."}
          </p>
          <Button onClick={() => nav("/home")}>식단 만들러 가기</Button>
        </div>
      )}
    </Shell>
  );
}
