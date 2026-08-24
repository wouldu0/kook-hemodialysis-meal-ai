import { useEffect, useMemo, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { BottomNav } from "../../components/layout/BottomNav";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { currentUser, deleteEverywhere, isValidIntake, loadEverywhere, storage } from "../../services/api";
import type { SavedItem } from "../../types";
import { MEAL_TIMES, todayISO } from "../../utils/date";
import { fmt, intakeToDisplay, nmeta } from "../../utils/nutrition";

const DOW = ["일", "월", "화", "수", "목", "금", "토"];

function itemDate(item: SavedItem) {
  return item.mealDate || item.createdAt.slice(0, 10);
}

// year/month(0-indexed)의 달력 셀을 7일씩 끊어 반환한다. 그 달에 속하지 않는 칸은 null.
function monthGrid(year: number, month: number): (string | null)[][] {
  const startDow = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const cells: (string | null)[] = Array(startDow).fill(null);
  for (let d = 1; d <= daysInMonth; d++) {
    cells.push(`${year}-${String(month + 1).padStart(2, "0")}-${String(d).padStart(2, "0")}`);
  }
  while (cells.length % 7 !== 0) cells.push(null);
  const rows: (string | null)[][] = [];
  for (let i = 0; i < cells.length; i += 7) rows.push(cells.slice(i, i + 7));
  return rows;
}

// 그 날 기록 중 intake가 유효한 것만 합산한다(구버전/미완성 기록은 건너뜀). 하나도
// 없으면 undefined — "0"으로 오해될 만한 합계를 보여주지 않기 위해서다.
function daySummary(dayItems: SavedItem[]): Record<string, number> | undefined {
  let sum: Record<string, number> | undefined;
  for (const item of dayItems) {
    if (!isValidIntake(item.intake)) continue;
    const d = intakeToDisplay(item.intake);
    if (!sum) sum = { energy: 0, protein: 0, potassium: 0, phosphorus: 0, sodium: 0 };
    for (const k of Object.keys(d)) sum[k] += d[k as keyof typeof d];
  }
  return sum;
}

function NutritionLine({
  values,
  className = "intake-line",
}: {
  values: Record<string, number>;
  className?: string;
}) {
  return (
    <div className={className}>
      {nmeta.map((n) => (
        <span key={n.key}>
          {n.icon} {fmt(values[n.key] ?? 0)}
          {n.unit}
        </span>
      ))}
    </div>
  );
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
  const intake = mode === "history" && isValidIntake(item.intake) ? item.intake : undefined;
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
        {intake && <NutritionLine values={intakeToDisplay(intake)} />}
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
  const [calendarMonth, setCalendarMonth] = useState(() => new Date());
  const [selectedDate, setSelectedDate] = useState(() => todayISO());

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

  // 달력에 점으로 표시할, 기록이 있는 날짜 집합.
  const historyDateSet = useMemo(() => {
    if (mode !== "history") return new Set<string>();
    return new Set(items.map(itemDate));
  }, [items, mode]);

  // 달력에서 선택한 날짜(selectedDate)의 기록만 뽑는다 — 아침/점심/저녁 순 정렬.
  const selectedDayItems = useMemo(() => {
    if (mode !== "history") return [] as SavedItem[];
    return items
      .filter((item) => itemDate(item) === selectedDate)
      .sort((a, b) => {
        const ai = a.mealTime ? MEAL_TIMES.indexOf(a.mealTime) : MEAL_TIMES.length;
        const bi = b.mealTime ? MEAL_TIMES.indexOf(b.mealTime) : MEAL_TIMES.length;
        if (ai !== bi) return ai - bi;
        return new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
      });
  }, [items, mode, selectedDate]);

  const shiftMonth = (delta: number) => {
    setCalendarMonth((prev) => new Date(prev.getFullYear(), prev.getMonth() + delta, 1));
  };

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
          <div className="calendar-inline">
            <div className="calendar-nav">
              <button onClick={() => shiftMonth(-1)} aria-label="이전 달">
                ‹
              </button>
              <b>
                {calendarMonth.getFullYear()}년 {calendarMonth.getMonth() + 1}월
              </b>
              <button onClick={() => shiftMonth(1)} aria-label="다음 달">
                ›
              </button>
            </div>
            <div className="calendar-dow">
              {DOW.map((d) => (
                <span key={d}>{d}</span>
              ))}
            </div>
            <div className="calendar-grid">
              {monthGrid(calendarMonth.getFullYear(), calendarMonth.getMonth())
                .flat()
                .map((date, i) => {
                  if (!date) return <span className="calendar-cell empty" key={`e${i}`} />;
                  const has = historyDateSet.has(date);
                  const isToday = date === todayISO();
                  const isSelected = date === selectedDate;
                  return (
                    <button
                      key={date}
                      className={`calendar-cell${has ? " has-record" : ""}${isToday ? " today" : ""}${isSelected ? " selected" : ""}`}
                      onClick={() => setSelectedDate(date)}
                    >
                      {Number(date.slice(-2))}
                    </button>
                  );
                })}
            </div>
          </div>

          <h2 className="section-title">
            {new Date(selectedDate).toLocaleDateString("ko-KR", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
            {selectedDayItems.length > 0 && (
              <span className="slot-count">{selectedDayItems.length}</span>
            )}
          </h2>

          {selectedDayItems.length > 0 ? (
            <>
              {daySummary(selectedDayItems) && (
                <NutritionLine
                  values={daySummary(selectedDayItems)!}
                  className="intake-line day-total"
                />
              )}
              <div className="saved-list">
                {selectedDayItems.map((x) => (
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
            </>
          ) : (
            <div className="empty-state day-empty">
              <div>◷</div>
              <p>이 날은 기록한 식사가 없어요.</p>
            </div>
          )}
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
