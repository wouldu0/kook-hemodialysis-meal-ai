import { useEffect, useMemo, useState } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { BottomNav } from "../../components/layout/BottomNav";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import {
  currentUser,
  deleteEverywhere,
  fetchDayTargets,
  isValidIntake,
  loadEverywhere,
  storage,
  type DayTargets,
} from "../../services/api";
import { useApp } from "../../hooks/useApp";
import type { SavedItem } from "../../types";
import { MEAL_TIMES, todayISO } from "../../utils/date";
import {
  displayTarget,
  displayValue,
  fmt,
  intakeToDisplay,
  minTargetOf,
  nmeta,
  STATUS_CLASS,
  statusOf,
} from "../../utils/nutrition";

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
          {/* 나트륨은 displayValue()가 sodium_total(총나트륨)이 있으면 그걸 우선 쓴다 —
              찜한 메뉴 배지도 화면 전체와 같은 총나트륨 기준으로 통일(2026-08). */}
          {n.icon} {fmt(displayValue(values, n.key))}
          {n.unit}
        </span>
      ))}
    </div>
  );
}

// 하루 합계(values)를 그 사람의 실제 하루 기준(targets, /day_targets)과 비교해
// 칸이 얼마나 채워졌는지 막대로 보여준다. targetOf/minTargetOf/statusOf는 끼니별
// 화면(Nutrients)에서 이미 쓰는 것과 같은 함수라 판정 기준이 화면마다 다르지 않다.
function DayProgress({
  values,
  targets,
}: {
  values: Record<string, number>;
  targets: DayTargets;
}) {
  return (
    <div className="day-progress">
      {nmeta.map((n) => {
        const hi = displayTarget(targets, n.key);
        const lo = minTargetOf(targets, n.key);
        const v = displayValue(values, n.key);
        // 막대 너비는 트랙을 벗어날 수 없어 100%에서 자르지만, 옆 숫자는 실제 초과분이
        // 보이도록 그대로 둔다 — "100%"만 보이면 얼마나 넘었는지 알 수 없다.
        const rawPct = hi > 0 ? Math.round((v / hi) * 100) : 0;
        const barPct = Math.min(100, rawPct);
        const status = statusOf(v, lo, hi);
        return (
          <div className={`day-progress-row ${STATUS_CLASS[status]}`} key={n.key}>
            <span className="day-progress-label">
              {n.icon} {n.label}
            </span>
            <div className="day-progress-track">
              <div className="day-progress-fill" style={{ width: `${barPct}%` }} />
            </div>
            <span className="day-progress-value">
              {fmt(v)}/{fmt(hi)}
              {n.unit} · {rawPct}%
            </span>
          </div>
        );
      })}
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
  // 메뉴 단위 찜은 item.nutrition이 이미 NutrientKey 체계라 intakeToDisplay() 변환이 필요 없다.
  const menuNutrition = mode === "favorites" && item.kind === "menu" ? item.nutrition : undefined;
  // 식사 기록은 title이 밥·국·반찬 중 국 하나(menus[1])라 국만 굵게 보였다 — 기록은 항상
  // 한 끼 전체(콤보)이므로 밥·국·반찬을 전부 굵게 보여준다. subtitle도 원래 같은 내용
  // (menus.join(" · "))이라 중복 표시를 피하려고 이때만 따로 안 보여준다.
  const isHistoryCombo = mode === "history" && !!item.menus?.length;
  const mainText = isHistoryCombo ? item.menus!.join(" · ") : item.title;
  return (
    <article>
      <div className="saved-thumb">{mode === "documents" ? "PDF" : "KOOK"}</div>
      <button className="saved-main" onClick={() => onOpen(item)}>
        <b>{mainText}</b>
        {!isHistoryCombo && <span>{item.subtitle}</span>}
        {mode !== "favorites" && (
          <small>
            {showMealTime && item.mealTime ? `${item.mealTime} · ` : ""}
            {new Date(itemDate(item)).toLocaleDateString("ko-KR")}
          </small>
        )}
        {intake && <NutritionLine values={intakeToDisplay(intake)} />}
        {menuNutrition && <NutritionLine values={menuNutrition} />}
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
  const { profile } = useApp();
  const key = `fook:${mode}`;
  const [items, setItems] = useState<SavedItem[]>(storage.get(key, []));
  const [loading, setLoading] = useState(true);
  const [calendarMonth, setCalendarMonth] = useState(() => new Date());
  const [selectedDate, setSelectedDate] = useState(() => todayISO());
  const [dayTargets, setDayTargets] = useState<DayTargets | null>(null);

  useEffect(() => {
    if (mode !== "history") return;
    let live = true;
    const body: Record<string, unknown> = { weight: Number(profile.weight) || 60 };
    if (profile.height) {
      body.height = Number(profile.height);
      body.sex = profile.gender === "남성" ? "남" : "여";
    }
    if (profile.customTargets && Object.keys(profile.customTargets).length > 0) {
      body.custom_targets = profile.customTargets;
    }
    // 실패해도(오프라인 등) 그냥 원래대로 숫자만 보여준다 — 기준 없이 %를 지어내지 않는다.
    fetchDayTargets(body)
      .then((d) => live && setDayTargets(d))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [mode, profile.weight, profile.height, profile.gender]);

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
    favorites: ["찜한 메뉴", "마음에 든 메뉴를 모아두고 다시 볼 수 있어요"],
    documents: ["PDF 보관함", "생성한 레시피 문서 기록"],
  }[mode];

  const sortedItems = useMemo(
    () =>
      [...items]
        // 콤보(식단 전체) 찜은 더 이상 만들지 않는다 — 예전에 저장된 콤보 항목이 남아있어도
        // "찜한 메뉴" 화면에는 메뉴 단위 찜만 보여준다.
        .filter((x) => mode !== "favorites" || x.kind === "menu")
        .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()),
    [items, mode],
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
    if (x.kind === "menu" && x.menuName) {
      nav(`/menu/${encodeURIComponent(x.menuName)}`);
      return;
    }
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
              {daySummary(selectedDayItems) &&
                (dayTargets ? (
                  <DayProgress values={daySummary(selectedDayItems)!} targets={dayTargets} />
                ) : (
                  <NutritionLine
                    values={daySummary(selectedDayItems)!}
                    className="intake-line day-total"
                  />
                ))}
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

      {!loading && sortedItems.length > 0 && mode === "favorites" && (
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

      {!loading && (mode === "favorites" ? sortedItems.length === 0 : items.length === 0) && (
        <div className="empty-state">
          <div>
            {mode === "favorites" ? "♡" : mode === "documents" ? "PDF" : "◷"}
          </div>
          <b>
            {mode === "favorites"
              ? "아직 찜한 메뉴가 없어요."
              : mode === "history"
                ? "아직 기록한 식사가 없어요."
                : "아직 기록된 항목이 없어요."}
          </b>
          <p>
            {mode === "favorites"
              ? "마음에 드는 메뉴를 찜해두면 여기에서 다시 볼 수 있어요."
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
