import { useState } from "react";
import type { MealTime } from "../../types";
import { defaultMealTime, MEAL_TIMES, todayISO } from "../../utils/date";

export function MealSlotDialog({
  onCancel,
  onConfirm,
}: {
  onCancel: () => void;
  onConfirm: (date: string, time: MealTime) => void;
}) {
  const [date, setDate] = useState(todayISO());
  const [time, setTime] = useState<MealTime>(defaultMealTime());
  return (
    <div className="modal-bg" onClick={onCancel}>
      <div className="modal ask slot" onClick={(e) => e.stopPropagation()}>
        <span className="modal-mark">🍽</span>
        <h2>언제 먹은 식단인가요?</h2>
        <p>
          실제로 먹은 날짜와 끼니를 기록하면
          <br />
          식단 관리에 저장되고, 오늘 기록은 다음 식단
          <br />
          추천의 남은 영양 기준에 반영돼요.
        </p>
        <label className="slot-date">
          <span>날짜</span>
          <input
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </label>
        <div className="slot-times">
          {MEAL_TIMES.map((t) => (
            <button
              key={t}
              type="button"
              className={`slot-chip${t === time ? " on" : ""}`}
              aria-pressed={t === time}
              onClick={() => setTime(t)}
            >
              {t}
            </button>
          ))}
        </div>
        <div className="ask-actions">
          <button className="ask-no" onClick={onCancel}>
            취소
          </button>
          <button className="ask-yes" onClick={() => onConfirm(date, time)}>
            기록하기
          </button>
        </div>
      </div>
    </div>
  );
}
