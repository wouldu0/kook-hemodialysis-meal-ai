import type { MealTime } from "../types";

// YYYY-MM-DD 생년월일로 만 나이를 계산한다. 형식이 이상하면 null.
export function ageFromBirthdate(birthdate: string): number | null {
  const m = birthdate.match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return null;
  const b = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  if (isNaN(b.getTime())) return null;
  const today = new Date();
  let age = today.getFullYear() - b.getFullYear();
  const beforeBirthday =
    today.getMonth() < b.getMonth() ||
    (today.getMonth() === b.getMonth() && today.getDate() < b.getDate());
  if (beforeBirthday) age -= 1;
  return age >= 1 && age <= 120 ? age : null;
}

// 끼니 구분 — 식단 관리 화면에서 아침/점심/저녁 섹션으로 나누는 기준
export const MEAL_TIMES: MealTime[] = ["아침", "점심", "저녁"];

// 지금 시각으로 기본 끼니를 고른다 (10시 전 아침, 15시 전 점심, 그 뒤 저녁)
export function defaultMealTime(): MealTime {
  const h = new Date().getHours();
  return h < 10 ? "아침" : h < 15 ? "점심" : "저녁";
}

// <input type="date">에 넣을 오늘 날짜 (YYYY-MM-DD)
export function todayISO(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate(),
  ).padStart(2, "0")}`;
}
