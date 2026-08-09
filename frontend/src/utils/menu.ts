import { dietPlans, menuData } from "../fookData";
import type { MenuRecord, Plan } from "../types";

export const menuMap = new Map(menuData.map((m) => [m.name, m as MenuRecord]));

export const fallbackPlan =
  (dietPlans.find((p) => p.menus.includes("시금치된장국")) as Plan) ||
  (dietPlans[0] as Plan);

export const labels = ["밥", "국", "어육류", "밑반찬", "김치류"];

// 목업 기준 표기.
//  · 식단 생성 화면(2/5): 밥 / 국 / 반찬1 / 반찬2 / 반찬3
//  · 레시피 화면: 밥 / 국 / 찬 1 (어육류) / 찬 2 (밑반찬) / 찬 3 (김치류)
export function roleShort(i: number) {
  if (i === 0) return "밥";
  if (i === 1) return "국";
  return `반찬${i - 1}`;
}

export function roleLong(i: number) {
  if (i === 0) return "밥";
  if (i === 1) return "국";
  return `찬 ${i - 1} (${labels[i] || "반찬"})`;
}

// 서버 미연결(오프라인) 폴백 데이터의 재료는 "시금치, 생것 37.5g"처럼 양이 문자열에 붙어 있다.
// 서버 응답과 같은 [이름, 양] 형태로 쪼개서, 표시할 때 동일하게 소수점 2자리로 맞춘다.
export function parseLocalIngredient(raw: string): [string, number] {
  const m = raw.match(/^(.*?)\s+([\d.]+)\s*g$/);
  return m ? [m[1], Number(m[2])] : [raw, 0];
}
