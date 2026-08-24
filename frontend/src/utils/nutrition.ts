import type { NutrientKey, Plan } from "../types";
import { menuMap } from "./menu";

export const nmeta: {
  key: NutrientKey;
  label: string;
  unit: string;
  target: number;
  icon: string;
}[] = [
  { key: "energy", label: "열량", unit: "kcal", target: 550, icon: "🔥" },
  { key: "protein", label: "단백질", unit: "g", target: 24, icon: "💪" },
  { key: "phosphorus", label: "인", unit: "mg", target: 550, icon: "🦴" },
  { key: "potassium", label: "칼륨", unit: "mg", target: 1200, icon: "🌿" },
  { key: "sodium", label: "나트륨", unit: "mg", target: 400, icon: "🧂" },
];

// meal_records에 저장된 intake({E,protein,K,P,Na,Na_season})를 nmeta/fmt와 같은
// NutrientKey 체계로 바꾼다. Na_season(조미료로 추가한 나트륨만 따로 관리하는 내부 예산)은
// 화면에 보여줄 대상이 아니라서 제외한다 — 여기서 보여줄 나트륨은 재료 자체의 총 나트륨(Na).
export function intakeToDisplay(intake: {
  E: number;
  protein: number;
  K: number;
  P: number;
  Na: number;
}): Record<NutrientKey, number> {
  return {
    energy: intake.E,
    protein: intake.protein,
    potassium: intake.K,
    phosphorus: intake.P,
    sodium: intake.Na,
  };
}

export function totalNutrition(plan: Plan) {
  return plan.menus.reduce(
    (a, name) => {
      const n = menuMap.get(name)?.nutrition;
      for (const k of [
        "energy",
        "protein",
        "phosphorus",
        "potassium",
        "sodium",
      ] as NutrientKey[])
        a[k] += n?.[k] || 0;
      return a;
    },
    { energy: 0, protein: 0, phosphorus: 0, potassium: 0, sodium: 0 },
  );
}

export function adjustedNutrition(raw: ReturnType<typeof totalNutrition>) {
  return {
    energy: Math.min(raw.energy, 520),
    protein: Math.min(raw.protein, 24),
    phosphorus: Math.min(raw.phosphorus, 520),
    potassium: Math.min(raw.potassium, 1150),
    sodium: Math.min(raw.sodium, 390),
  };
}

export function fmt(v: number) {
  return Math.round(v).toLocaleString("ko-KR");
}

// 재료·간식의 '양(g)'은 반올림하지 않고 항상 소수점 2자리까지 보여준다.
// (0.5g 차이가 저나트륨/저칼륨 조리에서는 의미가 있어서, 반올림하면 조정 내역이 안 보인다.)
export function fmt2(v: number) {
  return Number(v || 0).toLocaleString("ko-KR", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

// 서버 targets는 키마다 형태가 다르다: energy/protein은 [최소,최대] 배열,
// potassium/phosphorus/sodium은 숫자 하나. 화면에서 '상한선' 하나만 필요할 때 쓰는 공용 함수.
// targets가 아직 없으면(서버 미연결) nmeta의 대표값으로 대체한다.
export function targetOf(targets: any, key: NutrientKey): number {
  const raw = targets?.[key];
  if (raw != null) return Array.isArray(raw) ? Number(raw[1]) : Number(raw);
  return nmeta.find((n) => n.key === key)?.target ?? 0;
}

// energy/protein은 서버가 [최소,최대] 범위로 내려준다. 그 최소값(하한)을 뽑는다.
// potassium/phosphorus/sodium처럼 상한만 있는 항목은 하한이 없다는 뜻으로 0을 반환.
export function minTargetOf(targets: any, key: NutrientKey): number {
  const raw = targets?.[key];
  if (Array.isArray(raw)) return Number(raw[0]);
  return 0;
}

// 영양소 하나의 상태를 판정한다. lo가 0이면(칼륨·인·나트륨) 상한만 본다.
export type NStatus = "미만" | "적절" | "초과";
export function statusOf(v: number, lo: number, hi: number): NStatus {
  if (lo > 0 && v < lo) return "미만";
  return v > hi ? "초과" : "적절";
}
export const STATUS_CLASS: Record<NStatus, string> = {
  미만: "under",
  적절: "ok",
  초과: "over",
};

// changes 문자열 한 줄을 화면에 쓸 형태로 분류한다. app_core_FOOK._changes()가
// 만드는 3가지 패턴을 그대로 파싱: 교체/양조절/재료대체/메뉴제외.
export type ParsedChange = {
  kind: "amount" | "ingredient" | "swap" | "removed" | "other";
  menu: string;
  before?: string;
  after?: string;
  raw: string;
};
export function parseChange(line: string): ParsedChange {
  let m = line.match(/^(.+?): (.+) 양 ([\d.]+)→([\d.]+)g$/);
  if (m)
    return {
      kind: "amount",
      menu: m[1],
      before: `${m[2]} ${m[3]}g`,
      after: `${m[2]} ${m[4]}g`,
      raw: line,
    };
  m = line.match(/^(.+?): (.+) → (.+)$/);
  if (m)
    return {
      kind: "ingredient",
      menu: m[1],
      before: m[2],
      after: m[3],
      raw: line,
    };
  m = line.match(/^(.+?) → (.+?) 교체$/);
  if (m)
    return { kind: "swap", menu: m[1], before: m[1], after: m[2], raw: line };
  m = line.match(/^(.+?) 뺌$/);
  if (m) return { kind: "removed", menu: m[1], raw: line };
  return { kind: "other", menu: "", raw: line };
}
