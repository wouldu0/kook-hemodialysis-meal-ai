// buildTodayMealContext()(frontend/src/services/api.ts)에 대한 단위 테스트.
//
// frontend에는 아직 vitest/jest 같은 테스트 러너가 없고, 이 한 함수 때문에 새 npm
// 의존성을 추가하고 싶지 않았다. 대신 이미 devDependency로 딸려오는 esbuild(vite가
// 의존)의 JS API로 src/services/api.ts를 그 자리에서(파일을 남기지 않고, 메모리에서)
// 번들해 실제 프로덕션 소스를 node에서 그대로 실행/검증한다 — 로직을 다시 베껴 적어
// 테스트하면 구현이 바뀌어도 테스트가 따라 안 바뀌는 위험이 있어 피했다.
//
// 실행: cd frontend && node scripts/test-today-meal-context.mjs
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import { writeFile, unlink } from "node:fs/promises";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const esbuild = require("esbuild");

const projectRoot = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const entry = path.join(projectRoot, "src/services/api.ts");

const result = esbuild.buildSync({
  entryPoints: [entry],
  bundle: true,
  platform: "node",
  format: "esm",
  write: false,
  // api.ts 최상단이 (import.meta as any).env.VITE_API_URL을 읽는다 — vite 밖(순수 node)에서는
  // import.meta.env 자체가 없으므로, 테스트 목적으로만 빈 객체로 치환해 그 줄이 예외 없이
  // "" || "http://127.0.0.1:8000"으로 안전하게 폴백하게 한다(실제 값 자체는 이 테스트가
  // 검증하는 buildTodayMealContext()와 무관 — API 상수를 쓰는 다른 함수는 호출하지 않는다).
  define: { "import.meta.env": "{}" },
});

const bundledCode = result.outputFiles[0].text;
const tmpFile = path.join(tmpdir(), `api-bundle-${Date.now()}.mjs`);
await writeFile(tmpFile, bundledCode, "utf-8");
let buildTodayMealContext;
try {
  ({ buildTodayMealContext } = await import(pathToFileURL(tmpFile).href));
} finally {
  await unlink(tmpFile).catch(() => {});
}

// todayISO()(frontend/src/utils/date.ts)와 동일한 방식(로컬 기준)으로 오늘 날짜를 만든다.
// new Date().toISOString()(UTC)을 쓰면 KST 자정 근처에서 실제 함수와 어긋날 수 있어 피한다.
function localISO(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate(),
  ).padStart(2, "0")}`;
}
const TODAY = localISO(new Date());
const YESTERDAY = localISO(new Date(Date.now() - 24 * 60 * 60 * 1000));

const validIntake = (E = 500) => ({
  E,
  protein: 20,
  K: 800,
  P: 400,
  Na: 600,
  Na_season: 100,
});

let passed = 0;
let failed = 0;
function check(name, fn) {
  try {
    fn();
    passed++;
    console.log(`PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`FAIL  ${name}`);
    console.log(`      ${e.message}`);
  }
}

// ---- 1. 0 records today ----
check("1) 오늘 기록 0개 -> consumed/used_today 없음, mealsLeft=3", () => {
  const ctx = buildTodayMealContext([]);
  assert.equal(ctx.consumed, undefined);
  assert.equal(ctx.used_today, undefined);
  assert.equal(ctx.mealsLeft, 3);
  assert.equal(ctx.usedSlotCount, 0);
});

// ---- 2. 1 valid 아침 record ----
check("2) 유효한 아침 기록 1개 -> intake가 그대로 consumed, mealsLeft=2", () => {
  const breakfast = {
    id: "a1",
    title: "아침",
    subtitle: "",
    createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY,
    mealTime: "아침",
    raw_menus: ["된장찌개", "현미밥"],
    intake: validIntake(500),
  };
  const ctx = buildTodayMealContext([breakfast]);
  assert.deepEqual(ctx.consumed, validIntake(500));
  assert.deepEqual(new Set(ctx.used_today), new Set(["된장찌개", "현미밥"]));
  assert.equal(ctx.mealsLeft, 2);
  assert.equal(ctx.usedSlotCount, 1);
});

// ---- 3. 아침+점심 ----
check("3) 아침+점심 -> 합산 intake, 합집합 raw_menus, mealsLeft=1", () => {
  const breakfast = {
    id: "a1", title: "아침", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["된장찌개", "현미밥"], intake: validIntake(500),
  };
  const lunch = {
    id: "l1", title: "점심", subtitle: "", createdAt: `${TODAY}T05:00:00.000Z`,
    mealDate: TODAY, mealTime: "점심",
    raw_menus: ["고등어조림", "현미밥"], intake: validIntake(700),
  };
  const ctx = buildTodayMealContext([breakfast, lunch]);
  assert.deepEqual(ctx.consumed, {
    E: 1200, protein: 40, K: 1600, P: 800, Na: 1200, Na_season: 200,
  });
  assert.deepEqual(
    new Set(ctx.used_today),
    new Set(["된장찌개", "현미밥", "고등어조림"]),
  );
  assert.equal(ctx.mealsLeft, 1);
  assert.equal(ctx.usedSlotCount, 2);
});

// ---- 4. 아침 기록 2개(다른 createdAt) -> 최신만 카운트 ----
check("4) 같은 슬롯(아침) 기록 2개 -> 최신 createdAt 하나만 반영(중복 합산 없음)", () => {
  const older = {
    id: "a1", title: "아침(옛날)", subtitle: "", createdAt: `${TODAY}T00:00:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["김치찌개"], intake: validIntake(300),
  };
  const newer = {
    id: "a2", title: "아침(최신)", subtitle: "", createdAt: `${TODAY}T00:30:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["된장찌개"], intake: validIntake(500),
  };
  for (const arr of [[older, newer], [newer, older]]) {
    const ctx = buildTodayMealContext(arr);
    assert.deepEqual(ctx.consumed, validIntake(500), "최신(newer) intake만 반영돼야 함");
    assert.deepEqual(ctx.used_today, ["된장찌개"], "최신(newer) raw_menus만 반영돼야 함");
    assert.equal(ctx.mealsLeft, 2);
    assert.equal(ctx.usedSlotCount, 1, "같은 슬롯 2개가 1개로 dedup 되어야 함");
  }
});

// ---- 5. 어제 기록 -> 완전히 제외 ----
check("5) 어제 날짜 기록 -> 오늘 집계에 전혀 영향 없음", () => {
  const yesterdayRecord = {
    id: "y1", title: "어제 저녁", subtitle: "", createdAt: `${YESTERDAY}T10:00:00.000Z`,
    mealDate: YESTERDAY, mealTime: "저녁",
    raw_menus: ["삼겹살"], intake: validIntake(900),
  };
  const ctx = buildTodayMealContext([yesterdayRecord]);
  assert.equal(ctx.consumed, undefined);
  assert.equal(ctx.used_today, undefined);
  assert.equal(ctx.mealsLeft, 3);
  assert.equal(ctx.usedSlotCount, 0);
});

// ---- 6. intake 누락/NaN/음수 -> 영양 합산·mealsLeft 모두에서 제외되지만 raw_menus는 별개로 취급 ----
// (2026-08 수정: 예전엔 intake가 무효여도 "슬롯에 기록이 있었다"는 이유로 mealsLeft가
// 줄었는데, 이는 사용자 피드백에 따라 잘못된 동작으로 확인되어 고쳤다 — 아래 A/B/C 참고.)
check("6) intake가 결측/NaN/음수인 기록 -> consumed·mealsLeft 모두 제외, raw_menus만 유효", () => {
  const missingIntake = {
    id: "old1", title: "구버전 기록", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["옛날메뉴"],
    // intake 필드 자체가 없음(구버전 저장 데이터)
  };
  const ctxMissing = buildTodayMealContext([missingIntake]);
  assert.equal(ctxMissing.consumed, undefined, "intake 없는 기록은 합산에서 제외");
  assert.deepEqual(ctxMissing.used_today, ["옛날메뉴"], "raw_menus는 intake 검증과 무관하게 살아있어야 함");
  assert.equal(ctxMissing.mealsLeft, 3, "intake가 무효한 슬롯은 meals_left 계산에서도 빠져야 함");
  assert.equal(ctxMissing.usedSlotCount, 0);

  const nanIntake = {
    id: "nan1", title: "NaN 기록", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, mealTime: "점심",
    raw_menus: ["이상한메뉴"],
    intake: { E: NaN, protein: 20, K: 800, P: 400, Na: 600, Na_season: 100 },
  };
  const ctxNan = buildTodayMealContext([nanIntake]);
  assert.equal(ctxNan.consumed, undefined, "NaN이 섞인 intake는 통째로 제외");
  assert.deepEqual(ctxNan.used_today, ["이상한메뉴"]);
  assert.equal(ctxNan.mealsLeft, 3);
  assert.equal(ctxNan.usedSlotCount, 0);

  const negativeIntake = {
    id: "neg1", title: "음수 기록", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, mealTime: "저녁",
    raw_menus: ["음수메뉴"],
    intake: { E: -10, protein: 20, K: 800, P: 400, Na: 600, Na_season: 100 },
  };
  const ctxNeg = buildTodayMealContext([negativeIntake]);
  assert.equal(ctxNeg.consumed, undefined, "음수가 섞인 intake는 통째로 제외");
  assert.deepEqual(ctxNeg.used_today, ["음수메뉴"]);
  assert.equal(ctxNeg.mealsLeft, 3);
  assert.equal(ctxNeg.usedSlotCount, 0);

  // 셋(결측/NaN/음수)을 동시에 넣으면: consumed도 없고, 세 슬롯 다 intake가 무효라
  // usedSlotCount=0 -> mealsLeft = max(1, 3-0) = 3(구버전 기록만 있으면 기존 생성과 동일).
  // used_today(raw_menus)는 intake 유효성과 무관하게 셋 다 합쳐진다(메뉴 중복 방지는 별개 관심사).
  const ctxAll = buildTodayMealContext([missingIntake, nanIntake, negativeIntake]);
  assert.equal(ctxAll.consumed, undefined);
  assert.equal(ctxAll.usedSlotCount, 0);
  assert.equal(ctxAll.mealsLeft, 3);
  assert.deepEqual(
    new Set(ctxAll.used_today),
    new Set(["옛날메뉴", "이상한메뉴", "음수메뉴"]),
  );
});

// ---- A. 오늘 아침 기록 1개지만 intake 없음 -> consumed 없음, meals_left=3(기존 생성과 동일) ----
check("A) 아침 기록 1개, intake 없음 -> consumed 없음, meals_left=3", () => {
  const invalidBreakfast = {
    id: "a1", title: "아침(구버전)", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["된장찌개"],
    // intake 없음
  };
  const ctx = buildTodayMealContext([invalidBreakfast]);
  assert.equal(ctx.consumed, undefined);
  assert.equal(ctx.mealsLeft, 3, "유효한 intake가 없으므로 meals_left는 3이어야 함(줄면 안 됨)");
  assert.equal(ctx.usedSlotCount, 0);
});

// ---- B. 아침 invalid + 점심 valid -> 점심 intake만 consumed, meals_left=2(1이 아님) ----
check("B) 아침 invalid + 점심 valid -> 점심만 consumed 반영, meals_left=2", () => {
  const invalidBreakfast = {
    id: "a1", title: "아침(구버전)", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["된장찌개"],
    // intake 없음
  };
  const validLunch = {
    id: "l1", title: "점심", subtitle: "", createdAt: `${TODAY}T05:00:00.000Z`,
    mealDate: TODAY, mealTime: "점심",
    raw_menus: ["고등어조림"], intake: validIntake(700),
  };
  const ctx = buildTodayMealContext([invalidBreakfast, validLunch]);
  assert.deepEqual(ctx.consumed, validIntake(700), "점심 intake만 합산돼야 함(아침은 무효라 제외)");
  assert.equal(ctx.mealsLeft, 2, "유효한 슬롯은 점심 1개뿐이므로 meals_left=2여야 함(1이면 안 됨)");
  assert.equal(ctx.usedSlotCount, 1);
});

// ---- C. 같은 끼니: 최신 기록이 invalid, 과거 기록이 valid -> 과거 기록으로 fallback 금지 ----
check("C) 같은 슬롯(아침) 최신=invalid, 과거=valid -> 과거로 fallback하지 않고 그 슬롯 전체 제외", () => {
  const olderValid = {
    id: "a1", title: "아침(과거, 정상)", subtitle: "", createdAt: `${TODAY}T00:00:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["김치찌개"], intake: validIntake(300),
  };
  const newerInvalid = {
    id: "a2", title: "아침(최신, 구버전)", subtitle: "", createdAt: `${TODAY}T00:30:00.000Z`,
    mealDate: TODAY, mealTime: "아침",
    raw_menus: ["된장찌개"],
    // intake 없음 — "최신 기록 1개만 사용" 규칙에 따라 이 기록이 그 슬롯을 대표한다.
  };
  const ctx = buildTodayMealContext([olderValid, newerInvalid]);
  // "최신 기록만 쓴다"는 선택 규칙 자체는 그대로 유지 — 과거의 유효했던 300kcal로
  // 임의로 대체(fallback)하지 않는다. 그 결과 이 슬롯은 intake 관점에서 통째로 무효
  // 취급되어 consumed/meals_left 어디에도 기여하지 않는다.
  assert.equal(ctx.consumed, undefined, "과거 valid 기록으로 fallback하면 안 됨 -> consumed 없음");
  assert.equal(ctx.mealsLeft, 3, "이 슬롯은 무효로 취급되어 meals_left에 반영되면 안 됨");
  assert.equal(ctx.usedSlotCount, 0);
  // raw_menus는 여전히 "최신 기록"(된장찌개) 기준 -> 과거 기록(김치찌개)은 안 쓰임.
  assert.deepEqual(ctx.used_today, ["된장찌개"], "used_today도 최신 기록 기준이어야 함(과거 기록 아님)");
});

// ---- 7 (추가). mealTime이 비표준 값이거나 없는 경우 -> 슬롯 집계에서 아예 제외 ----
check("7) mealTime이 없거나 비표준 값이면 집계에서 완전히 제외(슬롯 카운트도 안 늘어남)", () => {
  const noMealTime = {
    id: "n1", title: "끼니 미지정", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, // mealTime 없음
    raw_menus: ["아무거나"], intake: validIntake(500),
  };
  const weirdMealTime = {
    id: "w1", title: "간식", subtitle: "", createdAt: `${TODAY}T00:10:00.000Z`,
    mealDate: TODAY, mealTime: "간식", // 표준 3값이 아님
    raw_menus: ["과자"], intake: validIntake(200),
  };
  const ctx = buildTodayMealContext([noMealTime, weirdMealTime]);
  assert.equal(ctx.consumed, undefined);
  assert.equal(ctx.used_today, undefined);
  assert.equal(ctx.mealsLeft, 3);
  assert.equal(ctx.usedSlotCount, 0);
});

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
