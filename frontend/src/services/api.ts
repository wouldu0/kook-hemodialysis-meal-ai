// 백엔드 통신 + 로컬/서버 동기화 저장소. App.tsx의 화면 컴포넌트들은 이 모듈의
// 함수만 통해서 서버와 이야기한다(직접 fetch를 새로 쓰지 않는다).
import type { ApiResult, DayPlanResult, MealTime, SavedItem, SavedUser } from "../types";
import { todayISO } from "../utils/date";

export const storage = {
  get<T>(key: string, fallback: T): T {
    try {
      return JSON.parse(localStorage.getItem(key) || "") as T;
    } catch {
      return fallback;
    }
  },
  set(key: string, value: unknown) {
    localStorage.setItem(key, JSON.stringify(value));
  },
};

// 백엔드 주소. 빌드 시점에 박힌 값(.env.production의 VITE_API_URL)을 그대로 쓴다.
// (예전엔 ?api=로 배포 중에 다른 백엔드로 갈아탈 수 있는 비상 우회 기능이 있었지만,
// 로그인 토큰을 그 주소로 보내는 구조 자체가 위험 요소라 판단해 제거했다. 백엔드
// 주소를 바꿔야 하면 VITE_API_URL을 바꾸고 다시 배포한다.)
export const API =
  (import.meta as any).env.VITE_API_URL || "http://127.0.0.1:8000";

export const currentUser = () => storage.get<SavedUser | null>("fook:user", null);
export const authToken = () => localStorage.getItem("fook:token") || "";

export type ApiFetchOptions = RequestInit & {
  // 대부분 응답은 JSON이지만 /tts는 오디오(mp3)를 blob으로 돌려준다.
  responseType?: "json" | "blob";
  // /generate_day처럼 유난히 느린 엔드포인트는 기본 타임아웃(75초)보다 더 여유를 준다.
  timeoutMs?: number;
};

export async function apiFetch(path: string, options: ApiFetchOptions = {}) {
  const {
    responseType = "json",
    timeoutMs = 75000,
    signal: externalSignal,
    ...init
  } = options;
  const headers = new Headers(init.headers || {});
  if (!headers.has("Content-Type") && init.body)
    headers.set("Content-Type", "application/json");
  const token = authToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);

  // 요청별 timeout과 호출부의 AbortSignal을 하나의 controller로 합친다.
  // 이전에는 apiFetch가 자체 signal로 덮어써서 화면 이탈 시 외부 abort가 실제 fetch까지
  // 전달되지 않았다.
  const controller = new AbortController();
  let timedOut = false;
  const abortFromExternal = () => controller.abort();
  if (externalSignal?.aborted) controller.abort();
  else externalSignal?.addEventListener("abort", abortFromExternal, { once: true });

  // 무료 호스팅은 요청이 없으면 잠들고, 깨어나는 데 1분 가까이 걸린다.
  const tid = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const r = await fetch(`${API}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });
    if (!r.ok) {
      let msg = "요청을 처리하지 못했습니다.";
      try {
        const d = await r.json();
        msg = d.detail || msg;
      } catch {}
      throw new Error(msg);
    }
    if (responseType === "blob") return r.blob();
    return r.status === 204 ? null : r.json();
  } catch (e: any) {
    if (e.name === "AbortError") {
      if (externalSignal?.aborted && !timedOut) throw e;
      throw new Error(
        "서버 응답 속도가 느려 시간 초과되었습니다. 잠시 후 다시 시도해주세요.",
      );
    }
    // 네트워크 자체가 안 닿으면 브라우저는 "Failed to fetch"라는 원문만 준다.
    // (백엔드 미실행, 또는 서버가 아직 로딩 중이라 포트가 안 열린 경우)
    if (e instanceof TypeError)
      throw new Error(
        "서버에 연결하지 못했습니다. 백엔드가 실행 중인지 확인해주세요.",
      );
    throw e;
  } finally {
    clearTimeout(tid);
    externalSignal?.removeEventListener("abort", abortFromExternal);
  }
}

export function saveSession(data: any) {
  localStorage.setItem("fook:token", data.token);
  storage.set("fook:user", data.user);
}

export const addSaved = (key: string, item: SavedItem) => {
  const list = storage.get<SavedItem[]>(key, []);
  storage.set(key, [item, ...list.filter((x) => x.id !== item.id)]);
};

// 로컬스토리지 키 'fook:favorites'|'fook:history'|'fook:documents' ↔ 서버 리소스 이름 매핑.
// 서버는 /me/{meal-records|favorites|documents} 로 실제 DB에 저장한다(회원별로 기기 간 동기화됨).
export const RESOURCE_KEY_MAP: Record<string, string> = {
  "fook:history": "meal-records",
  "fook:favorites": "favorites",
  "fook:documents": "documents",
};

// 저장: 로그인 상태면 서버(DB)에도 저장한다. 서버 실패해도 로컬엔 남겨서 화면은 끊기지 않는다.
//
// item.id는 호출부(FinalMealPage 등)가 `meal-${Date.now()}`처럼 임시로 만든 클라이언트 id다.
// 서버 저장이 성공하면 서버가 진짜 id(UUID, POST /me/{resource} 응답의 `id`)를 돌려주는데,
// 이걸 무시하고 클라이언트 id를 그대로 로컬에 남겨두면 이후 loadEverywhere()의 서버 목록과
// 이 로컬 항목의 id가 영영 어긋나서, deleteEverywhere()가 로컬에서 엉뚱한(또는 아무) 항목도
// 못 지우고 서버에서 지운 항목이 다음 조회 때 로컬 잔재로 되살아나는 문제가 생긴다. 그래서
// 성공 시 로컬에 방금 저장해둔 그 항목의 id를 서버 id로 바로 맞춰 놓는다.
export async function saveEverywhere(key: string, item: SavedItem) {
  addSaved(key, item);
  const resource = RESOURCE_KEY_MAP[key];
  if (!resource || !authToken()) return;
  try {
    const res = await apiFetch(`/me/${resource}`, {
      method: "POST",
      body: JSON.stringify({
        title: item.title,
        subtitle: item.subtitle,
        payload: {
          menus: item.menus,
          createdAt: item.createdAt,
          mealDate: item.mealDate,
          mealTime: item.mealTime,
          raw_menus: item.raw_menus,
          intake: item.intake,
          dish_ingredients: item.dish_ingredients,
        },
      }),
    });
    if (res?.id && res.id !== item.id) {
      const list = storage.get<SavedItem[]>(key, []);
      storage.set(
        key,
        list.map((x) => (x.id === item.id ? { ...x, id: res.id } : x)),
      );
    }
  } catch {
    // 서버 저장 실패는 조용히 무시 — 로컬엔 이미 클라이언트 id로 저장돼 있으니(local-only
    // fallback) 화면은 정상 동작. id 동기화는 다음 성공한 저장에서나 일어난다.
  }
}

// 조회: 로그인 상태면 서버(DB) 목록을 우선 사용, 아니면 로컬 목록을 그대로 쓴다.
export async function loadEverywhere(key: string): Promise<SavedItem[]> {
  const resource = RESOURCE_KEY_MAP[key];
  const local = storage.get<SavedItem[]>(key, []);
  if (!resource || !authToken()) return local;
  try {
    const d = await apiFetch(`/me/${resource}`);
    const items: SavedItem[] = (d?.items || []).map((r: any) => ({
      id: r.id,
      title: r.title,
      subtitle: r.subtitle || "",
      createdAt: r.created_at,
      menus: r.payload?.menus || [],
      mealDate: r.payload?.mealDate,
      mealTime: r.payload?.mealTime,
      raw_menus: r.payload?.raw_menus,
      intake: r.payload?.intake,
      dish_ingredients: r.payload?.dish_ingredients,
    }));
    // 서버 저장이 실패했던 항목은 로컬에만 남아 있다. 서버 목록으로 덮어쓰면
    // 방금 저장한 게 사라져 보이므로, 서버에 없는 로컬 항목은 함께 보여준다.
    //
    // saveEverywhere()가 성공 시 로컬 id를 서버 id로 맞춰두므로, 정상 동기화된 항목은
    // id만 비교해도 정확히 걸러진다(주 판별 기준). title|subtitle|createdAt 방식은
    // 이 id 동기화가 생기기 전에 저장된 예전 로컬 항목이나 서버 저장이 실패해 클라이언트
    // id를 그대로 갖고 있는 항목까지 놓치지 않기 위한 보조 판별로 남겨둔다.
    const serverIds = new Set(items.map((x) => x.id));
    const seen = new Set(
      items.map((x) => `${x.title}|${x.subtitle}|${x.createdAt?.slice(0, 16)}`),
    );
    const extras = local.filter(
      (x) =>
        !serverIds.has(x.id) &&
        !seen.has(`${x.title}|${x.subtitle}|${x.createdAt?.slice(0, 16)}`),
    );
    return [...extras, ...items].sort((a, b) =>
      String(b.createdAt).localeCompare(String(a.createdAt)),
    );
  } catch {
    return local;
  }
}

export async function deleteEverywhere(key: string, id: string) {
  storage.set(
    key,
    storage.get<SavedItem[]>(key, []).filter((x) => x.id !== id),
  );
  const resource = RESOURCE_KEY_MAP[key];
  if (!resource || !authToken()) return;
  try {
    await apiFetch(`/me/${resource}/${id}`, { method: "DELETE" });
  } catch {
    // 무시
  }
}

// ────────────────────────── 오늘 기록한 식사 → /generate 컨텍스트 ──────────────────────────
// "기록하기"(FinalMealPage)로 저장된 fook:history(=meal-records)에는 이미 raw_menus/intake가
// 함께 들어있는데(FinalMealPage.tsx 참고), /generate는 지금까지 이걸 전혀 안 보고 매번
// meals_left: 3(오늘 아무것도 안 먹은 것처럼)으로만 호출해왔다. 여기서는 그 이력 배열 하나를
// 받아 순수하게 {consumed, used_today, mealsLeft, usedSlotCount}로 접어주기만 한다 — fetch나
// localStorage 접근이 전혀 없는 순수 함수라 단위 테스트가 쉽다(호출부는 GeneratingPage.tsx).

// server_FOOK.CONSUMED_KEYS / parse_consumed()와 반드시 같은 키 목록이어야 한다 — 다르면
// consumed를 만들어도 백엔드가 422로 거부한다.
const CONSUMED_KEYS = ["E", "protein", "K", "P", "Na", "Na_season"] as const;

const STANDARD_MEAL_TIMES: MealTime[] = ["아침", "점심", "저녁"];

function isValidIntake(
  intake: SavedItem["intake"],
): intake is NonNullable<SavedItem["intake"]> {
  if (!intake) return false;
  return CONSUMED_KEYS.every((k) => {
    const v = (intake as Record<string, unknown>)[k];
    return typeof v === "number" && Number.isFinite(v) && v >= 0;
  });
}

export type TodayMealContext = {
  // 유효한 intake를 가진 오늘 기록이 하나도 없으면 아예 안 넣는다(fail-open — history가
  // 없는 것과 동일하게 동작).
  consumed?: NonNullable<SavedItem["intake"]>;
  used_today?: string[];
  // 이번 요청(이번 끼 포함) 기준 남은 끼니 수. 항상 1~3.
  mealsLeft: number;
  // 오늘 "유효한 intake를 가진" 최신 끼니 슬롯 수(아침/점심/저녁 중 최대 3) — consumed
  // 합산에 실제로 기여한 슬롯 수와 정확히 같다. UI 힌트("N끼를 반영해...")와 mealsLeft
  // 계산의 근거가 되는 값이다. intake가 없거나 구버전이라 consumed에서 제외된 슬롯은
  // 여기에도 포함하지 않는다 — "기록은 있지만 영양 데이터가 없는" 끼니 때문에 meals_left가
  // 부당하게 줄어들면 안 되기 때문(2026-08 수정 — 이전에는 intake 유효성과 무관하게
  // 슬롯 존재 여부만 셌었다).
  usedSlotCount: number;
};

// today의 fook:history 레코드 배열 → /generate에 실어보낼 컨텍스트.
// - 오늘(todayISO()) 날짜가 아닌 기록은 전부 제외.
// - 같은 mealTime(아침/점심/저녁) 안에서는 createdAt이 가장 늦은 기록 하나만 남긴다(중복 방지).
//   mealTime이 없거나 아침/점심/저녁이 아닌 기록은 이 집계에서 제외한다(기록 자체는 그대로 둠).
//   이 "최신 1건" 선택은 그 기록의 intake 유효성과 무관하다 — 같은 슬롯에 예전의 유효한
//   기록이 있어도, 최신 기록이 intake가 없으면 그 과거 기록으로 임의로 대체(fallback)하지
//   않는다(최신 기록이 실제로 그 끼니를 대표한다고 보기 때문).
// - consumed: 선택된 기록 중 intake의 6개 키(E/protein/K/P/Na/Na_season)가 전부 있고, 전부
//   유한하고 0 이상인 숫자인 기록만 합산한다. 하나도 없으면 consumed를 아예 안 넣는다.
// - used_today: 선택된 기록들의 raw_menus를 합쳐 중복 제거(빈 값 제외). 비어 있으면 안 넣는다.
// - mealsLeft: max(1, 3 - usedSlotCount). usedSlotCount는 선택된 기록 중 intake가 유효한
//   것만 센다(=consumed 합산에 실제로 기여한 슬롯 수) — intake 없는/구버전 기록만 있는
//   슬롯은 "아직 기록 안 한 것"과 동일하게 취급한다. 예: 오늘 아침 기록이 있어도 intake가
//   없으면 그 아침은 meals_left 계산에서도, consumed 합산에서도 빠진다.
export function buildTodayMealContext(history: SavedItem[]): TodayMealContext {
  const today = todayISO();
  const todays = (history || []).filter(
    (item) => item && item.mealDate === today,
  );

  const latestBySlot = new Map<MealTime, SavedItem>();
  for (const item of todays) {
    const slot = item.mealTime;
    if (!slot || !STANDARD_MEAL_TIMES.includes(slot)) continue;
    const existing = latestBySlot.get(slot);
    if (!existing || String(item.createdAt) > String(existing.createdAt)) {
      latestBySlot.set(slot, item);
    }
  }
  const selected = Array.from(latestBySlot.values());

  let consumed: NonNullable<SavedItem["intake"]> | undefined;
  let usedSlotCount = 0;
  for (const item of selected) {
    if (!isValidIntake(item.intake)) continue;
    usedSlotCount += 1;
    if (!consumed) consumed = { E: 0, protein: 0, K: 0, P: 0, Na: 0, Na_season: 0 };
    for (const k of CONSUMED_KEYS) consumed[k] += item.intake![k];
  }

  const usedMenus = new Set<string>();
  for (const item of selected) {
    for (const m of item.raw_menus || []) {
      if (m) usedMenus.add(m);
    }
  }

  return {
    consumed,
    used_today: usedMenus.size ? Array.from(usedMenus) : undefined,
    mealsLeft: Math.max(1, 3 - usedSlotCount),
    usedSlotCount,
  };
}

// ────────────────────────── 엔드포인트별 함수 ──────────────────────────
// 화면 컴포넌트는 아래 함수만 부르면 된다 — URL·타임아웃·헤더·에러 처리는
// 전부 apiFetch()가 책임진다.

let backendWarmupPromise: Promise<unknown> | null = null;
let backendReady = false;

export function isBackendReady() {
  return backendReady;
}

export function warmupBackend() {
  // 앱 전체에서 하나의 /health 요청만 공유한다. 서버가 깨어나는 중 생성 화면을 나갔다가
  // 다시 들어와도 새 /health 요청을 쌓지 않고 이미 진행 중인 warm-up을 그대로 기다린다.
  if (backendReady) return Promise.resolve();
  if (backendWarmupPromise) return backendWarmupPromise;

  backendWarmupPromise = apiFetch("/health", { timeoutMs: 120000 })
    .then((result) => {
      backendReady = true;
      return result;
    })
    .catch((error) => {
      // 실패한 Promise를 영구 캐시하면 이후 재시도가 불가능하므로 다시 시도할 수 있게 비운다.
      backendWarmupPromise = null;
      throw error;
    });
  return backendWarmupPromise;
}

export function getMenus(): Promise<{ menus: string[] }> {
  return apiFetch("/menus");
}

export function getIngredients(): Promise<{ ingredients: string[] }> {
  return apiFetch("/ingredients");
}

export function getMenusByIngredient(q: string): Promise<{ menus: string[] }> {
  return apiFetch(`/menus_by_ingredient?q=${encodeURIComponent(q)}`);
}

export function generateMeal(
  body: Record<string, unknown>,
  options: { signal?: AbortSignal } = {},
): Promise<ApiResult> {
  // cold start는 warmupBackend() 단계에서 먼저 끝낸다. 이 120초는 네트워크 이상 등
  // 예외 상황에서 생성 요청이 무한정 매달리지 않게 하는 안전장치다.
  return apiFetch("/generate", {
    method: "POST",
    body: JSON.stringify(body),
    timeoutMs: 120000,
    signal: options.signal,
  });
}

export function generateDayPlan(body: Record<string, unknown>): Promise<DayPlanResult> {
  // 세 끼를 이어서 계산하므로 한 끼 생성보다 훨씬 오래 걸린다.
  return apiFetch("/generate_day", { method: "POST", body: JSON.stringify(body), timeoutMs: 90000 });
}

export function generateRecipe(payload: {
  menu: string;
  ingredients: [string, number][];
  source?: string;
}): Promise<{ menu: string; steps?: string; error?: string }> {
  return apiFetch("/recipe", { method: "POST", body: JSON.stringify(payload) });
}

export function textToSpeech(text: string): Promise<Blob> {
  return apiFetch("/tts", {
    method: "POST",
    body: JSON.stringify({ text }),
    responseType: "blob",
  });
}

export function getPotassiumTips(): Promise<{
  tips: { category: string; steps: { title: string; detail: string }[] }[];
}> {
  return apiFetch("/veg_potassium_tips");
}

export function login(username: string, password: string) {
  return apiFetch("/auth/login", {
    method: "POST",
    body: JSON.stringify({ username, password }),
  });
}

export function signup(name: string, username: string, password: string) {
  return apiFetch("/auth/signup", {
    method: "POST",
    body: JSON.stringify({ name, username, password }),
  });
}

export function findId(name: string, birthdate: string): Promise<{ usernames: string[] }> {
  return apiFetch("/auth/find-id", {
    method: "POST",
    body: JSON.stringify({ name, birthdate }),
  });
}

export function resetPassword(
  username: string,
  name: string,
  birthdate: string,
  newPassword: string,
) {
  return apiFetch("/auth/reset-password", {
    method: "POST",
    body: JSON.stringify({ username, name, birthdate, new_password: newPassword }),
  });
}

export function updateProfile(payload: {
  gender: string;
  birthdate: string;
  height: number;
  weight: number;
  dialysis: string;
}) {
  return apiFetch("/me/profile", { method: "PUT", body: JSON.stringify(payload) });
}

export function getMe(): Promise<{ user: any; profile: any }> {
  return apiFetch("/me");
}

// RAG 영양 상담 챗봇. 현재는 개인화(weight/consumed/meals_left) 없이 일반 질문만 보낸다 —
// "오늘 이미 먹은 양(consumed)"을 로그인 유저 기준으로 정확히 집계하는 방법이 프론트에 아직
// 없어서(끼니별 기록은 있어도 "오늘 하루 합계"를 만들어주는 화면/엔드포인트가 없음), 근거 없는
// 값을 지어내 보내느니 일반 질문 모드로만 동작시킨다(백엔드 chat()도 weight/consumed가 없으면
// 자동으로 일반 답변 모드로 폴백한다).
//
// contextFood(선택, 2026-08-14 추가): 대화 이력 전체가 아니라 "직전 응답이 어떤 재료에 대한
// 답이었는가" 딱 그 값 하나만 다음 요청에 그대로 되돌려 보낸다 — "그럼 몇 조각?" 같은 한 턴짜리
// 음식 후속 질문 지원용(백엔드 FOOK_rag_chatbot.answer_with_context() 참고). null/undefined면
// 요청 바디에 아예 안 넣는다(기존 요청 형태를 그대로 유지 — 하위호환).
export function askChat(
  question: string,
  contextFood?: string | null
): Promise<{ answer: string; sources: string[]; context_food?: string | null }> {
  const body: Record<string, unknown> = { question };
  if (contextFood) body.context_food = contextFood;
  return apiFetch("/chat", {
    method: "POST",
    body: JSON.stringify(body),
    timeoutMs: 45000,
  });
}

export function logout() {
  return apiFetch("/auth/logout", { method: "POST" });
}
