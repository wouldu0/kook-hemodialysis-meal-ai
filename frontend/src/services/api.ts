// 백엔드 통신 + 로컬/서버 동기화 저장소. App.tsx의 화면 컴포넌트들은 이 모듈의
// 함수만 통해서 서버와 이야기한다(직접 fetch를 새로 쓰지 않는다).
import type { ApiResult, DayPlanResult, SavedItem, SavedUser } from "../types";

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
  const { responseType = "json", timeoutMs = 75000, ...init } = options;
  const headers = new Headers(init.headers || {});
  if (!headers.has("Content-Type") && init.body)
    headers.set("Content-Type", "application/json");
  const token = authToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const controller = new AbortController();
  // 무료 호스팅은 요청이 없으면 잠들고, 깨어나는 데 1분 가까이 걸린다.
  // 12초로 끊으면 잠든 직후의 첫 요청이 무조건 실패하므로 넉넉히 잡는다.
  const tid = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const r = await fetch(`${API}${path}`, {
      ...init,
      headers,
      signal: controller.signal,
    });
    clearTimeout(tid);
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
    clearTimeout(tid);
    if (e.name === "AbortError")
      throw new Error(
        "서버 응답 속도가 느려 시간 초과되었습니다. 잠시 후 다시 시도해주세요.",
      );
    // 네트워크 자체가 안 닿으면 브라우저는 "Failed to fetch"라는 원문만 준다.
    // (백엔드 미실행, 또는 서버가 아직 로딩 중이라 포트가 안 열린 경우)
    if (e instanceof TypeError)
      throw new Error(
        "서버에 연결하지 못했습니다. 백엔드가 실행 중인지 확인해주세요.",
      );
    throw e;
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
export async function saveEverywhere(key: string, item: SavedItem) {
  addSaved(key, item);
  const resource = RESOURCE_KEY_MAP[key];
  if (!resource || !authToken()) return;
  try {
    await apiFetch(`/me/${resource}`, {
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
  } catch {
    // 서버 저장 실패는 조용히 무시 — 로컬엔 이미 저장됐으니 화면은 정상 동작
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
    const seen = new Set(
      items.map((x) => `${x.title}|${x.subtitle}|${x.createdAt?.slice(0, 16)}`),
    );
    const extras = local.filter(
      (x) => !seen.has(`${x.title}|${x.subtitle}|${x.createdAt?.slice(0, 16)}`),
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

// ────────────────────────── 엔드포인트별 함수 ──────────────────────────
// 화면 컴포넌트는 아래 함수만 부르면 된다 — URL·타임아웃·헤더·에러 처리는
// 전부 apiFetch()가 책임진다.

export function warmupBackend() {
  // 무료 호스팅은 한동안 요청이 없으면 잠든다. 온보딩을 보는 동안 미리 깨워둔다.
  return apiFetch("/health").catch(() => {});
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

export function generateMeal(body: Record<string, unknown>): Promise<ApiResult> {
  // 조건에 맞는 조합을 찾을 때까지 재시도하는 구조라 원래 느리다.
  return apiFetch("/generate", { method: "POST", body: JSON.stringify(body), timeoutMs: 60000 });
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
