// 백엔드 통신 + 로컬/서버 동기화 저장소. App.tsx의 화면 컴포넌트들은 이 모듈의
// 함수만 통해서 서버와 이야기한다(직접 fetch를 새로 쓰지 않는다).
import type { SavedItem, SavedUser } from "../types";

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

// 백엔드 주소. 평소에는 빌드 시점에 박힌 값(.env.production)을 쓴다.
//
// 비상용 우회로: 배포된 백엔드가 죽었을 때 주소 뒤에 ?api=... 를 붙이면
// 그 백엔드로 갈아탄다. 한 번 붙이면 브라우저에 기억되므로 이후에는 그냥 열면 된다.
//   예) https://kook-omega.vercel.app/?api=https://abc-def.trycloudflare.com
// 원래대로 되돌리려면  ?api=reset  으로 열면 된다.
//
// ⚠️ 보안: 이 값을 그대로 신뢰하면 안 된다 — apiFetch()가 로그인 토큰을
// `Authorization: Bearer ...`로 이 주소에 보내므로, 누군가 조작한 링크(?api=공격자서버)를
// 열면 로그인 세션이 그쪽으로 새어나갈 수 있다. 그래서 (1) https만 허용하고
// (2) 실제로 바뀔 때만 confirm()으로 주소를 보여주고 동의를 받은 뒤 저장한다.
export function isValidApiOverride(raw: string): string | null {
  try {
    const url = new URL(raw);
    if (url.protocol !== "https:") return null;
    return raw.replace(/\/+$/, "");
  } catch {
    return null;
  }
}

export const API = (() => {
  const fallback =
    (import.meta as any).env.VITE_API_URL || "http://127.0.0.1:8000";
  try {
    const q = new URLSearchParams(location.search).get("api");
    if (q === "reset") {
      localStorage.removeItem("fook:api");
      return fallback;
    }
    if (q) {
      const url = isValidApiOverride(q);
      const saved = localStorage.getItem("fook:api");
      if (!url) return saved || fallback; // https가 아니면 그냥 무시
      if (url === saved) return url; // 이미 동의하고 저장된 주소면 다시 안 물어봄
      const ok = window.confirm(
        `백엔드 서버 주소를 다음으로 바꿉니다:\n\n${url}\n\n` +
          "이 주소를 신뢰할 수 있는 경우에만 확인을 누르세요. " +
          "로그인 상태라면 인증 정보가 이 서버로 전송됩니다.",
      );
      if (!ok) return saved || fallback;
      localStorage.setItem("fook:api", url);
      return url;
    }
    return localStorage.getItem("fook:api") || fallback;
  } catch {
    return fallback;
  }
})();

export const currentUser = () => storage.get<SavedUser | null>("fook:user", null);
export const authToken = () => localStorage.getItem("fook:token") || "";

export async function apiFetch(path: string, options: RequestInit = {}) {
  const headers = new Headers(options.headers || {});
  if (!headers.has("Content-Type") && options.body)
    headers.set("Content-Type", "application/json");
  const token = authToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  const controller = new AbortController();
  // 무료 호스팅은 요청이 없으면 잠들고, 깨어나는 데 1분 가까이 걸린다.
  // 12초로 끊으면 잠든 직후의 첫 요청이 무조건 실패하므로 넉넉히 잡는다.
  const tid = setTimeout(() => controller.abort(), 75000);
  try {
    const r = await fetch(`${API}${path}`, {
      ...options,
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
