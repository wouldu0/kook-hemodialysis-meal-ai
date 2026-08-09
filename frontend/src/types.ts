// 앱 전역에서 쓰는 타입 모음.

export type Profile = {
  gender: string;
  birthdate: string; // YYYY-MM-DD. age는 여기서 계산해서 화면에만 보여준다.
  age: string;
  height: string;
  weight: string;
  dialysis: "혈액투석";
};

export type MenuRecord = {
  name: string;
  ingredients: readonly string[];
  steps: readonly string[];
  nutrition: {
    energy: number;
    protein: number;
    potassium: number;
    phosphorus: number;
    sodium: number;
  };
};

export type Plan = {
  id: string;
  day: number;
  slot: string;
  menus: readonly string[];
};

export type NutrientKey =
  | "energy"
  | "protein"
  | "phosphorus"
  | "potassium"
  | "sodium";

export type SavedUser = { id: string; username: string; name: string };

// 끼니 구분 — 식단 관리 화면에서 아침/점심/저녁 섹션으로 나누는 기준
export type MealTime = "아침" | "점심" | "저녁";

export type SavedItem = {
  id: string;
  title: string;
  subtitle: string;
  createdAt: string;
  menus?: readonly string[];
  mealDate?: string; // 사용자가 고른 날짜 (YYYY-MM-DD)
  mealTime?: MealTime; // 사용자가 고른 끼니
};

// backend/app_core_FOOK.py의 meal_result()가 실제로 내려주는 응답 형태.
// (backend/README.md, docs/DEVLOG.md와 실제 /generate 응답을 대조해 맞췄다.)
export type NutritionBlock = {
  energy: number;
  protein: number;
  potassium: number;
  phosphorus: number;
  sodium: number;
  sodium_total: number;
};

export interface ApiResult {
  mode?: "menu" | "ingredient" | "random";
  anchor?: string | null;
  meal?: string[];
  raw_menus?: string[];
  nutrition?: NutritionBlock;
  // 레버로 재조정하기 전(원본 레시피) 영양값 — '재구성 전/후' 비교 화면용.
  nutrition_before?: NutritionBlock;
  targets?: {
    energy: [number, number];
    protein: [number, number];
    potassium: number;
    phosphorus: number;
    sodium: number;
  };
  passed?: boolean;
  note?: string;
  warning?: string;
  // 레버가 원본 대비 뭘 바꿨는지 사람이 읽을 문장 목록 (app_core_FOOK._changes()).
  changes?: string[];
  // 메뉴별 최종 재료: {메뉴명: [[재료명, 양g], ...]}
  dish_ingredients?: Record<string, [string, number][]>;
  dish_nutrition?: Record<string, NutritionBlock>;
  snacks?: { menu: string; amount_g: number }[];
  // 재료 교체로 이름이 바뀐 메뉴의 원본명: {표시명: 원본메뉴명}
  recipe_source?: Record<string, string>;
  intake?: {
    E: number;
    protein: number;
    K: number;
    P: number;
    Na: number;
    Na_season: number;
  };
}

export type AppState = {
  profile: Profile;
  setProfile: (p: Profile) => void;
  plan: Plan;
  setPlan: (p: Plan) => void;
  query: string;
  setQuery: (q: string) => void;
  searchMode: "menu" | "ingredient" | "random";
  setSearchMode: (m: "menu" | "ingredient" | "random") => void;
  apiResult: ApiResult | null;
  setApiResult: (r: ApiResult | null) => void;
  // true = 서버 생성이 실패해서 내장 예시 데이터로 화면을 보여주는 중.
  // 이때는 영양 판정 뱃지가 실제 개인 맞춤 계산이 아니라는 걸 화면에 계속 표시해야 한다.
  usingFallback: boolean;
  setUsingFallback: (v: boolean) => void;
  // 메뉴별로 LLM이 재구성한 조리법(steps)을 캐싱해둔다. 레시피 상세 화면에서 한 번 펼쳐본
  // 메뉴는 여기 남아있어서, PDF 미리보기가 같은 메뉴를 또 /recipe로 불러오지 않고 재사용한다.
  dishSteps: Record<string, string[]>;
  setDishSteps: (menu: string, steps: string[]) => void;
};
