import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { useApp } from "../../hooks/useApp";
import { fetchDayTargets, type DayTargets, updateProfile } from "../../services/api";
import type { CustomTargets } from "../../types";
import { fmt } from "../../utils/nutrition";

type Draft = {
  energyLo: string;
  energyHi: string;
  proteinLo: string;
  proteinHi: string;
  potassium: string;
  phosphorus: string;
  sodium: string;
};

function draftFromCustomTargets(ct?: CustomTargets): Draft {
  return {
    energyLo: ct?.energy?.[0]?.toString() ?? "",
    energyHi: ct?.energy?.[1]?.toString() ?? "",
    proteinLo: ct?.protein?.[0]?.toString() ?? "",
    proteinHi: ct?.protein?.[1]?.toString() ?? "",
    potassium: ct?.potassium?.toString() ?? "",
    phosphorus: ct?.phosphorus?.toString() ?? "",
    sodium: ct?.sodium?.toString() ?? "",
  };
}

// draft(입력 중인 문자열) → 실제 서버에 보낼 CustomTargets. 빈 칸은 그 항목을 "자동 산출값
// 그대로 사용"으로 취급해 아예 안 담는다 — 값을 지어내 보내지 않는다.
// validateDraft()를 먼저 통과한 draft만 여기 들어온다는 전제라, 여기서는 파싱만 한다.
function customTargetsFromDraft(d: Draft): CustomTargets {
  const ct: CustomTargets = {};
  if (d.energyLo && d.energyHi) ct.energy = [Number(d.energyLo), Number(d.energyHi)];
  if (d.proteinLo && d.proteinHi) ct.protein = [Number(d.proteinLo), Number(d.proteinHi)];
  if (d.potassium) ct.potassium = Number(d.potassium);
  if (d.phosphorus) ct.phosphorus = Number(d.phosphorus);
  if (d.sodium) ct.sodium = Number(d.sodium);
  return ct;
}

// 저장 전 클라이언트 쪽 검증. 통과하지 못하면 에러 문구를 반환(그대로 화면에 표시)하고
// 저장을 아예 안 보낸다 — 잘못 입력한 값이 조용히 사라지는 대신(예: 최소만 입력하고
// 최대는 비워둔 채 저장) 무엇이 문제인지 알려준다. 최종 판단은 여전히 백엔드(schemas.
// CustomTargets)가 하지만, 여기서 명백한 실수는 네트워크를 타기 전에 걸러준다.
function validateDraft(d: Draft): string | null {
  const isValidPositive = (v: string) => {
    const n = Number(v);
    return Number.isFinite(n) && n > 0 && n <= 10000;
  };
  // 열량/단백질은 최소·최대가 한 쌍 — 하나만 입력된 채로는 저장하지 않는다.
  if ((!!d.energyLo) !== (!!d.energyHi)) {
    return "열량은 최소·최대를 함께 입력하거나, 둘 다 비워주세요.";
  }
  if ((!!d.proteinLo) !== (!!d.proteinHi)) {
    return "단백질은 최소·최대를 함께 입력하거나, 둘 다 비워주세요.";
  }

  const fields: [string, string][] = [
    ["열량", d.energyLo], ["열량", d.energyHi],
    ["단백질", d.proteinLo], ["단백질", d.proteinHi],
    ["칼륨", d.potassium], ["인", d.phosphorus], ["나트륨", d.sodium],
  ];
  for (const [label, v] of fields) {
    if (v && !isValidPositive(v)) {
      return `${label} 값이 올바르지 않아요. 0보다 크고 10000 이하인 숫자를 입력해주세요.`;
    }
  }
  if (d.energyLo && d.energyHi && Number(d.energyLo) > Number(d.energyHi)) {
    return "열량은 최소값이 최대값보다 클 수 없어요.";
  }
  if (d.proteinLo && d.proteinHi && Number(d.proteinLo) > Number(d.proteinHi)) {
    return "단백질은 최소값이 최대값보다 클 수 없어요.";
  }
  return null;
}

// 이 페이지 안에서만 쓰는 "영양소 1개 = 입력칸 1~2개" 필드 정의. nmeta처럼 아이콘/라벨/단위를
// 그대로 재사용해 다른 화면과 표기를 통일한다.
const FIELDS: {
  key: "energy" | "protein" | "potassium" | "phosphorus" | "sodium";
  icon: string;
  label: string;
  unit: string;
  range: boolean;
  note?: string;
}[] = [
  { key: "energy", icon: "🔥", label: "열량", unit: "kcal", range: true },
  { key: "protein", icon: "💪", label: "단백질", unit: "g", range: true },
  { key: "potassium", icon: "🌿", label: "칼륨", unit: "mg", range: false },
  { key: "phosphorus", icon: "🦴", label: "인", unit: "mg", range: false },
  { key: "sodium", icon: "🧂", label: "나트륨", unit: "mg", range: false, note: "자연재료 포함 총나트륨 기준" },
];

export function NutritionTargetsPage() {
  const nav = useNavigate();
  const { profile, setProfile } = useApp();
  const [draft, setDraft] = useState<Draft>(() => draftFromCustomTargets(profile.customTargets));
  const [defaults, setDefaults] = useState<DayTargets | null>(null);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    // custom_targets는 일부러 안 보낸다 — 여기서 보여줄 "기본값"은 override 없는 순수
    // 자동 산출값이어야 하기 때문(이미 저장된 override와 무관하게 항상 같은 기준선).
    const body: Record<string, unknown> = { weight: Number(profile.weight) || 60 };
    if (profile.height) {
      body.height = Number(profile.height);
      body.sex = profile.gender === "남성" ? "남" : "여";
    }
    fetchDayTargets(body)
      .then(setDefaults)
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const resetField = (keys: (keyof Draft)[]) => {
    setDraft((prev) => {
      const next = { ...prev };
      for (const k of keys) next[k] = "";
      return next;
    });
  };

  const resetAll = () => setDraft(draftFromCustomTargets(undefined));

  const save = async () => {
    setError("");
    setSaved(false);
    const invalid = validateDraft(draft);
    if (invalid) {
      setError(invalid);
      return;
    }
    setSaving(true);
    const customTargets = customTargetsFromDraft(draft);
    try {
      await updateProfile({
        gender: profile.gender,
        birthdate: profile.birthdate,
        height: Number(profile.height),
        weight: Number(profile.weight),
        dialysis: profile.dialysis,
        customTargets: Object.keys(customTargets).length > 0 ? customTargets : null,
      });
      setProfile({ ...profile, customTargets });
      setSaved(true);
    } catch (e: any) {
      setError(e.message || "저장하지 못했어요. 잠시 후 다시 시도해주세요.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <Shell
      header={false}
      footer={
        <Button disabled={saving} onClick={save}>
          {saving ? "저장 중..." : "저장"}
        </Button>
      }
    >
      <BackHeader title="영양 기준 설정" />
      <p className="eyebrow">MY KOOK</p>
      <h1>
        의료진에게 안내받은
        <br />
        기준이 있다면 입력하세요.
      </h1>
      <p className="sub">
        입력한 항목만 자동 산출값(성별·키 기준) 대신 사용돼요. 비워두면 그 항목은 그대로
        자동 산출값을 씁니다.
      </p>

      <div className="form">
        {FIELDS.map((f) => {
          const hi = defaults ? (f.key === "sodium" ? defaults.sodium_total_target : (defaults as any)[f.key]) : null;
          const lo =
            f.range && defaults
              ? f.key === "energy"
                ? defaults.energy[0]
                : defaults.protein[0]
              : null;
          const hiDisplay =
            f.range && defaults
              ? f.key === "energy"
                ? defaults.energy[1]
                : defaults.protein[1]
              : hi;
          return (
            <label key={f.key}>
              <span>
                {f.icon} {f.label}
                {f.note && <small className="field-hint"> · {f.note}</small>}
              </span>
              {defaults && (
                <small className="field-hint">
                  자동 산출값: {lo != null ? `${fmt(lo)}~` : ""}
                  {fmt(hiDisplay ?? 0)}
                  {f.unit}
                </small>
              )}
              {f.range ? (
                <div className="segments">
                  <div className="field">
                    <input
                      inputMode="decimal"
                      placeholder="최소"
                      value={f.key === "energy" ? draft.energyLo : draft.proteinLo}
                      onChange={(e) =>
                        setDraft({
                          ...draft,
                          [f.key === "energy" ? "energyLo" : "proteinLo"]: e.target.value,
                        })
                      }
                    />
                    <span>{f.unit}</span>
                  </div>
                  <div className="field">
                    <input
                      inputMode="decimal"
                      placeholder="최대"
                      value={f.key === "energy" ? draft.energyHi : draft.proteinHi}
                      onChange={(e) =>
                        setDraft({
                          ...draft,
                          [f.key === "energy" ? "energyHi" : "proteinHi"]: e.target.value,
                        })
                      }
                    />
                    <span>{f.unit}</span>
                  </div>
                </div>
              ) : (
                <div className="field">
                  <input
                    inputMode="decimal"
                    placeholder="상한"
                    value={(draft as any)[f.key]}
                    onChange={(e) => setDraft({ ...draft, [f.key]: e.target.value })}
                  />
                  <span>{f.unit}</span>
                </div>
              )}
              <button
                type="button"
                className="field-reset"
                onClick={() =>
                  resetField(
                    f.range
                      ? f.key === "energy"
                        ? ["energyLo", "energyHi"]
                        : ["proteinLo", "proteinHi"]
                      : [f.key as "potassium" | "phosphorus" | "sodium"],
                  )
                }
              >
                기본값으로 되돌리기
              </button>
            </label>
          );
        })}
        {error && <p className="form-error">{error}</p>}
        {saved && !error && <p className="field-hint">저장했어요.</p>}
      </div>
      <Button secondary onClick={resetAll}>
        전체 기본값으로 되돌리기
      </Button>
    </Shell>
  );
}
