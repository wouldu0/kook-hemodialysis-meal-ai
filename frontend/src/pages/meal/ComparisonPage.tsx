import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { CheckIcon, DocIcon } from "../../components/icons";
import { useApp } from "../../hooks/useApp";
import {
  adjustedNutrition,
  displayTarget,
  displayValue,
  fmt,
  minTargetOf,
  nmeta,
  parseChange,
  STATUS_CLASS,
  statusOf,
  totalNutrition,
} from "../../utils/nutrition";

export function ComparisonPage() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  // after = 재구성 후(서버 최종 수치), before = 재구성 전(서버가 nutrition_before로 내려줌).
  // 서버 미연결이면 로컬 메뉴 영양 합계를 '전', 보정값을 '후'로 써서 화면 구조를 유지한다.
  const after = apiResult?.nutrition || adjustedNutrition(totalNutrition(plan));
  const before = apiResult?.nutrition_before || totalNutrition(plan);
  const changes: string[] = apiResult?.changes || [];
  const parsed = changes.map(parseChange);
  const hasReal = changes.length > 0;
  const speech = hasReal
    ? `판정에 따라 레시피를 재구성했습니다. ${parsed
        .slice(0, 5)
        .map((c) =>
          c.kind === "removed"
            ? `${c.menu} 제외`
            : `${c.menu}, ${c.before}에서 ${c.after}로 변경`,
        )
        .join(". ")}.`
    : "이번 식단은 별도 조정이 없었습니다. 생성된 조합이 처음부터 영양 기준을 만족했습니다.";
  // 음식별로 묶는다. 서버 changes의 메뉴명은 '교체 전' 이름일 수 있어서(예: 배추김치 →
  // 저염물김치 교체), 교체 후 이름(c.after)으로도 매칭해야 카드가 비지 않는다.
  const perMenu = plan.menus.map((menu) => {
    const mine = parsed.filter(
      (c) =>
        c.menu === menu ||
        (c.kind === "swap" && c.after === menu) ||
        c.raw.startsWith(`${menu}:`),
    );
    const tags: string[] = [];
    const lines: string[] = [];
    for (const c of mine) {
      if (c.kind === "amount") {
        if (!tags.includes("양 조절")) tags.push("양 조절");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "ingredient") {
        if (!tags.includes("재료 대체")) tags.push("재료 대체");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "swap" && apiResult?.recipe_source?.[c.after || ""]) {
        // 서버는 '재료를 바꿔서 메뉴 이름까지 달라진 경우'(시금치나물 → 양배추나물)도
        // "교체"라는 한 문장으로 내려준다. 이때만 recipe_source에 원래 이름이 남으므로,
        // 그걸로 '재료 대체'와 '저염 메뉴로 통째 교체'를 구분한다.
        if (!tags.includes("재료 대체")) tags.push("재료 대체");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "swap") {
        if (!tags.includes("저염 메뉴 교체")) tags.push("저염 메뉴 교체");
        lines.push(`${c.before} → ${c.after}`);
      } else if (c.kind === "removed") {
        if (!tags.includes("메뉴 제외")) tags.push("메뉴 제외");
        lines.push("기준을 넘겨 이번 식단에서 제외했어요.");
      }
    }
    return { menu, tags, lines, details: mine };
  });
  // 어떤 메뉴에도 붙지 않은 변경(메뉴 교체·제외 등)은 따로 모아 보여준다.
  const matched = new Set(perMenu.flatMap((m) => m.details.map((c) => c.raw)));
  const unmatched = parsed.filter((c) => !matched.has(c.raw));
  // 카드가 위에서부터 하나씩 나타나는 등장 애니메이션
  const [revealed, setRevealed] = useState(0);
  useEffect(() => {
    const t = setInterval(
      () => setRevealed((v) => (v >= perMenu.length ? v : v + 1)),
      220,
    );
    return () => clearInterval(t);
  }, [perMenu.length]);
  return (
    <Shell
      header={false}
      footer={
        <button className="btn" onClick={() => nav("/final")}>
          <i className="btn-icon">
            <DocIcon />
          </i>{" "}
          재료와 조리과정 보러가기 <i className="btn-arrow">›</i>
        </button>
      }
    >
      <BackHeader onBack={() => nav("/analysis")} dot />
      <div className="hero-center">
        <span className="hero-check">
          <CheckIcon />
        </span>
        <h1>
          레시피 <b>재구성 완료</b>
        </h1>
        <p className="sub center">
          개인 프로필 영양 섭취 기준에 맞게
          <br />
          식단을 재구성했어요.
        </p>
      </div>
      <h2 className="section-title">음식별 재구성 내용</h2>
      <div className="permenu-list">
        {perMenu.map((m, i) => (
          <article
            className={i < revealed ? "permenu-card show" : "permenu-card"}
            key={m.menu}
          >
            <div className="permenu-head">
              <b>{m.menu}</b>
              <span className="permenu-tags">
                {m.tags.length ? (
                  m.tags.map((t) => (
                    <em className="tag-adjust" key={t}>
                      {t}
                    </em>
                  ))
                ) : (
                  <em className="tag-keep">조정 내용 없음</em>
                )}
              </span>
            </div>
            {m.lines.length ? (
              <ul className="permenu-changes">
                {m.lines.map((l, k) => {
                  const [b, a] = l.split(" → ");
                  return (
                    <li key={`${l}-${k}`}>
                      {a ? (
                        <>
                          <span className="pc-before">{b}</span>
                          <i>→</i>
                          <span className="pc-after">{a}</span>
                        </>
                      ) : (
                        <span className="pc-plain">{l}</span>
                      )}
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p>변경 없이 그대로 유지했어요.</p>
            )}
          </article>
        ))}
      </div>
      {!hasReal && (
        <div className="info-box">
          <b>이번 식단은 별도 조정이 없었어요</b>
          <span>생성된 조합이 처음부터 영양 기준을 만족했어요.</span>
        </div>
      )}
      {unmatched.length > 0 && (
        <section className="change-section">
          <div className="change-heading">
            <span>+</span>
            <div>
              <b>그 밖의 조정</b>
              <small>메뉴 교체·제외처럼 개별 음식에 묶이지 않는 변경이에요.</small>
            </div>
          </div>
          {unmatched.map((c) => (
            <article className="change-card reveal-in" key={c.raw}>
              <div className="change-card-title">
                <b>{c.menu || "식단"}</b>
                <span>
                  {c.kind === "swap"
                    ? "메뉴 교체"
                    : c.kind === "removed"
                      ? "메뉴 제외"
                      : "재료 교체"}
                </span>
              </div>
              <div className="amount-flow mini">
                {c.after ? (
                  <>
                    <strong>{c.before}</strong>
                    <i>→</i>
                    <strong className="highlight">{c.after}</strong>
                  </>
                ) : (
                  <strong>{c.raw}</strong>
                )}
              </div>
            </article>
          ))}
        </section>
      )}
      {apiResult?.note && (
        <div className="info-box">
          <b>AI 안내</b>
          <span>{apiResult.note}</span>
        </div>
      )}
      {apiResult?.warning && (
        <div className="warning-box">
          {/* warning은 나트륨 전용이 아니라 열량·단백질·칼륨·인·나트륨 중 기준을 못 맞춘
              항목의 범용 안내라 라벨도 그에 맞춘다(2026-08, 예전엔 "나트륨 안내"로 고정돼
              단백질 미달 등 다른 영양소 문구가 떠도 나트륨 문제처럼 보였음). */}
          <b>⚠ 영양 안내</b>
          <span>{apiResult.warning}</span>
        </div>
      )}
      <section className="change-section">
        <div className="change-heading">
          <span>✓</span>
          <div>
            <b>최종 영양소</b>
            <small>레시피 재구성 전과 후를 나란히 비교했어요.</small>
          </div>
        </div>
        <div className="ba-list">
          {nmeta.map((n) => {
            const b = displayValue(before, n.key);
            const a = displayValue(after, n.key);
            const hi = displayTarget(apiResult?.targets, n.key);
            const lo = minTargetOf(apiResult?.targets, n.key);
            const aStatus = statusOf(a, lo, hi);
            const diff = a - b;
            const same = Math.round(Math.abs(diff)) === 0;
            const dir = same ? "same" : diff < 0 ? "down" : "up";
            return (
              <div className="ba-item" key={n.key}>
                <span className="ba-item-name">
                  <n.icon /> {n.label}
                </span>
                {/* 증감 뱃지는 화살표 안에 쌓지 않고 아래 행으로 뺀다.
                    그래야 전/후 숫자가 화살표와 같은 줄에 정렬된다. */}
                <div className="ba-flow">
                  <span className="ba-from">
                    {fmt(b)}
                    <small>{n.unit}</small>
                  </span>
                  <span className={`ba-arrow ${dir}`}>
                    <svg viewBox="0 0 40 16" aria-hidden="true">
                      <path d="M2 8h30" />
                      <path d="M27 3l6 5-6 5" />
                    </svg>
                  </span>
                  <span className={`ba-to ${STATUS_CLASS[aStatus]}`}>
                    {fmt(a)}
                    <small>{n.unit}</small>
                  </span>
                  {!same && (
                    <em className={`ba-delta ${dir}`}>
                      {diff < 0 ? "−" : "+"}
                      {fmt(Math.abs(diff))}
                    </em>
                  )}
                </div>
              </div>
            );
          })}
        </div>
        {!apiResult?.nutrition_before && (
          <p className="ba-note">
            ⓘ 서버가 재구성 전 수치를 주지 않아, 원본 메뉴 기준 예시값으로 비교했어요.
          </p>
        )}
      </section>
    </Shell>
  );
}
