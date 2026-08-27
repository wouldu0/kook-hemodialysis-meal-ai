import {
  displayTarget,
  displayValue,
  fmt,
  minTargetOf,
  nmeta,
  STATUS_CLASS,
  statusOf,
  totalNutrition,
} from "../../utils/nutrition";

// 영양소 목록. 목업 요청 반영:
//  · 원 그래프(도넛) 없음
//  · 한 줄에 "아이콘 이름 / 수치 (기준 얼마) / 판정 뱃지"
//  · 기준을 넘으면 카드 전체가 빨간 배경, 적절하면 초록 배경
export function Nutrients({
  values,
  targets,
  isFallback = false,
}: {
  values: ReturnType<typeof totalNutrition>;
  targets?: any;
  // true면 실제 서버 판정이 아니라 내장 예시 데이터라는 뜻 — 뱃지를 '적절/초과' 대신
  // 중립적인 '예시'로 표시해서, 개인 맞춤 판정처럼 보이지 않게 한다.
  isFallback?: boolean;
}) {
  return (
    <div className="nutri-list">
      {nmeta.map((n) => {
        const v = displayValue(values, n.key);
        const hi = displayTarget(targets, n.key);
        const lo = minTargetOf(targets, n.key);
        const status = statusOf(v, lo, hi);
        const cls = isFallback ? "demo" : STATUS_CLASS[status];
        return (
          <div className={`nutri-card ${cls}`} key={n.key}>
            <span className="nutri-icon">{n.icon}</span>
            <span className="nutri-body">
              <b className="nutri-name">{n.label}</b>
              <span className="nutri-value">
                {fmt(v)}
                <small> {n.unit}</small>
                <em className="nutri-target">
                  {lo > 0
                    ? `(기준 ${fmt(lo)}~${fmt(hi)} ${n.unit})`
                    : `(기준 ${fmt(hi)} ${n.unit} 이하)`}
                </em>
              </span>
            </span>
            <i className={`nutri-badge ${cls}`}>{isFallback ? "예시" : status}</i>
          </div>
        );
      })}
    </div>
  );
}
