// 흐름 화면 하단의 '이전 / 단계 점 / 다음' 네비게이션.
// 어느 단계에서든 앞 화면으로 다시 돌아가 볼 수 있어야 한다는 요구사항을 이걸로 처리한다.
export function FlowFooter({
  step,
  total,
  onPrev,
  onNext,
  nextLabel = "다음",
  prevLabel = "이전",
}: {
  step: number;
  total: number;
  onPrev?: () => void;
  onNext?: () => void;
  nextLabel?: string;
  prevLabel?: string;
}) {
  return (
    <div className="flow-footer">
      {onPrev ? (
        <button className="flow-btn ghost" onClick={onPrev}>
          <i>‹</i> {prevLabel}
        </button>
      ) : (
        <span />
      )}
      <div className="flow-dots">
        {Array.from({ length: total }, (_, i) => (
          <span key={i} className={i + 1 === step ? "fdot on" : "fdot"} />
        ))}
      </div>
      {onNext ? (
        <button className="flow-btn solid" onClick={onNext}>
          {nextLabel} <i>›</i>
        </button>
      ) : (
        <span />
      )}
    </div>
  );
}
