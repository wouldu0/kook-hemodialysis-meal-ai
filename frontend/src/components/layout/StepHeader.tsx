export function StepHeader({ step, total }: { step: number; total: number }) {
  return (
    <div className="step-header">
      <div className="step-dots">
        {Array.from({ length: total }, (_, i) => i + 1).map((n, i) => (
          <div className="step-dot-wrap" key={n}>
            <span className={n <= step ? "step-circle done" : "step-circle"}>
              {n}
            </span>
            {i < total - 1 && (
              <span className={n < step ? "step-line done" : "step-line"} />
            )}
          </div>
        ))}
      </div>
      <span className="step-count">
        {step} / {total}
      </span>
    </div>
  );
}
