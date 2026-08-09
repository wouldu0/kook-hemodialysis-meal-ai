// 브랜드 로고. 사용자가 직접 준비한 kook-logo.png가 있으면 그걸 쓰고,
// 없으면 같은 폴더의 kook-logo.svg(코드로 그린 동일 디자인)로 자동 대체한다.
export function Logo({ className = "" }: { className?: string }) {
  return (
    <img
      className={className}
      src="/assets/kook-logo.png"
      alt="KOOK"
      onError={(e) => {
        const img = e.currentTarget;
        if (!img.src.endsWith(".svg")) img.src = "/assets/kook-logo.svg";
      }}
    />
  );
}
