import { useApp } from "../../hooks/useApp";
import { Header } from "./Header";

export function Shell({
  children,
  footer,
  header = true,
  full = false,
}: {
  children: any;
  footer?: any;
  header?: boolean;
  full?: boolean; // 이미지 한 장이 화면 전체를 채울 때 본문 여백을 없앤다
}) {
  const { usingFallback } = useApp();
  return (
    <main className="page">
      <div className="phone">
        {header && <Header />}
        <section className={full ? "screen screen-full" : "screen"}>
          {usingFallback && (
            <div className="warning-box fallback-banner">
              <b>⚠ 예시 데이터를 보고 있어요</b>
              <span>
                서버 연결에 실패해서 실제 개인 맞춤 계산이 아닌 내장 예시 값으로
                화면을 보여주고 있습니다. 아래 영양 판정·재구성 내역은 참고용이
                아니라 그저 화면 구성을 보여주기 위한 예시입니다.
              </span>
            </div>
          )}
          {children}
        </section>
        {footer && <footer className="footer">{footer}</footer>}
      </div>
    </main>
  );
}
