import { useNavigate } from "react-router-dom";
import { BellIcon, UserIcon } from "../icons";
import { Logo } from "../Logo";

export function BackHeader({
  title,
  onBack,
  dot = false,
}: {
  title?: string;
  onBack?: () => void;
  dot?: boolean;
}) {
  const nav = useNavigate();
  return (
    <header className="header detail-header">
      <div className="header-lead">
        <button
          className="icon-back"
          aria-label="뒤로"
          onClick={() => (onBack ? onBack() : nav(-1))}
        >
          ‹
        </button>
        {/* 어느 화면에 있어도 브랜드가 보이도록 왼쪽 위에 로고를 둔다. 누르면 홈으로. */}
        <button
          className="header-brand"
          aria-label="푹 홈으로"
          onClick={() => nav("/home")}
        >
          <Logo />
        </button>
        {title && <b className="header-lead-title">{title}</b>}
      </div>
      <div className="header-actions">
        <button
          className={dot ? "icon-ghost has-dot" : "icon-ghost"}
          aria-label="알림"
          onClick={() => nav("/history")}
        >
          <BellIcon />
        </button>
        <button
          className="icon-avatar"
          aria-label="내 정보"
          onClick={() => nav("/account")}
        >
          <UserIcon />
        </button>
      </div>
    </header>
  );
}
