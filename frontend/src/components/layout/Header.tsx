import { useNavigate } from "react-router-dom";
import { BellIcon, UserIcon } from "../icons";
import { Logo } from "../Logo";

export function Header() {
  const nav = useNavigate();
  return (
    <header className="header">
      <button className="logo-button" onClick={() => nav("/home")}>
        <Logo />
      </button>
      <div className="header-actions">
        <button className="icon-ghost" aria-label="알림" onClick={() => nav("/history")}>
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
