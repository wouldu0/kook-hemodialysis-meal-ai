import { useNavigate } from "react-router-dom";
import { ClipboardIcon, HomeIcon, UserIcon } from "../icons";

export function BottomNav({
  active,
}: {
  active: "home" | "history" | "favorites" | "account";
}) {
  const nav = useNavigate();
  // 목업 하단 탭: 홈 / 식단 관리 / 프로필 3개
  const items = [
    ["home", <HomeIcon key="h" />, "홈", "/home"],
    ["favorites", <ClipboardIcon key="c" />, "식단 관리", "/favorites"],
    ["account", <UserIcon key="u" />, "프로필", "/account"],
  ] as const;
  return (
    <nav className="bottom-nav three">
      {items.map(([k, icon, label, path]) => (
        <button
          key={k}
          className={active === k ? "active" : ""}
          onClick={() => nav(path)}
        >
          <i>{icon}</i>
          <span>{label}</span>
        </button>
      ))}
    </nav>
  );
}
