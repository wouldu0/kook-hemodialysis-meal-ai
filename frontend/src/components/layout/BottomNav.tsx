import { useNavigate } from "react-router-dom";
import { ClipboardIcon, HomeIcon, UserIcon } from "../icons";

export function BottomNav({
  active,
}: {
  active: "home" | "history" | "favorites" | "account";
}) {
  const nav = useNavigate();
  // 핵심 사용 흐름에 맞춰 홈 / 식사 기록 / 프로필을 하단의 주 내비게이션으로 둔다.
  const items = [
    ["home", <HomeIcon key="h" />, "홈", "/home"],
    ["history", <ClipboardIcon key="c" />, "식사 기록", "/history"],
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
