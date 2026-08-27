import { useEffect } from "react";
import { Navigate, useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { BottomNav } from "../../components/layout/BottomNav";
import { Shell } from "../../components/layout/Shell";
import { useApp } from "../../hooks/useApp";
import { currentUser, getMe, logout } from "../../services/api";

export function AccountPage() {
  const nav = useNavigate();
  const { profile, setProfile } = useApp();
  const user = currentUser();
  if (!user) return <Navigate to="/login" replace />;
  // 계정 화면에 들어올 때마다 서버의 최신 프로필을 한 번 불러와 화면에 반영한다.
  // (다른 기기에서 수정했거나, 세션이 오래된 경우에도 항상 최신 값을 보여주기 위함)
  useEffect(() => {
    getMe()
      .then((d) => {
        if (!d?.profile) return;
        const p = d.profile;
        setProfile({
          ...profile,
          gender: p.gender || profile.gender,
          birthdate: p.birthdate || profile.birthdate,
          age: p.age != null ? String(p.age) : profile.age,
          height: p.height_cm != null ? String(p.height_cm) : profile.height,
          weight: p.weight_kg != null ? String(p.weight_kg) : profile.weight,
          dialysis: p.dialysis_type || profile.dialysis,
          customTargets: p.custom_targets || profile.customTargets,
        });
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  const handleLogout = async () => {
    try {
      await logout();
    } catch {}
    localStorage.removeItem("fook:user");
    localStorage.removeItem("fook:token");
    nav("/start");
  };
  return (
    <Shell header={false} footer={<BottomNav active="account" />}>
      <BackHeader title="내 정보" />
      <div className="profile-summary">
        <div className="avatar">{user.name.slice(0, 1)}</div>
        <div>
          <h2>{user.name}님</h2>
          <p>@{user.username}</p>
        </div>
        <button onClick={() => nav("/profile")}>수정</button>
      </div>
      <div className="metric-grid">
        <div>
          <b>{profile.age}세</b>
          <span>나이</span>
        </div>
        <div>
          <b>{profile.height}cm</b>
          <span>신장</span>
        </div>
        <div>
          <b>{profile.weight}kg</b>
          <span>체중</span>
        </div>
        <div>
          <b>{profile.dialysis}</b>
          <span>투석 유형</span>
        </div>
      </div>
      <div className="menu-list">
        <button onClick={() => nav("/nutrition-targets")}>
          <span>영양 기준 설정</span>
          <i>›</i>
        </button>
        <button onClick={() => nav("/favorites")}>
          <span>찜한 메뉴</span>
          <i>›</i>
        </button>
        <button onClick={() => nav("/documents")}>
          <span>PDF 보관함</span>
          <i>›</i>
        </button>
        <button onClick={() => nav("/tips")}>
          <span>칼륨 낮추는 조리 팁</span>
          <i>›</i>
        </button>
        <button onClick={() => nav("/chat")}>
          <span>영양 상담 챗봇</span>
          <i>›</i>
        </button>
      </div>
      <button className="logout" onClick={handleLogout}>
        로그아웃
      </button>
    </Shell>
  );
}
