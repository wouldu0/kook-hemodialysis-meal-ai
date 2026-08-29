import { useEffect, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import {
  BookmarkIcon,
  ClipboardIcon,
  DocIcon,
  LeafIcon,
  SearchIcon,
} from "../../components/icons";
import { Logo } from "../../components/Logo";
import { FlowFooter } from "../../components/layout/FlowFooter";
import { Shell } from "../../components/layout/Shell";
import { Nutrients } from "../../components/meal/Nutrients";
import { NutrientIconRow } from "../../components/meal/NutrientIconRow";

// 온보딩 5단계. [번호, 제목, 설명, 미리보기 종류]
const slides = [
  [
    "1.",
    "원하는 음식 및 재료 검색",
    "먹고 싶은 음식이나 재료를 선택하세요",
    "search",
  ],
  [
    "2.",
    "한 끼 식단 생성",
    "선택한 음식과 어울리는 식단을 생성해드려요",
    "meal",
  ],
  [
    "3.",
    "영양 적합성 판정",
    "혈액투석 환자 개인 프로필 기준으로\n영양 적합성을 판정합니다.",
    "nutrition",
  ],
  [
    "4.",
    "판정에 따른 레시피 재구성",
    "초과되는 영양소에 대하여\n재료의 양을 조절하거나 대체하여\n혈액투석 환자 맞춤형으로 레시피를 재구성해줍니다",
    "adjust",
  ],
  [
    "5.",
    "오늘의 한 끼, 완성!",
    "오늘의 한 끼가 완성되었어요.\n선택한 식단의 재료와 조리과정을 확인해보세요.",
    "final",
  ],
] as const;

// 첫 화면: "오늘 뭐 해 먹지?" 스플래시. 시작하기를 누르면 온보딩 5단계로 들어간다.
// 온보딩 1~5단계와 같은 틀(.onboarding-page > .onboarding-photo, 하단 pill 버튼)을 그대로
// 써서, 이 화면만 사진이 여백 없이 화면 전체를 채우다가 다음 화면부터 갑자기 카드 안에
// 작게 들어가 보이던 크기 불일치를 없앤다. 제목("오늘 뭐 먹지?")은 사진 안에 이미 그려져
// 있어(onboarding-home.png), 다른 단계처럼 실시간 텍스트 제목을 그 위에 또 얹지 않는다
// (얹으면 같은 문구가 두 번 보임) — 대신 .splash-photo가 그 자리만큼 위쪽 여백을 더 준다.
function Splash({ onStart }: { onStart: () => void }) {
  const [shotFailed, setShotFailed] = useState(false);
  return (
    <Shell
      header={false}
      footer={
        <button className="btn btn-pill" onClick={onStart}>
          시작하기 <i className="btn-arrow">→</i>
        </button>
      }
      full
    >
      <div className="onboarding-page">
        <div className="onboarding-photo splash-photo">
          {!shotFailed ? (
            <img
              src="/assets/onboarding-home.png"
              alt="오늘 뭐 해 먹지? 고민에 푹 빠질 땐 푹이 도와드립니다"
              onError={() => setShotFailed(true)}
            />
          ) : (
            <div className="visual-card splash-fallback">
              <Logo className="splash-logo" />
              <h1 className="splash-title">오늘 뭐 먹지?</h1>
              <p className="splash-sub">
                고민에 <b>푹</b> 빠질 땐, <b>푹</b>이 도와드립니다
              </p>
              <div className="thought-bubble">
                {[
                  ["🥣", "건강한 식단"],
                  ["📋", "영양 균형"],
                  ["🍲", "간편한 관리"],
                ].map(([icon, text]) => (
                  <div key={text}>
                    <span>{icon}</span>
                    <small>{text}</small>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </Shell>
  );
}

// 온보딩 각 단계의 미리보기. 실제 화면을 축소해 보여주는 용도라 동작하지 않는다.
const PREVIEW_MENUS = [
  ["백미밥", "밥"],
  ["시금치 된장국", "국"],
  ["닭가슴살 채소볶음", "반찬"],
  ["애호박전", "반찬"],
  ["저염 배추김치", "반찬"],
] as const;

// 온보딩 단계별 이미지(실제 앱 화면 캡처). public/assets 에 파일이 있으면 그 이미지를 쓰고,
// 없으면(파일 미준비) 아래의 CSS 미리보기로 자동 대체된다.
// 이미지에는 제목·n/5 배지를 굽지 않는다 — 화면 위쪽에 실시간 텍스트로 항상 얹는다.
const ONBOARDING_IMAGE: Record<string, string> = {
  search: "/assets/onboarding-1.png",
  meal: "/assets/onboarding-2.png",
  nutrition: "/assets/onboarding-3.png",
  adjust: "/assets/onboarding-4.png",
  final: "/assets/onboarding-5.png",
};

function OnboardingVisual({ type }: { type: string }) {
  if (type === "search")
    return (
      <div className="visual-card preview-card">
        <div className="tabs three-tabs preview-tabs">
          <span className="tab active">⌕ 음식 검색</span>
          <span className="tab">🥕 재료 검색</span>
          <span className="tab">⇄ 랜덤 추천</span>
        </div>
        <div className="fake-search">
          <SearchIcon />
          <span>음식 이름을 입력하세요</span>
          <b>🎙</b>
        </div>
        <p className="preview-label">추천 검색어</p>
        <div className="chips">
          {["된장찌개", "닭가슴살", "연어구이", "비빔밥", "두부조림", "미역국"].map(
            (x) => (
              <button key={x} disabled>
                {x}
              </button>
            ),
          )}
        </div>
        <p className="preview-label">최근 검색</p>
        <div className="chips recent">
          {["된장찌개", "연어구이", "두부조림"].map((x) => (
            <button key={x} disabled>
              ◷ {x} ✕
            </button>
          ))}
        </div>
        <span className="preview-cta">✨ 선택하기</span>
      </div>
    );
  if (type === "meal")
    return (
      <div className="visual-card preview-card">
        <span className="compose-pill">
          구성 : <b>밥, 국, 반찬 3가지</b>
        </span>
        <div className="meal-list">
          {PREVIEW_MENUS.map(([name, role]) => (
            <div className="meal-row" key={name}>
              <b className="meal-row-name">{name}</b>
              <span className="meal-row-sub">{role}</span>
            </div>
          ))}
        </div>
      </div>
    );
  if (type === "nutrition")
    return (
      <div className="visual-card preview-card">
        <NutrientIconRow caption="푹 AI가 영양 기준 적합 여부를 판정합니다." />
        <Nutrients
          values={{
            energy: 620,
            protein: 22,
            phosphorus: 210,
            potassium: 1200,
            sodium: 1800,
          }}
          targets={{
            energy: [600, 700],
            protein: [22, 24],
            phosphorus: 333,
            potassium: 1000,
            sodium: 1000,
          }}
        />
        <p className="gauge-note">
          ⓘ 혈액투석 환자 남자 65세 키 170cm, 몸무게 60kg 기준
        </p>
      </div>
    );
  if (type === "adjust")
    return (
      <div className="visual-card preview-card">
        <div className="adjust-loader">
          <span className="adjust-ring">
            <ClipboardIcon />
          </span>
          <b>레시피 재구성 중입니다</b>
          <small>
            푹 AI가 영양 밸런스를 맞추고 있어요...
            <br />
            잠시만 기다려주세요
          </small>
        </div>
        <p className="preview-label with-icon">
          <ClipboardIcon /> 레시피 재구성 내용
        </p>
        <div className="lever-list">
          {[
            [<LeafIcon key="l" />, "재료 대체", "칼륨, 인 함량이 높은 재료를 저함량 재료로 대체했어요."],
            [<span key="s">🧂</span>, "조미료량 조정", "나트륨 섭취를 줄이기 위해 소금, 간장 등 조미료의 양을 조절했어요."],
            [<span key="c">🥛</span>, "재료 양 조절", "초과되는 영양소의 섭취를 줄이기 위해 재료의 양을 조절했어요."],
          ].map(([icon, title, desc]: any) => (
            <div className="lever-row" key={title}>
              <span className="lever-icon">{icon}</span>
              <div>
                <b>{title}</b>
                <small>{desc}</small>
              </div>
              <i>›</i>
            </div>
          ))}
        </div>
      </div>
    );
  return (
    <div className="visual-card preview-card">
      <div className="final-plate">
        <span>🍚 🍲 🥘 🥗 🥬</span>
      </div>
      <div className="preview-actions">
        <span>
          <BookmarkIcon /> 기록하기
        </span>
        <span>
          <DocIcon /> PDF 다운로드
        </span>
      </div>
      <p className="preview-label with-icon">📖 레시피 보러가기</p>
      <div className="meal-list">
        {PREVIEW_MENUS.map(([name]) => (
          <div className="meal-row" key={name}>
            <b className="meal-row-name">{name}</b>
            <i className="meal-row-arrow">›</i>
          </div>
        ))}
      </div>
    </div>
  );
}

export function OnboardingPage() {
  // URL로 단계를 구분한다. /onboarding = 스플래시, /onboarding/1~5 = 각 단계 화면.
  const { step } = useParams();
  const nav = useNavigate();
  const [shotFailed, setShotFailed] = useState(false);
  useEffect(() => setShotFailed(false), [step]);
  const finish = () => nav("/login");
  if (!step) return <Splash onStart={() => nav("/onboarding/1")} />;
  const n = Number(step);
  // 범위를 벗어난 주소(/onboarding/9 등)는 스플래시로 되돌린다.
  if (!Number.isInteger(n) || n < 1 || n > slides.length)
    return <Navigate to="/onboarding" replace />;
  const i = n - 1;
  const s = slides[i];
  const last = i === slides.length - 1;
  const footer = (
    <FlowFooter
      step={n}
      total={slides.length}
      prevLabel="이전"
      onPrev={() => nav(i > 0 ? `/onboarding/${i}` : "/onboarding")}
      onNext={last ? finish : () => nav(`/onboarding/${n + 1}`)}
      nextLabel={last ? "시작" : "다음"}
    />
  );
  // 제목·건너뛰기·n/5 배지는 항상 실시간 텍스트로 그리고, 그 아래에 실제 화면
  // 스크린샷(준비돼 있으면) 또는 CSS 미리보기를 채운다.
  const shot = ONBOARDING_IMAGE[s[3]];
  const head = (
    <>
      <div className="onboarding-top">
        <button className="skip" onClick={finish}>
          건너뛰기
        </button>
        <span className="step-count">
          {i + 1} / {slides.length}
        </span>
      </div>
      <h1 className="onboarding-title">
        <b>{s[0]}</b> {s[1]}
      </h1>
      <p className="sub center">{s[2]}</p>
    </>
  );
  return (
    <Shell header={false} footer={footer} full>
      <div className="onboarding-page">
        <div className="onboarding-head">{head}</div>
        <div className="onboarding-photo">
          {shot && !shotFailed ? (
            <img
              src={shot}
              alt={`${s[0]} ${s[1]} — ${s[2]}`}
              onError={() => setShotFailed(true)}
            />
          ) : (
            <OnboardingVisual type={s[3]} />
          )}
        </div>
      </div>
    </Shell>
  );
}
