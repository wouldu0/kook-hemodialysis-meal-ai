import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { BackHeader } from "../../components/layout/BackHeader";
import { Button } from "../../components/layout/Button";
import { Shell } from "../../components/layout/Shell";
import { Logo } from "../../components/Logo";
import { useApp } from "../../hooks/useApp";
import { currentUser, saveEverywhere } from "../../services/api";
import { labels, menuMap, parseLocalIngredient } from "../../utils/menu";
import {
  adjustedNutrition,
  fmt,
  fmt2,
  minTargetOf,
  nmeta,
  targetOf,
  totalNutrition,
} from "../../utils/nutrition";

export function PdfPreviewPage() {
  const nav = useNavigate();
  const { plan, apiResult } = useApp();
  const ref = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = useState(false);
  // 서버가 실제로 계산한 값이 있으면 그걸 쓰고, 없을 때만(오프라인) 폴백 계산을 쓴다.
  const values =
    apiResult?.nutrition || adjustedNutrition(totalNutrition(plan));
  const targets = apiResult?.targets;
  const download = async () => {
    if (!ref.current) return;
    setLoading(true);
    try {
      const [{ default: html2canvas }, { jsPDF }] = await Promise.all([
        import("html2canvas"),
        import("jspdf"),
      ]);
      const canvas = await html2canvas(ref.current, {
        scale: 2,
        backgroundColor: "#fff",
      });
      const pdf = new jsPDF("p", "mm", "a4");
      // A4 한 장(210×297mm) 안에 반드시 들어가게 한다.
      // 가로를 먼저 맞춰보고, 그래도 세로가 넘치면 세로 기준으로 다시 줄인다.
      const PAGE_W = 210;
      const PAGE_H = 297;
      const MARGIN = 8;
      const maxW = PAGE_W - MARGIN * 2;
      const maxH = PAGE_H - MARGIN * 2;
      let w = maxW;
      let h = (canvas.height * w) / canvas.width;
      if (h > maxH) {
        h = maxH;
        w = (canvas.width * h) / canvas.height;
      }
      pdf.addImage(
        canvas.toDataURL("image/jpeg", 0.96),
        "JPEG",
        (PAGE_W - w) / 2,
        MARGIN,
        w,
        h,
      );
      pdf.save(`KOOK_${plan.menus[1]}_맞춤한끼.pdf`);
      if (currentUser())
        await saveEverywhere("fook:documents", {
          id: `pdf-${Date.now()}`,
          title: `${plan.menus[1]} 맞춤 한 끼 PDF`,
          subtitle: "레시피 · 영양정보",
          createdAt: new Date().toISOString(),
          menus: plan.menus,
        });
    } finally {
      setLoading(false);
    }
  };
  return (
    <Shell
      header={false}
      footer={
        <div className="pdf-footer-actions">
          <Button secondary onClick={() => nav(-1)}>
            ‹ 이전으로
          </Button>
          <Button onClick={download}>
            {loading ? "PDF 생성 중..." : "PDF 다운로드"}
          </Button>
        </div>
      }
    >
      <BackHeader title="PDF 미리보기" />
      <div className="pdf-page" ref={ref}>
        <div className="pdf-brand">
          <Logo />
          <div>
            <b>KOOK 맞춤 한 끼 레시피</b>
            <span>혈액투석 환자용 식단 참고 자료</span>
          </div>
        </div>
        <h2>{plan.menus[1]} 한 끼</h2>
        {/* PDF는 다운로드되면 앱 화면(Shell의 예시 데이터 배너)과 분리된 별도 파일이 되므로,
            서버 미연결로 예시 데이터를 쓴 경우 이 안내를 파일 안에도 반드시 함께 남긴다. */}
        {!apiResult && (
          <p className="pdf-fallback-notice">
            ⚠ 서버 연결에 실패해 내장 예시 데이터로 생성된 문서입니다. 실제 개인
            맞춤 계산 결과가 아닙니다.
          </p>
        )}
        {/* 사진 자리(pdf-hero-space)는 뺐다. A4 한 장에 담기지 않아서 표를 위로 당긴다. */}
        <h3>한 끼 전체 레시피</h3>
        <table className="meal-recipe-table">
          <thead>
            <tr>
              <th>구분</th>
              <th>음식</th>
              <th>재료</th>
              <th>조리 과정</th>
            </tr>
          </thead>
          <tbody>
            {plan.menus.map((name, i) => {
              const m = menuMap.get(name); // 오프라인 폴백용
              const serverIngs: [string, number][] | undefined =
                apiResult?.dish_ingredients?.[name];
              const ingList = (
                serverIngs || (m?.ingredients || []).map(parseLocalIngredient)
              ).map(([ing, amt]) => (amt > 0 ? `${ing} ${fmt2(amt)}g` : ing));
              return (
                <tr key={name}>
                  <th>{labels[i]}</th>
                  <td>{name}</td>
                  <td>
                    <ul>
                      {ingList.map((x) => (
                        <li key={x}>{x}</li>
                      ))}
                    </ul>
                  </td>
                  <td>
                    <ol>
                      {(
                        m?.steps || [
                          "레시피 상세 화면에서 AI 조리법을 확인하세요.",
                        ]
                      ).map((x, j) => (
                        <li key={j}>{x}</li>
                      ))}
                    </ol>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <h3>한 끼 영양 정보</h3>
        <table className="nutrition-pdf-table">
          <thead>
            <tr>
              <th>영양소</th>
              <th>섭취량</th>
              <th>판정</th>
            </tr>
          </thead>
          <tbody>
            {nmeta.map((n) => {
              const t = targetOf(targets, n.key);
              const lo = minTargetOf(targets, n.key);
              const v = values[n.key];
              const ok = v <= t && (lo === 0 || v >= lo);
              return (
                <tr key={n.key}>
                  <th>{n.label}</th>
                  <td>
                    {fmt(v)}
                    {n.unit} (기준 {lo > 0 ? `${fmt(lo)}~${fmt(t)}` : `${fmt(t)} 이하`}
                    {n.unit})
                  </td>
                  <td>{!apiResult ? "예시" : ok ? "적정" : "주의"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Shell>
  );
}
