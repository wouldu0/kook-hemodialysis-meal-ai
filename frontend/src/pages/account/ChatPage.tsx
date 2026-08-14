import { useRef, useState } from "react";
import { BackHeader } from "../../components/layout/BackHeader";
import { Shell } from "../../components/layout/Shell";
import { askChat } from "../../services/api";

type ChatMessage = {
  role: "user" | "assistant";
  text: string;
  sources?: string[];
  error?: boolean;
};

const SUGGESTIONS = [
  "칼륨이 많은 채소는 뭐가 있나요?",
  "하루 나트륨 섭취는 얼마나 줄여야 하나요?",
  "바나나 먹어도 되나요?",
];

export function ChatPage() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  // 대화 이력 전체가 아니라 "직전 응답이 어떤 재료 얘기였는가" 딱 그 값 하나만 들고 있는다 —
  // "그럼 몇 조각?" 같은 한 턴짜리 음식 후속 질문 지원용(services/api.ts::askChat 주석 참고).
  // 다른 채팅 상태와 마찬가지로 메모리(React state)에만 있고 새로고침하면 사라진다 — 영구
  // 저장(localStorage) 안 함.
  const [foodContext, setFoodContext] = useState<string | null>(null);
  const threadEnd = useRef<HTMLDivElement>(null);

  const send = async (question: string) => {
    const q = question.trim();
    if (!q || loading) return;
    setMessages((prev) => [...prev, { role: "user", text: q }]);
    setInput("");
    setLoading(true);
    try {
      // 개인화(weight/consumed) 없이 일반 질문으로 보낸다 — services/api.ts::askChat 주석 참고.
      const d = await askChat(q, foodContext);
      // 응답의 context_food로 항상 덮어쓴다(병합/유지 아님) — 화제가 음식이 아닌 쪽으로 바뀌면
      // 서버가 null을 돌려주고, 그 순간 다음 질문에서 stale 컨텍스트가 새지 않도록 그대로 지운다.
      setFoodContext(d.context_food ?? null);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: d.answer, sources: d.sources || [] },
      ]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: e?.message || "답변을 받아오지 못했어요. 잠시 후 다시 시도해주세요.",
          error: true,
        },
      ]);
    } finally {
      setLoading(false);
      setTimeout(() => threadEnd.current?.scrollIntoView({ behavior: "smooth" }), 50);
    }
  };

  return (
    <Shell
      header={false}
      footer={
        <div className="chat-input-row">
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send(input);
              }
            }}
            placeholder="궁금한 영양 정보를 물어보세요"
            disabled={loading}
          />
          <button onClick={() => send(input)} disabled={loading || !input.trim()}>
            전송
          </button>
        </div>
      }
    >
      <BackHeader title="영양 상담" />
      <p className="eyebrow">AI 영양 상담</p>
      <h1>
        궁금한 식이 정보를
        <br />
        물어보세요.
      </h1>
      <div className="info-box">
        <b>일반적인 원칙만 안내해요</b>
        <span>
          임상 영양 자료를 근거로 답하는 참고용 정보이며, 개인별 검사 수치·약물에 따른 판단은
          담당 의료진·영양사와 상담해주세요.
        </span>
      </div>
      {messages.length === 0 && (
        <div className="chips" style={{ marginTop: 16 }}>
          {SUGGESTIONS.map((s) => (
            <button key={s} onClick={() => send(s)}>
              {s}
            </button>
          ))}
        </div>
      )}
      <div className="chat-thread" style={{ marginTop: 18 }}>
        {messages.map((m, i) => (
          <div key={i} className={`chat-msg ${m.role}`}>
            <div className={m.error ? "chat-bubble warning-box" : "chat-bubble"}>
              {m.text}
            </div>
            {m.sources && m.sources.length > 0 && (
              <div className="chat-sources">
                <b>참고 자료</b>
                {m.sources.map((s) => (
                  <span key={s}>{s}</span>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-msg assistant">
            <div className="recipe-loading">
              <div className="spinner" />
              <span>답변을 준비하고 있어요...</span>
            </div>
          </div>
        )}
        <div ref={threadEnd} />
      </div>
    </Shell>
  );
}
