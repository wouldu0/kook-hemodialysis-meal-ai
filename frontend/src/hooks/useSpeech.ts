import { useEffect, useRef, useState } from "react";
import { textToSpeech } from "../services/api";

// ── 음성 안내 ────────────────────────────────────────────────────────────────
// 요구사항: "눌러야 나온다" — 화면에 들어왔다고 저절로 읽지 않고, 버튼을 누른 순간에만 읽는다.
// 브라우저 내장 음성합성(무료·즉시 재생·오프라인)을 우선 쓰고, 그게 없는 환경에서만
// 서버 /tts(OpenAI)를 부른다. 서버 TTS는 OPENAI_API_KEY가 없으면 실패하므로 폴백 순서가 중요하다.
export function useSpeech() {
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stop = () => {
    if (typeof window !== "undefined" && window.speechSynthesis)
      window.speechSynthesis.cancel();
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setSpeaking(false);
  };
  useEffect(() => stop, []);
  const speak = async (text: string) => {
    stop();
    const clean = text.replace(/\s+/g, " ").trim();
    if (!clean) return;
    const synth = typeof window !== "undefined" ? window.speechSynthesis : null;
    if (synth) {
      // 음성 목록은 비동기로 채워지는 브라우저가 있어서, 비어 있으면 한 번 기다렸다 다시 본다.
      let voices = synth.getVoices();
      if (!voices.length) {
        await new Promise<void>((res) => {
          const t = setTimeout(res, 800);
          synth.onvoiceschanged = () => {
            clearTimeout(t);
            res();
          };
        });
        voices = synth.getVoices();
      }
      const u = new SpeechSynthesisUtterance(clean);
      const ko = voices.find((v) => /^ko/i.test(v.lang));
      if (ko) u.voice = ko; // lang만 지정하면 무시하는 브라우저가 있어 음성을 직접 지정한다
      u.lang = "ko-KR";
      u.rate = 0.9; // 어르신이 따라오기 쉽도록 조금 느리게
      u.onend = () => setSpeaking(false);
      u.onerror = () => setSpeaking(false);
      setSpeaking(true);
      synth.speak(u);
      return;
    }
    setSpeaking(true);
    try {
      const blob = await textToSpeech(clean);
      const audio = new Audio(URL.createObjectURL(blob));
      audioRef.current = audio;
      audio.onended = () => setSpeaking(false);
      await audio.play();
    } catch {
      setSpeaking(false);
    }
  };
  return { speak, stop, speaking };
}
