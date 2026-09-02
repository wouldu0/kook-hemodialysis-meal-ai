// 목업 아이콘 (선화 스타일, currentColor로 색을 상속받는다)
const svg = (d: any, extra: any = {}) => (
  <svg
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
    {...extra}
  >
    {d}
  </svg>
);

export const BellIcon = () =>
  svg(
    <>
      <path d="M18 8a6 6 0 1 0-12 0c0 7-2 8-2 8h16s-2-1-2-8" />
      <path d="M13.7 20a2 2 0 0 1-3.4 0" />
    </>,
  );
export const UserIcon = () =>
  svg(
    <>
      <path d="M19 21v-1a7 7 0 0 0-14 0v1" />
      <circle cx="12" cy="7.5" r="3.8" />
    </>,
  );
export const SearchIcon = () =>
  svg(
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m20 20-3.6-3.6" />
    </>,
  );
export const LeafIcon = () =>
  svg(
    <>
      <path
        d="M11 20A7 7 0 0 1 4 13c0-6 7-9 16-9 0 9-3 16-9 16Z"
        fill="currentColor"
        stroke="none"
      />
      <path d="M4.5 20c3.5-6 8.5-9.5 13.5-11.5" stroke="#fff" />
    </>,
  );
export const DiceIcon = () =>
  svg(
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="4" fill="currentColor" stroke="none" />
      <g fill="#fff">
        <circle cx="8.2" cy="8.2" r="1.5" />
        <circle cx="15.8" cy="8.2" r="1.5" />
        <circle cx="12" cy="12" r="1.5" />
        <circle cx="8.2" cy="15.8" r="1.5" />
        <circle cx="15.8" cy="15.8" r="1.5" />
      </g>
    </>,
  );
export const SlidersIcon = () =>
  svg(
    <>
      <path d="M4 7h11M18.5 7H20M4 17h5M12.5 17H20" />
      <circle cx="16" cy="7" r="1.9" />
      <circle cx="10.5" cy="17" r="1.9" />
    </>,
  );
export const CheckIcon = () => svg(<path d="m5 12.5 4.5 4.5L19 7" />, { strokeWidth: 2.4 });
export const BowlIcon = () =>
  svg(
    <>
      <path d="M3.5 11h17a8.5 8.5 0 0 1-17 0Z" />
      <path d="M9 6c0-1.2 1-1.6 1-2.8M12.5 6c0-1.6 1.2-2 1.2-3.4M16 6c0-1.2 1-1.6 1-2.6" />
    </>,
  );
export const HomeIcon = () =>
  svg(<path d="M4 10.5 12 4l8 6.5V20a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1Z" />);
export const ClipboardIcon = () =>
  svg(
    <>
      <rect x="5" y="4" width="14" height="17" rx="2.5" />
      <path d="M9 4.5A1.5 1.5 0 0 1 10.5 3h3A1.5 1.5 0 0 1 15 4.5V6H9Z" />
      <path d="M9 11h6M9 15h4" />
    </>,
  );
export const DocIcon = () =>
  svg(
    <>
      <path d="M6 3h7l5 5v13a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" />
      <path d="M13 3v5h5M8.5 13h7M8.5 17h5" />
    </>,
  );
export const BookmarkIcon = () =>
  svg(<path d="M7 3h10a1 1 0 0 1 1 1v17l-6-4-6 4V4a1 1 0 0 1 1-1Z" />);
export const SpeakerIcon = () =>
  svg(
    <>
      <path d="M4 9.5h3.5L12 5.5v13L7.5 14.5H4Z" />
      <path d="M15.5 9a4 4 0 0 1 0 6M18 6.5a7.5 7.5 0 0 1 0 11" />
    </>,
  );
export const RefreshIcon = () =>
  svg(
    <>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <path d="M20 4v4h-4" />
    </>,
  );
export const ChartIcon = () =>
  svg(
    <>
      <rect x="3.5" y="3.5" width="17" height="17" rx="3" />
      <path d="M8 16v-4M12 16V8M16 16v-2.5" />
    </>,
  );

// ── 영양소 아이콘 (열량·단백질·인·칼륨·나트륨) ──
// 기존에 🔥💪🦴🌿🧂 이모지로 쓰던 자리를 대체한다. 다른 아이콘과 같은 선화 스타일
// (currentColor 상속)이라 어떤 배경·다크모드에도 이모지처럼 OS마다 다르게 보이지 않는다.
// 아이콘만으로 의미를 판단하게 하지 않도록, 쓰는 곳에서는 항상 텍스트 라벨(n.label)을
// 같이 보여준다 — 아이콘은 보조 시각 신호일 뿐이다.
export const FlameIcon = () =>
  svg(
    <path
      d="M12 2.5c2.2 3 .3 4.4.9 7 .5-1 .6-1.8.6-2.6 1.7 1.6 2.5 3.4 2.5 5.1a5 5 0 1 1-10 0c0-1.8.9-3 1.7-4.1.2 1 .7 1.6 1.3 1.9-.6-2.7.2-4.8 3-7.3Z"
      fill="currentColor"
      stroke="none"
    />,
  );
export const ProteinIcon = () =>
  svg(
    <>
      <path d="M4 9v6M2.3 10.6v2.8M7 6.8v10.4M17 6.8v10.4M21.7 10.6v2.8M20 9v6" />
      <path d="M7 12h10" />
    </>,
  );
export const PhosphorusIcon = () =>
  svg(
    <>
      <circle cx="12" cy="12" r="2.1" fill="currentColor" stroke="none" />
      <circle cx="12" cy="4.3" r="1.6" />
      <circle cx="19" cy="16.3" r="1.6" />
      <circle cx="5" cy="16.3" r="1.6" />
      <path d="M12 6.4v3.6M17.4 15.2 13.8 13M10.2 13 6.6 15.2" />
    </>,
  );
export const PotassiumIcon = () =>
  svg(
    <>
      <path d="M17 3a14 14 0 0 0-14 14v4h4A14 14 0 0 0 21 7V3h-4Z" />
      <path d="M7 17c3-6 8-9 12-12" />
    </>,
  );
export const SaltIcon = () =>
  svg(
    <>
      <path d="M9.3 3h5.4l1.6 3.6c.3.5.5 1 .5 1.6V19a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V8.2c0-.6.2-1.1.5-1.6L9.3 3Z" />
      <circle cx="10.3" cy="12.2" r=".6" fill="currentColor" stroke="none" />
      <circle cx="13.7" cy="12.2" r=".6" fill="currentColor" stroke="none" />
      <circle cx="12" cy="15.4" r=".6" fill="currentColor" stroke="none" />
      <circle cx="12" cy="9.6" r=".6" fill="currentColor" stroke="none" />
    </>,
  );
