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
