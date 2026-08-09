export function Button({
  children,
  onClick,
  secondary = false,
  disabled = false,
}: any) {
  return (
    <button
      className={secondary ? "btn btn-secondary" : "btn"}
      onClick={onClick}
      disabled={disabled}
    >
      {children}
    </button>
  );
}
