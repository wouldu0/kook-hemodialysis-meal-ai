export function MealListRow({
  name,
  role,
  onClick,
}: {
  name: string;
  role: string;
  onClick: () => void;
}) {
  return (
    <button className="meal-row" onClick={onClick}>
      <span className="meal-row-role">{role}</span>
      <b className="meal-row-name">{name}</b>
      <i className="meal-row-arrow">›</i>
    </button>
  );
}
