import { nmeta } from "../../utils/nutrition";

export function NutrientIconRow({ caption }: { caption?: string }) {
  return (
    <div className="nutrient-icon-row">
      <div className="nutrient-icons">
        {nmeta.map((n) => (
          <div key={n.key}>
            <span>{n.icon}</span>
            <small>{n.label}</small>
          </div>
        ))}
      </div>
      {caption && <p className="nutrient-icon-caption">{caption}</p>}
    </div>
  );
}
