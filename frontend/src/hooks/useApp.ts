import { createContext, useContext } from "react";
import type { AppState } from "../types";

export const AppContext = createContext<AppState | null>(null);

export const useApp = () => {
  const v = useContext(AppContext);
  if (!v) throw new Error("AppContext missing");
  return v;
};
