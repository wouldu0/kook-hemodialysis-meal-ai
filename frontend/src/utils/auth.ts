import type { useNavigate } from "react-router-dom";
import { currentUser } from "../services/api";

export const requireUser = (nav: ReturnType<typeof useNavigate>) => {
  if (!currentUser()) {
    nav("/login");
    return false;
  }
  return true;
};
