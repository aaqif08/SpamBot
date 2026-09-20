import { createContext, useContext, type ReactNode } from "react";

import { useApi, type ApiState } from "@/hooks/useApi";
import { api } from "@/services/api";
import type { HealthResponse } from "@/types/api";

const HealthContext = createContext<ApiState<HealthResponse> | null>(null);

export function HealthProvider({ children }: { children: ReactNode }) {
  const state = useApi(() => api.health(), []);
  return <HealthContext.Provider value={state}>{children}</HealthContext.Provider>;
}

export function useHealth(): ApiState<HealthResponse> {
  const ctx = useContext(HealthContext);
  if (!ctx) throw new Error("useHealth must be used within HealthProvider");
  return ctx;
}
