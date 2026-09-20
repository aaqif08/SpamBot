import { useEffect, useRef, useState } from "react";

import { api, errorMessage } from "@/services/api";
import type { JobResponse } from "@/types/api";

const TERMINAL = new Set(["COMPLETED", "FAILED", "CANCELLED"]);

/** Poll a background job (/jobs/{id}) until it reaches a terminal status. */
export function useJobPolling(jobId: string | null, intervalMs = 1000) {
  const [status, setStatus] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    setStatus(null);
    setError(null);
    if (!jobId) return;
    let stopped = false;
    const tick = async () => {
      try {
        const s = await api.job(jobId);
        if (stopped) return;
        setStatus(s);
        if (TERMINAL.has(s.status)) return;
      } catch (e) {
        if (stopped) return;
        setError(errorMessage(e));
        return;
      }
      timer.current = window.setTimeout(tick, intervalMs);
    };
    void tick();
    return () => {
      stopped = true;
      if (timer.current) window.clearTimeout(timer.current);
    };
  }, [jobId, intervalMs]);

  return { status, error, done: status !== null && TERMINAL.has(status.status) };
}
