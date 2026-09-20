import { useEffect, useRef, useState } from "react";

import { api, errorMessage } from "@/services/api";
import type { TrainStatusResponse } from "@/types/api";

/** Poll a background job (training or dataset import) until it finishes. */
export function useJobPolling(jobId: string | null, kind: "train" | "dataset" = "train", intervalMs = 1000) {
  const [status, setStatus] = useState<TrainStatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    setStatus(null);
    setError(null);
    if (!jobId) return;
    let stopped = false;
    const tick = async () => {
      try {
        const s = kind === "train" ? await api.trainStatus(jobId) : await api.jobStatus(jobId);
        if (stopped) return;
        setStatus(s);
        if (s.status === "completed" || s.status === "failed") return;
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
  }, [jobId, kind, intervalMs]);

  return { status, error, done: status?.status === "completed" || status?.status === "failed" };
}
