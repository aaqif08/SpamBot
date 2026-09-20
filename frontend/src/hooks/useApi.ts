import { useCallback, useEffect, useRef, useState } from "react";

import { errorMessage } from "@/services/api";

export interface ApiState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  status: number | null;
  reload: () => void;
  setData: (updater: T | ((prev: T | null) => T | null)) => void;
}

/**
 * Fetch-on-mount hook with loading / error / reload. `deps` re-run the fetch.
 * Pass `enabled=false` to defer.
 */
export function useApi<T>(fetcher: () => Promise<T>, deps: unknown[] = [], enabled = true): ApiState<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<number | null>(null);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetcherRef
      .current()
      .then((d) => {
        if (!cancelled) {
          setData(d);
          setStatus(200);
        }
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          setError(errorMessage(e));
          setStatus((e as { status?: number }).status ?? null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, tick, ...deps]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  const set = useCallback((updater: T | ((prev: T | null) => T | null)) => {
    setData((prev) => (typeof updater === "function" ? (updater as (p: T | null) => T | null)(prev) : updater));
  }, []);
  return { data, loading, error, status, reload, setData: set };
}

/** Imperative async action with loading/error state (for buttons / forms). */
export function useAction<TArgs extends unknown[], TResult>(action: (...args: TArgs) => Promise<TResult>) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<TResult | null>(null);
  const run = useCallback(
    async (...args: TArgs): Promise<TResult | null> => {
      setLoading(true);
      setError(null);
      try {
        const r = await action(...args);
        setResult(r);
        return r;
      } catch (e) {
        setError(errorMessage(e));
        return null;
      } finally {
        setLoading(false);
      }
    },
    [action],
  );
  return { run, loading, error, result, setError, setResult };
}
