import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiRequestError, api, errorMessage, tokenStore } from "@/services/api";

function mockFetch(status: number, body: unknown, ok = status < 400) {
  const fn = vi.fn(async (_input: string, _init?: RequestInit) => ({
    ok,
    status,
    headers: new Headers(),
    text: async () => (body === undefined ? "" : JSON.stringify(body)),
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("sends the access token as a Bearer header and cookies for refresh", async () => {
    const fn = mockFetch(200, { id: "u1" });
    tokenStore.set("tok");
    await api.me();
    const init = fn.mock.calls[0][1] as RequestInit;
    expect((init.headers as Headers).get("Authorization")).toBe("Bearer tok");
    expect(init.credentials).toBe("include");
    tokenStore.set(null);
  });

  it("retries once after a silent refresh on 401", async () => {
    let calls = 0;
    vi.stubGlobal("fetch", vi.fn(async (input: string) => {
      calls += 1;
      if (String(input).endsWith("/auth/refresh")) return { ok: true, status: 200, headers: new Headers(), text: async () => JSON.stringify({ access_token: "new", token_type: "bearer", expires_in: 1800, user: { id: "u1" } }) };
      if (calls === 1) return { ok: false, status: 401, headers: new Headers(), text: async () => JSON.stringify({ detail: "expired", code: "unauthorized" }) };
      return { ok: true, status: 200, headers: new Headers(), text: async () => JSON.stringify({ version: "2.0.0" }) };
    }));
    const rt = await api.runtime();
    expect(rt.version).toBe("2.0.0");
    expect(calls).toBe(3);
    expect(tokenStore.get()).toBe("new");
    tokenStore.set(null);
  });

  it("parses successful JSON responses", async () => {
    const fn = mockFetch(200, { status: "ok", version: "2.0.0" });
    const res = await api.health();
    expect(res.version).toBe("2.0.0");
    expect(fn).toHaveBeenCalledWith("/api/v1/health", expect.any(Object));
  });

  it("raises ApiRequestError with backend detail and code", async () => {
    mockFetch(409, { detail: "No production model is configured for this organization.", code: "model_not_available" });
    await expect(api.analyze({ verified: false, friends_count: 0, followers_count: 0, listed_count: 0, favorites_count: 0, statuses_count: 0, default_profile: false, default_profile_image: false, geo_enabled: false, profile_background_tile: false, has_profile_banner: false, tweets: [] })).rejects.toMatchObject({
      status: 409,
      code: "model_not_available",
      message: "No production model is configured for this organization.",
    });
  });

  it("formats validation errors with field locations", () => {
    const err = new ApiRequestError(422, { detail: "Request validation failed", code: "validation_error", errors: [{ loc: ["body", "account", "followers_count"], msg: "must be >= 0", type: "greater_than_equal" }] }, "x");
    expect(errorMessage(err)).toContain("account.followers_count — must be >= 0");
  });

  it("maps network failures to a backend-unavailable message", () => {
    expect(errorMessage(new TypeError("Failed to fetch"))).toMatch(/API is not reachable/);
  });

  it("maps a non-JSON proxy 5xx to a backend-unavailable message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 500, headers: new Headers(), text: async () => "Error occurred while trying to proxy" })));
    await expect(api.health()).rejects.toMatchObject({ status: 500, message: expect.stringMatching(/API is not reachable/) });
  });

  it("builds history query strings and skips empty filters", async () => {
    const fn = mockFetch(200, { items: [], total: 0, page: 1, page_size: 25 });
    await api.history({ prediction: "BOT", min_risk: 60, search: "", page: 2 });
    const url = String(fn.mock.calls[0][0]);
    expect(url).toContain("/api/v1/analyses?");
    expect(url).toContain("prediction=BOT");
    expect(url).toContain("min_risk=60");
    expect(url).toContain("page=2");
    expect(url).not.toContain("search=");
  });

  it("returns undefined for 204 responses", async () => {
    mockFetch(204, undefined);
    await expect(api.deleteAnalysis("abc")).resolves.toBeUndefined();
  });
});
