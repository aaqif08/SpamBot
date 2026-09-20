import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiRequestError, api, errorMessage } from "@/services/api";

function mockFetch(status: number, body: unknown, ok = status < 400) {
  const fn = vi.fn(async (_input: string, _init?: RequestInit) => ({
    ok,
    status,
    text: async () => (body === undefined ? "" : JSON.stringify(body)),
  }));
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("parses successful JSON responses", async () => {
    const fn = mockFetch(200, { status: "ok", n_features: 31 });
    const res = await api.health();
    expect(res.n_features).toBe(31);
    expect(fn).toHaveBeenCalledWith("/api/health", expect.any(Object));
  });

  it("raises ApiRequestError with backend detail and code", async () => {
    mockFetch(503, { detail: "No trained model available.", code: "http_503" });
    await expect(api.predict({ verified: false, friends_count: 0, followers_count: 0, listed_count: 0, favorites_count: 0, statuses_count: 0, default_profile: false, default_profile_image: false, geo_enabled: false, profile_background_tile: false, has_profile_banner: false, tweets: [] })).rejects.toMatchObject({
      status: 503,
      code: "http_503",
      message: "No trained model available.",
    });
  });

  it("formats validation errors with field locations", () => {
    const err = new ApiRequestError(422, { detail: "Request validation failed", code: "validation_error", errors: [{ loc: ["body", "account", "followers_count"], msg: "must be >= 0", type: "greater_than_equal" }] }, "x");
    expect(errorMessage(err)).toContain("account.followers_count — must be >= 0");
  });

  it("maps network failures to a backend-unavailable message", () => {
    expect(errorMessage(new TypeError("Failed to fetch"))).toMatch(/Backend unavailable/);
  });

  it("maps a non-JSON proxy 5xx to a backend-unavailable message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 500, text: async () => "Error occurred while trying to proxy" })));
    await expect(api.health()).rejects.toMatchObject({ status: 500, message: expect.stringMatching(/Backend unavailable/) });
  });

  it("builds history query strings and skips empty filters", async () => {
    const fn = mockFetch(200, { items: [], total: 0, page: 1, page_size: 25 });
    await api.history({ prediction: "BOT", high_risk: true, search: "", page: 2 });
    const url = String(fn.mock.calls[0][0]);
    expect(url).toContain("prediction=BOT");
    expect(url).toContain("high_risk=true");
    expect(url).toContain("page=2");
    expect(url).not.toContain("search=");
  });

  it("returns undefined for 204 responses", async () => {
    mockFetch(204, undefined);
    await expect(api.deleteHistory("abc")).resolves.toBeUndefined();
  });
});
