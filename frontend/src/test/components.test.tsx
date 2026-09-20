import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PredictionResult } from "@/components/PredictionResult";
import { ConfusionMatrixView, ProportionBar } from "@/components/charts/basic";
import { EmptyState, ErrorState, RiskBadge, StatusBadge } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import type { AnalysisResponse } from "@/types/api";

const analysis: AnalysisResponse = {
  prediction_id: "p1",
  account_identifier: "acct_1",
  account_ref: "acct_1",
  prediction: "BOT",
  bot_probability: 0.94,
  human_probability: 0.06,
  confidence: 0.94,
  risk_score: 94,
  risk_band: "critical",
  risk_score_note: "Risk score = round(100 × estimated bot probability); a model output, not a verified fact.",
  model: { id: "m1", name: "LightGBM", version: 1 },
  input_summary: {},
  features: { hashtag_count: 25, ffratio: 0.12 },
  feature_groups: { content: { hashtag_count: 25 }, engagement: { ffratio: 0.12 } },
  auxiliary: {},
  top_features: [
    { feature: "hashtag_count", group: "content", description: "Total hashtags", value: 25, impact: 0.31, direction: "BOT" },
    { feature: "verified", group: "user_profile", description: "Verified", value: 0, impact: -0.15, direction: "HUMAN" },
  ],
  shap_explanation: {
    explainer: "TreeExplainer",
    output_scale: "probability",
    base_value: 0.5,
    model_output: 0.94,
    sum_positive: 0.6,
    sum_negative: -0.16,
    contributions: [
      { feature: "hashtag_count", group: "content", description: "Total hashtags", value: 25, value_scaled: 0.8, shap: 0.31, direction: "BOT", cumulative: 0.81 },
      { feature: "verified", group: "user_profile", description: "Verified", value: 0, value_scaled: 0, shap: -0.15, direction: "HUMAN", cumulative: 0.66 },
    ],
  },
  lime_explanation: {
    class_names: ["HUMAN", "BOT"],
    prediction_probabilities: { HUMAN: 0.1, BOT: 0.9 },
    intercept: 0.4,
    local_prediction: 0.88,
    surrogate_r2: 0.7,
    num_samples: 3000,
    bot_contribution: 0.4,
    human_contribution: 0.1,
    bot_indicators: [{ feature: "hashtag_count", group: "content", description: "", rule: "hashtag_count > 0.50", weight: 0.4, direction: "BOT", value: 25, value_scaled: 0.8 }],
    human_indicators: [{ feature: "verified", group: "user_profile", description: "", rule: "verified <= 0.00", weight: -0.1, direction: "HUMAN", value: 0, value_scaled: 0 }],
    items: [
      { feature: "hashtag_count", group: "content", description: "", rule: "hashtag_count > 0.50", weight: 0.4, direction: "BOT", value: 25, value_scaled: 0.8 },
      { feature: "verified", group: "user_profile", description: "", rule: "verified <= 0.00", weight: -0.1, direction: "HUMAN", value: 0, value_scaled: 0 },
    ],
  },
  explanation_errors: {},
  explanation_status: { shap: "available", lime: "available" },
  interpretation: { summary: "Model prediction: BOT. Estimated bot probability 94.0%.", recommendation: "Recommend manual review.", bot_indicators: [], human_indicators: [], disclaimer: "" },
  source: "manual",
  status: "COMPLETED",
  batch_id: null,
  label_true: null,
  inference_ms: 12,
  created_at: "2026-01-01T00:00:00Z",
  created_by: "u1",
};

describe("PredictionResult", () => {
  it("renders prediction, probability, risk, SHAP and LIME sections from real payload values", () => {
    render(<PredictionResult result={analysis} />);
    expect(screen.getByText("Classified as BOT")).toBeInTheDocument();
    expect(screen.getAllByText("94").length).toBeGreaterThan(0);
    expect(screen.getByText(/Why was this account classified as BOT/)).toBeInTheDocument();
    expect(screen.getByText(/Local explanation — LIME/)).toBeInTheDocument();
    expect(screen.getByText("+0.3100")).toBeInTheDocument();
    expect(screen.getByText("Model prediction: BOT. Estimated bot probability 94.0%.")).toBeInTheDocument();
  });

  it("shows explanation-unavailable notices when SHAP/LIME are missing", () => {
    render(<PredictionResult result={{ ...analysis, shap_explanation: null, lime_explanation: null, explanation_errors: { shap: "boom", lime: "bang" } }} />);
    expect(screen.getByText(/SHAP explanation unavailable: boom/)).toBeInTheDocument();
    expect(screen.getByText(/LIME explanation unavailable: bang/)).toBeInTheDocument();
  });
});

describe("UI states", () => {
  it("ErrorState calls retry", async () => {
    const onRetry = vi.fn();
    render(<ErrorState message="Backend unavailable" onRetry={onRetry} />);
    await userEvent.click(screen.getByRole("button", { name: /retry/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("alert")).toHaveTextContent("Backend unavailable");
  });

  it("EmptyState, StatusBadge and RiskBadge render their copy", () => {
    render(
      <>
        <EmptyState title="No production model configured" description="Train and activate a model first." />
        <StatusBadge status="PRODUCTION" />
        <RiskBadge band="high" score={72} />
      </>,
    );
    expect(screen.getByText("No production model configured")).toBeInTheDocument();
    expect(screen.getByText("production")).toBeInTheDocument();
    expect(screen.getByText("high")).toBeInTheDocument();
    expect(screen.getByText("72")).toBeInTheDocument();
  });

  it("ProportionBar and ConfusionMatrixView show counts", () => {
    render(
      <>
        <ProportionBar bots={30} humans={70} />
        <ConfusionMatrixView cm={{ labels: ["HUMAN", "BOT"], matrix: [[75, 0], [3, 72]], tn: 75, fp: 0, fn: 3, tp: 72, false_positive_rate: 0, false_negative_rate: 0.04 }} />
      </>,
    );
    expect(screen.getByRole("img", { name: "30 bots, 70 humans" })).toBeInTheDocument();
    expect(screen.getByText("75")).toBeInTheDocument();
    expect(screen.getByText("False negative")).toBeInTheDocument();
  });
});

function Probe({ fetcher }: { fetcher: () => Promise<{ v: number }> }) {
  const s = useApi(fetcher, []);
  if (s.loading) return <p>loading</p>;
  if (s.error) return <p>error: {s.error}</p>;
  return <p>value {s.data?.v}</p>;
}

describe("useApi", () => {
  it("transitions loading → data", async () => {
    render(<Probe fetcher={async () => ({ v: 7 })} />);
    expect(screen.getByText("loading")).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("value 7")).toBeInTheDocument());
  });

  it("transitions loading → error with a readable message", async () => {
    render(<Probe fetcher={async () => { throw new TypeError("Failed to fetch"); }} />);
    await waitFor(() => expect(screen.getByText(/error: The API is not reachable/)).toBeInTheDocument());
  });
});
