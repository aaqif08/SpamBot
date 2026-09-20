import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { PredictionResult } from "@/components/PredictionResult";
import { ConfusionMatrixView, ProportionBar } from "@/components/charts/basic";
import { DemoBanner, EmptyState, ErrorState, RiskBadge } from "@/components/ui";
import { useApi } from "@/hooks/useApi";
import type { PredictResponse } from "@/types/api";

const sampleResult: PredictResponse = {
  prediction_id: "p1",
  account_identifier: "demo_promo_bot",
  prediction: "BOT",
  bot_probability: 0.94,
  human_probability: 0.06,
  confidence: 0.94,
  risk_score: 94,
  risk_band: "critical",
  risk_score_note: "Risk Score is an application-level representation of the model probability.",
  model: { id: "m1", name: "LightGBM", algorithm: "lightgbm", version: "1", feature_version: "paper-31-v1", n_features: 31 },
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
  interpretation: { summary: "Model prediction: BOT. Estimated bot probability 94.0%.", recommendation: "Recommend manual review.", bot_indicators: [], human_indicators: [], disclaimer: "" },
  is_demo: true,
  created_at: "2026-01-01T00:00:00Z",
};

describe("PredictionResult", () => {
  it("renders prediction, probability, risk, SHAP and LIME sections from real payload values", () => {
    render(<PredictionResult result={sampleResult} />);
    expect(screen.getByText(/BOT \/ spambot-like/)).toBeInTheDocument();
    expect(screen.getAllByText("94").length).toBeGreaterThan(0);
    expect(screen.getByText(/Why was this account classified as BOT/)).toBeInTheDocument();
    expect(screen.getByText(/Local explanation — LIME/)).toBeInTheDocument();
    expect(screen.getAllByText(/DEMO DATA — NOT REAL SOCIAL MEDIA DATA/).length).toBeGreaterThan(0);
    expect(screen.getByText("+0.3100")).toBeInTheDocument();
    expect(screen.getByText("Model prediction: BOT. Estimated bot probability 94.0%.")).toBeInTheDocument();
  });

  it("shows explanation-unavailable notices when SHAP/LIME are missing", () => {
    render(<PredictionResult result={{ ...sampleResult, shap_explanation: null, lime_explanation: null, explanation_errors: { shap: "boom", lime: "bang" } }} />);
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

  it("EmptyState and DemoBanner render their copy", () => {
    render(
      <>
        <EmptyState title="No trained model available" description="Run the training pipeline to populate evaluation results." />
        <DemoBanner />
        <RiskBadge band="high" score={72} />
      </>,
    );
    expect(screen.getByText("No trained model available")).toBeInTheDocument();
    expect(screen.getByText(/DEMO DATA — NOT REAL SOCIAL MEDIA DATA/)).toBeInTheDocument();
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
    await waitFor(() => expect(screen.getByText(/error: Backend unavailable/)).toBeInTheDocument());
  });
});
