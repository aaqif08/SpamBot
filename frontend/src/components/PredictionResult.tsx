import { Bot, Gauge, Sparkles, UserRound } from "lucide-react";
import { useState } from "react";

import { LimeChart, ProbabilityBars, ShapContributionChart, ShapWaterfall } from "@/components/charts/explain";
import { Badge, Card, DemoBanner, Notice, PredictionBadge, RiskBadge, Table, Tabs, Td, Th, riskColor } from "@/components/ui";
import type { PredictResponse } from "@/types/api";
import { featureLabel, groupLabel, num, pct, signed } from "@/utils/format";

function RiskMeter({ score, band }: { score: number; band: string }) {
  return (
    <div>
      <div className="flex items-end justify-between">
        <span className="text-4xl font-semibold tabular-nums text-ink">{score}</span>
        <span className="text-xs text-ink-3">/ 100</span>
      </div>
      <div className="mt-2 h-2.5 w-full overflow-hidden rounded-full bg-surface-2">
        <div className="h-full rounded-full transition-[width]" style={{ width: `${score}%`, background: riskColor(band) }} />
      </div>
      <div className="mt-2 flex items-center justify-between text-[11px] text-ink-3">
        <span>minimal</span>
        <span>low</span>
        <span>medium</span>
        <span>high</span>
        <span>critical</span>
      </div>
    </div>
  );
}

export function PredictionResult({ result }: { result: PredictResponse }) {
  const [shapView, setShapView] = useState<"bars" | "waterfall">("bars");
  const isBot = result.prediction === "BOT";
  const shap = result.shap_explanation;
  const lime = result.lime_explanation;

  return (
    <div className="space-y-4 animate-fade-in">
      {result.is_demo && <DemoBanner compact text="This prediction involves demonstration data: a hand-written sample account and/or a model trained on synthetic data. It is not an observation from a real social network." />}

      {/* Prediction / confidence / risk */}
      <div className="grid gap-4 md:grid-cols-3">
        <Card title="Model prediction" subtitle={`${result.model.name} · ${result.model.n_features} features`}>
          <div className="flex items-center gap-3">
            <span className="rounded-xl p-3" style={{ background: isBot ? "var(--bot-soft)" : "var(--human-soft)" }}>
              {isBot ? <Bot className="h-6 w-6" style={{ color: "var(--bot)" }} aria-hidden /> : <UserRound className="h-6 w-6" style={{ color: "var(--human)" }} aria-hidden />}
            </span>
            <div>
              <div className="text-2xl font-semibold text-ink">{isBot ? "BOT / spambot-like" : "HUMAN / legitimate-like"}</div>
              <div className="mt-1 flex items-center gap-2">
                <PredictionBadge label={result.prediction} />
                <span className="text-xs text-ink-3">for {result.account_identifier}</span>
              </div>
            </div>
          </div>
        </Card>
        <Card title="Estimated bot probability" subtitle="Model confidence = max(P(bot), P(human))">
          <ProbabilityBars human={result.human_probability} bot={result.bot_probability} />
          <p className="mt-3 text-xs text-ink-2">
            Confidence <strong className="font-mono text-ink">{pct(result.confidence, 1)}</strong>
          </p>
        </Card>
        <Card title="Risk score" subtitle="Application-level view of P(bot)" actions={<RiskBadge band={result.risk_band} />}>
          <RiskMeter score={result.risk_score} band={result.risk_band} />
        </Card>
      </div>

      <Notice>
        <Gauge className="mr-1 inline h-3.5 w-3.5" aria-hidden />
        {result.risk_score_note}
      </Notice>

      {/* Interpretation */}
      <Card title="Interpretation" subtitle="Plain-language summary generated from the SHAP contributions of this prediction">
        <p className="text-sm text-ink">{result.interpretation.summary}</p>
        <p className="mt-2 text-sm text-ink-2">
          <Sparkles className="mr-1 inline h-3.5 w-3.5" aria-hidden />
          {result.interpretation.recommendation}
        </p>
      </Card>

      {/* SHAP */}
      <Card
        title={`Why was this account classified as ${result.prediction}? — SHAP`}
        subtitle={shap ? `${shap.explainer} · base value ${num(shap.base_value, 3)} → model output ${num(shap.model_output, 3)} (${shap.output_scale.replace("_", " ")})` : "Local SHAP explanation"}
        actions={shap && <Tabs tabs={[{ key: "bars", label: "Contributions" }, { key: "waterfall", label: "Waterfall" }]} value={shapView} onChange={setShapView} />}
      >
        {shap ? (
          <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
            {shapView === "bars" ? <ShapContributionChart shap={shap} /> : <ShapWaterfall shap={shap} />}
            <div>
              <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-2">Feature impact table</h4>
              <Table className="max-h-[360px] overflow-y-auto">
                <thead>
                  <tr>
                    <Th>Feature</Th>
                    <Th align="right">Value</Th>
                    <Th align="right">Impact</Th>
                    <Th>Direction</Th>
                  </tr>
                </thead>
                <tbody>
                  {shap.contributions.slice(0, 12).map((c) => (
                    <tr key={c.feature}>
                      <Td>{featureLabel(c.feature)}</Td>
                      <Td align="right" mono>{num(c.value, 3)}</Td>
                      <Td align="right" mono>{signed(c.shap, 4)}</Td>
                      <Td>
                        <Badge tone={c.direction === "BOT" ? "bot" : c.direction === "HUMAN" ? "human" : "neutral"}>{c.direction}</Badge>
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          </div>
        ) : (
          <Notice tone="warning">SHAP explanation unavailable{result.explanation_errors.shap ? `: ${result.explanation_errors.shap}` : "."}</Notice>
        )}
      </Card>

      {/* LIME */}
      <Card title="Local explanation — LIME" subtitle={lime ? `Surrogate fitted around this account · bot contribution ${signed(lime.bot_contribution, 3)} · human contribution ${signed(lime.human_contribution, 3)}` : "LIME local explanation"}>
        {lime ? (
          <div className="grid gap-6 lg:grid-cols-[1fr_1.4fr]">
            <div className="space-y-4">
              <div>
                <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-2">Prediction probabilities (LIME)</h4>
                <ProbabilityBars human={lime.prediction_probabilities.HUMAN} bot={lime.prediction_probabilities.BOT} />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg border border-border p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-ink">
                    <span className="h-2 w-2 rounded-full" style={{ background: "var(--bot)" }} aria-hidden />
                    Top positive indicators (BOT)
                  </div>
                  <ul className="space-y-1 text-xs text-ink-2">
                    {lime.bot_indicators.slice(0, 5).map((i) => (
                      <li key={i.rule} className="flex justify-between gap-2">
                        <span className="truncate" title={i.rule}>{i.rule}</span>
                        <span className="font-mono text-ink">{signed(i.weight, 3)}</span>
                      </li>
                    ))}
                    {!lime.bot_indicators.length && <li>none</li>}
                  </ul>
                </div>
                <div className="rounded-lg border border-border p-3">
                  <div className="mb-1 flex items-center gap-1.5 text-xs font-semibold text-ink">
                    <span className="h-2 w-2 rounded-full" style={{ background: "var(--human)" }} aria-hidden />
                    Top negative indicators (HUMAN)
                  </div>
                  <ul className="space-y-1 text-xs text-ink-2">
                    {lime.human_indicators.slice(0, 5).map((i) => (
                      <li key={i.rule} className="flex justify-between gap-2">
                        <span className="truncate" title={i.rule}>{i.rule}</span>
                        <span className="font-mono text-ink">{signed(i.weight, 3)}</span>
                      </li>
                    ))}
                    {!lime.human_indicators.length && <li>none</li>}
                  </ul>
                </div>
              </div>
            </div>
            <LimeChart lime={lime} />
          </div>
        ) : (
          <Notice tone="warning">LIME explanation unavailable{result.explanation_errors.lime ? `: ${result.explanation_errors.lime}` : "."}</Notice>
        )}
      </Card>

      {/* Key indicators + features */}
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Key indicators" subtitle="Highest-impact features for this prediction">
          <ul className="divide-y divide-border">
            {result.top_features.map((f) => (
              <li key={f.feature} className="flex items-center justify-between gap-3 py-2 text-sm">
                <div className="min-w-0">
                  <div className="font-medium text-ink">{featureLabel(f.feature)}</div>
                  <div className="truncate text-[11px] text-ink-3">{f.description}</div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <span className="font-mono text-xs text-ink-2">{f.value === null ? "—" : num(f.value, 3)}</span>
                  {f.impact !== null && (
                    <Badge tone={f.direction === "BOT" ? "bot" : "human"}>
                      {signed(f.impact, 3)} {f.direction}
                    </Badge>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </Card>
        <Card title="Feature vector" subtitle={`${Object.values(result.feature_groups).reduce((a, g) => a + Object.keys(g).length, 0)} of the paper's 31 features are used by ${result.model.name}${result.model.n_features < 31 ? " (features that were constant in its training data were dropped; see Evaluation)" : ""}, grouped as in Table 4`}>
          <div className="scrollbar-thin max-h-[380px] space-y-3 overflow-y-auto pr-1">
            {Object.entries(result.feature_groups).map(([g, feats]) => (
              <div key={g}>
                <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-2">{groupLabel(g)}</div>
                <dl className="grid grid-cols-2 gap-x-4 gap-y-0.5 text-xs sm:grid-cols-3">
                  {Object.entries(feats).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2 border-b border-border py-0.5">
                      <dt className="truncate text-ink-2">{featureLabel(k)}</dt>
                      <dd className="font-mono tabular-nums text-ink">{Number.isInteger(v) ? v : num(v, 3)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
