import { Eraser, Play, Plug, ScanSearch } from "lucide-react";
import { useCallback, useMemo, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";

import { PredictionResult } from "@/components/PredictionResult";
import { Badge, Button, Card, EmptyState, ErrorState, Field, Input, Notice, PageHeader, Select, Textarea, Toggle } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { api } from "@/services/api";
import type { AccountInput, AnalysisResponse, ProviderFetchResponse, TweetInput } from "@/types/api";
import { dateTime } from "@/utils/format";

interface FormState {
  account_id: string;
  name: string;
  description: string;
  location: string;
  url: string;
  verified: boolean;
  default_profile: boolean;
  default_profile_image: boolean;
  geo_enabled: boolean;
  profile_background_tile: boolean;
  has_profile_banner: boolean;
  friends_count: string;
  followers_count: string;
  listed_count: string;
  favorites_count: string;
  statuses_count: string;
  hashtag_count: string;
  mentions_count: string;
  retweet_count: string;
  reply_count: string;
  url_count: string;
  favorite_count_received: string;
  tweets_observed: string;
  tweetsText: string;
  deriveFromTweets: boolean;
}

const EMPTY: FormState = {
  account_id: "", name: "", description: "", location: "", url: "",
  verified: false, default_profile: false, default_profile_image: false, geo_enabled: false, profile_background_tile: false, has_profile_banner: false,
  friends_count: "", followers_count: "", listed_count: "", favorites_count: "", statuses_count: "",
  hashtag_count: "", mentions_count: "", retweet_count: "", reply_count: "", url_count: "", favorite_count_received: "", tweets_observed: "",
  tweetsText: "", deriveFromTweets: true,
};

const toInt = (s: string): number => {
  const n = parseInt(s, 10);
  return Number.isFinite(n) && n >= 0 ? n : 0;
};
const toOpt = (s: string): number | null => (s.trim() === "" ? null : toInt(s));

function fromAccount(a: AccountInput): FormState {
  return {
    ...EMPTY,
    account_id: a.account_id ?? a.screen_name ?? "",
    name: a.name ?? "",
    description: a.description ?? "",
    location: a.location ?? "",
    url: a.url ?? "",
    verified: a.verified,
    default_profile: a.default_profile,
    default_profile_image: a.default_profile_image,
    geo_enabled: a.geo_enabled,
    profile_background_tile: a.profile_background_tile,
    has_profile_banner: a.has_profile_banner,
    friends_count: String(a.friends_count),
    followers_count: String(a.followers_count),
    listed_count: String(a.listed_count),
    favorites_count: String(a.favorites_count),
    statuses_count: String(a.statuses_count),
    hashtag_count: a.hashtag_count?.toString() ?? "",
    mentions_count: a.mentions_count?.toString() ?? "",
    retweet_count: a.retweet_count?.toString() ?? "",
    reply_count: a.reply_count?.toString() ?? "",
    url_count: a.url_count?.toString() ?? "",
    favorite_count_received: a.favorite_count_received?.toString() ?? "",
    tweets_observed: a.tweets_observed?.toString() ?? "",
    tweetsText: (a.tweets ?? []).map((t) => (typeof t === "string" ? t : t.text)).join("\n"),
    deriveFromTweets: !(a.hashtag_count != null || a.retweet_count != null),
  };
}

function toAccount(f: FormState, fetchedTweets: TweetInput[] | null): AccountInput {
  const lines = f.tweetsText.split("\n").map((l) => l.trim()).filter(Boolean);
  const tweets: (TweetInput | string)[] = fetchedTweets && fetchedTweets.map((t) => t.text).join("\n") === lines.join("\n") ? fetchedTweets : lines;
  const derive = f.deriveFromTweets && lines.length > 0;
  return {
    account_id: f.account_id || null, screen_name: f.account_id || null, name: f.name || null,
    verified: f.verified, friends_count: toInt(f.friends_count), followers_count: toInt(f.followers_count), listed_count: toInt(f.listed_count), favorites_count: toInt(f.favorites_count), statuses_count: toInt(f.statuses_count),
    description: f.description || null, location: f.location || null, url: f.url || null,
    default_profile: f.default_profile, default_profile_image: f.default_profile_image, geo_enabled: f.geo_enabled, profile_background_tile: f.profile_background_tile, has_profile_banner: f.has_profile_banner,
    hashtag_count: derive ? null : toOpt(f.hashtag_count), mentions_count: derive ? null : toOpt(f.mentions_count), url_count: derive ? null : toOpt(f.url_count),
    retweet_count: toOpt(f.retweet_count), reply_count: toOpt(f.reply_count), favorite_count_received: toOpt(f.favorite_count_received), tweets_observed: toOpt(f.tweets_observed),
    tweets,
  };
}

export function AnalyzePage() {
  const models = useApi(() => api.models(), []);
  const providers = useApi(() => api.providers(), []);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [fetchedTweets, setFetchedTweets] = useState<TweetInput[] | null>(null);
  const [live, setLive] = useState<ProviderFetchResponse | null>(null);
  const [modelId, setModelId] = useState("");
  const [handle, setHandle] = useState("");
  const [result, setResult] = useState<AnalysisResponse | null>(null);

  const analyze = useAction(useCallback((account: AccountInput, source: "manual" | "x_api") => api.analyze(account, { model_id: modelId || null, source }), [modelId]));
  const fetchLive = useAction(useCallback((identifier: string) => api.providerFetch("x_api", identifier), []));
  const xProvider = providers.data?.find((p) => p.name === "x_api") ?? null;
  const prodId = models.data?.production_model_id ?? null;
  const usable = (models.data?.models ?? []).filter((m) => m.status === "READY" || m.status === "PRODUCTION" || m.status === "DEPRECATED");
  const noModel = models.data !== null && !prodId && usable.length === 0;
  const activeModel = usable.find((m) => m.id === (modelId || prodId)) ?? null;

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) => setForm((f) => ({ ...f, [k]: v }));
  const tweetCount = useMemo(() => form.tweetsText.split("\n").filter((l) => l.trim()).length, [form.tweetsText]);

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const r = await analyze.run(toAccount(form, fetchedTweets), live ? "x_api" : "manual");
    if (r) setResult(r);
  };
  const onFetch = async () => {
    if (!handle.trim()) return;
    const r = await fetchLive.run(handle);
    if (r) {
      setForm(fromAccount(r.account));
      setFetchedTweets(r.account.tweets.map((t) => (typeof t === "string" ? { text: t } : t)));
      setLive(r);
      setResult(null);
    }
  };
  const clear = () => { setForm(EMPTY); setFetchedTweets(null); setLive(null); setResult(null); };

  return (
    <div className="space-y-6">
      <PageHeader title="Analyze Account" description="Enter the account's profile data or fetch it from a configured data provider. The API derives the research features, runs the production model and explains the classification with SHAP and LIME." actions={<Button variant="ghost" icon={Eraser} onClick={clear}>Clear</Button>} />

      {models.error && <ErrorState message={models.error} onRetry={models.reload} />}
      {noModel && (
        <Card>
          <EmptyState icon={ScanSearch} title="No production model configured" description="Analyses need a trained model. Train one on a labelled dataset and activate it." action={<Link to="/training"><Button>Go to Training</Button></Link>} />
        </Card>
      )}

      {live && (
        <Notice>
          Live data from X for <strong>@{live.account.screen_name}</strong> (fetched {dateTime(live.fetched_at)}{live.cached ? ", cached" : ""}): {live.tweets_fetched} recent tweet{live.tweets_fetched === 1 ? "" : "s"}{live.protected ? " · protected account" : ""}.{" "}
          {live.tweets_error ? `Tweets unavailable: ${live.tweets_error} ` : ""}Not exposed by API v2 (sent as false): {live.unavailable_fields.join(", ")}.
        </Notice>
      )}

      <form onSubmit={onSubmit} className="grid gap-4 xl:grid-cols-[1.1fr_1fr]" aria-label="Account analysis form">
        <div className="space-y-4">
          <Card title="1 · Account information" subtitle="Profile fields as shown on the account">
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Account ID / handle"><Input value={form.account_id} onChange={(e) => set("account_id", e.target.value)} placeholder="e.g. jdoe_92" /></Field>
              <Field label="Display name"><Input value={form.name} onChange={(e) => set("name", e.target.value)} /></Field>
              <Field label="Description (bio)" className="sm:col-span-2"><Textarea value={form.description} onChange={(e) => set("description", e.target.value)} /></Field>
              <Field label="Location"><Input value={form.location} onChange={(e) => set("location", e.target.value)} /></Field>
              <Field label="Profile URL"><Input value={form.url} onChange={(e) => set("url", e.target.value)} /></Field>
            </div>
            <div className="mt-3 grid gap-2 sm:grid-cols-2">
              <Toggle label="Verified" checked={form.verified} onChange={(v) => set("verified", v)} />
              <Toggle label="Default profile theme" checked={form.default_profile} onChange={(v) => set("default_profile", v)} />
              <Toggle label="Default profile image" checked={form.default_profile_image} onChange={(v) => set("default_profile_image", v)} />
              <Toggle label="Geo enabled" checked={form.geo_enabled} onChange={(v) => set("geo_enabled", v)} />
              <Toggle label="Profile background tiled" checked={form.profile_background_tile} onChange={(v) => set("profile_background_tile", v)} />
              <Toggle label="Has profile banner" checked={form.has_profile_banner} onChange={(v) => set("has_profile_banner", v)} />
            </div>
          </Card>
          <Card title="2 · Behavioral metrics" subtitle="Raw profile counts">
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label="Followers"><Input type="number" min={0} value={form.followers_count} onChange={(e) => set("followers_count", e.target.value)} /></Field>
              <Field label="Following (friends)"><Input type="number" min={0} value={form.friends_count} onChange={(e) => set("friends_count", e.target.value)} /></Field>
              <Field label="Listed"><Input type="number" min={0} value={form.listed_count} onChange={(e) => set("listed_count", e.target.value)} /></Field>
              <Field label="Likes given"><Input type="number" min={0} value={form.favorites_count} onChange={(e) => set("favorites_count", e.target.value)} /></Field>
              <Field label="Statuses (total posts)"><Input type="number" min={0} value={form.statuses_count} onChange={(e) => set("statuses_count", e.target.value)} /></Field>
              <Field label="Posts observed" hint="Denominator for per-post averages when no post text is given"><Input type="number" min={0} value={form.tweets_observed} onChange={(e) => set("tweets_observed", e.target.value)} /></Field>
            </div>
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="3 · Content" subtitle="Recent posts, one per line. Hashtags, mentions, URLs, linguistic and sentiment features are extracted server-side; post text itself is not stored.">
            {activeModel && activeModel.n_features < 31 && (
              <div className="mb-3"><Notice>{activeModel.name} v{activeModel.version} uses {activeModel.n_features} of the 31 research features (features constant in its training data were dropped). Post-level counts may not influence its classification; text still feeds the linguistic and sentiment features.</Notice></div>
            )}
            <Textarea value={form.tweetsText} onChange={(e) => set("tweetsText", e.target.value)} rows={8} className="min-h-[180px] font-mono text-xs" aria-label="Recent posts" />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-2">
              <span>{tweetCount} post{tweetCount === 1 ? "" : "s"}</span>
              <Toggle label="Derive hashtag / mention / URL counts from post text" checked={form.deriveFromTweets} onChange={(v) => set("deriveFromTweets", v)} />
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <Field label="Hashtags (total)"><Input type="number" min={0} disabled={form.deriveFromTweets && tweetCount > 0} value={form.hashtag_count} onChange={(e) => set("hashtag_count", e.target.value)} /></Field>
              <Field label="Mentions (total)"><Input type="number" min={0} disabled={form.deriveFromTweets && tweetCount > 0} value={form.mentions_count} onChange={(e) => set("mentions_count", e.target.value)} /></Field>
              <Field label="URLs (total)"><Input type="number" min={0} disabled={form.deriveFromTweets && tweetCount > 0} value={form.url_count} onChange={(e) => set("url_count", e.target.value)} /></Field>
              <Field label="Reposts received"><Input type="number" min={0} value={form.retweet_count} onChange={(e) => set("retweet_count", e.target.value)} /></Field>
              <Field label="Replies received"><Input type="number" min={0} value={form.reply_count} onChange={(e) => set("reply_count", e.target.value)} /></Field>
              <Field label="Likes received"><Input type="number" min={0} value={form.favorite_count_received} onChange={(e) => set("favorite_count_received", e.target.value)} /></Field>
            </div>
          </Card>

          <Card title="Run analysis" subtitle="Classification, SHAP and LIME are computed by the API for this exact input and stored in your history">
            <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
              <Field label="Model">
                <Select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                  <option value="">Production model{prodId ? ` (${usable.find((m) => m.id === prodId)?.name} v${usable.find((m) => m.id === prodId)?.version})` : " — none configured"}</option>
                  {usable.map((m) => <option key={m.id} value={m.id}>{m.name} v{m.version} · {m.status.toLowerCase()} · {m.dataset_name}</option>)}
                </Select>
              </Field>
              <Button type="submit" icon={Play} loading={analyze.loading} disabled={noModel}>Analyze account</Button>
            </div>
            {analyze.error && <div className="mt-3"><ErrorState title="Analysis failed" message={analyze.error} /></div>}
          </Card>

          <Card title="External data provider" subtitle="Fetch a public profile and recent posts from a configured provider." actions={xProvider && <Badge tone={xProvider.configured ? "good" : "neutral"} dot={xProvider.configured ? "var(--status-good)" : "var(--text-3)"}>X API v2 · {xProvider.configured ? "configured" : "not configured"}</Badge>}>
            {xProvider?.configured ? (
              <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
                <Field label="X username"><Input value={handle} onChange={(e) => setHandle(e.target.value)} placeholder="@username" onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void onFetch(); } }} /></Field>
                <Button type="button" variant="secondary" icon={Plug} loading={fetchLive.loading} disabled={!handle.trim()} onClick={onFetch}>Fetch</Button>
              </div>
            ) : (
              <Notice>External API integration is not configured. {xProvider?.configuration_hint ?? "Set BOTSHIELD_X_BEARER_TOKEN on the backend and restart the API."} Until then, enter account data manually or use batch CSV analysis.</Notice>
            )}
            {fetchLive.error && <p className="mt-2 text-xs text-status-critical" role="alert">{fetchLive.error}</p>}
          </Card>
        </div>
      </form>

      <div id="result">
        {result ? <PredictionResult result={result} /> : !analyze.loading && <EmptyState icon={ScanSearch} title="No analysis yet" description="Fill in the account data (or fetch it from a provider) and run the analysis. The classification, probabilities, risk indicator, SHAP and LIME explanations appear here." />}
      </div>
    </div>
  );
}
