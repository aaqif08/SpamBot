import { Eraser, FlaskConical, Play, Plug } from "lucide-react";
import { useCallback, useMemo, useState, type FormEvent } from "react";

import { PredictionResult } from "@/components/PredictionResult";
import { Badge, Button, Card, DemoBanner, EmptyState, ErrorState, Field, Input, Notice, PageHeader, Select, Textarea, Toggle } from "@/components/ui";
import { useAction, useApi } from "@/hooks/useApi";
import { useHealth } from "@/hooks/useHealth";
import { api } from "@/services/api";
import type { AccountInput, AdapterFetchResponse, PredictResponse, TweetInput } from "@/types/api";
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
  account_id: "",
  name: "",
  description: "",
  location: "",
  url: "",
  verified: false,
  default_profile: false,
  default_profile_image: false,
  geo_enabled: false,
  profile_background_tile: false,
  has_profile_banner: false,
  friends_count: "",
  followers_count: "",
  listed_count: "",
  favorites_count: "",
  statuses_count: "",
  hashtag_count: "",
  mentions_count: "",
  retweet_count: "",
  reply_count: "",
  url_count: "",
  favorite_count_received: "",
  tweets_observed: "",
  tweetsText: "",
  deriveFromTweets: true,
};

const toInt = (s: string): number => {
  const n = parseInt(s, 10);
  return Number.isFinite(n) && n >= 0 ? n : 0;
};
const toOpt = (s: string): number | null => (s.trim() === "" ? null : toInt(s));

function fromAccount(a: AccountInput): FormState {
  const tweets = (a.tweets ?? []).map((t) => (typeof t === "string" ? t : t.text)).join("\n");
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
    tweetsText: tweets,
    deriveFromTweets: !(a.hashtag_count != null || a.retweet_count != null),
  };
}

function toAccount(f: FormState, sampleTweets: TweetInput[] | null): AccountInput {
  const lines = f.tweetsText
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  // Keep per-tweet engagement counts from a loaded sample when the text was not edited.
  const tweets: (TweetInput | string)[] = sampleTweets && sampleTweets.map((t) => t.text).join("\n") === lines.join("\n") ? sampleTweets : lines;
  const derive = f.deriveFromTweets && lines.length > 0;
  return {
    account_id: f.account_id || null,
    screen_name: f.account_id || null,
    name: f.name || null,
    verified: f.verified,
    friends_count: toInt(f.friends_count),
    followers_count: toInt(f.followers_count),
    listed_count: toInt(f.listed_count),
    favorites_count: toInt(f.favorites_count),
    statuses_count: toInt(f.statuses_count),
    description: f.description || null,
    location: f.location || null,
    url: f.url || null,
    default_profile: f.default_profile,
    default_profile_image: f.default_profile_image,
    geo_enabled: f.geo_enabled,
    profile_background_tile: f.profile_background_tile,
    has_profile_banner: f.has_profile_banner,
    hashtag_count: derive ? null : toOpt(f.hashtag_count),
    mentions_count: derive ? null : toOpt(f.mentions_count),
    url_count: derive ? null : toOpt(f.url_count),
    retweet_count: toOpt(f.retweet_count),
    reply_count: toOpt(f.reply_count),
    favorite_count_received: toOpt(f.favorite_count_received),
    tweets_observed: toOpt(f.tweets_observed),
    tweets,
  };
}

export function AnalyzePage() {
  const health = useHealth();
  const samples = useApi(() => api.sampleAccounts(), []);
  const models = useApi(() => api.models(), []);
  const [form, setForm] = useState<FormState>(EMPTY);
  const [sampleTweets, setSampleTweets] = useState<TweetInput[] | null>(null);
  const [loadedSample, setLoadedSample] = useState<string | null>(null);
  const [modelId, setModelId] = useState<string>("");
  const [result, setResult] = useState<PredictResponse | null>(null);
  const adapters = useApi(() => api.adapters(), []);
  const [adapterName, setAdapterName] = useState<string>("x_api");
  const [adapterId, setAdapterId] = useState("");
  const [liveFetch, setLiveFetch] = useState<AdapterFetchResponse | null>(null);

  const predict = useAction(useCallback((account: AccountInput, source: "manual" | "sample" | "adapter") => api.predict(account, { model_id: modelId || null, source }), [modelId]));
  const fetchAdapter = useAction(useCallback((name: string, identifier: string) => api.adapterFetch(name, identifier), []));
  const selectedAdapter = adapters.data?.find((a) => a.name === adapterName) ?? null;

  const set = <K extends keyof FormState>(k: K, v: FormState[K]) => setForm((f) => ({ ...f, [k]: v }));

  const loadSample = (sampleId: string) => {
    const s = samples.data?.find((x) => x.sample_id === sampleId);
    if (!s) return;
    setForm(fromAccount(s.account));
    setSampleTweets(s.account.tweets.map((t) => (typeof t === "string" ? { text: t } : t)));
    setLoadedSample(s.sample_id);
    setLiveFetch(null);
    setResult(null);
  };

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const source = loadedSample ? "sample" : liveFetch ? "adapter" : "manual";
    const r = await predict.run(toAccount(form, sampleTweets), source);
    if (r) setResult(r);
  };

  const onAdapterFetch = async () => {
    if (!adapterId.trim()) return;
    const r = await fetchAdapter.run(adapterName, adapterId);
    if (r) {
      setForm(fromAccount(r.account));
      setSampleTweets(r.account.tweets.map((t) => (typeof t === "string" ? { text: t } : t)));
      if (r.source === "x_api") {
        setLoadedSample(null);
        setLiveFetch(r);
      } else {
        setLoadedSample(r.sample_id ?? "sample");
        setLiveFetch(null);
      }
      setResult(null);
    }
  };

  const tweetCount = useMemo(() => form.tweetsText.split("\n").filter((l) => l.trim()).length, [form.tweetsText]);
  const noModel = health.data ? !health.data.model_available : false;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Analyze Account"
        description="Enter a social-network account profile. The backend derives the 31 paper features (ratios, averages, linguistic and sentiment features), runs the active model and explains the prediction with SHAP and LIME."
        actions={
          <>
            <Select value="" onChange={(e) => e.target.value && loadSample(e.target.value)} className="w-56" aria-label="Load sample account">
              <option value="">Load sample account (DEMO)…</option>
              {samples.data?.map((s) => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.sample_id} — {s.label_hint}
                </option>
              ))}
            </Select>
            <Button variant="ghost" icon={Eraser} onClick={() => { setForm(EMPTY); setSampleTweets(null); setLoadedSample(null); setResult(null); }}>
              Clear
            </Button>
          </>
        }
      />

      {noModel && (
        <Notice tone="warning">
          No trained model available. Go to <strong>Training</strong>, create the demo dataset (or import Cresci) and train a model before analysing accounts.
        </Notice>
      )}
      {loadedSample && <DemoBanner compact text={`Loaded sample "${loadedSample}". Sample accounts are hand-written demonstration data; nothing was fetched from X/Twitter.`} />}
      {liveFetch && (
        <Notice>
          Live data from the X API v2 for <strong>@{liveFetch.account.screen_name}</strong> (fetched {dateTime(liveFetch.fetched_at)}{liveFetch.cached ? ", cached" : ""}): {liveFetch.tweets_fetched} recent tweet{liveFetch.tweets_fetched === 1 ? "" : "s"}
          {liveFetch.protected ? " - protected account" : ""}. {liveFetch.tweets_error ? `Tweets not available: ${liveFetch.tweets_error} ` : ""}
          Not exposed by API v2 (sent as false): {liveFetch.unavailable_fields.join(", ")}.
        </Notice>
      )}

      <form onSubmit={onSubmit} className="grid gap-4 xl:grid-cols-[1.1fr_1fr]">
        <div className="space-y-4">
          <Card title="1 · Account information" subtitle="Profile fields as shown on the account">
            <div className="grid gap-3 sm:grid-cols-2">
              <Field label="Account ID / screen name"><Input value={form.account_id} onChange={(e) => set("account_id", e.target.value)} placeholder="e.g. jdoe_92" /></Field>
              <Field label="Display name"><Input value={form.name} onChange={(e) => set("name", e.target.value)} /></Field>
              <Field label="Description (bio)" className="sm:col-span-2"><Textarea value={form.description} onChange={(e) => set("description", e.target.value)} placeholder="Empty bio → imputed as 'missing' (paper §III-B)" /></Field>
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

          <Card title="2 · Behavioral metrics" subtitle="Raw profile counts (user-profile feature group)">
            <div className="grid gap-3 sm:grid-cols-3">
              <Field label="Followers"><Input type="number" min={0} value={form.followers_count} onChange={(e) => set("followers_count", e.target.value)} /></Field>
              <Field label="Following (friends)"><Input type="number" min={0} value={form.friends_count} onChange={(e) => set("friends_count", e.target.value)} /></Field>
              <Field label="Listed"><Input type="number" min={0} value={form.listed_count} onChange={(e) => set("listed_count", e.target.value)} /></Field>
              <Field label="Favorites given (likes)"><Input type="number" min={0} value={form.favorites_count} onChange={(e) => set("favorites_count", e.target.value)} /></Field>
              <Field label="Statuses (total tweets)"><Input type="number" min={0} value={form.statuses_count} onChange={(e) => set("statuses_count", e.target.value)} /></Field>
              <Field label="Tweets observed" hint="Denominator for per-tweet averages when no tweet text is given"><Input type="number" min={0} value={form.tweets_observed} onChange={(e) => set("tweets_observed", e.target.value)} /></Field>
            </div>
          </Card>
        </div>

        <div className="space-y-4">
          <Card title="3 · Content analysis" subtitle="Paste recent tweets (one per line). Hashtags, mentions, URLs, linguistic and sentiment features are extracted server-side.">
            {health.data?.active_model && health.data.active_model.n_features < 31 && (
              <div className="mb-3">
                <Notice>
                  The active model ({health.data.active_model.name}) uses {health.data.active_model.n_features} of the 31 paper features: it was trained on user-level data without tweet files, so hashtag / mention / URL / retweet / reply counts do not influence its prediction. Tweet text still feeds the linguistic and sentiment features.
                </Notice>
              </div>
            )}
            <Textarea value={form.tweetsText} onChange={(e) => { set("tweetsText", e.target.value); }} rows={8} placeholder={"One tweet per line…\nGET FOLLOWERS FAST http://bit.ly/x #followback"} className="min-h-[180px] font-mono text-xs" />
            <div className="mt-2 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-2">
              <span>{tweetCount} tweet{tweetCount === 1 ? "" : "s"}</span>
              <Toggle label="Derive hashtag / mention / URL counts from tweet text" checked={form.deriveFromTweets} onChange={(v) => set("deriveFromTweets", v)} />
            </div>
            <div className="mt-3 grid gap-3 sm:grid-cols-3">
              <Field label="Hashtags (total)"><Input type="number" min={0} disabled={form.deriveFromTweets && tweetCount > 0} value={form.hashtag_count} onChange={(e) => set("hashtag_count", e.target.value)} /></Field>
              <Field label="Mentions (total)"><Input type="number" min={0} disabled={form.deriveFromTweets && tweetCount > 0} value={form.mentions_count} onChange={(e) => set("mentions_count", e.target.value)} /></Field>
              <Field label="URLs (total)"><Input type="number" min={0} disabled={form.deriveFromTweets && tweetCount > 0} value={form.url_count} onChange={(e) => set("url_count", e.target.value)} /></Field>
              <Field label="Retweets received"><Input type="number" min={0} value={form.retweet_count} onChange={(e) => set("retweet_count", e.target.value)} /></Field>
              <Field label="Replies received"><Input type="number" min={0} value={form.reply_count} onChange={(e) => set("reply_count", e.target.value)} /></Field>
              <Field label="Likes received"><Input type="number" min={0} value={form.favorite_count_received} onChange={(e) => set("favorite_count_received", e.target.value)} /></Field>
            </div>
          </Card>

          <Card title="Run analysis" subtitle="Prediction, SHAP and LIME are computed by the backend for this exact input">
            <div className="grid gap-3 sm:grid-cols-[1fr_auto] sm:items-end">
              <Field label="Model">
                <Select value={modelId} onChange={(e) => setModelId(e.target.value)}>
                  <option value="">Active model{models.data?.models.find((m) => m.is_active) ? ` (${models.data.models.find((m) => m.is_active)?.name})` : ""}</option>
                  {models.data?.models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} · {m.dataset_name}{m.is_demo ? " (demo)" : ""}
                    </option>
                  ))}
                </Select>
              </Field>
              <Button type="submit" icon={Play} loading={predict.loading} disabled={noModel}>
                Analyze account
              </Button>
            </div>
            {predict.error && <div className="mt-3"><ErrorState title="Prediction failed" message={predict.error} /></div>}
          </Card>

          <Card
            title="Fetch a live account"
            subtitle="Social-network adapters pull a profile and recent tweets straight into the form above."
            actions={
              selectedAdapter ? (
                <Badge tone={selectedAdapter.is_sample ? "demo" : selectedAdapter.configured ? "good" : "neutral"} dot={selectedAdapter.configured ? "var(--status-good)" : "var(--text-3)"}>
                  {selectedAdapter.name} - {selectedAdapter.configured ? "configured" : "not configured"}
                </Badge>
              ) : null
            }
          >
            <div className="grid gap-3 sm:grid-cols-[180px_1fr_auto] sm:items-end">
              <Field label="Adapter">
                <Select value={adapterName} onChange={(e) => { setAdapterName(e.target.value); setAdapterId(""); }}>
                  {adapters.data?.map((a) => (
                    <option key={a.name} value={a.name}>{a.name === "x_api" ? "X (Twitter) API v2" : "Sample accounts (DEMO)"}</option>
                  ))}
                </Select>
              </Field>
              <Field label={adapterName === "x_api" ? "X username" : "Sample id"}>
                <Input value={adapterId} onChange={(e) => setAdapterId(e.target.value)} placeholder={adapterName === "x_api" ? "@username" : "demo-human-1 | demo-spambot-1 | demo-fake-follower-1"} onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); void onAdapterFetch(); } }} />
              </Field>
              <Button type="button" variant="secondary" icon={Plug} loading={fetchAdapter.loading} disabled={!adapterId.trim() || (selectedAdapter ? !selectedAdapter.configured : true)} onClick={onAdapterFetch}>
                Fetch
              </Button>
            </div>
            {fetchAdapter.error && <p className="mt-2 text-xs text-status-critical">{fetchAdapter.error}</p>}
            <p className="mt-2 text-[11px] text-ink-3">
              {adapterName === "x_api"
                ? selectedAdapter?.configured
                  ? "Reads the public profile (/2/users/by/username) and up to 100 recent tweets (/2/users/:id/tweets) with the configured bearer token. Reading tweets needs the Basic tier or higher; fields that API v2 does not expose are reported."
                  : "Not configured: set BOTSHIELD_X_BEARER_TOKEN in .env (app-only bearer token from developer.x.com) and restart the backend. Reading tweets requires the Basic tier or higher."
                : "Sample accounts are hand-written demonstration data; nothing is fetched from X."}
            </p>
          </Card>
        </div>
      </form>

      <div id="result">
        {result ? (
          <PredictionResult result={result} />
        ) : (
          !predict.loading && (
            <EmptyState icon={FlaskConical} title="No analysis yet" description="Fill in the form or load a sample account, then click Analyze account. Sections 4–10 (prediction, confidence, risk, SHAP, LIME, key indicators, interpretation) appear here." />
          )
        )}
      </div>
    </div>
  );
}
