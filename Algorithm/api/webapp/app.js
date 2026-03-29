const qs = (sel) => document.querySelector(sel);
const qsa = (sel) => Array.from(document.querySelectorAll(sel));

const state = {
  apiKey: "",
};

const isoDate = (date) => date.toISOString().slice(0, 10);

const setDefaults = () => {
  const now = new Date();
  const toDate = isoDate(now);
  const from = new Date(now);
  from.setDate(from.getDate() - 365);
  qs("#to-date-input").value = toDate;
  qs("#as-of-date-input").value = toDate;
  qs("#from-date-input").value = isoDate(from);
};

const setStatus = (text, isError = false) => {
  const el = qs("#status-line");
  el.textContent = text;
  el.classList.toggle("error", !!isError);
};

const setActiveTab = (tabId) => {
  qsa(".tab").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabId));
  qsa(".tab-panel").forEach((panel) => panel.classList.toggle("hidden", panel.id !== tabId));
};

const getHeaders = () => {
  const headers = { "Content-Type": "application/json" };
  if (state.apiKey) headers["X-API-Key"] = state.apiKey;
  return headers;
};

const getInputs = () => ({
  symbol: qs("#symbol-input").value.trim().toUpperCase(),
  lookback: Number(qs("#lookback-input").value || 252),
  from_date: qs("#from-date-input").value,
  to_date: qs("#to-date-input").value,
  as_of_date: qs("#as-of-date-input").value,
});

const parseJsonSafe = async (resp) => {
  try {
    return await resp.json();
  } catch {
    return null;
  }
};

const callJson = async (url, options = {}) => {
  const resp = await fetch(url, options);
  const data = await parseJsonSafe(resp);
  if (!resp.ok) {
    const msg = data?.detail || data?.error?.message || `Request failed: ${resp.status}`;
    throw new Error(msg);
  }
  return data;
};

const renderSignal = (signal) => {
  qs("#signal-symbol").textContent = signal?.symbol || "N/A";
  qs("#signal-asof").textContent = signal?.as_of || "N/A";
  qs("#signal-direction").textContent = signal?.signal || "N/A";
  qs("#signal-score").textContent = signal?.score ?? "N/A";
  qs("#signal-confidence").textContent = signal?.confidence ?? "N/A";
  qs("#signal-regime").textContent = signal?.regime || "N/A";
  qs("#signal-json").textContent = JSON.stringify(signal || {}, null, 2);
};

const renderFeatures = (features) => {
  qs("#features-symbol").textContent = features?.symbol || "N/A";
  qs("#features-asof").textContent = features?.as_of || "N/A";
  qs("#features-lookback").textContent = features?.lookback ?? "N/A";
  qs("#features-json").textContent = JSON.stringify(features || {}, null, 2);
};

const refreshOfficialLatest = async () => {
  const { symbol, lookback } = getInputs();
  if (!symbol) {
    setStatus("Symbol is required for latest signal/features.", true);
    return;
  }
  setStatus(`Refreshing latest official v1 rows for ${symbol}...`);
  try {
    const [signal, features] = await Promise.all([
      callJson(`/prosperity/latest/signal?symbol=${encodeURIComponent(symbol)}&lookback=${lookback}`),
      callJson(`/prosperity/latest/features?symbol=${encodeURIComponent(symbol)}&lookback=${lookback}`),
    ]);
    renderSignal(signal);
    renderFeatures(features);
    setStatus(`Loaded latest signal/features for ${symbol}.`);
  } catch (err) {
    setStatus(`Refresh failed: ${err.message}`, true);
  }
};

qs("#bootstrap-btn").addEventListener("click", async () => {
  setStatus("Running POST /prosperity/bootstrap...");
  try {
    await callJson("/prosperity/bootstrap", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({}),
    });
    setStatus("Bootstrap complete.");
  } catch (err) {
    setStatus(`Bootstrap failed: ${err.message}`, true);
  }
});

qs("#snapshot-btn").addEventListener("click", async () => {
  const { symbol, lookback, from_date, to_date, as_of_date } = getInputs();
  if (!symbol || !from_date || !to_date || !as_of_date) {
    setStatus("symbol/from_date/to_date/as_of_date are required.", true);
    return;
  }

  setStatus(`Running POST /prosperity/snapshot/run for ${symbol}...`);
  try {
    await callJson("/prosperity/snapshot/run", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({
        symbols: [symbol],
        from_date,
        to_date,
        as_of_date,
        lookback,
      }),
    });
    setStatus(`Snapshot run complete for ${symbol}. Refreshing latest signal/features...`);
    await refreshOfficialLatest();
  } catch (err) {
    setStatus(`Snapshot failed: ${err.message}`, true);
  }
});

qs("#refresh-btn").addEventListener("click", refreshOfficialLatest);

qs("#daily-snapshot-btn").addEventListener("click", async () => {
  setStatus("Running optional POST /jobs/prosperity/daily-snapshot...");
  try {
    await callJson("/jobs/prosperity/daily-snapshot", {
      method: "POST",
      headers: getHeaders(),
      body: JSON.stringify({}),
    });
    setStatus("Daily snapshot job request accepted.");
  } catch (err) {
    setStatus(`Daily snapshot job failed: ${err.message}`, true);
  }
});

qs("#api-key-input").addEventListener("input", (e) => {
  state.apiKey = e.target.value.trim();
});

qsa(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setActiveTab(tab.dataset.tab));
});

setDefaults();
setActiveTab("signal");
