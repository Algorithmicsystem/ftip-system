/* Universe Screen / Opportunities Panel */

const DEFAULT_UNIVERSE = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSLA','JPM','V','UNH'];

async function loadOpportunities() {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  body.innerHTML = DEFAULT_UNIVERSE.map(() =>
    '<div class="loading-skeleton skeleton-line full" style="margin-bottom:6px;height:28px;border-radius:4px;"></div>'
  ).join('');

  try {
    const rows = [];
    // Try to pull top opportunities from morning briefing cache
    const briefing = await API.get('/jobs/briefing/morning').catch(() => null);
    const tops = briefing?.top_opportunities || [];

    if (tops.length > 0) {
      renderOpportunitiesList(tops.map(t => ({
        symbol: t.symbol,
        dau: t.dau,
        signal_label: t.regime ? (t.dau > 65 ? 'BUY' : t.dau > 40 ? 'HOLD' : 'SELL') : 'HOLD',
        regime: t.regime || 'unknown',
      })));
    } else {
      renderOpportunitiesList(DEFAULT_UNIVERSE.map(s => ({ symbol: s, dau: null, signal_label: null, regime: null })));
    }
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not load opportunities.</div>`;
  }
}

function renderOpportunitiesList(symbols) {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  if (!symbols || symbols.length === 0) {
    body.innerHTML = '<div class="text-muted text-sm">No opportunities available.</div>';
    return;
  }

  body.innerHTML = symbols.map(item => {
    const dau = item.dau ?? 0;
    const sig = (item.signal_label || 'HOLD').toLowerCase().replace(' ', '-');
    const pct = Math.min(100, Math.max(0, dau));
    const barCls = pct >= 65 ? 'high' : pct >= 40 ? 'mid' : 'low';
    return `
      <div class="opp-row" onclick="loadSymbolIntelligence('${item.symbol}')">
        <span class="opp-row__symbol">${item.symbol}</span>
        <span class="signal-badge ${sig}" style="font-size:10px;padding:1px 7px;">${(item.signal_label || '—').toUpperCase()}</span>
        <div class="opp-row__bar">
          <div class="dau-bar__track">
            <div class="dau-bar__fill ${barCls}" style="width:${pct.toFixed(0)}%;"></div>
          </div>
        </div>
        <span class="opp-row__dau">${dau > 0 ? dau.toFixed(0) : '—'}</span>
      </div>`;
  }).join('');
}
