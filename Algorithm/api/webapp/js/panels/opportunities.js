/* Universe Screen / Opportunities Panel */

async function loadOpportunities() {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  body.innerHTML = Array(9).fill(
    '<div class="loading-skeleton skeleton-line full" style="margin-bottom:6px;height:28px;border-radius:4px;"></div>'
  ).join('');

  try {
    const rows = await API.get('/intelligence/universe/scores').catch(() => null);
    console.log('[AXIOM] universe scores:', rows?.length, 'symbols');
    if (rows && rows.length > 0) {
      const withData = rows.filter(r => r.dau !== null);
      const noData   = rows.filter(r => r.dau === null);
      console.log('[AXIOM] symbols with DAU data:', withData.length);
      if (withData.length === 0) {
        body.innerHTML = `<div class="alert-banner warning" style="font-size:12px;">
          Pipeline running — scores will appear in ~15 minutes.
        </div>`;
        return;
      }
      const display  = [...withData.slice(0, 9), ...noData].slice(0, 9);
      renderOpportunitiesList(display);
    } else {
      const DEFAULT_UNIVERSE = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSLA','JPM','V','UNH'];
      renderOpportunitiesList(DEFAULT_UNIVERSE.map(s => ({ symbol: s, dau: null, signal: null })));
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
    const sig = (item.signal || 'HOLD').toLowerCase().replace(/[_ ]/g, '-');
    const pct = Math.min(100, Math.max(0, dau));
    const barCls = pct >= 65 ? 'high' : pct >= 40 ? 'mid' : 'low';
    return `
      <div class="opp-row" onclick="loadSymbolIntelligence('${item.symbol}')">
        <span class="opp-row__symbol">${item.symbol}</span>
        <span class="signal-badge ${sig}" style="font-size:10px;padding:1px 7px;">${(item.signal || '—').replace('_', ' ')}</span>
        ${item.regime_label ? `<span style="font-size:9px;color:var(--text-muted);padding:1px 5px;border-radius:3px;background:var(--bg-elevated);margin-left:2px;">${item.regime_label}</span>` : ''}
        <div class="opp-row__bar">
          <div class="dau-bar__track">
            <div class="dau-bar__fill ${barCls}" style="width:${pct.toFixed(0)}%;"></div>
          </div>
        </div>
        <span class="opp-row__dau">${dau > 0 ? dau.toFixed(0) : '—'}</span>
      </div>`;
  }).join('');
}
