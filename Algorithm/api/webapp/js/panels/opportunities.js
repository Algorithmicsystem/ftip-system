/* Universe Screen / Opportunities Panel */

async function loadOpportunities() {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  body.innerHTML = Array(10).fill(
    '<div class="loading-skeleton skeleton-line full" style="margin-bottom:6px;height:28px;border-radius:4px;"></div>'
  ).join('');

  try {
    const rows = await API.get('/intelligence/universe/scores').catch(() => null);
    console.log('[AXIOM] universe scores:', rows?.length, 'symbols');
    if (rows && rows.length > 0) {
      const withData = rows.filter(r => r.dau !== null);
      console.log('[AXIOM] symbols with DAU data:', withData.length);
      const noData   = rows.filter(r => r.dau === null);
      if (withData.length === 0) {
        body.innerHTML = `<div class="alert-banner warning" style="font-size:12px;">
          Pipeline running — scores will appear in ~15 minutes.
        </div>`;
        return;
      }
      // Sort: BUY first, then HOLD, then SELL — within each group by DAU desc
      const sigOrder = { BUY: 0, HOLD: 1, SELL: 2, NO_DATA: 3 };
      const sorted = [...withData].sort((a, b) => {
        const so = (sigOrder[a.signal] ?? 3) - (sigOrder[b.signal] ?? 3);
        return so !== 0 ? so : (b.dau || 0) - (a.dau || 0);
      });
      const display = [...sorted, ...noData].slice(0, 10);
      renderOpportunitiesList(display, rows);
    } else {
      const DEFAULT_UNIVERSE = ['AAPL','MSFT','NVDA','GOOG','AMZN','META','TSLA','JPM','V','UNH'];
      renderOpportunitiesList(DEFAULT_UNIVERSE.map(s => ({ symbol: s, dau: null, signal: null })), []);
    }
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not load opportunities.</div>`;
  }
}

function renderOpportunitiesList(symbols, allRows) {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  if (!symbols || symbols.length === 0) {
    body.innerHTML = '<div class="text-muted text-sm">No opportunities available.</div>';
    return;
  }

  // Build summary bar from full dataset
  const withData = (allRows || symbols).filter(r => r.dau !== null);
  let summaryHTML = '';
  if (withData.length > 0) {
    const nBuy  = withData.filter(r => r.signal === 'BUY').length;
    const nHold = withData.filter(r => r.signal === 'HOLD').length;
    const nSell = withData.filter(r => r.signal === 'SELL').length;
    const avgDau = (withData.reduce((s, x) => s + (x.dau || 0), 0) / withData.length).toFixed(1);
    summaryHTML = `
      <div style="font-size:10px;color:var(--text-muted);margin-bottom:8px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;">
        <span>${withData.length} symbols</span>
        <span style="color:var(--signal-buy);font-weight:600;">${nBuy} BUY</span>
        <span style="color:var(--signal-hold, #f59e0b);font-weight:600;">${nHold} HOLD</span>
        <span style="color:var(--signal-sell);font-weight:600;">${nSell} SELL</span>
        <span style="margin-left:auto;">Avg DAU: <strong>${avgDau}</strong></span>
      </div>`;
  }

  const rowsHTML = symbols.map(item => {
    const dau = item.dau ?? 0;
    const sig = item.signal || 'HOLD';
    const sigLower = sig.toLowerCase().replace(/[_ ]/g, '-');
    const pct = Math.min(100, Math.max(0, dau));
    const barCls = pct >= 65 ? 'high' : pct >= 40 ? 'mid' : 'low';
    const borderColor = sig === 'BUY' ? 'var(--signal-buy)' : sig === 'SELL' ? 'var(--signal-sell)' : 'var(--signal-hold, #f59e0b)';
    return `
      <div class="opp-row" onclick="loadSymbolIntelligence('${item.symbol}')" style="border-left:3px solid ${borderColor};padding-left:8px;cursor:pointer;">
        <span class="opp-row__symbol">${item.symbol}</span>
        ${item.regime_label ? `<span style="font-size:9px;color:var(--text-muted);margin-left:2px;">${item.regime_label.replace(/_/g,' ')}</span>` : ''}
        <span class="signal-badge ${sigLower}" style="font-size:10px;padding:1px 7px;">${(sig || '—').replace('_', ' ')}</span>
        <div class="opp-row__bar">
          <div class="dau-bar__track">
            <div class="dau-bar__fill ${barCls}" style="width:${pct.toFixed(0)}%;"></div>
          </div>
        </div>
        <span class="opp-row__dau">${dau > 0 ? dau.toFixed(0) : '—'}</span>
      </div>`;
  }).join('');

  body.innerHTML = summaryHTML + rowsHTML;
}
