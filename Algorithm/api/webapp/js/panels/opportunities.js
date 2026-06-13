/* Universe Screen / Opportunities Panel */

let _OPP_ALL_ROWS = [];
let _OPP_SHOW_ALL = false;
const OPP_DEFAULT_LIMIT = 50;

async function loadOpportunities() {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  _OPP_SHOW_ALL = false;
  body.innerHTML = Array(10).fill(
    '<div class="loading-skeleton skeleton-line full" style="margin-bottom:6px;height:28px;border-radius:4px;"></div>'
  ).join('');

  try {
    const rows = await API.get('/intelligence/universe/scores').catch(() => null);
    console.log('[AXIOM] universe scores:', rows?.length, 'symbols');
    if (rows && rows.length > 0) {
      const withData = rows.filter(r => r.dau !== null && r.dau !== undefined && !isNaN(Number(r.dau)));
      console.log('[AXIOM] symbols with DAU data:', withData.length);
      if (withData.length === 0) {
        body.innerHTML = `<div class="alert-banner warning" style="font-size:12px;">
          Pipeline running — scores will appear in ~15 minutes.
        </div>`;
        setTimeout(() => loadOpportunities(), 60000);
        return;
      }
      // Sort by DAU descending: BUY signals first, SELL last
      const sorted = [...withData].sort((a, b) => (b.dau || 0) - (a.dau || 0));
      const noData = rows.filter(r => r.dau === null || r.dau === undefined || isNaN(Number(r.dau)));
      _OPP_ALL_ROWS = [...sorted, ...noData];
      renderOpportunitiesList(_OPP_ALL_ROWS, '');
    } else {
      body.innerHTML = '<div class="text-muted text-sm">No scored symbols — run the pipeline to generate scores.</div>';
    }
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not load opportunities.</div>`;
  }
}

function renderOpportunitiesList(allRows, filterText) {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  if (!allRows || allRows.length === 0) {
    body.innerHTML = '<div class="text-muted text-sm">No opportunities available.</div>';
    return;
  }

  const withData = allRows.filter(r => r.dau !== null && r.dau !== undefined && !isNaN(Number(r.dau)));
  const nBuy  = withData.filter(r => r.signal === 'BUY').length;
  const nHold = withData.filter(r => r.signal === 'HOLD').length;
  const nSell = withData.filter(r => r.signal === 'SELL').length;
  const avgDau = withData.length
    ? (withData.reduce((s, x) => s + (x.dau || 0), 0) / withData.length).toFixed(1)
    : '—';

  const q = (filterText || '').trim().toUpperCase();
  const filtered = q ? allRows.filter(r => r.symbol && r.symbol.includes(q)) : allRows;
  const display  = _OPP_SHOW_ALL ? filtered : filtered.slice(0, OPP_DEFAULT_LIMIT);
  const hasMore  = !_OPP_SHOW_ALL && filtered.length > OPP_DEFAULT_LIMIT;

  const summaryHTML = `
    <div style="font-size:10px;color:var(--text-muted);margin-bottom:6px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
      <strong style="color:var(--text-primary);">${withData.length} signals</strong>
      <span style="color:var(--signal-buy);font-weight:600;">${nBuy} BUY</span>
      <span style="color:var(--signal-hold,#f59e0b);font-weight:600;">${nHold} HOLD</span>
      <span style="color:var(--signal-sell);font-weight:600;">${nSell} SELL</span>
      <span style="margin-left:auto;">Avg DAU: <strong>${avgDau}</strong></span>
    </div>
    <input id="opp-search" type="text" placeholder="Filter by symbol…" value="${q}"
      style="width:100%;box-sizing:border-box;padding:5px 8px;margin-bottom:6px;font-size:11px;
             background:var(--bg-elevated);border:1px solid var(--border-default);border-radius:4px;
             color:var(--text-primary);outline:none;"
      oninput="renderOpportunitiesList(_OPP_ALL_ROWS,this.value)" />`;

  const rowsHTML = display.map(item => {
    const dau = item.dau ?? 0;
    const sig = item.signal || 'HOLD';
    const sigLower = sig.toLowerCase().replace(/[_ ]/g, '-');
    const pct = Math.min(100, Math.max(0, dau));
    const barCls = pct >= 65 ? 'high' : pct >= 40 ? 'mid' : 'low';
    const borderColor = sig === 'BUY'
      ? 'var(--signal-buy)'
      : sig === 'SELL' ? 'var(--signal-sell)' : 'var(--signal-hold, #f59e0b)';
    return `
      <div class="opp-row" onclick="loadSymbolIntelligence('${item.symbol}')"
           style="border-left:3px solid ${borderColor};padding-left:8px;cursor:pointer;">
        <span class="opp-row__symbol">${item.symbol}</span>
        ${item.regime_label
          ? `<span style="font-size:9px;color:var(--text-muted);margin-left:2px;">${item.regime_label.replace(/_/g,' ')}</span>`
          : ''}
        <span class="signal-badge ${sigLower}" style="font-size:10px;padding:1px 7px;">${(sig || '—').replace('_',' ')}</span>
        <div class="opp-row__bar">
          <div class="dau-bar__track">
            <div class="dau-bar__fill ${barCls}" style="width:${pct.toFixed(0)}%;"></div>
          </div>
        </div>
        <span class="opp-row__dau">${dau > 0 ? dau.toFixed(0) : '—'}</span>
      </div>`;
  }).join('');

  const showAllHTML = hasMore ? `
    <div style="text-align:center;margin-top:8px;">
      <button
        onclick="_OPP_SHOW_ALL=true;renderOpportunitiesList(_OPP_ALL_ROWS,document.getElementById('opp-search')?.value||'')"
        style="font-size:11px;padding:4px 16px;background:var(--bg-elevated);
               border:1px solid var(--border-default);border-radius:4px;
               color:var(--text-secondary);cursor:pointer;">
        Show all ${filtered.length} symbols
      </button>
    </div>` : '';

  body.innerHTML = summaryHTML + rowsHTML + showAllHTML;
}
