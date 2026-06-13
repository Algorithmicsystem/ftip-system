/* Universe Screen Panel — full scored-universe ranked view */

async function loadUniverseScreen() {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  body.innerHTML = '<div class="loading-skeleton skeleton-line full" style="height:20px;margin-bottom:4px;"></div>'.repeat(8);

  try {
    const rows = await API.get('/intelligence/universe/scores');
    renderUniverseScreen(rows);
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Universe data unavailable.</div>`;
  }
}

function renderUniverseScreen(rows) {
  const body = document.getElementById('opportunities-body');
  if (!body) return;

  if (!rows || rows.length === 0) {
    body.innerHTML = '<div class="text-muted text-sm">No universe data available.</div>';
    return;
  }

  const sigColor = s => s === 'BUY' ? 'var(--signal-buy)' : s === 'SELL' ? 'var(--signal-sell)' : 'var(--signal-hold)';

  const scored   = rows.filter(r => Number(r.dau) > 0);
  const unscored = rows.filter(r => !(Number(r.dau) > 0));
  const avgDau   = scored.length ? (scored.reduce((s, r) => s + (r.dau || 0), 0) / scored.length).toFixed(1) : '—';
  const unscoredList = unscored.map(r => r.symbol).join(', ') || 'none';
  const summaryBar = `
    <div style="font-size:10px;color:var(--text-muted);margin-bottom:6px;padding:4px 6px;background:var(--bg-elevated);border-radius:4px;display:flex;gap:12px;flex-wrap:wrap;">
      <span><strong style="color:var(--text-primary);">${scored.length}</strong> scored</span>
      ${unscored.length ? `<span><strong style="color:var(--accent-warning);">${unscored.length}</strong> unscored (${unscoredList})</span>` : ''}
      <span style="margin-left:auto;">Avg DAU: <strong style="color:var(--text-primary);">${avgDau}</strong></span>
    </div>`;

  body.innerHTML = summaryBar + `
    <div style="overflow-y:auto;max-height:360px;">
      <table style="width:100%;border-collapse:collapse;font-size:11px;">
        <thead>
          <tr style="border-bottom:1px solid var(--border-default);color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;">
            <th style="text-align:left;padding:4px 6px;font-weight:600;">Symbol</th>
            <th style="text-align:center;padding:4px 6px;font-weight:600;">Signal</th>
            <th style="text-align:right;padding:4px 6px;font-weight:600;">DAU</th>
            <th style="text-align:left;padding:4px 6px;font-weight:600;min-width:100px;">Bar</th>
            <th style="text-align:right;padding:4px 6px;font-weight:600;">EIS</th>
            <th style="text-align:right;padding:4px 6px;font-weight:600;">CAPS</th>
            <th style="text-align:left;padding:4px 6px;font-weight:600;">Regime</th>
            <th style="text-align:right;padding:4px 6px;font-weight:600;">Updated</th>
          </tr>
        </thead>
        <tbody>
          ${rows.map(r => {
            const dau = r.dau ?? 0;
            const pct = Math.min(100, Math.max(0, dau));
            const sig = r.signal || 'NO_DATA';
            const barCls = pct >= 65 ? 'high' : pct >= 40 ? 'mid' : 'low';
            const dateStr = r.as_of_date ? r.as_of_date.slice(5) : '—';
            return `
              <tr style="border-bottom:1px solid rgba(255,255,255,0.04);cursor:pointer;"
                  onclick="loadSymbolIntelligence('${r.symbol}')"
                  onmouseover="this.style.background='var(--bg-elevated)'"
                  onmouseout="this.style.background=''">
                <td style="padding:5px 6px;font-family:var(--font-mono);font-weight:700;color:var(--text-primary);">${r.symbol}</td>
                <td style="padding:5px 6px;text-align:center;">
                  <span style="font-size:10px;font-weight:700;color:${sigColor(sig)};">${sig}</span>
                </td>
                <td style="padding:5px 6px;text-align:right;font-family:var(--font-mono);color:var(--text-secondary);">
                  ${dau > 0 ? dau.toFixed(1) : '—'}
                </td>
                <td style="padding:5px 6px;">
                  <div class="dau-bar__track" style="height:5px;">
                    <div class="dau-bar__fill ${barCls}" style="width:${pct.toFixed(0)}%;height:5px;"></div>
                  </div>
                </td>
                <td style="padding:5px 6px;text-align:right;font-family:var(--font-mono);color:var(--text-muted);font-size:10px;">${r.eis_score != null ? r.eis_score.toFixed(0) : '50'}</td>
                <td style="padding:5px 6px;text-align:right;font-family:var(--font-mono);color:var(--text-muted);font-size:10px;">${r.caps_score != null ? r.caps_score.toFixed(0) : '50'}</td>
                <td style="padding:5px 6px;color:var(--text-muted);font-size:10px;">${r.regime_label || '—'}</td>
                <td style="padding:5px 6px;text-align:right;color:var(--text-muted);font-size:10px;">${dateStr}</td>
              </tr>`;
          }).join('')}
        </tbody>
      </table>
    </div>`;
}
