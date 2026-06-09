/* Symbol Intelligence Deep-Dive Panel */

let _currentSymbol = null;

async function loadSymbolIntelligence(symbol) {
  if (!symbol) symbol = document.getElementById('symbol-search')?.value?.trim()?.toUpperCase();
  if (!symbol) return;

  _currentSymbol = symbol;

  // Update header
  const badgeEl = document.getElementById('symbol-signal-badge');
  const ratingEl = document.getElementById('symbol-analyst-rating');
  const body = document.getElementById('symbol-body');

  if (badgeEl) { badgeEl.className = 'signal-badge hold'; badgeEl.textContent = 'Loading…'; }
  if (ratingEl) ratingEl.textContent = symbol;

  // Show skeleton in active tab
  const activePane = document.querySelector('.tab-pane.active');
  if (activePane) {
    activePane.innerHTML = `
      <div class="loading-skeleton skeleton-line full" style="margin-bottom:8px;"></div>
      <div class="loading-skeleton skeleton-line medium" style="margin-bottom:8px;"></div>
      <div class="loading-skeleton skeleton-line short"></div>`;
  }

  try {
    const [intel, explain] = await Promise.all([
      API.get(`/intelligence/universal/${symbol}`).catch(() => null),
      API.get(`/explain/signal/${symbol}`).catch(() => null),
    ]);

    renderIntelligenceTab(intel, symbol);
    renderRiskTab(intel, symbol);
    renderExplanationTab(explain, intel);

    // Update header
    if (intel && badgeEl) {
      const sig = (intel.signal_label || 'HOLD').toLowerCase().replace('_', '-').replace(' ', '-');
      badgeEl.className = `signal-badge ${sig}`;
      badgeEl.textContent = intel.signal_label || 'HOLD';
    }
    if (intel && ratingEl) {
      ratingEl.textContent = `${symbol} · ${intel.analyst_rating || '—'} · DAU ${(intel.dau || 0).toFixed(1)}`;
    }
  } catch (err) {
    if (activePane) activePane.innerHTML = `<div class="alert-banner warning">Could not load intelligence for ${symbol}: ${err.message}</div>`;
    if (badgeEl) { badgeEl.className = 'signal-badge hold'; badgeEl.textContent = 'HOLD'; }
  }
}

function renderIntelligenceTab(data, symbol) {
  const el = document.getElementById('tab-intelligence');
  if (!el) return;

  if (!data) {
    el.innerHTML = `<div class="alert-banner info">No AXIOM intelligence available for ${symbol}.</div>`;
    return;
  }

  const isDefault = data.ic_state === 'INSUFFICIENT' || (data.intelligence_quality_score === 0) || (data.dau === 50 && data.eis_score === 50 && data.caps_score === 50);
  const defaultBanner = isDefault ? `<div class="alert-banner info" style="margin-bottom:10px;">Pipeline data not yet available — scores reflect system defaults.</div>` : '';

  const scores = `
    ${scoreBarHTML('EIS Score', data.eis_score, '#10b981')}
    ${scoreBarHTML('CAPS Score', data.caps_score, '#3b82f6')}
    ${scoreBarHTML('Factor Composite', data.factor_composite_score, '#8b5cf6')}
    ${data.osms_score != null ? scoreBarHTML('OSMS (Alt Data)', data.osms_score, '#f59e0b') : ''}
    ${data.ias_score != null ? scoreBarHTML('IAS (Insider)', data.ias_score, '#f59e0b') : ''}`;

  const evidenceItems = data.top_supporting_evidence || data.key_reasons || [];
  const evidenceHTML = evidenceItems.length > 0
    ? evidenceItems.map(r => {
        const text = typeof r === 'string' ? r
          : r.factor ? `${r.factor.replace(/_/g, ' ')}: ${r.contribution > 0 ? '+' : ''}${r.contribution}`
          : JSON.stringify(r);
        return `
          <div class="evidence-item supporting">
            <span class="evidence-item__icon">✓</span>
            <span class="evidence-item__text">${text}</span>
          </div>`;
      }).join('')
    : `<div class="text-muted text-sm">Evidence builds with each pipeline run.</div>`;

  el.innerHTML = `
    ${defaultBanner}
    <div id="scores-section" style="margin-bottom:14px;">${scores}</div>
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Evidence</div>
    <div>${evidenceHTML}</div>
    <div style="margin-top:12px;padding:8px 12px;background:var(--bg-tertiary);border-radius:6px;font-size:12px;">
      <span class="text-muted" style="font-size:10px;text-transform:uppercase;letter-spacing:.06em;">Primary Driver · </span>
      <span style="color:var(--text-primary);">${(data.primary_driver && data.primary_driver.toLowerCase() !== 'unknown') ? data.primary_driver : 'Factor Composite'}</span>
    </div>`;
}

function renderRiskTab(data, symbol) {
  const el = document.getElementById('tab-risk-tab');
  if (!el) return;

  if (!data) {
    el.innerHTML = `<div class="alert-banner info">No risk data for ${symbol}.</div>`;
    return;
  }

  const frag = data.fragility_score ?? 0;
  const scps = data.scps_score ?? 0;
  const bfs  = data.bfs_score  ?? 0;

  const caAdj = data.cross_asset_adjusted_dau;
  const rawDau = data.dau ?? 0;
  const adjDiff = caAdj != null ? (caAdj - rawDau) : null;
  const adjSign = adjDiff != null && adjDiff > 0 ? '+' : '';

  const conditions = (data.invalidation_conditions && data.invalidation_conditions.length > 0)
    ? data.invalidation_conditions
    : [
        frag >= 60 ? `Fragility already elevated (${frag.toFixed(0)}) — spike above 75 triggers stop` : 'Fragility spike above 70',
        data.top_risk || 'Regime shift to HIGH_VOL',
        'IC gate degradation to WEAK',
      ];

  el.innerHTML = `
    <div style="margin-bottom:14px;">
      ${scoreBarHTML('Fragility', frag, frag >= 70 ? '#ef4444' : frag >= 50 ? '#f59e0b' : '#10b981')}
      ${scoreBarHTML('SCPS', scps, scps >= 65 ? '#ef4444' : '#3b82f6')}
      ${scoreBarHTML('BFS', bfs, '#8b5cf6')}
    </div>
    ${data.var_1d_99 != null ? `
    <div class="metric-card" style="flex-direction:row;align-items:center;justify-content:space-between;margin-bottom:10px;">
      <span class="metric-card__label">VaR 1d 99%</span>
      <span class="metric-card__value" style="font-size:16px;color:var(--signal-sell);">${(data.var_1d_99 * 100).toFixed(2)}%</span>
    </div>` : ''}
    ${caAdj != null ? `
    <div class="metric-card" style="flex-direction:row;align-items:center;justify-content:space-between;margin-bottom:10px;">
      <span class="metric-card__label">Cross-Asset Adj DAU</span>
      <span style="font-size:13px;font-family:var(--font-mono);">
        ${caAdj.toFixed(1)}
        <span style="font-size:11px;color:${adjDiff >= 0 ? 'var(--signal-buy)' : 'var(--signal-sell)'};">(${adjSign}${adjDiff.toFixed(1)})</span>
      </span>
    </div>` : ''}
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px;">Invalidation Conditions</div>
    <div style="font-size:12px;color:var(--text-secondary);line-height:1.7;">
      ${conditions.map(c => `<div>• ${c}</div>`).join('')}
    </div>`;
}

function renderExplanationTab(explain, intel) {
  const el = document.getElementById('tab-explanation');
  if (!el) return;

  if (!explain && !intel) {
    el.innerHTML = `<div class="alert-banner info">No explanation data available.</div>`;
    return;
  }

  const explanationText = explain?.explanation_text || '';
  const chain = explain?.reasoning_steps || [];
  const chainHTML = chain.length > 0
    ? chain.map((step, i) => `
        <div class="reasoning-step">
          <span class="reasoning-step__num">${i + 1}</span>
          <div class="reasoning-step__content">
            <div class="reasoning-step__premise">${step.claim || step.premise || step.factor || '—'}</div>
            <div class="reasoning-step__conclusion">${step.evidence || step.conclusion || step.reasoning || '—'}</div>
          </div>
        </div>`).join('')
    : `<div class="text-muted text-sm">No reasoning chain available.</div>`;

  const counter = explain?.contradicting_factors || explain?.counterfactuals || [];
  const counterHTML = counter.length > 0
    ? counter.map(c => {
        const text = typeof c === 'string' ? c : c.description || c.scenario || JSON.stringify(c);
        return `
          <div class="evidence-item contradicting">
            <span class="evidence-item__icon">↺</span>
            <span class="evidence-item__text">${text}</span>
          </div>`;
      }).join('')
    : `<div class="text-muted text-sm">No contradicting factors available.</div>`;

  const batting = intel?.signal_batting_average;
  const invalidConds = explain?.invalidation_conditions || [];

  const aiSynthesis = explain?.ai_synthesis || null;
  el.innerHTML = `
    ${aiSynthesis ? `
    <div style="background:linear-gradient(135deg,rgba(59,130,246,0.1),rgba(139,92,246,0.1));
                border:1px solid rgba(59,130,246,0.3);border-radius:8px;
                padding:12px;margin-bottom:14px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;
                  color:var(--accent-primary);margin-bottom:6px;">
        ✦ AI SYNTHESIS (GPT-4o-mini)
      </div>
      <div style="font-size:13px;color:var(--text-primary);line-height:1.7;">
        ${aiSynthesis}
      </div>
    </div>` : ''}
    ${explanationText ? `
    <div style="font-size:12px;color:var(--text-secondary);line-height:1.7;margin-bottom:14px;padding:10px;background:var(--bg-tertiary);border-radius:6px;">
      ${explanationText.split('\n\n').map(p => `<p style="margin-bottom:4px;">${p}</p>`).join('')}
    </div>` : ''}
    ${batting != null ? `
    <div style="margin-bottom:14px;">
      <div class="metric-card" style="text-align:center;">
        <span class="metric-card__label">Signal Batting Average</span>
        <span class="metric-card__value" style="font-size:18px;color:var(--signal-buy);">${(batting * 100).toFixed(0)}%</span>
      </div>
    </div>` : ''}
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">Reasoning Chain</div>
    ${chainHTML}
    ${invalidConds.length > 0 ? `
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-top:14px;margin-bottom:6px;">Invalidation Conditions</div>
    <div style="font-size:12px;color:var(--text-secondary);line-height:1.7;">
      ${invalidConds.map(c => `<div>• ${c}</div>`).join('')}
    </div>` : ''}
    <div style="font-size:11px;font-weight:600;color:var(--text-muted);text-transform:uppercase;letter-spacing:.08em;margin-top:14px;margin-bottom:8px;">What Would Flip This Signal?</div>
    ${counterHTML}`;
}

async function loadPeersTab(symbol) {
  const el = document.getElementById('peers-content');
  if (!el || !symbol) return;
  el.innerHTML = '<div class="loading-skeleton skeleton-line full" style="height:40px;"></div>';
  try {
    const data = await API.get(`/competitive/${symbol}`).catch(() => null);
    if (!data || !data.competitors || data.competitors.length === 0) {
      el.innerHTML = `<div class="alert-banner info">No peer data for ${symbol} in sector ${data?.sector || '—'}.</div>`;
      return;
    }
    const rows = data.competitors.map(c => {
      const advCls = c.dau_advantage > 0 ? 'color:var(--signal-buy)' : 'color:var(--signal-sell)';
      return `<tr>
        <td style="font-weight:600;cursor:pointer;" onclick="loadSymbolIntelligence('${c.competitor_symbol}')">${c.competitor_symbol}</td>
        <td style="${advCls}">${c.dau_advantage > 0 ? '+' : ''}${(c.dau_advantage || 0).toFixed(1)}</td>
        <td>${(c.competitive_position_score || 0).toFixed(0)}</td>
        <td style="font-size:11px;color:var(--text-muted)">${c.key_advantage || '—'}</td>
      </tr>`;
    }).join('');
    el.innerHTML = `
      <div style="font-size:11px;color:var(--text-muted);margin-bottom:8px;">
        Sector: <strong>${data.sector}</strong> &nbsp;·&nbsp;
        Rank: <strong>${data.sector_dau_rank}/${data.sector_size}</strong> &nbsp;·&nbsp;
        Position: <strong>${data.competitive_position}</strong>
      </div>
      <table style="width:100%;font-size:12px;border-collapse:collapse;">
        <thead><tr style="color:var(--text-muted);font-size:10px;text-transform:uppercase;">
          <th style="text-align:left;padding:4px 0;">Peer</th>
          <th style="text-align:left;padding:4px 0;">DAU Adv</th>
          <th style="text-align:left;padding:4px 0;">Score</th>
          <th style="text-align:left;padding:4px 0;">Key Edge</th>
        </tr></thead>
        <tbody>${rows}</tbody>
      </table>`;
  } catch (err) {
    el.innerHTML = `<div class="alert-banner warning">Peer data unavailable: ${err.message}</div>`;
  }
}

function handleSymbolSearch() {
  const val = document.getElementById('symbol-search')?.value?.trim()?.toUpperCase();
  if (val) loadSymbolIntelligence(val);
}

function handleGlobalSymbolSearch() {
  const val = document.getElementById('global-symbol-input')?.value?.trim()?.toUpperCase();
  if (val) {
    // Sync to symbol search box
    const symInput = document.getElementById('symbol-search');
    if (symInput) symInput.value = val;
    switchPanel('symbol');
    loadSymbolIntelligence(val);
  }
}
