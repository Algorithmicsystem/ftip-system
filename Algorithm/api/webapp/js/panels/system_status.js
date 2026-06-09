/* System Status Panel — operator view of AXIOM health */

async function loadSystemStatus() {
  const body = document.getElementById('pipeline-body');
  const lastRunEl = document.getElementById('pipeline-last-run');
  if (!body) return;

  body.innerHTML = '<div class="loading-skeleton skeleton-line full" style="margin-bottom:6px;height:20px;"></div>'.repeat(5);

  try {
    const [status, pipeline] = await Promise.all([
      API.get('/system/status').catch(() => null),
      API.get('/orchestration/pipeline/status').catch(() => null),
    ]);

    renderSystemStatus(status, pipeline);

    if (lastRunEl && (status?.last_pipeline_run || pipeline?.started_at)) {
      const ts = status?.last_pipeline_run || pipeline?.started_at;
      lastRunEl.textContent = `Last run: ${new Date(ts).toLocaleString()}`;
    }

    if (status?.server === 'healthy') {
      updateHealthIndicator('healthy');
    }
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">System status unavailable.</div>`;
  }
}

function renderSystemStatus(status, pipeline) {
  const body = document.getElementById('pipeline-body');
  if (!body) return;

  if (!status) {
    body.innerHTML = `<div class="alert-banner warning">Could not load system status.</div>`;
    return;
  }

  const axiomCount  = status.axiom_scores_count ?? 0;
  const coverage    = axiomCount > 0 ? ((axiomCount / 30) * 100).toFixed(1) : '0.0';
  const dbOk        = status.db_connected ? '✓ Connected' : '✗ Offline';
  const dbColor     = status.db_connected ? 'var(--signal-buy)' : 'var(--signal-sell)';
  const warnings    = status.warnings || [];
  const sched       = status.scheduler_running ? '✓ Running' : '✗ Stopped';
  const schedColor  = status.scheduler_running ? 'var(--signal-buy)' : 'var(--signal-sell)';

  // Pipeline stage summary
  const stages     = pipeline?.stages || {};
  const stageNames = Object.keys(stages);
  const succeeded  = stageNames.filter(k => stages[k]?.status === 'success').length;
  const failed     = stageNames.filter(k => stages[k]?.status === 'failed').length;
  const overallSts = pipeline?.overall_status || 'unknown';
  const stsColor   = overallSts === 'success' ? 'var(--signal-buy)'
                   : overallSts === 'failed'  ? 'var(--signal-sell)' : 'var(--signal-hold)';

  const warningsHTML = warnings.length > 0
    ? warnings.map(w => `<div class="alert-banner warning" style="margin-bottom:4px;padding:4px 10px;font-size:11px;">${w}</div>`).join('')
    : `<div style="color:var(--signal-buy);font-size:11px;">✓ All systems operational</div>`;

  // Performance
  const perf       = status.performance || {};
  const p95        = perf.system_p95_ms ?? 0;
  const slaOk      = perf.meets_sla ?? (p95 < 200);
  const slaLabel   = slaOk ? '✓ PASSING' : '✗ BREACHED';
  const slaColor   = slaOk ? 'var(--signal-buy)' : 'var(--signal-sell)';
  const p95Color   = p95 < 200 ? 'var(--signal-buy)' : p95 < 500 ? 'var(--signal-hold)' : 'var(--signal-sell)';

  // Readiness
  const acq        = status.acquisition_readiness || {};
  const rdPassed   = acq.readiness_passed ?? 0;
  const rdTotal    = acq.readiness_total  ?? 20;
  const rdConf     = acq.deployment_confidence || 'unknown';
  const rdColor    = rdConf === 'high' ? 'var(--signal-buy)' : rdConf === 'medium' ? 'var(--signal-hold)' : 'var(--signal-sell)';
  const acqScore   = acq.score ?? 0;
  const acqTier    = (acq.tier || 'unknown').replace(/_/g,' ');
  const acqColor   = acqScore >= 80 ? 'var(--signal-buy)' : acqScore >= 60 ? 'var(--signal-hold)' : 'var(--signal-sell)';

  // Moat
  const moatSt     = status.moat_status || {};
  const moatScore  = moatSt.moat_score ?? 0;
  const moatColor  = moatScore >= 70 ? 'var(--signal-buy)' : moatScore >= 50 ? 'var(--signal-hold)' : 'var(--signal-sell)';

  body.innerHTML = `
    <!-- Section A: Data Health -->
    <div style="margin-bottom:12px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;">Data Health</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        ${metricChipHTML('AXIOM Scores', axiomCount)}
        ${metricChipHTML('Universe Coverage', `${coverage}%`)}
        ${metricChipHTML('DB', dbOk, dbColor)}
        ${metricChipHTML('Scheduler', sched, schedColor)}
        ${metricChipHTML('Migrations', status.migrations_applied ?? '—')}
        ${metricChipHTML('Version', status.version || '—')}
      </div>
    </div>

    <!-- Section B: Performance -->
    <div style="margin-bottom:12px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;">Performance</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        ${metricChipHTML('p95 Latency', p95 > 0 ? `${p95.toFixed(0)}ms` : '—', p95Color)}
        ${metricChipHTML('SLA Target', '<200ms')}
        ${metricChipHTML('SLA Status', slaLabel, slaColor)}
        ${metricChipHTML('Requests', perf.total_requests ?? 0)}
      </div>
    </div>

    <!-- Section C: Readiness + Moat -->
    <div style="margin-bottom:12px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;">Acquisition Readiness</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;">
        ${metricChipHTML('Score', `${acqScore}/100`, acqColor)}
        ${metricChipHTML('Tier', acqTier, acqColor)}
        ${metricChipHTML('Checks Passed', `${rdPassed}/${rdTotal}`, rdColor)}
        ${metricChipHTML('Confidence', rdConf, rdColor)}
        ${metricChipHTML('Moat Score', moatScore > 0 ? moatScore.toFixed(1) : '—', moatColor)}
      </div>
    </div>

    <!-- Section D: Pipeline -->
    <div style="margin-bottom:12px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;display:flex;align-items:center;justify-content:space-between;">
        <span>Last Pipeline Run</span>
        <button onclick="triggerPipeline()" style="font-size:10px;padding:3px 10px;background:var(--accent-primary);color:#fff;border:none;border-radius:4px;cursor:pointer;font-weight:600;letter-spacing:.04em;">▶ Run Pipeline</button>
      </div>
      <div id="pipeline-trigger-status" style="font-size:11px;margin-bottom:6px;"></div>
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
        <span style="font-family:var(--font-mono);font-size:14px;font-weight:700;color:${stsColor};">${overallSts.toUpperCase()}</span>
        <span style="font-size:11px;color:var(--text-muted);">${succeeded}/${stageNames.length} stages succeeded</span>
        ${failed > 0 ? `<span style="font-size:11px;color:var(--signal-sell);">${failed} failed</span>` : ''}
      </div>
      ${stageNames.slice(0, 8).map(s => {
        const st = stages[s] || {};
        const icon = st.status === 'success' ? '✓' : st.status === 'failed' ? '✗' : '○';
        const col  = st.status === 'success' ? 'var(--signal-buy)' : st.status === 'failed' ? 'var(--signal-sell)' : 'var(--text-muted)';
        const dur  = st.duration_ms != null ? `${(st.duration_ms/1000).toFixed(1)}s` : '';
        return `<div style="display:flex;align-items:center;gap:6px;font-size:11px;padding:2px 0;">
          <span style="color:${col};font-family:var(--font-mono);">${icon}</span>
          <span style="flex:1;color:var(--text-secondary);">${s.replace(/_/g,' ')}</span>
          <span style="color:var(--text-muted);font-size:10px;">${dur}</span>
        </div>`;
      }).join('')}
    </div>

    <!-- Section E: Warnings -->
    <div style="margin-bottom:12px;">
      <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;">System Alerts</div>
      ${warningsHTML}
    </div>

    <!-- Section F: Acquisition links -->
    <div style="margin-top:4px;padding:8px 10px;background:var(--bg-tertiary);border-radius:6px;font-size:11px;">
      <span style="color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;font-size:10px;">Due Diligence · </span>
      <a href="/developer/due-diligence" target="_blank" style="color:var(--accent-primary);text-decoration:none;margin-right:12px;">Technical DD</a>
      <a href="/developer/ip-audit" target="_blank" style="color:var(--accent-primary);text-decoration:none;margin-right:12px;">IP Audit</a>
      <a href="/developer/formula-registry" target="_blank" style="color:var(--accent-primary);text-decoration:none;">Formula Registry</a>
    </div>`;
}

async function triggerPipeline() {
  const statusEl = document.getElementById('pipeline-trigger-status');
  if (statusEl) { statusEl.textContent = 'Triggering pipeline…'; statusEl.style.color = 'var(--text-muted)'; }
  try {
    const result = await API.post('/orchestration/pipeline/run', {});
    if (statusEl) {
      statusEl.textContent = `Pipeline triggered — run ID: ${result.run_id || 'unknown'}`;
      statusEl.style.color = 'var(--signal-buy)';
    }
    setTimeout(loadSystemStatus, 3000);
  } catch (err) {
    if (statusEl) {
      const msg = err?.statusCode === 401 ? 'Unauthorized — pipeline requires authentication.'
                : err?.statusCode === 403 ? 'Access denied — enterprise tier required.'
                : 'Pipeline trigger failed — please retry or check system logs.';
      statusEl.textContent = msg;
      statusEl.style.color = 'var(--signal-sell)';
    }
  }
}

function metricChipHTML(label, value, valueColor) {
  const col = valueColor || 'var(--text-primary)';
  return `
    <div class="metric-card" style="padding:6px 10px;">
      <span class="metric-card__label">${label}</span>
      <span class="metric-card__value" style="font-size:13px;color:${col};">${value}</span>
    </div>`;
}
