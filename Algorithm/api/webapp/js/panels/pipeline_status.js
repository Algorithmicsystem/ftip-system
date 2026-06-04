/* Pipeline Status Panel */

const PIPELINE_STAGE_NAMES = [
  'bar_ingestion', 'feature_computation', 'signal_generation', 'axiom_scoring',
  'alt_data_update', 'factor_computation', 'ml_inference', 'pnl_compute',
  'ic_computation', 'calibration_update', 'ic_gate_update', 'breadth_computation',
  'sri_computation', 'memory_consolidation', 'cache_refresh',
];

async function loadPipelineStatus() {
  const body = document.getElementById('pipeline-body');
  const lastRunEl = document.getElementById('pipeline-last-run');
  if (!body) return;

  body.innerHTML = '<div class="loading-skeleton skeleton-line full" style="margin-bottom:6px;height:20px;"></div>'.repeat(5);

  try {
    const [health, pipelineRun] = await Promise.all([
      API.get('/orchestration/health').catch(() => null),
      API.get('/orchestration/pipeline/status').catch(() => null),
    ]);

    renderPipelinePanel(health, pipelineRun);

    if (lastRunEl && pipelineRun?.started_at) {
      lastRunEl.textContent = `Last run: ${new Date(pipelineRun.started_at).toLocaleString()}`;
    }

    // Update header health dot
    if (health) {
      updateHealthIndicator(health.overall_status || 'healthy');
    }
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Pipeline data unavailable.</div>`;
  }
}

function renderPipelinePanel(health, run) {
  const body = document.getElementById('pipeline-body');
  if (!body) return;

  const stages = run?.stages || {};
  const overallStatus = run?.overall_status || 'unknown';
  const statusColor = overallStatus === 'success' ? 'var(--signal-buy)' :
                      overallStatus === 'partial'  ? 'var(--signal-hold)' :
                      overallStatus === 'failed'   ? 'var(--signal-sell)' : 'var(--text-muted)';

  const stagesHTML = PIPELINE_STAGE_NAMES.map(stage => {
    const s = stages[stage] || {};
    const status = s.status || 'pending';
    const icon = status === 'success' ? '✓' : status === 'failed' ? '✗' : status === 'running' ? '◌' : '○';
    const durationMs = s.duration_ms;
    const dur = durationMs != null ? `${(durationMs / 1000).toFixed(1)}s` : '';
    return `
      <div class="pipeline-stage ${status}">
        <span class="pipeline-stage__icon">${icon}</span>
        <span class="pipeline-stage__name">${stage.replace(/_/g, ' ')}</span>
        <span class="pipeline-stage__time">${dur}</span>
      </div>`;
  }).join('');

  const mlHealth = health?.ml_model_health || {};
  const icState  = health?.ic_state || 'INSUFFICIENT';
  const icCls = icState === 'STRONG' ? 'strong' : icState === 'MODERATE' ? 'moderate' :
                icState === 'WEAK'   ? 'weak'   : icState === 'DEGRADED'  ? 'degraded' : 'insufficient';

  body.innerHTML = `
    <!-- Overall status -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
      <div>
        <div class="metric-card__label">Pipeline Status</div>
        <div style="font-family:var(--font-mono);font-size:16px;font-weight:700;color:${statusColor};margin-top:2px;">
          ${overallStatus.toUpperCase()}
        </div>
      </div>
      <div style="text-align:right;">
        <div class="metric-card__label">IC State</div>
        <span class="ic-state-badge ${icCls}" style="margin-top:4px;">${icState}</span>
      </div>
    </div>

    <!-- ML Model -->
    ${mlHealth.model_version ? `
    <div style="display:flex;gap:12px;margin-bottom:10px;">
      <div class="metric-card" style="flex:1;padding:8px 10px;">
        <span class="metric-card__label">Model Version</span>
        <span class="metric-card__value" style="font-size:13px;">${mlHealth.model_version}</span>
      </div>
      <div class="metric-card" style="flex:1;padding:8px 10px;">
        <span class="metric-card__label">PSI Score</span>
        <span class="metric-card__value" style="font-size:13px;color:${(mlHealth.psi_score||0) > 0.2 ? 'var(--signal-sell)' : 'var(--signal-buy)'};">${(mlHealth.psi_score||0).toFixed(3)}</span>
      </div>
    </div>` : ''}

    <!-- Stage list -->
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--text-muted);margin-bottom:6px;">Pipeline Stages</div>
    <div style="max-height:220px;overflow-y:auto;">
      ${stagesHTML}
    </div>

    <!-- Health summary -->
    ${health ? `
    <div style="margin-top:10px;font-size:11px;color:var(--text-muted);">
      Health score: <span style="color:${health.overall_score >= 70 ? 'var(--signal-buy)' : 'var(--signal-hold)'};font-family:var(--font-mono);">${(health.overall_score||0).toFixed(0)}</span> ·
      Status: <span style="color:var(--text-secondary);">${health.overall_status || 'unknown'}</span>
    </div>` : ''}`;
}

async function triggerPipelineRun() {
  const body = document.getElementById('pipeline-body');
  if (!body) return;
  try {
    body.innerHTML = `<div class="alert-banner info">Triggering pipeline run…</div>`;
    await API.post('/orchestration/pipeline/run', {});
    setTimeout(loadPipelineStatus, 1000);
  } catch (err) {
    body.innerHTML = `<div class="alert-banner warning">Could not trigger pipeline: ${err.message}</div>`;
  }
}

function updateHealthIndicator(status) {
  const dot = document.getElementById('health-dot');
  const label = document.getElementById('health-label');
  if (!dot || !label) return;
  dot.className = `health-dot ${status}`;
  label.textContent = status.charAt(0).toUpperCase() + status.slice(1);
}
