/* AXIOM Chart rendering utilities — wraps Chart.js */

const _chartInstances = {};

function _destroyChart(id) {
  if (_chartInstances[id]) {
    _chartInstances[id].destroy();
    delete _chartInstances[id];
  }
}

/**
 * renderDAUBar — horizontal colored progress bar in a container.
 * @param {string} containerId
 * @param {number} value  0-100
 * @param {number} max    default 100
 */
function renderDAUBar(containerId, value, max = 100) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const pct = Math.min(100, Math.max(0, (value / max) * 100));
  const cls = pct >= 65 ? 'high' : pct >= 40 ? 'mid' : 'low';
  container.innerHTML = `
    <div class="dau-bar">
      <div class="dau-bar__track">
        <div class="dau-bar__fill ${cls}" style="width:${pct.toFixed(1)}%;"></div>
      </div>
      <span class="dau-bar__label">${value.toFixed(0)}</span>
    </div>`;
}

/**
 * renderSRIGauge — display SRI value with colored bar and label.
 * @param {string} containerId
 * @param {number} value  0-100
 */
function renderSRIGauge(containerId, value) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const label = value >= 85 ? 'Critical' : value >= 70 ? 'High Alert' : value >= 50 ? 'Warning' : value >= 25 ? 'Elevated' : 'Stable';
  const cls   = value >= 70 ? 'high' : value >= 40 ? 'medium' : 'low';
  const barCls = value >= 85 ? 'critical' : value >= 70 ? 'warning' : value >= 40 ? 'elevated' : 'stable';
  container.innerHTML = `
    <div class="gauge-container">
      <span class="gauge-label">Systemic Risk</span>
      <span class="gauge-value ${cls}">${value.toFixed(1)}</span>
      <div class="sri-bar" style="width:100%;margin-top:4px;">
        <div class="sri-bar__fill ${barCls}" style="width:${value.toFixed(1)}%;"></div>
      </div>
      <span class="text-sm text-muted" style="margin-top:4px;">${label}</span>
    </div>`;
}

/**
 * renderSparkline — mini time series using Chart.js line chart.
 * @param {string} containerId
 * @param {number[]} values
 * @param {string} color  CSS color
 */
function renderSparkline(containerId, values, color = '#3b82f6') {
  const container = document.getElementById(containerId);
  if (!container || !values || values.length === 0) return;

  _destroyChart(containerId);
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  canvas.height = 60;

  if (typeof Chart === 'undefined') return;

  const ctx = canvas.getContext('2d');
  const last = values[values.length - 1];
  const first = values[0];
  const lineColor = last > first ? '#10b981' : last < first ? '#ef4444' : color;

  _chartInstances[containerId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: values.map((_, i) => i),
      datasets: [{
        data: values,
        borderColor: lineColor,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: {
          target: 'origin',
          above: lineColor.replace(')', ',0.08)').replace('rgb', 'rgba').replace('#10b981', 'rgba(16,185,129,0.08)').replace('#ef4444', 'rgba(239,68,68,0.08)').replace('#3b82f6', 'rgba(59,130,246,0.08)'),
        },
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: { display: false },
      },
    },
  });
}

/**
 * renderHeatmap — factor weight heatmap as colored bars.
 * @param {string} containerId
 * @param {{ name: string, value: number, type: string }[]} data
 */
function renderHeatmap(containerId, data) {
  const container = document.getElementById(containerId);
  if (!container || !data) return;

  const maxAbs = Math.max(...data.map(d => Math.abs(d.value)), 1);
  container.innerHTML = data.map(d => {
    const pct = (Math.abs(d.value) / maxAbs * 100).toFixed(1);
    const cls = d.type === 'favored' ? 'favored' : d.type === 'unfavored' ? 'unfavored' : 'neutral';
    return `
      <div class="factor-row">
        <span class="factor-row__name">${d.name}</span>
        <div class="factor-row__bar">
          <div class="factor-row__fill ${cls}" style="width:${pct}%;"></div>
        </div>
        <span class="factor-row__val">${(d.value * 100).toFixed(0)}%</span>
      </div>`;
  }).join('');
}

/**
 * renderDAUSparkline — mini DAU history line chart.
 * @param {string} containerId
 * @param {{ as_of_date: string, dau: number }[]} dauHistory
 */
function renderDAUSparkline(containerId, dauHistory) {
  const container = document.getElementById(containerId);
  if (!container || !dauHistory || dauHistory.length === 0) return;

  _destroyChart(containerId);
  container.innerHTML = '<canvas></canvas>';
  const canvas = container.querySelector('canvas');
  canvas.height = 60;

  if (typeof Chart === 'undefined') return;

  const values = dauHistory.map(d => d.dau ?? 0);
  const labels = dauHistory.map(d => d.as_of_date || '');
  const last = values[values.length - 1];
  const first = values[0];
  const lineColor = last > first ? '#10b981' : last < first ? '#ef4444' : '#3b82f6';

  const ctx = canvas.getContext('2d');
  _chartInstances[containerId] = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: lineColor,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        tension: 0.3,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } },
    },
  });
}

/**
 * renderScoreBar — named horizontal bar inside a container.
 * @param {string} label
 * @param {number} value 0-100
 * @param {string} color  CSS color override
 */
function scoreBarHTML(label, value, color = '#3b82f6') {
  const v = typeof value === 'number' ? value : parseFloat(value) || 0;
  return `
    <div class="score-bar">
      <span class="score-bar__label">${label}</span>
      <div class="score-bar__track">
        <div class="score-bar__fill" style="width:${v.toFixed(1)}%;background:${color};"></div>
      </div>
      <span class="score-bar__num">${v.toFixed(0)}</span>
    </div>`;
}
