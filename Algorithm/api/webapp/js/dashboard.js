/* Dashboard Controller — orchestrates panels, navigation, and auto-refresh */

const REFRESH_INTERVAL_MS = 5 * 60 * 1000; // 5 minutes

let _activePanel = 'briefing';
let _refreshTimer = null;
let _panelLoadState = {};

// Panel definitions: id → { load: fn, label }
const PANELS = {
  briefing:      { load: loadMorningBriefing,   label: 'Morning Briefing' },
  opportunities: { load: loadUniverseScreen,    label: 'Universe Screen' },
  symbol:        { load: () => {},              label: 'Symbol Intelligence' },
  risk:          { load: loadRiskMonitor,       label: 'Risk Monitor' },
  factors:       { load: loadFactorEnvironment, label: 'Factor Environment' },
  pipeline:      { load: loadSystemStatus,      label: 'System Status' },
};

// Grid panels live inside view-dashboard; pe/smb are overlay divs
const _GRID_PANELS    = new Set(['briefing', 'opportunities', 'symbol', 'risk', 'factors', 'pipeline']);
const _OVERLAY_PANELS = new Set(['pe', 'smb']);

// ── Initialization ────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  await autoConfigureAPIKey();
  initAPIKey();
  setupSidebarNav();
  setupTabNav();
  setupKeyboardShortcuts();
  switchPanel('briefing');
  scheduleAutoRefresh();
  startHealthMonitor();
});

async function autoConfigureAPIKey() {
  try {
    const res = await fetch('/config/client');
    if (!res.ok) return;
    const cfg = await res.json();
    if (cfg.api_key) {
      localStorage.setItem('ftip_api_key', cfg.api_key);
      const input = document.getElementById('api-key-input');
      if (input) input.value = cfg.api_key;
    }
  } catch (_) {
    // Server not reachable or no key configured — no-op
  }
}

// ── Sidebar Navigation ────────────────────────────────────────────────────────

function setupSidebarNav() {
  document.querySelectorAll('.nav-item[data-panel]').forEach(item => {
    item.addEventListener('click', () => {
      const panel = item.dataset.panel;
      if (panel) switchPanel(panel);
    });
  });
}

function switchPanel(panelId) {
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  const navItem = document.querySelector(`.nav-item[data-panel="${panelId}"]`);
  if (navItem) navItem.classList.add('active');

  _activePanel = panelId;

  if (_OVERLAY_PANELS.has(panelId)) {
    // Show overlay, hide dashboard grid and other overlays
    document.getElementById('view-dashboard')?.classList.add('hidden');
    _OVERLAY_PANELS.forEach(id => {
      const el = document.getElementById(`view-${id}`);
      if (el) el.classList.toggle('hidden', id !== panelId);
    });
  } else {
    // Reveal dashboard grid, hide all overlays, scroll to panel
    document.getElementById('view-dashboard')?.classList.remove('hidden');
    _OVERLAY_PANELS.forEach(id => document.getElementById(`view-${id}`)?.classList.add('hidden'));
    const panel = document.getElementById(`panel-${panelId}`);
    if (panel) {
      panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
      panel.style.boxShadow = '0 0 0 2px var(--accent-primary)';
      setTimeout(() => { panel.style.boxShadow = ''; }, 1500);
    }
  }

  if (!_panelLoadState[panelId]) {
    loadPanel(panelId);
  }
}

function loadPanel(panelId) {
  const def = PANELS[panelId];
  if (!def) return;

  _panelLoadState[panelId] = Date.now();
  try {
    def.load();
  } catch (err) {
    console.error(`[dashboard] Failed to load panel ${panelId}:`, err);
  }
}

// ── Tab Navigation ────────────────────────────────────────────────────────────

function setupTabNav() {
  document.querySelectorAll('.tab-btn[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      const container = btn.closest('.panel');
      if (!container || !tab) return;
      switchTab(container, tab);
    });
  });
}

function switchTab(container, tabId) {
  container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  container.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

  const btn = container.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  const pane = container.querySelector(`#tab-${tabId}`);

  if (btn)  btn.classList.add('active');
  if (pane) pane.classList.add('active');
}

// ── PE / SMB Handlers ─────────────────────────────────────────────────────────

function handlePELoad() {
  const input = document.getElementById('pe-org-input');
  const orgId = input?.value?.trim();
  if (orgId) loadPEPortfolio(orgId);
}

function handleSMBLoad() {
  const input = document.getElementById('smb-entity-input');
  const entityId = input?.value?.trim();
  if (entityId) loadSMBIntelligence(entityId);
}

function showPEView() {
  switchPanel('pe');
}

function showSMBView() {
  switchPanel('smb');
}

// ── Auto-Refresh ──────────────────────────────────────────────────────────────

function scheduleAutoRefresh() {
  if (_refreshTimer) clearInterval(_refreshTimer);
  _refreshTimer = setInterval(refreshActivePanels, REFRESH_INTERVAL_MS);
}

function refreshActivePanels() {
  // Invalidate load state so panels reload on next visit
  _panelLoadState = {};

  // Immediately reload the active panel
  if (_activePanel && PANELS[_activePanel]) {
    loadPanel(_activePanel);
  }
}

function forceRefreshAll() {
  _panelLoadState = {};
  loadPanel(_activePanel);
}

// ── Keyboard Shortcuts ────────────────────────────────────────────────────────

function setupKeyboardShortcuts() {
  document.addEventListener('keydown', e => {
    // Cmd/Ctrl+K → focus global symbol search
    if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
      e.preventDefault();
      const input = document.getElementById('global-symbol-input');
      if (input) { input.focus(); input.select(); }
      return;
    }

    // Enter on global symbol search
    if (e.key === 'Enter' && document.activeElement?.id === 'global-symbol-input') {
      handleGlobalSymbolSearch();
      return;
    }

    // Enter on symbol search within panel
    if (e.key === 'Enter' && document.activeElement?.id === 'symbol-search') {
      handleSymbolSearch();
      return;
    }

    // Escape → blur any input
    if (e.key === 'Escape') {
      document.activeElement?.blur();
    }
  });
}

// ── Utility ───────────────────────────────────────────────────────────────────

function setLoadingState(elementId, loading) {
  const el = document.getElementById(elementId);
  if (!el) return;
  if (loading) {
    el.classList.add('loading');
  } else {
    el.classList.remove('loading');
  }
}

// ── System Health Monitor ─────────────────────────────────────────────────────

let _healthTimer = null;
const HEALTH_POLL_MS = 60000; // 1 minute

async function updateSystemHealth() {
  try {
    const data = await API.get('/system/status');
    const dot = document.getElementById('health-dot');
    const label = document.getElementById('health-label');
    if (dot) {
      dot.className = 'health-dot';
      if (data.server === 'healthy' && data.db_connected) {
        dot.classList.add('healthy');
      } else if (data.warnings && data.warnings.length > 0) {
        dot.classList.add('degraded');
      } else {
        dot.classList.add('critical');
      }
    }
    if (label) {
      label.textContent = data.db_connected ? 'Live' : 'Degraded';
    }
    const versionEl = document.getElementById('system-version');
    if (versionEl && data.version) versionEl.textContent = `v${data.version}`;
  } catch (err) {
    const dot = document.getElementById('health-dot');
    if (dot) { dot.className = 'health-dot critical'; }
    const label = document.getElementById('health-label');
    if (label) label.textContent = 'Offline';
  }
}

function startHealthMonitor() {
  updateSystemHealth();
  if (_healthTimer) clearInterval(_healthTimer);
  _healthTimer = setInterval(updateSystemHealth, HEALTH_POLL_MS);
}

function showGlobalError(message) {
  const existing = document.getElementById('global-error-banner');
  if (existing) existing.remove();

  const banner = document.createElement('div');
  banner.id = 'global-error-banner';
  banner.className = 'alert-banner warning';
  banner.style.cssText = 'position:fixed;top:56px;left:50%;transform:translateX(-50%);z-index:9999;max-width:480px;';
  banner.textContent = message;
  document.body.appendChild(banner);
  setTimeout(() => banner.remove(), 5000);
}
