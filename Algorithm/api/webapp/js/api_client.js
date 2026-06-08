/* AXIOM API Client — centralized HTTP layer */

// Self-init: fetch server config and populate API key before any panel loads
(async function axiomSelfInit() {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await fetch('/config/client');
      if (!res.ok) {
        await new Promise(r => setTimeout(r, 400 * (attempt + 1)));
        continue;
      }
      const cfg = await res.json();
      if (cfg.api_key && cfg.api_key.length > 0) {
        localStorage.setItem('ftip_api_key', cfg.api_key);
        const el = document.getElementById('api-key-input');
        if (el) {
          el.value = '•'.repeat(Math.min(cfg.api_key.length, 20));
          el.placeholder = 'Configured ✓';
          el.style.color = 'var(--signal-buy)';
        }
      }
      return; // success
    } catch (_) {
      await new Promise(r => setTimeout(r, 400 * (attempt + 1)));
    }
  }
})();

const API = {
  base: '',

  headers() {
    const key = localStorage.getItem('ftip_api_key') || '';
    const h = { 'Content-Type': 'application/json', 'Accept': 'application/json' };
    if (key) h['X-FTIP-API-Key'] = key;
    return h;
  },

  async get(path, params = {}) {
    const url = new URL(this.base + path, window.location.origin);
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== null) url.searchParams.set(k, v);
    });
    const res = await fetch(url.toString(), { method: 'GET', headers: this.headers() });
    if (!res.ok) {
      const text = await res.text();
      throw new APIError(res.status, text);
    }
    return res.json();
  },

  async post(path, body = {}) {
    const res = await fetch(this.base + path, {
      method: 'POST',
      headers: this.headers(),
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new APIError(res.status, text);
    }
    return res.json();
  },
};

class APIError extends Error {
  constructor(statusCode, message) {
    super(message);
    this.name = 'APIError';
    this.statusCode = statusCode;
  }
}

/* Sync API key input with localStorage */
function initAPIKey() {
  const input = document.getElementById('api-key-input');
  if (!input) return;
  const stored = localStorage.getItem('ftip_api_key') || '';
  input.value = stored;
  input.addEventListener('change', () => {
    localStorage.setItem('ftip_api_key', input.value.trim());
  });
}

document.addEventListener('DOMContentLoaded', initAPIKey);
