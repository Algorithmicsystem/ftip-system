/* AXIOM Floating Chatbot — bottom-right corner, proxies to /llm/chat */

(function () {
  'use strict';

  if (document.getElementById('axiom-chatbot-btn')) return;

  function init() {
    const styleEl = document.createElement('style');
    styleEl.textContent = `
      #axiom-chatbot-btn {
        position: fixed; bottom: 24px; right: 24px;
        width: 48px; height: 48px; border-radius: 50%;
        background: var(--accent-primary, #3b82f6);
        color: #fff; border: none; cursor: pointer;
        font-size: 20px; display: flex; align-items: center; justify-content: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.4); z-index: 9000;
        transition: transform .15s;
      }
      #axiom-chatbot-btn:hover { transform: scale(1.08); }
      #axiom-chatbot-window {
        position: fixed; bottom: 84px; right: 24px;
        width: 340px; max-height: 480px;
        background: var(--bg-secondary, #1a1f2e);
        border: 1px solid var(--border-primary, rgba(255,255,255,0.1));
        border-radius: 12px; box-shadow: 0 8px 32px rgba(0,0,0,0.5);
        display: none; flex-direction: column; z-index: 9000; overflow: hidden;
      }
      #axiom-chatbot-window.open { display: flex; }
      #axiom-chatbot-header {
        padding: 10px 14px;
        background: var(--bg-tertiary, #242938);
        border-bottom: 1px solid var(--border-primary, rgba(255,255,255,0.1));
        font-size: 13px; font-weight: 600; color: var(--text-primary, #e2e8f0);
        display: flex; align-items: center; justify-content: space-between;
      }
      #axiom-chatbot-close {
        background: none; border: none;
        color: var(--text-muted, #718096); cursor: pointer;
        font-size: 16px; padding: 0; line-height: 1;
      }
      #axiom-chatbot-messages {
        flex: 1; overflow-y: auto; padding: 12px;
        display: flex; flex-direction: column; gap: 8px;
      }
      .axiom-msg {
        font-size: 12px; line-height: 1.6;
        padding: 8px 10px; border-radius: 8px; max-width: 90%;
      }
      .axiom-msg.user {
        align-self: flex-end;
        background: var(--accent-primary, #3b82f6); color: #fff;
      }
      .axiom-msg.assistant {
        align-self: flex-start;
        background: var(--bg-tertiary, #242938);
        color: var(--text-primary, #e2e8f0);
        border: 1px solid var(--border-primary, rgba(255,255,255,0.08));
      }
      .axiom-msg.error {
        align-self: flex-start;
        background: rgba(239,68,68,0.1); color: var(--signal-sell, #ef4444);
        border: 1px solid rgba(239,68,68,0.2); font-size: 11px;
      }
      #axiom-chatbot-input-row {
        display: flex; gap: 6px; padding: 10px 12px;
        border-top: 1px solid var(--border-primary, rgba(255,255,255,0.1));
        background: var(--bg-tertiary, #242938);
      }
      #axiom-chatbot-input {
        flex: 1;
        background: var(--bg-primary, #0f1117);
        border: 1px solid var(--border-primary, rgba(255,255,255,0.1));
        border-radius: 6px; color: var(--text-primary, #e2e8f0);
        font-size: 12px; padding: 6px 10px; outline: none;
      }
      #axiom-chatbot-send {
        background: var(--accent-primary, #3b82f6); color: #fff;
        border: none; border-radius: 6px;
        padding: 6px 12px; font-size: 12px; cursor: pointer; font-weight: 600;
      }
      #axiom-chatbot-send:disabled { opacity: 0.5; cursor: default; }
    `;
    document.head.appendChild(styleEl);

    const btn = document.createElement('button');
    btn.id = 'axiom-chatbot-btn';
    btn.title = 'AXIOM AI Assistant';
    btn.innerHTML = '&#10022;';

    const win = document.createElement('div');
    win.id = 'axiom-chatbot-window';
    win.innerHTML = `
      <div id="axiom-chatbot-header">
        <span>&#10022; AXIOM Assistant</span>
        <button id="axiom-chatbot-close" title="Close">&#x2715;</button>
      </div>
      <div id="axiom-chatbot-messages">
        <div class="axiom-msg assistant">
          Hello. I'm your AXIOM Intelligence assistant. Ask me about signals, regimes, risk, or portfolio strategy.
        </div>
      </div>
      <div id="axiom-chatbot-input-row">
        <input id="axiom-chatbot-input" type="text" placeholder="Ask about signals, risk, regime…" maxlength="500" />
        <button id="axiom-chatbot-send">Send</button>
      </div>
    `;

    document.body.appendChild(btn);
    document.body.appendChild(win);

    const messagesEl = document.getElementById('axiom-chatbot-messages');
    const inputEl    = document.getElementById('axiom-chatbot-input');
    const sendBtn    = document.getElementById('axiom-chatbot-send');

    btn.addEventListener('click', () => {
      win.classList.toggle('open');
      if (win.classList.contains('open')) inputEl.focus();
    });

    document.getElementById('axiom-chatbot-close').addEventListener('click', () => {
      win.classList.remove('open');
    });

    function appendMessage(role, text) {
      const div = document.createElement('div');
      div.className = `axiom-msg ${role}`;
      div.textContent = text;
      messagesEl.appendChild(div);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      return div;
    }

    async function buildContext(userMessage) {
      const ctx = {};
      try {
        const scores = await fetch('/intelligence/universe/scores').then(r => r.ok ? r.json() : []);
        const withData = (scores || []).filter(s => s.dau !== null);
        const buys  = withData.filter(s => (s.dau || 0) >= 65).length;
        const sells = withData.filter(s => (s.dau || 0) < 40).length;
        const avgDau = withData.length ? (withData.reduce((a, x) => a + (x.dau || 50), 0) / withData.length).toFixed(1) : 0;
        ctx.universe_summary = { total: withData.length, buys, sells, avg_dau: Number(avgDau) };
        ctx.top_signals = withData.sort((a, b) => (b.dau || 0) - (a.dau || 0)).slice(0, 3).map(s => ({
          symbol: s.symbol, dau: s.dau, signal: s.signal,
        }));
      } catch (_) {}
      const symMatch = userMessage.match(/\b([A-Z]{2,5})\b/);
      const sym = symMatch?.[1];
      if (sym && !['I','AND','OR','THE','AI','PE','ML','IC','DAU','SLA','BUY','ETF'].includes(sym)) {
        try {
          const intel = await fetch(`/intelligence/universal/${sym}`).then(r => r.ok ? r.json() : null);
          if (intel && intel.dau) ctx.symbol_intel = { symbol: sym, dau: intel.dau, signal: intel.signal_label, eis: intel.eis_score };
        } catch (_) {}
      }
      return ctx;
    }

    async function sendMessage() {
      const text = inputEl.value.trim();
      if (!text || sendBtn.disabled) return;
      inputEl.value = '';
      sendBtn.disabled = true;

      appendMessage('user', text);
      const thinkingEl = appendMessage('assistant', '…');

      try {
        const context = await buildContext(text);
        const res = await fetch('/llm/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: text, context }),
        });
        const data = await res.json();
        if (!res.ok) {
          thinkingEl.className = 'axiom-msg error';
          thinkingEl.textContent = data?.error?.message || 'Request failed. Please try again.';
        } else {
          thinkingEl.textContent = data.reply || '(no response)';
        }
      } catch (_err) {
        thinkingEl.className = 'axiom-msg error';
        thinkingEl.textContent = 'Network error. Check your connection.';
      } finally {
        sendBtn.disabled = false;
        inputEl.focus();
      }
    }

    sendBtn.addEventListener('click', sendMessage);
    inputEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
