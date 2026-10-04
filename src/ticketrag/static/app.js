(() => {
  'use strict';

  const $ = (sel, root = document) => root.querySelector(sel);
  const make = (tag, cls, text) => {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;  // textContent only: never innerHTML with server data
    return node;
  };

  const EXAMPLES = [
    ['Heating', 'There is no heat in my apartment for 3 days now and it is freezing. I have a baby at home.'],
    ['Noise', 'My neighbours upstairs are blasting music again at 3am. I cannot sleep and I have work in the morning.'],
    ['Street', 'A huge pothole on my street destroyed my tyre yesterday.'],
    ['Parking', 'Some car has been parked across my driveway all day and I cannot get out to go to work.'],
    ['Out of scope', "My broadband drops every evening around 8 and I've already restarted the router twice. I work from home and this is costing me."],
  ];
  const SEVERITY = ['', 'Minor', 'Nuisance', 'Significant disruption', 'Health or safety risk', 'Emergency'];
  const TIER_HELP = {
    fast_lookup: 'History is highly consistent (90% or more, at least 30 tickets). Answered from a template, no AI drafting.',
    rag_light: 'Fairly consistent history. AI-drafted from the retrieved tickets.',
    rag_core: 'Mixed outcomes in the history. Full retrieval-augmented draft.',
  };
  const METHOD = {
    llm: ['AI-drafted, checked against sources', ''],
    template: ['Rendered from ticket statistics', ''],
    template_degraded: ['AI unavailable: statistics only', 'warn'],
  };

  const state = { gate: 0.74, recent: [], lastJson: null };
  const ui = {
    input: $('#complaint'), ask: $('#ask'), count: $('#count'), result: $('#result'), status: $('#status'),
    examples: $('#examples'), recent: $('#recent'), key: $('#apikey'), keyBox: $('.advanced'),
  };

  // ---------- header status ----------
  async function loadStatus() {
    try {
      const r = await fetch('/ready');
      if (!r.ok) throw new Error(String(r.status));
      const d = await r.json();
      state.gate = d.gate;
      ui.status.className = 'status ok';
      ui.status.textContent = `${d.patterns.toLocaleString()} ticket types · ${d.vectors.toLocaleString()} vectors · ${d.embedder.split('/').pop()} · ${d.llm}`;
    } catch (e) {
      ui.status.className = 'status down';
      ui.status.textContent = 'service not ready';
    }
  }

  // ---------- input column ----------
  EXAMPLES.forEach(([tag, text]) => {
    const li = make('li');
    const b = make('button');
    b.type = 'button';
    b.append(make('span', 'tag', tag), document.createTextNode(text.length > 80 ? text.slice(0, 77) + '...' : text));
    b.addEventListener('click', () => { ui.input.value = text; updateCount(); ui.input.focus(); });
    li.append(b);
    ui.examples.append(li);
  });

  function updateCount() { ui.count.textContent = `${ui.input.value.length} / 2000`; }
  ui.input.addEventListener('input', updateCount);
  ui.input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) submit(); });
  ui.ask.addEventListener('click', submit);

  function renderRecent() {
    ui.recent.replaceChildren();
    if (!state.recent.length) { ui.recent.append(make('li', 'muted', 'Nothing asked yet.')); return; }
    state.recent.forEach((item) => {
      const li = make('li');
      const b = make('button');
      b.type = 'button';
      b.append(document.createTextNode(item.text.length > 70 ? item.text.slice(0, 67) + '...' : item.text),
        make('span', 'meta', item.res.abstained ? 'abstained' : `${item.res.category} · severity ${item.res.severity}`));
      b.addEventListener('click', () => { ui.input.value = item.text; updateCount(); render(item.res); });
      li.append(b);
      ui.recent.append(li);
    });
  }

  // ---------- request ----------
  async function submit() {
    const text = ui.input.value.trim();
    if (text.length < 5) { showError('Please enter at least a few words of the complaint.'); return; }
    ui.ask.disabled = true;
    const started = performance.now();
    const loading = make('div', 'loading');
    ui.result.replaceChildren(loading);
    const tick = setInterval(() => { loading.textContent = `Searching history and drafting... ${((performance.now() - started) / 1000).toFixed(1)} s`; }, 100);
    try {
      const headers = { 'Content-Type': 'application/json' };
      if (ui.key.value) headers['X-API-Key'] = ui.key.value;
      const r = await fetch('/ask', { method: 'POST', headers, body: JSON.stringify({ complaint: text }) });
      if (r.status === 401) { ui.keyBox.open = true; ui.key.focus(); throw new Error('The server requires an access key. Enter it under "Access key".'); }
      if (!r.ok) {
        let detail = `Request failed (${r.status}).`;
        try { const j = await r.json(); if (typeof j.detail === 'string') detail = j.detail; } catch (_) { /* keep default */ }
        throw new Error(detail);
      }
      const res = await r.json();
      state.recent.unshift({ text, res });
      state.recent = state.recent.slice(0, 6);
      renderRecent();
      render(res);
    } catch (e) {
      showError(e.message || 'Could not reach the service.');
    } finally {
      clearInterval(tick);
      ui.ask.disabled = false;
    }
  }

  function showError(message) {
    const card = make('div', 'note bad', message);
    card.setAttribute('role', 'alert');
    ui.result.replaceChildren(card);
  }

  // ---------- rendering ----------
  function kv(label, valueNode, title) {
    const box = make('div', 'kv');
    if (title) box.title = title;
    box.append(make('div', 'k', label), valueNode);
    return box;
  }
  function value(text, small) { return make('div', small ? 'v small' : 'v', text); }

  function severityNode(level) {
    const wrap = make('div', 'v');
    const pips = make('span', 'pips');
    for (let i = 1; i <= 5; i++) pips.append(make('span', `pip${i <= level ? ` on l${level}` : ''}`));
    wrap.append(pips, document.createTextNode(`${level} · ${SEVERITY[level] || ''}`));
    return wrap;
  }

  function meterNode(score) {
    const wrap = make('div');
    wrap.append(make('div', 'v', score.toFixed(3)));
    const meter = make('div', 'meter');
    const fill = make('span', 'fill');
    const scale = (x) => Math.max(0, Math.min(1, (x - 0.5) / 0.5)) * 100;
    fill.style.setProperty('width', `${scale(score)}%`);
    const gate = make('span', 'gate');
    gate.style.setProperty('left', `${scale(state.gate)}%`);
    gate.title = `abstention threshold ${state.gate}`;
    meter.append(fill, gate);
    wrap.append(meter);
    return wrap;
  }

  function citeChips(ids, sources) {
    const wrap = make('span', 'cites');
    (ids || []).forEach((id) => {
      const b = make('button', 'cite', id);
      b.type = 'button';
      b.title = 'Show source';
      b.addEventListener('click', () => focusSource(id));
      wrap.append(b);
    });
    return wrap;
  }

  function focusSource(id) {
    const details = $('#sources-card');
    if (details) details.open = true;
    const target = document.getElementById(`src-${id}`);
    if (!target) return;
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.classList.remove('flash');
    void target.offsetWidth;  // restart the animation
    target.classList.add('flash');
  }

  function card(title) { const c = make('section', 'card'); if (title) c.append(make('h3', null, title)); return c; }

  function candidatesCard(candidates, open) {
    const c = make('details', 'card');
    if (open) c.open = true;
    c.append(make('summary', null, 'Nearest ticket types'));
    candidates.forEach(([label, score]) => {
      const row = make('div', 'cand');
      row.append(make('span', null, label), make('span', 'score', score.toFixed(3)));
      c.append(row);
    });
    return c;
  }

  function renderAbstain(res) {
    const frag = document.createDocumentFragment();
    const c = card();
    c.append(make('p', 'abstain-title', 'No close match in the ticket history'));
    const why = res.abstain_reason && res.abstain_reason.startsWith('low_similarity')
      ? `The most similar past ticket type scores ${res.top_score.toFixed(3)}, below the ${state.gate} threshold. Nothing in the history describes this problem well enough to ground an answer.`
      : 'A closer look at the nearest ticket types found that none of them describes this problem.';
    c.append(make('p', null, why), make('p', 'muted', 'Route this to a human agent. The nearest types are shown below in case they help.'));
    frag.append(c, candidatesCard(res.candidates, true), footer(res));
    return frag;
  }

  function renderAnswer(res) {
    const frag = document.createDocumentFragment();

    // triage strip
    const tri = card();
    const strip = make('div', 'triage');
    strip.append(
      kv('Category', value(res.category)),
      kv('Product', value(res.product, true)),
      kv('Severity', severityNode(res.severity || 0), 'Model judgement, not validated'),
      kv('Sentiment', value(res.sentiment), 'Model judgement, not validated'),
      kv('Routing', (() => { const v = make('div', 'v'); v.append(make('span', 'badge accent', res.tier.replace('_', ' '))); return v; })(), TIER_HELP[res.tier]),
      kv('Match confidence', meterNode(res.top_score), `Cosine similarity; tick marks the ${state.gate} threshold`),
    );
    tri.append(strip);
    frag.append(tri);

    // notes
    const notes = [...(res.policy_notes || [])];
    const truncated = Object.values(res.sources || {}).some((s) => s.truncated_in_source);
    if (truncated) notes.push('One of the cited resolution texts is cut off in the source data. Check the full text before relying on it.');
    if (res.method === 'template_degraded') notes.unshift('AI drafting is unavailable. Showing historical statistics only.');
    if (notes.length) {
      const box = make('div', 'notes');
      notes.forEach((n) => box.append(make('div', /911|Emergency|Degraded|unavailable/.test(n) ? 'note bad' : 'note', n)));
      frag.append(box);
    }

    // answer
    const a = res.answer || {};
    const ans = card('Suggested approach');
    if (a.opening) ans.append(make('p', 'opening', a.opening));
    if (a.summary) {
      const p = make('p', 'summary', a.summary.text);
      p.append(citeChips(a.summary.citations, res.sources));
      ans.append(p);
    }
    const steps = make('ol', 'steps');
    (a.steps || []).forEach((s) => {
      const li = make('li', null, s.text);
      li.append(citeChips(s.citations, res.sources));
      steps.append(li);
    });
    ans.append(steps);
    frag.append(ans);

    // outcomes
    if ((a.outcomes || []).length) {
      const oc = card('What happened in similar past tickets');
      const list = make('div', 'outcomes');
      a.outcomes.forEach((o) => {
        const m = /^(\d+)% of tickets:\s*([\s\S]*)$/.exec(o.text);
        const share = m ? Number(m[1]) : 0;
        const item = make('div', 'outcome');
        const head = make('div', 'head');
        head.append(make('span', 'pct', m ? `${share}%` : ''), make('span', 'txt', m ? m[2] : o.text), citeChips(o.citations, res.sources));
        const bar = make('div', 'bar');
        const fill = make('span');
        fill.style.setProperty('width', `${share}%`);
        bar.append(fill);
        item.append(head, bar);
        list.append(item);
      });
      oc.append(list);
      frag.append(oc);
    }

    // sources
    const src = make('details', 'card');
    src.id = 'sources-card';
    src.append(make('summary', null, 'Sources'));
    const ids = Object.keys(res.sources || {}).sort((x, y) => (x[0] === y[0] ? Number(x.slice(1)) - Number(y.slice(1)) : x[0] < y[0] ? 1 : -1));
    ids.forEach((id) => {
      const s = res.sources[id];
      const row = make('div', 'source');
      row.id = `src-${id}`;
      row.append(make('span', 'sid', id));
      if (s.kind === 'pattern') {
        row.append(make('span', 'stxt', `Past closed tickets of type: ${s.label}`));
      } else {
        row.append(make('span', 'stxt', s.text));
        if (s.truncated_in_source) row.append(document.createTextNode(' '), make('span', 'badge warn', 'cut off in source data'));
      }
      src.append(row);
    });
    frag.append(src, candidatesCard(res.candidates, false), footer(res));
    return frag;
  }

  function footer(res) {
    const f = make('div', 'foot');
    const [methodLabel, methodClass] = METHOD[res.method] || ['', ''];
    if (res.method) f.append(make('span', `badge ${methodClass}`, methodLabel));
    const g = res.grounding || {};
    if (g.method === 'llm') {
      f.append(make('span', null, `validator: ${g.attempts} attempt${g.attempts === 1 ? '' : 's'}, ${(g.remaining_violations || []).length} rule violations left`));
    }
    const t = res.timing_ms || {};
    const parts = ['retrieve', 'parse', 'generate'].filter((k) => k in t).map((k) => `${k} ${t[k]} ms`);
    if (t.total !== undefined) f.append(make('span', null, `${parts.join(' · ')}${parts.length ? ' · ' : ''}total ${(t.total / 1000).toFixed(1)} s`));
    const u = res.llm_usage || {};
    if (u.prompt_tokens) f.append(make('span', null, `${u.prompt_tokens + u.completion_tokens} tokens`));
    f.append(make('span', null, `id ${res.request_id}`));
    const copy = make('button', null, 'Copy JSON');
    copy.type = 'button';
    copy.addEventListener('click', async () => {
      try { await navigator.clipboard.writeText(JSON.stringify(res, null, 2)); copy.textContent = 'Copied'; }
      catch (_) { copy.textContent = 'Copy failed'; }
      setTimeout(() => { copy.textContent = 'Copy JSON'; }, 1500);
    });
    f.append(copy);
    return f;
  }

  function render(res) {
    state.lastJson = res;
    ui.result.replaceChildren(res.abstained ? renderAbstain(res) : renderAnswer(res));
    ui.result.scrollIntoView({ block: 'nearest' });
  }

  loadStatus();
  renderRecent();
})();
