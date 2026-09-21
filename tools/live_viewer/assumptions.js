// Панель «Параметры модели и приборов»: просмотр, редактирование, сброс, экспорт и импорт значений.
(function () {
  const SRC_TEXT = {
    GOST: 'норматив (ГОСТ)', REQUIREMENT: 'нормативное требование', TASK: 'требование ТЗ или уточнение заказчика',
    DATA: 'получено по данным', LIT: 'значение из документа-источника',
  };
  const EFFECT_TEXT = {
    cycle: 'применяется при следующем расчёте',
    ensemble: 'применяется при следующем расчёте',
    state: 'пересчёт фильтра серы и рядов состояния (до 30 с)',
  };
  const REF_KIND = {
    official: 'официальный документ', literature: 'литература, не норматив', data: 'данные проекта',
    none: 'официального источника нет',
  };
  const $ = (id) => document.getElementById(id);
  let model = null;
  let activeTab = null;
  let dirty = {};
  let busy = false;
  let lastCycle = null;
  const BLEND_TAB = 'Блендинг';
  const pct = (x) => (Math.round(x * 1000) / 10) + ' %';

  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  const num = (x) => (typeof x === 'number' ? String(Number(x.toPrecision(6))) : x);
  const show = (x) => typeof x === 'boolean' ? (x ? 'включён' : 'исключён') : Array.isArray(x) ? x.map(num).join(' … ') : (x === null || x === undefined ? '-' : String(num(x)));
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined) e.textContent = text;
    return e;
  };

  function setStatus(text, kind) {
    const s = $('aStatus');
    s.textContent = text || '';
    s.className = 'a-status' + (kind ? ' ' + kind : '');
  }

  function infoIcon(p) {
    const i = el('span', 'info', 'i');
    i.tabIndex = 0;
    const tip = el('span', 'tip');
    tip.append(el('span', '', p.hint || ''));
    const meta = el('span', 'tip-meta', 'Тип значения: ' + (SRC_TEXT[p.src] || p.src) + (p.note ? '. ' + p.note : ''));
    tip.append(meta);
    (p.refs || []).forEach((r) => {
      tip.append(el('span', 'tip-meta', `Источник (${REF_KIND[r.kind]}): ${r.title}. ${r.note || ''}`));
    });
    i.append(tip);
    return i;
  }

  function numberInput(p, value, onChange) {
    const inp = el('input');
    inp.type = 'number';
    if (p.min !== null) inp.min = p.min;
    if (p.max !== null) inp.max = p.max;
    if (p.step !== null) inp.step = p.step;
    inp.value = num(value);
    inp.addEventListener('input', () => onChange(inp.value === '' ? NaN : Number(inp.value)));
    return inp;
  }

  function buildInput(p, current, onChange) {
    const wrap = el('div', 'a-input');
    if (p.kind === 'readonly') {
      wrap.append(el('span', 'a-fixed', show(current)));
    } else if (p.kind === 'bool') {
      const box = el('input');
      box.type = 'checkbox';
      box.checked = !!current;
      const txt = el('span', 'a-mini', current ? 'включён' : 'исключён');
      box.addEventListener('change', () => { txt.textContent = box.checked ? 'включён' : 'исключён'; onChange(box.checked); });
      wrap.append(box, txt);
    } else if (p.kind === 'enum') {
      const sel = el('select');
      p.labels.forEach((o) => { const opt = el('option', '', o); opt.value = o; sel.append(opt); });
      sel.value = current;
      sel.addEventListener('change', () => onChange(sel.value));
      wrap.append(sel);
    } else if (p.kind === 'range' || p.kind === 'vector') {
      const vals = current.slice();
      vals.forEach((x, i) => {
        if (p.kind === 'range') wrap.append(el('span', 'a-mini', i === 0 ? 'от' : 'до'));
        else wrap.append(el('span', 'a-mini', p.labels[i]));
        wrap.append(numberInput(p, x, (n) => { vals[i] = n; onChange(vals.slice()); }));
      });
    } else {
      wrap.append(numberInput(p, current, onChange));
    }
    if (p.unit && p.kind !== 'enum' && p.kind !== 'bool') wrap.append(el('span', 'a-unit', p.unit));
    return wrap;
  }

  function buildRow(p) {
    const row = el('div', 'a-row');
    row.dataset.id = p.id;
    const current = p.id in dirty ? dirty[p.id] : p.value;
    if (p.id in dirty) row.classList.add('edited');
    else if (p.changed) row.classList.add('changed');

    const label = el('div', 'a-label');
    label.append(el('span', '', p.label), infoIcon(p));
    label.append(el('span', 'src src-' + p.src, SRC_TEXT[p.src] || p.src));
    const refs = el('div', 'a-refs');
    refs.append(document.createTextNode('Источник: '));
    (p.refs || []).forEach((r, i) => {
      if (i) refs.append(document.createTextNode('; '));
      const href = r.url || ((r.file || '').startsWith('docs/') ? '/' + encodeURI(r.file) : '');
      if (href) {
        const a = el('a', '', r.short);
        a.href = href; a.target = '_blank'; a.rel = 'noopener'; a.title = r.title;
        refs.append(a);
      } else {
        refs.append(el('span', 'ref-none', r.short));
      }
    });
    label.append(refs);
    row.append(label);

    const input = buildInput(p, current, (val) => {
      if (same(val, p.value) || (Array.isArray(val) ? val.some(Number.isNaN) : Number.isNaN(val))) delete dirty[p.id];
      else dirty[p.id] = val;
      row.classList.toggle('edited', p.id in dirty);
      updateButtons();
    });
    row.append(input);

    const meta = el('div', 'a-meta');
    if (p.kind !== 'readonly') {
      meta.append(el('span', '', 'по умолчанию: ' + show(p.default)));
      if (p.changed) {
        const b = el('button', 'link', 'вернуть');
        b.addEventListener('click', () => send({ reset: [p.id] }));
        meta.append(b);
      }
      meta.append(el('span', 'a-effect', EFFECT_TEXT[p.effect]));
    } else {
      meta.append(el('span', '', 'норматив, не редактируется'));
    }
    row.append(meta, el('div', 'a-err'));
    return row;
  }

  function renderBody() {
    const body = $('aBody');
    body.textContent = '';
    let section = null;
    model.params.filter((p) => p.tab === activeTab).forEach((p) => {
      if (p.section !== section) {
        section = p.section;
        body.append(el('div', 'a-section', section));
      }
      body.append(buildRow(p));
    });
  }

  function renderTabs() {
    const box = $('aTabs');
    box.textContent = '';
    model.tabs.forEach((name) => {
      const n = model.params.filter((p) => p.tab === name && p.changed).length;
      const b = el('button', 'tab' + (name === activeTab ? ' active' : ''), name + (n ? ` (${n})` : ''));
      b.addEventListener('click', () => { activeTab = name; renderTabs(); renderBody(); renderStrip(); });
      box.append(b);
    });
  }

  function updateButtons() {
    const n = Object.keys(dirty).length;
    $('aApply').disabled = busy || n === 0;
    $('aCancel').disabled = busy || n === 0;
    $('aResetAll').disabled = busy || Object.keys(model.overrides).length === 0;
    $('aApply').textContent = n ? `Применить изменения (${n})` : 'Применить изменения';
  }

  function updateBanner() {
    const n = Object.keys(model.overrides).length;
    $('assumptionsBanner').style.display = n ? 'block' : 'none';
    $('assumptionsCount').textContent = n;
    $('aSummary').textContent = 'Параметры модели и приборов' + (n ? ` (изменено: ${n})` : '');
  }

  // краткий результат блендинга над формой: эффект правок виден без прокрутки к карточке
  function renderStrip() {
    const strip = $('aStrip');
    strip.style.display = activeTab === BLEND_TAB ? 'block' : 'none';
    if (activeTab !== BLEND_TAB) return;
    const b = lastCycle && lastCycle.blocks['8. Блендинг'];
    if (!b || !(b['состав'] || []).length) {
      strip.textContent = 'На выбранный момент расчёт блендинга не выполнялся: рекомендация не сформирована.';
      return;
    }
    const recipe = b['состав'].filter((p) => p['доля'] > 0).map((p) => `${p['компонент']} ${pct(p['доля'])}`).join(' · ');
    const doses = (b['дозы'] || []).filter((d) => d['доза, кг/т'] > 0).map((d) => `${d['присадка']} ${d['доза, кг/т']} кг/т`).join(' · ');
    const bad = b['не выполнено'] || [];
    strip.textContent = `Результат на ${lastCycle.t.replace('T', ' ').slice(0, 16)}: режим ${b['режим']}; ${recipe}` +
      `${doses ? '; ' + doses : ''}; ${bad.length ? 'нарушено: ' + bad.join(', ') : 'нормы выполнены'}.`;
  }

  function render() {
    if (!activeTab) activeTab = model.tabs[0];
    renderTabs();
    renderBody();
    renderStrip();
    updateButtons();
    updateBanner();
  }

  function showErrors(errors) {
    Object.entries(errors).forEach(([id, msg]) => {
      const row = document.querySelector(`.a-row[data-id="${CSS.escape(id)}"]`);
      if (row) { row.classList.add('invalid'); row.querySelector('.a-err').textContent = msg; }
    });
    const first = Object.keys(errors)[0];
    const p = model.params.find((x) => x.id === first);
    if (p && p.tab !== activeTab) { activeTab = p.tab; renderTabs(); renderBody(); renderStrip(); showErrors(errors); }
  }

  async function send(payload) {
    if (busy) return;
    busy = true;
    updateButtons();
    const ids = (payload.reset === 'all' ? model.params.filter((p) => p.changed).map((p) => p.id) : (payload.reset || []))
      .concat(Object.keys(payload.changes || {}));
    const needsState = ids.some((id) => (model.params.find((p) => p.id === id) || {}).effect === 'state');
    setStatus(needsState ? 'Пересчёт фильтра серы и рядов состояния…' : 'Применение…', 'busy');
    try {
      const res = await fetch('/api/assumptions', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (data.error) { setStatus('Ошибка: ' + data.error, 'error'); return; }
      if (!data.ok) {
        setStatus('Значения не приняты: проверьте выделенные поля.', 'error');
        showErrors(data.errors);
        return;
      }
      model = data;
      dirty = {};
      render();
      setStatus(`Применено (${data.recomputed ? 'пересчёт ' + data.elapsed_s + ' с' : 'без пересчёта рядов'}).`, 'ok');
      if (typeof loadAt === 'function' && currentT) await loadAt(currentT, true);
    } catch (e) {
      setStatus('Ошибка связи с сервером: ' + e, 'error');
    } finally {
      busy = false;
      if (model) updateButtons();
    }
  }

  async function importFile(file) {
    try {
      const data = JSON.parse(await file.text());
      if (!data.overrides || typeof data.overrides !== 'object') throw new Error('нет поля overrides');
      await send({ reset: 'all', changes: data.overrides });
    } catch (e) {
      setStatus('Файл не распознан: ' + e.message, 'error');
    }
  }

  async function init() {
    model = await (await fetch('/api/assumptions')).json();
    render();
    $('aApply').addEventListener('click', () => send({ changes: dirty }));
    $('aCancel').addEventListener('click', () => { dirty = {}; render(); setStatus(''); });
    $('aResetAll').addEventListener('click', () => send({ reset: 'all' }));
    $('bannerReset').addEventListener('click', () => send({ reset: 'all' }));
    $('aExport').addEventListener('click', () => { window.location = '/api/assumptions/export'; });
    $('aImport').addEventListener('change', (e) => { if (e.target.files[0]) importFile(e.target.files[0]); e.target.value = ''; });
    $('openAssumptions').addEventListener('click', () => {
      $('assumptionsCard').open = true;
      $('assumptionsCard').scrollIntoView({ behavior: 'smooth' });
    });
    $('bannerOpen').addEventListener('click', () => $('openAssumptions').click());
  }

  // открыть панель на указанной вкладке (используется кнопкой в карточке блендинга)
  window.onCycleRendered = (data) => { lastCycle = data; if (model) renderStrip(); };

  window.openAssumptionsTab = (tab) => {
    if (model && tab) { activeTab = tab; render(); }
    $('assumptionsCard').open = true;
    $('assumptionsCard').scrollIntoView({ behavior: 'smooth' });
  };

  init().catch((e) => setStatus('Не удалось загрузить параметры: ' + e, 'error'));
})();
