/* Shared behaviour for every page. No dependencies.
   All numbers below are copied from the thesis tables (Chapter 4). */
(function () {
  'use strict';
  var SVGNS = 'http://www.w3.org/2000/svg';

  function $(sel, root) { return (root || document).querySelector(sel); }
  function $$(sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); }
  function el(tag, attrs, parent) {
    var e = document.createElementNS(SVGNS, tag);
    for (var k in attrs) if (attrs[k] !== undefined) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function txt(parent, x, y, s, cls, anchor) {
    var t = el('text', { x: x, y: y, 'class': cls || '', 'text-anchor': anchor || 'start' }, parent);
    t.textContent = s;
    return t;
  }
  function pressGroup(buttons, active) {
    buttons.forEach(function (b) { b.setAttribute('aria-pressed', b === active ? 'true' : 'false'); });
  }

  /* ---------------- Theme toggle ---------------- */
  function initTheme() {
    var btn = $('.theme-toggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var root = document.documentElement;
      var current = root.getAttribute('data-theme') ||
        (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      var next = current === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
    });
  }

  /* ---------------- Mobile menu ---------------- */
  function initMenu() {
    var btn = $('.menu-toggle'), side = $('.sidebar');
    if (!btn || !side) return;
    btn.addEventListener('click', function () {
      var open = side.classList.toggle('open');
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      btn.textContent = open ? 'Close' : 'Menu';
    });
  }

  /* ---------------- Lightbox for figures ---------------- */
  function initLightbox() {
    $$('.plate').forEach(function (plate) {
      plate.addEventListener('click', function (ev) {
        ev.preventDefault();
        var img = $('img', plate);
        if (!img) return;
        var box = document.createElement('div');
        box.className = 'lightbox';
        box.setAttribute('role', 'dialog');
        box.setAttribute('aria-label', img.alt || 'Figure');
        box.innerHTML = '<div class="lb-inner"><img alt=""></div><button class="lb-close" type="button" aria-label="Close">×</button>';
        $('img', box).src = img.currentSrc || img.src;
        $('img', box).alt = img.alt;
        function close() { box.remove(); document.removeEventListener('keydown', onKey); plate.focus(); }
        function onKey(e) { if (e.key === 'Escape') close(); }
        box.addEventListener('click', close);
        document.addEventListener('keydown', onKey);
        document.body.appendChild(box);
        $('.lb-close', box).focus();
      });
    });
  }

  /* ---------------- Copy buttons ---------------- */
  function initCopy() {
    $$('.copy-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var target = document.getElementById(btn.getAttribute('data-copy'));
        if (!target) return;
        var text = target.textContent;
        function done(msg) { btn.textContent = msg; setTimeout(function () { btn.textContent = 'Copy'; }, 1600); }
        function selectIt() {
          var r = document.createRange(); r.selectNodeContents(target);
          var s = window.getSelection(); s.removeAllRanges(); s.addRange(r);
          done('Selected');
        }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () { done('Copied'); }, selectIt);
        } else { selectIt(); }
      });
    });
  }

  /* ---------------- Home: specimen viewer ---------------- */
  function initSpecimen() {
    var card = $('.specimen');
    if (!card) return;
    var modes = $$('.specimen-foot button', card);
    var state = $$('.specimen-state button', card);
    var imgRgb = $('.layer-rgb', card), imgIr = $('.layer-ir', card);
    var chan = $('.chan', card), gasPanel = $('.specimen-gaslist', card), view = $('.specimen-view', card);
    var label = $('.specimen-label', card);
    var cls = 'notspoiled', mode = 'rgb', timer = null;

    function render() {
      imgRgb.src = 'assets/dataset/banana_' + cls + '_rgb.jpg';
      imgIr.src = 'assets/dataset/banana_' + cls + '_ir.jpg';
      card.setAttribute('data-mode', mode === 'ir' ? 'ir' : 'rgb');
      chan.textContent = mode === 'ir' ? 'Thermal camera' : 'sRGB camera';
      view.hidden = mode === 'gas';
      gasPanel.hidden = mode !== 'gas';
      label.textContent = cls === 'spoiled' ? 'spoiled' : 'not spoiled';
      pressGroup(modes, modes.filter(function (b) { return b.dataset.mode === mode; })[0]);
      pressGroup(state, state.filter(function (b) { return b.dataset.cls === cls; })[0]);
    }
    function stop() { if (timer) { clearInterval(timer); timer = null; } }
    modes.forEach(function (b) { b.addEventListener('click', function () { stop(); mode = b.dataset.mode; render(); }); });
    state.forEach(function (b) { b.addEventListener('click', function () { stop(); cls = b.dataset.cls; render(); }); });
    render();
    var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (!reduce) {
      var seq = ['rgb', 'ir'], i = 0;
      timer = setInterval(function () { i = (i + 1) % seq.length; mode = seq[i]; render(); }, 2600);
    }
  }

  /* ---------------- Comparison slider (sRGB | thermal) ---------------- */
  function bindCompare(box) {
    var range = $('input[type="range"]', box);
    if (!range || range._bound) return;
    range._bound = true;
    function set() { box.style.setProperty('--cut', range.value + '%'); }
    range.addEventListener('input', set);
    set();
  }

  /* ---------------- Dataset gallery ---------------- */
  var FRUITS = [
    { id: 'banana', name: 'Banana', days: 7, ns: 3, sp: 4, tr: 3, va: 2, te: 2, mean: 1.3, max: 4,
      note: 'Peel darkens from yellow with brown flecks to almost uniformly black. One of the fastest-spoiling items in TR-6.' },
    { id: 'carrot', name: 'Carrot', days: 9, ns: 2, sp: 7, tr: 6, va: 2, te: 1, mean: 0.6, max: 3,
      note: 'Only two not-spoiled days were recorded, so that stratum has no test day. Spoilage shows as dark, shrivelled patches.' },
    { id: 'guava', name: 'Guava', days: 10, ns: 4, sp: 6, tr: 6, va: 2, te: 2, mean: 1.4, max: 5,
      note: 'Brown lesions spread across the skin. Its not-spoiled test trajectory was one of the three false positives.' },
    { id: 'indian_gooseberry', name: 'Indian gooseberry', days: 30, ns: 18, sp: 12, tr: 21, va: 5, te: 4, mean: 6.7, max: 21,
      note: 'Second-largest share of the data, yet the hardest produce to generalise to in leave-one-fruit-out testing.' },
    { id: 'mango', name: 'Mango', days: 8, ns: 5, sp: 3, tr: 4, va: 2, te: 2, mean: 3.8, max: 9,
      note: 'Both of its test trajectories were classified correctly by the ensemble.' },
    { id: 'tomato', name: 'Tomato', days: 48, ns: 44, sp: 4, tr: 33, va: 8, te: 7, mean: 31.3, max: 65,
      note: 'Lasts by far the longest: up to 65 days before spoiling. This difference in timescale is the main reason the regression task is hard.' }
  ];

  function classCard(f, cls) {
    var spoiled = cls === 'spoiled';
    var base = 'assets/dataset/' + f.id + '_' + cls;
    var card = document.createElement('div');
    card.className = 'class-card';
    card.innerHTML =
      '<div class="class-card-head"><span>' + f.name + '</span><span class="label-chip ' + (spoiled ? 'spoiled' : 'fresh') + '">' +
      (spoiled ? 'Spoiled' : 'Not spoiled') + '</span></div>' +
      '<div class="compare" style="--cut:50%">' +
      '<img class="under bg" alt="" aria-hidden="true" src="' + base + '_rgb.jpg">' +
      '<img class="under" alt="' + f.name + ', ' + (spoiled ? 'spoiled' : 'not spoiled') + ', sRGB image" src="' + base + '_rgb.jpg">' +
      '<img class="over bg" alt="" aria-hidden="true" src="' + base + '_ir.jpg">' +
      '<img class="over" alt="' + f.name + ', ' + (spoiled ? 'spoiled' : 'not spoiled') + ', thermal image" src="' + base + '_ir.jpg">' +
      '<span class="handle"></span><span class="cap l">sRGB</span><span class="cap r">Thermal</span>' +
      '<input type="range" min="0" max="100" value="50" aria-label="Slide between sRGB and thermal image">' +
      '</div>' +
      '<div class="class-card-foot">' + (spoiled ? f.sp : f.ns) + ' day-level entries in this class. Drag across the image to compare the two cameras.</div>';
    var box = $('.compare', card), missing = 0;
    $$('img', box).forEach(function (img) {
      img.addEventListener('error', function () {
        img.hidden = true; missing++;
        if (missing === 1) {
          var ph = document.createElement('div');
          ph.className = 'compare-empty';
          ph.textContent = 'Sample image for this class not added yet. Run tools/export_dataset_samples.py to export it from TR-6.';
          box.appendChild(ph);
          $('.handle', box).hidden = true; $$('.cap', box).forEach(function (c) { c.hidden = true; });
          $('input', box).hidden = true;
        }
      });
    });
    bindCompare(box);
    return card;
  }

  function initGallery() {
    var root = $('#gallery');
    if (!root) return;
    var tabs = $('.gallery-tabs', root), pair = $('.class-pair', root), facts = $('.fruit-facts', root), note = $('.fruit-note', root);
    var buttons = FRUITS.map(function (f) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = f.name; b.dataset.id = f.id;
      tabs.appendChild(b);
      b.addEventListener('click', function () { show(f.id, true); });
      return b;
    });
    function show(id, scroll) {
      var f = FRUITS.filter(function (x) { return x.id === id; })[0] || FRUITS[0];
      pressGroup(buttons, buttons.filter(function (b) { return b.dataset.id === f.id; })[0]);
      pair.innerHTML = '';
      pair.appendChild(classCard(f, 'notspoiled'));
      pair.appendChild(classCard(f, 'spoiled'));
      facts.innerHTML =
        '<div><b>' + f.days + '</b><span>day-level entries</span></div>' +
        '<div><b>' + f.ns + ' / ' + f.sp + '</b><span>not spoiled / spoiled days</span></div>' +
        '<div><b>' + f.tr + ' · ' + f.va + ' · ' + f.te + '</b><span>train · val · test days</span></div>' +
        '<div><b>' + f.mean + ' <small>/ ' + f.max + '</small></b><span>mean / max days to spoilage</span></div>';
      note.textContent = f.note;
      if (scroll) root.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    $$('.contact button').forEach(function (b) { b.addEventListener('click', function () { show(b.dataset.id, true); }); });
    var start = (location.hash || '').replace('#', '');
    show(FRUITS.some(function (f) { return f.id === start; }) ? start : 'banana', false);
  }

  /* ---------------- Robustness explorer ---------------- */
  var LEVELS = [0, 10, 20, 30, 40, 50];
  var SWEEP = {
    val: {
      rgb: { full: [0.6934, 0.6770, 0.6763, 0.6276, 0.6120, 0.6008], early: [0.6370, 0.6391, 0.6378, 0.6397, 0.6344, 0.6260], late: [0.6048, 0.5989, 0.5955, 0.5991, 0.5816, 0.5830] },
      ir: { full: [0.6934, 0.6887, 0.6921, 0.6821, 0.6853, 0.6845], early: [0.6370, 0.6378, 0.6282, 0.6338, 0.6239, 0.6354], late: [0.6048, 0.6190, 0.6250, 0.6345, 0.6324, 0.6344] },
      gas: { full: [0.6934, 0.6881, 0.6969, 0.7082, 0.6857, 0.7004], early: [0.6370, 0.6413, 0.6268, 0.6260, 0.6250, 0.6234], late: [0.6048, 0.6036, 0.5968, 0.5962, 0.5897, 0.5881] },
      joint: { full: [0.6934, 0.6825, 0.6698, 0.6302, 0.6178, 0.6147], early: [0.6370, 0.6272, 0.6165, 0.5979, 0.5959, 0.5701], late: [0.6048, 0.6116, 0.6298, 0.6052, 0.6208, 0.5975] }
    },
    test: {
      rgb: { full: [0.6757, 0.6613, 0.6517, 0.6375, 0.5947, 0.5872], early: [0.6553, 0.6531, 0.6475, 0.6377, 0.6285, 0.6256], late: [0.6117, 0.5989, 0.6064, 0.6017, 0.5748, 0.5857] },
      ir: { full: [0.6757, 0.6694, 0.6685, 0.6811, 0.6738, 0.6735], early: [0.6553, 0.6508, 0.6446, 0.6345, 0.6188, 0.6218], late: [0.6117, 0.6211, 0.6448, 0.6269, 0.6434, 0.6422] },
      gas: { full: [0.6757, 0.6794, 0.6733, 0.6764, 0.6831, 0.6764], early: [0.6553, 0.6488, 0.6509, 0.6503, 0.6441, 0.6448], late: [0.6117, 0.6017, 0.6031, 0.6068, 0.6062, 0.5941] },
      joint: { full: [0.6757, 0.6703, 0.6511, 0.6219, 0.6083, 0.6040], early: [0.6553, 0.6291, 0.6165, 0.6097, 0.5759, 0.5789], late: [0.6117, 0.6127, 0.5956, 0.6117, 0.6084, 0.6080] }
    }
  };
  /* Bootstrap tests at 50% missingness. sig = survives Benjamini-Hochberg correction. */
  var SIG = {
    val: {
      rgb: [['early', '−0.025', 'p = 0.359', 'ns'], ['late', '+0.018', 'p = 0.629', 'ns']],
      ir: [['early', '+0.049', 'p = 0.002', 'sig'], ['late', '+0.050', 'p = 0.003', 'sig']],
      gas: [['early', '+0.077', 'p < 0.001', 'sig'], ['late', '+0.112', 'p < 0.001', 'sig']],
      joint: [['early', '+0.045', 'p = 0.148', 'ns'], ['late', '+0.017', 'p = 0.539', 'ns']]
    },
    test: {
      rgb: [['early', '−0.038', 'p = 0.249', 'ns'], ['late', '+0.002', 'p = 0.959', 'ns']],
      ir: [['early', '+0.052', 'p = 0.034', 'nom'], ['late', '+0.031', 'p = 0.109', 'ns']],
      gas: [['early', '+0.032', 'p = 0.201', 'ns'], ['late', '+0.082', 'p = 0.002', 'sig']],
      joint: [['early', '+0.025', 'p = 0.476', 'ns'], ['late', '−0.004', 'p = 0.883', 'ns']]
    }
  };
  var PATTERN_NAME = { rgb: 'sRGB camera', ir: 'thermal camera', gas: 'gas sensor', joint: 'all three sensors' };
  var MODEL_NAME = { full: 'Proposed model', early: 'Early fusion', late: 'Late fusion' };

  function drawLineChart(host, data, opts) {
    host.innerHTML = '';
    var W = 640, H = 300, m = { l: 48, r: 16, t: 14, b: 40 };
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': opts.label }, host);
    var y0 = 0.55, y1 = 0.72;
    function X(i) { return m.l + (W - m.l - m.r) * i / (LEVELS.length - 1); }
    function Y(v) { return m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0)); }
    var grid = el('g', { 'class': 'grid' }, svg);
    [0.56, 0.60, 0.64, 0.68, 0.72].forEach(function (v) {
      el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, grid);
      txt(svg, m.l - 8, Y(v) + 4, v.toFixed(2), '', 'end');
    });
    LEVELS.forEach(function (l, i) { txt(svg, X(i), H - m.b + 18, l + '%', '', 'middle'); });
    txt(svg, (m.l + W - m.r) / 2, H - 4, 'Share of days with the ' + PATTERN_NAME[opts.pattern] + ' switched off', '', 'middle');
    var yl = txt(svg, 12, (m.t + H - m.b) / 2, 'F1-score', '', 'middle');
    yl.setAttribute('transform', 'rotate(-90 12 ' + ((m.t + H - m.b) / 2) + ')');
    el('line', { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, 'class': 'zero', style: 'stroke-dasharray:none' }, svg);
    ['late', 'early', 'full'].forEach(function (k) {
      var d = data[k].map(function (v, i) { return (i ? 'L' : 'M') + X(i).toFixed(1) + ',' + Y(v).toFixed(1); }).join(' ');
      el('path', { d: d, 'class': 's-' + k }, svg);
      data[k].forEach(function (v, i) { el('circle', { cx: X(i), cy: Y(v), r: k === 'full' ? 3.6 : 2.8, 'class': 'd-' + k }, svg); });
    });
    // hover columns
    var cursor = el('line', { x1: 0, x2: 0, y1: m.t, y2: H - m.b, stroke: 'currentColor', 'stroke-opacity': 0, 'class': 'zero' }, svg);
    LEVELS.forEach(function (l, i) {
      var hit = el('rect', { x: X(i) - (W - m.l - m.r) / 10, y: m.t, width: (W - m.l - m.r) / 5, height: H - m.t - m.b, fill: 'transparent' }, svg);
      function on() {
        cursor.setAttribute('x1', X(i)); cursor.setAttribute('x2', X(i)); cursor.setAttribute('stroke-opacity', 1);
        opts.readout.innerHTML = '<b>' + l + '% missing</b> · Proposed ' + data.full[i].toFixed(3) +
          ' · Early fusion ' + data.early[i].toFixed(3) + ' · Late fusion ' + data.late[i].toFixed(3);
      }
      hit.addEventListener('mouseenter', on);
      hit.addEventListener('click', on);
    });
    svg.addEventListener('mouseleave', function () { cursor.setAttribute('stroke-opacity', 0); opts.readout.innerHTML = opts.restText; });
  }

  function initRobustness() {
    var root = $('#robustness-explorer');
    if (!root) return;
    var pBtns = $$('[data-pattern]', root), sBtns = $$('[data-split]', root);
    var host = $('.chart', root), readout = $('.chart-readout', root), sigBox = $('.sig-box', root);
    var pattern = 'gas', split = 'val';
    function render() {
      pressGroup(pBtns, pBtns.filter(function (b) { return b.dataset.pattern === pattern; })[0]);
      pressGroup(sBtns, sBtns.filter(function (b) { return b.dataset.split === split; })[0]);
      var d = SWEEP[split][pattern];
      var drop = function (a) { return ((a[5] - a[0]) / a[0] * 100); };
      var rest = 'At 50% missing: proposed ' + d.full[5].toFixed(3) + ' (' + fmtPct(drop(d.full)) + ' from its 0% value), early fusion ' +
        d.early[5].toFixed(3) + ' (' + fmtPct(drop(d.early)) + '), late fusion ' + d.late[5].toFixed(3) + ' (' + fmtPct(drop(d.late)) + ').';
      drawLineChart(host, d, { pattern: pattern, readout: readout, restText: rest, label: 'F1 against missingness for ' + PATTERN_NAME[pattern] });
      readout.innerHTML = rest;
      animateChart(host);
      sigBox.innerHTML = SIG[split][pattern].map(function (r) {
        var pill = r[3] === 'sig' ? '<span class="pill sig">significant after correction</span>' :
          r[3] === 'nom' ? '<span class="pill nom">nominal only, not after correction</span>' : '<span class="pill ns">not significant</span>';
        return '<div>vs. ' + MODEL_NAME[r[0]].toLowerCase() + ': <span class="mono">' + r[1] + ' F1, ' + r[2] + '</span> ' + pill + '</div>';
      }).join('');
    }
    function fmtPct(v) { return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + '%'; }
    pBtns.forEach(function (b) { b.addEventListener('click', function () { pattern = b.dataset.pattern; render(); }); });
    sBtns.forEach(function (b) { b.addEventListener('click', function () { split = b.dataset.split; render(); }); });
    render();
  }

  /* ---------------- Ablation forest plot ---------------- */
  var ABL = [
    ['Mean imputation instead of GRU-D', 0.109, -0.19, 0.45, 'p = 0.514'],
    ['LSTM instead of Transformer', 0.057, -0.14, 0.27, 'p = 0.766'],
    ['No CBAM', -0.050, -0.17, 0.00, 'p = 0.711'],
    ['Early fusion', -0.094, -0.25, 0.00, 'p = 0.230'],
    ['Late fusion', -0.133, -0.32, 0.00, 'p = 0.064'],
    ['No cross-modal attention', -0.133, -0.32, 0.00, 'p = 0.064']
  ];
  function initAblation() {
    var host = $('#ablation-chart');
    if (!host) return;
    var W = 640, rowH = 34, m = { l: 210, r: 80, t: 16, b: 40 }, H = m.t + m.b + rowH * ABL.length;
    var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Ablation: change in ensemble F1 with 95% bootstrap intervals' }, host);
    var x0 = -0.35, x1 = 0.5;
    function X(v) { return m.l + (W - m.l - m.r) * (v - x0) / (x1 - x0); }
    var g = el('g', { 'class': 'grid' }, svg);
    [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5].forEach(function (v) {
      el('line', { x1: X(v), x2: X(v), y1: m.t, y2: H - m.b }, g);
      txt(svg, X(v), H - m.b + 16, (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(1), '', 'middle');
    });
    el('line', { x1: X(0), x2: X(0), y1: m.t - 6, y2: H - m.b, 'class': 'zero' }, svg);
    txt(svg, (m.l + W - m.r) / 2, H - 6, 'Change in ensemble F1 vs. the full model (95% bootstrap CI)', '', 'middle');
    ABL.forEach(function (r, i) {
      var y = m.t + rowH * i + rowH / 2;
      txt(svg, m.l - 12, y + 4, r[0], 'lbl', 'end');
      el('line', { x1: X(r[2]), x2: X(r[3]), y1: y, y2: y, 'class': 'ci' }, svg);
      el('line', { x1: X(r[2]), x2: X(r[2]), y1: y - 5, y2: y + 5, 'class': 'ci' }, svg);
      el('line', { x1: X(r[3]), x2: X(r[3]), y1: y - 5, y2: y + 5, 'class': 'ci' }, svg);
      el('circle', { cx: X(r[1]), cy: y, r: 5.5, 'class': 'pt ' + (r[1] > 0 ? 'pos' : 'neg'), style: '--i:' + i }, svg);
      txt(svg, W - m.r + 10, y + 4, r[4], '', 'start');
    });
  }

  /* ---------------- Backbone scatter ---------------- */
  var BB = [
    ['ConvNeXt-Tiny', 0.7912, 32.47, 39.03, 38.0, 'chosen'],
    ['Swin-Tiny', 0.7887, 32.17, 27.14, 45.2, ''],
    ['EfficientNet-B0', 0.7701, 8.83, 6.63, 19.0, ''],
    ['RegNetY-1.6GF', 0.7469, 15.00, 16.48, 23.9, ''],
    ['DINOv2 ViT-S/14 (frozen)', 0.7467, 26.63, 47.52, 40.0, 'hl'],
    ['ResNet-18', 0.7414, 15.77, 17.91, 17.2, ''],
    ['MobileNetV3-Small', 0.7388, 5.54, 3.81, 14.3, ''],
    ['EfficientNetV2-S', 0.7362, 25.00, 26.53, 30.0, '']
  ];
  function initBackbone() {
    var root = $('#backbone-explorer');
    if (!root) return;
    var btns = $$('[data-x]', root), host = $('.chart', root), readout = $('.chart-readout', root);
    var AX = { flops: [3, 'GFLOPs of the complete model', 0, 50], params: [2, 'Parameters of the complete model (millions)', 0, 35], lat: [4, 'Inference latency per batch (ms)', 10, 50] };
    var mode = 'flops';
    var rest = 'Hover or tap a point for its numbers.';
    function render() {
      pressGroup(btns, btns.filter(function (b) { return b.dataset.x === mode; })[0]);
      host.innerHTML = '';
      var W = 640, H = 320, m = { l: 52, r: 24, t: 16, b: 44 };
      var a = AX[mode], x0 = a[2], x1 = a[3], y0 = 0.73, y1 = 0.80;
      function X(v) { return m.l + (W - m.l - m.r) * (v - x0) / (x1 - x0); }
      function Y(v) { return m.t + (H - m.t - m.b) * (1 - (v - y0) / (y1 - y0)); }
      var svg = el('svg', { viewBox: '0 0 ' + W + ' ' + H, role: 'img', 'aria-label': 'Validation F1 against ' + a[1] }, host);
      var g = el('g', { 'class': 'grid' }, svg);
      [0.73, 0.75, 0.77, 0.79].forEach(function (v) { el('line', { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g); txt(svg, m.l - 8, Y(v) + 4, v.toFixed(2), '', 'end'); });
      var step = (x1 - x0) / 5;
      for (var t = x0; t <= x1 + 1e-9; t += step) { el('line', { x1: X(t), x2: X(t), y1: m.t, y2: H - m.b }, g); txt(svg, X(t), H - m.b + 16, String(Math.round(t)), '', 'middle'); }
      txt(svg, (m.l + W - m.r) / 2, H - 6, a[1], '', 'middle');
      var yl = txt(svg, 12, (m.t + H - m.b) / 2, 'Mean validation F1 (10 seeds)', '', 'middle');
      yl.setAttribute('transform', 'rotate(-90 12 ' + ((m.t + H - m.b) / 2) + ')');
      BB.forEach(function (b) {
        var cx = X(b[a[0]]), cy = Y(b[1]);
        var c = el('circle', { cx: cx, cy: cy, r: 8, 'class': 'bub ' + b[5], tabindex: 0 }, svg);
        var right = cx < W - 170;
        if (b[5] || b[0] === 'EfficientNet-B0' || b[0] === 'Swin-Tiny') txt(svg, cx + (right ? 12 : -12), cy - 10, b[0], 'lbl', right ? 'start' : 'end');
        function on() { readout.innerHTML = '<b>' + b[0] + '</b> · F1 ' + b[1].toFixed(4) + ' · ' + b[2] + 'M params · ' + b[3] + ' GFLOPs · ' + b[4] + ' ms'; }
        c.addEventListener('mouseenter', on); c.addEventListener('focus', on); c.addEventListener('click', on);
      });
      svg.addEventListener('mouseleave', function () { readout.innerHTML = rest; });
      readout.innerHTML = rest;
    }
    btns.forEach(function (b) { b.addEventListener('click', function () { mode = b.dataset.x; render(); }); });
    render();
  }

  /* ---------------- Architecture explorer ---------------- */
  var ARCH = {
    rgbin: ['sRGB image', 'One colour photograph per day, resized to 224 × 224 and normalised with ImageNet statistics. Captures surface colour, browning, spots and texture.', 'input   3 × 224 × 224 per day\nsource  first available session image'],
    irin: ['Thermal image', 'One thermal (infrared fusion) image per day at 224 × 224. Surface temperature patterns change with respiration and microbial activity, sometimes before anything is visible.', 'input   3 × 224 × 224 per day'],
    gasin: ['Methane readings', 'Six features per day: mean concentration, its standard deviation, left and right sensor readings (ppm), their absolute difference, and the session. Readings can be missing on some days.', 'x_t     6 features\nm_t     1 if a reading exists\nδ_t     days since last reading'],
    rgbenc: ['ConvNeXt-Tiny + CBAM', 'ImageNet-pretrained ConvNeXt-Tiny, refined by a Convolutional Block Attention Module that reweights channels and spatial regions. Chosen from eight candidate backbones.', 'features 768 → global average pool\nCBAM     reduction 16, 7×7 spatial kernel\nunfreeze last block @6, last two @8, all @12'],
    irenc: ['EfficientNet-B0 + CBAM', 'A lightweight pretrained CNN for the thermal stream. Its backbone learns ten times slower (1e-5) because thermal images look nothing like the photographs it was pretrained on.', 'features 1280 → global average pool\nlr       1e-5 (rest of model 1e-4)'],
    gasenc: ['GRU-D encoder', 'A recurrent network built for irregular, gappy time series. Learned decay gates decide how fast a stale reading should fade toward the training mean, and the missing-value mask is itself an input.', 'γ_x = exp(−max(0, Wδ_t + b))\nx̂_t = m⊙x + (1−m)⊙(γx_{t−1} + (1−γ)x̄)\nhidden 64'],
    fusion: ['Mask-gated cross-modal attention', 'Each day becomes three tokens, one per sensor. Multi-head attention lets them inform each other, and a key-padding mask removes any sensor that is missing that day, so nothing is filled in with a placeholder.', 'tokens   3 × 64, 4 heads\nmask     ¬ available\nz_t      mean over available tokens'],
    temporal: ['Temporal Transformer', 'Two pre-norm Transformer layers with sinusoidal position encoding read the sequence of daily fused vectors. The output at the last real day summarises the whole trajectory.', 'layers 2 · heads 4 · FFN 512\npadding masked'],
    cls: ['Classification head', 'Spoiled or not spoiled. Weighted cross-entropy (2.3 on the spoiled class), with the decision threshold calibrated by 6-fold median on training + validation data: 0.660.', 'test F1  0.80 · AUC 0.73'],
    reg: ['Regression head', 'Days until spoilage, trained with a Huber loss. It stays close to predicting the mean (R² = 0.04), mostly because tomato lasts up to 65 days while banana and carrot last a few.', 'test MAE 0.92 days · R² 0.04'],
    recon: ['Gas reconstruction head', 'An auxiliary task that predicts the day’s six gas features from the sRGB and thermal embeddings, only where a real reading exists. It pushes the image features to carry chemical information.', 'loss  MSE on observed readings only'],
    loss: ['Uncertainty-weighted loss', 'Each task gets a learned uncertainty σ. Noisier tasks are automatically down-weighted, so no loss weights were tuned by hand.', 'L = Σ ½·exp(−2 log σ_i)·L_i + log σ_i']
  };
  function initArch() {
    var panel = $('#arch-panel');
    if (!panel) return;
    var nodes = $$('.arch-node');
    function show(id) {
      var a = ARCH[id];
      nodes.forEach(function (n) { n.classList.toggle('on', n.dataset.node === id); });
      panel.innerHTML = '<p class="kicker">Selected block</p><h3>' + a[0] + '</h3><p>' + a[1] + '</p><div class="spec">' + a[2] + '</div>';
    }
    nodes.forEach(function (n) {
      n.setAttribute('tabindex', '0'); n.setAttribute('role', 'button');
      n.addEventListener('click', function () { show(n.dataset.node); });
      n.addEventListener('keydown', function (e) { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); show(n.dataset.node); } });
    });
    show('fusion');
  }


  /* ---------------- Produce colours in running text ---------------- */
  var FRUIT_RE = /\b(Indian gooseberr(?:y|ies)|bananas?|carrots?|guavas?|mango(?:es|s)?|tomato(?:es)?)\b/gi;
  function fruitClass(word) {
    var w = word.toLowerCase();
    if (w.indexOf('gooseberr') >= 0) return 'gooseberry';
    if (w.indexOf('banana') === 0) return 'banana';
    if (w.indexOf('carrot') === 0) return 'carrot';
    if (w.indexOf('guava') === 0) return 'guava';
    if (w.indexOf('mango') === 0) return 'mango';
    return 'tomato';
  }
  function colourFruits() {
    var main = $('main');
    if (!main) return;
    var skip = 'svg, button, script, style, code, pre, .contact, .gallery-tabs, .fruit, .bar-row, .lofo-row, .class-card-head, input, .spec, thead, .chan';
    var walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {
      acceptNode: function (n) {
        if (!n.nodeValue || !FRUIT_RE.test(n.nodeValue)) { FRUIT_RE.lastIndex = 0; return NodeFilter.FILTER_REJECT; }
        FRUIT_RE.lastIndex = 0;
        return n.parentElement && n.parentElement.closest(skip) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      }
    });
    var nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(function (n) {
      var frag = document.createDocumentFragment(), text = n.nodeValue, last = 0, m;
      FRUIT_RE.lastIndex = 0;
      while ((m = FRUIT_RE.exec(text))) {
        frag.appendChild(document.createTextNode(text.slice(last, m.index)));
        var sp = document.createElement('span');
        sp.className = 'fruit f-' + fruitClass(m[0]);
        sp.textContent = m[0];
        frag.appendChild(sp);
        last = m.index + m[0].length;
      }
      frag.appendChild(document.createTextNode(text.slice(last)));
      n.parentNode.replaceChild(frag, n);
    });
    $$('.bar-row .bn, .lofo-row .ln, .contact .cn').forEach(function (e) {
      var m = e.textContent.match(FRUIT_RE); FRUIT_RE.lastIndex = 0;
      if (m) e.style.setProperty('--fc', 'var(--f-' + fruitClass(m[0]) + ')');
    });
  }

  /* ---------------- Scroll-in animations ----------------
     Elements already on screen at load are left alone, so the
     page is complete at rest; the rest replay their arrival. */
  var reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  function onArrive(selector, fn) {
    var els = $$(selector);
    if (!els.length || reduceMotion || !('IntersectionObserver' in window)) return;
    var vh = window.innerHeight;
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { if (en.isIntersecting) { io.unobserve(en.target); fn(en.target); } });
    }, { threshold: 0.2 });
    els.forEach(function (e) { if (e.getBoundingClientRect().top > vh * 0.9) io.observe(e); });
  }
  function countUp(el) {
    var final = el.textContent.trim();
    var m = final.match(/^([\d,]*\.?\d+)/);
    if (!m) return;
    var raw = m[1], num = parseFloat(raw.replace(/,/g, '')), dec = (raw.split('.')[1] || '').length, rest = final.slice(raw.length);
    var small = el.querySelector('small'), smallHTML = small ? small.outerHTML : '';
    var t0 = null, dur = 900;
    function fmt(v) {
      var s = v.toFixed(dec);
      if (raw.indexOf(',') >= 0) s = Math.round(v).toLocaleString('en-US');
      return s;
    }
    function step(ts) {
      if (!t0) t0 = ts;
      var k = Math.min(1, (ts - t0) / dur), e = 1 - Math.pow(1 - k, 3);
      el.innerHTML = fmt(num * e) + (smallHTML ? smallHTML : rest);
      if (k < 1) requestAnimationFrame(step); else el.innerHTML = fmt(num) + (smallHTML ? smallHTML : rest);
    }
    requestAnimationFrame(step);
  }
  function initMotion() {
    onArrive('.callout, figure.fig, .chart-card, .bars, .cm, .lofo, .dropout, .versus', function (el) { el.classList.add('in'); });
    $$('.cm .c').forEach(function (c, i) { c.style.setProperty('--i', i); });
    $$('.lofo-row').forEach(function (r) { $$('i.ok', r).forEach(function (d, i) { d.style.setProperty('--i', i); }); });
    onArrive('.metric .value', countUp);
    onArrive('#ablation-chart', function (el) { el.classList.add('drawn'); });
  }
  function animateChart(host) {
    if (reduceMotion) return;
    $$('path.s-full', host).forEach(function (p) { try { host.style.setProperty('--len', Math.ceil(p.getTotalLength())); } catch (e) {} });
    $$('circle.pt', host).forEach(function (c, i) { c.style.setProperty('--i', i); });
    host.classList.remove('drawn'); void host.offsetWidth; host.classList.add('drawn');
  }

  /* ---------------- Home: pull a sensor ---------------- */
  function initDropout() {
    var root = $('#dropout');
    if (!root) return;
    var btns = $$('.sensor', root), range = $('input', root), out = $('output', root), note = $('.race-note', root);
    var rows = { full: $('.race-row.full', root), early: $('.race-row.early', root), late: $('.race-row.late', root) };
    var pattern = 'gas', touched = false;
    var lo = 0.5, hi = 0.75;
    function render() {
      var i = Math.round(range.value / 10), d = SWEEP.val[pattern];
      out.textContent = range.value + '%';
      pressGroup(btns, btns.filter(function (b) { return b.dataset.pattern === pattern; })[0]);
      var best = 'full', bestV = -1;
      ['full', 'early', 'late'].forEach(function (k) {
        var v = d[k][i];
        $('.race-track span', rows[k]).style.width = Math.max(0, (v - lo) / (hi - lo) * 100).toFixed(1) + '%';
        $('.rv', rows[k]).textContent = v.toFixed(3);
        if (v > bestV) { bestV = v; best = k; }
      });
      ['full', 'early', 'late'].forEach(function (k) { rows[k].classList.toggle('lead', k === best); });
      var drop = ((d.full[i] - d.full[0]) / d.full[0] * 100);
      var who = PATTERN_NAME[pattern];
      if (i === 0) note.textContent = 'All sensors working. Drag the slider to switch the ' + who + ' off on more and more days.';
      else note.textContent = 'With the ' + who + ' off on ' + range.value + '% of days, the proposed model ' +
        (best === 'full' ? 'still leads' : 'falls behind ' + MODEL_NAME[best].toLowerCase()) + ' (' + (drop >= 0 ? '+' : '−') + Math.abs(drop).toFixed(1) + '% from its own starting F1).';
    }
    btns.forEach(function (b) { b.addEventListener('click', function () { touched = true; pattern = b.dataset.pattern; render(); }); });
    range.addEventListener('input', function () { touched = true; render(); });
    render();
    // A short demo run the first time the graphic scrolls into view
    if (!reduceMotion && 'IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (en) {
        if (!en[0].isIntersecting) return;
        io.disconnect();
        var v = 0;
        var t = setInterval(function () {
          if (touched) { clearInterval(t); return; }
          v += 10; range.value = v; render();
          if (v >= 50) clearInterval(t);
        }, 650);
      }, { threshold: 0.5 });
      io.observe(root);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    initMenu(); initLightbox(); initCopy(); initSpecimen(); initGallery();
    initRobustness(); initAblation(); initBackbone(); initArch();
    $$('.compare').forEach(bindCompare);
    initDropout(); colourFruits(); initMotion();
  });
})();
