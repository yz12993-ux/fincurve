/* fincurve showcase — renders the exported runs in window.FINCURVE with plain SVG. */
(function () {
  "use strict";

  const D = window.FINCURVE;
  const root = document.documentElement;
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage unavailable: preference is not remembered */ } },
  };
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const params = new URLSearchParams(location.search);
  const state = {
    lang: (params.get("lang") || store.get("fc-lang")) === "zh" ? "zh" : "en",
    caseIdx: 0, impactLog: false, libGroup: "all", libQuery: "",
  };
  if (params.get("theme") === "light" || params.get("theme") === "dark") root.setAttribute("data-theme", params.get("theme"));

  const tr = (en, zh) => (state.lang === "zh" ? zh : en);
  const L = (o) => (o && typeof o === "object" && ("en" in o || "zh" in o) ? (o[state.lang] != null ? o[state.lang] : o.en) : o);
  const cssVar = (n) => getComputedStyle(root).getPropertyValue(n).trim();
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  function fmtNum(v, digits = 3) {
    if (v == null || !isFinite(v)) return "–";
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(1);
    return String(+v.toPrecision(digits));
  }
  const fmtTick = (v) => (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString("en-US") : String(+v.toPrecision(4)));
  const pct = (v) => (v == null ? "–" : `${(v * 100).toFixed(v < 0.1 && v > 0 ? 1 : 0)}%`);

  // ---------------------------------------------------------------- static i18n
  const ZH = {
    "nav.pipeline": "流程", "nav.cases": "案例", "nav.scoreboard": "计分板", "nav.method": "方法", "nav.library": "函数库", "nav.start": "快速开始",
    "hero.eyebrow": "开源 · Python · MIT",
    "hero.title": "你的金融数据<span class=\"grad\">究竟</span>遵循哪条曲线？",
    "hero.lead": "fincurve 先给数据做画像，自动选择误差模型和验证方式，再拟合 32 种候选形式——从 Nelson–Siegel、SVI 到违约强度曲线和幂律——按<b>留出数据上的似然</b>排名。它也知道什么时候<b>不该</b>拟合曲线。",
    "hero.cta1": "在 GitHub 查看", "hero.cta2": "看它如何还原已知模型",
    "pipe.title": "六步：从原始数据到站得住脚的结论",
    "pipe.sub": "每个决定都来自数据本身并写进报告，可以逐条核查某个形式为什么胜出。",
    "pipe.s1": "数据画像", "pipe.s1d": "y 的类型、形状、噪声是否随水平变化、时间顺序，以及从列名读出的金融语境。",
    "pipe.s2": "似然", "pipe.s2d": "正态、对数正态、二项或 Poisson。拿不准时，加性与乘性误差一起竞争。",
    "pipe.s3": "验证", "pipe.s3d": "随机 K 折；时间序列用滚动验证；可按组切分；样本极少时用留一法。",
    "pipe.s4": "拟合", "pipe.s4d": "32 种形式 × 多起点有界最小二乘；先在网格上解线性系数，保证起点稳定。",
    "pipe.s5": "排名", "pipe.s5d": "按每行的留出负对数似然排名。并列需同时满足 2 个配对标准误和 0.1 nats；并列中最简单的胜出。",
    "pipe.s6": "护栏", "pipe.s6d": "随机游走、半边峰、外推分歧、厚尾、数据泄漏、波动聚集。",
    "cases.title": "四组答案已知的数据",
    "cases.sub": "每组数据都由已知模型模拟生成，所以能检验 fincurve 是否还原了真相。下面所有的点、曲线、排名和日志都导出自库的真实运行。",
    "score.title": "全部 11 个场景，包括没识别出来的",
    "score.sub": "同一随机种子、默认设置、不调参。没推荐真实形式时，表里写明它排第几、以及数据为什么分不出来。",
    "method.title": "胜者是怎么选出来的",
    "method.sub": "不看 R²。看留出似然、配对标准误和实际等价差距。",
    "m1.t": "每行的留出似然",
    "m1.d": "第 i 行所在的那一折不参与参数估计。正态、对数正态、二项和 Poisson 模型用同一把尺子，加性和乘性误差可以直接竞争。",
    "m2.t": "并列的两个条件",
    "m2.d": "配对标准误基于每行与最优模型的差值。0.1 nats 的差距（正态误差下约为 RMSE 相差 10%）防止小样本把所有模型都判成一样。并列模型中参数最少的被推荐。",
    "m3.t": "在每一折里重新搜索形状",
    "m3.d": "多特征模式中，形状与交互项的贪心搜索在每个训练折上重做，报告的分数不会被搜索过程美化。各折效应曲线的相关性说明形状是否真实。",
    "m4.t": "它会提醒什么",
    "m4.d": "<li>随机游走路径——不推荐曲线，转而分析收益率</li><li>用峰形函数拟合单调数据</li><li>外推后分歧很大的并列模型</li><li>厚尾残差和离群点</li><li>行号/ID 泄漏和高度共线的特征</li><li>收益率样本中的波动聚集</li>",
    "lib.title": "32 种候选形式",
    "lib.sub": "每种形式都带参数边界、定义域规则和基于网格的起始值。y 为 0/1、比例或计数时，自动加入 logit 或 log 连接的广义线性版本。",
    "start.title": "一分钟上手",
    "start.sub": "依赖只有 numpy、scipy、pandas 和 matplotlib。",
    "start.note": "文字报告目前是中文；结构化结果（report.ranking、report.recommended、report.warnings）不受语言限制。",
    "foot.gen": "数据生成于",
  };

  function applyStatic() {
    $$("[data-i18n]").forEach((el) => {
      if (el.dataset.en == null) el.dataset.en = el.innerHTML;
      const key = el.dataset.i18n;
      el.innerHTML = state.lang === "zh" && ZH[key] != null ? ZH[key] : el.dataset.en;
    });
    root.lang = state.lang === "zh" ? "zh-CN" : "en";
    $("#langBtn").textContent = state.lang === "zh" ? "EN" : "中文";
    $("#libSearch").placeholder = tr("search forms…", "搜索形式…");
  }

  // ---------------------------------------------------------------- svg helpers
  const NS = "http://www.w3.org/2000/svg";
  function S(tag, attrs, parent) {
    const e = document.createElementNS(NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function niceTicks(a, b, n) {
    if (!(b > a)) return [a];
    const raw = (b - a) / Math.max(1, n);
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const e = raw / mag;
    const step = (e >= 7.5 ? 10 : e >= 3.5 ? 5 : e >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(a / step) * step; v <= b + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
    return out;
  }
  function logTicks(a, b) {
    const out = [];
    const e0 = Math.floor(Math.log10(a)), e1 = Math.ceil(Math.log10(b));
    const mult = e1 - e0 <= 2 ? [1, 2, 5] : [1];
    for (let e = e0; e <= e1; e++) {
      for (const m of mult) {
        const v = m * Math.pow(10, e);
        if (v >= a * 0.999 && v <= b * 1.001) out.push(+v.toPrecision(6));
      }
    }
    return out;
  }
  function mkScale(type, d0, d1, r0, r1) {
    const f = type === "log" ? Math.log10 : (v) => v;
    const a = f(d0), b = f(d1);
    const s = (v) => r0 + ((f(v) - a) / (b - a || 1)) * (r1 - r0);
    s.invert = (p) => {
      const t = a + ((p - r0) / (r1 - r0)) * (b - a);
      return type === "log" ? Math.pow(10, t) : t;
    };
    return s;
  }
  function nearestIndex(xy, x) {
    let lo = 0, hi = xy.length - 1;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (xy[mid][0] < x) lo = mid; else hi = mid;
    }
    return Math.abs(xy[lo][0] - x) <= Math.abs(xy[hi][0] - x) ? lo : hi;
  }
  const measure = document.createElement("canvas").getContext("2d");
  function fitText(text, px, font) {
    measure.font = font || "12px Inter, system-ui, sans-serif";
    if (measure.measureText(text).width <= px) return text;
    let t = text;
    while (t.length > 1 && measure.measureText(t + "…").width > px) t = t.slice(0, -1);
    return t + "…";
  }
  const swLine = (c, dash) => `<svg width="22" height="10" aria-hidden="true"><line x1="1" y1="5" x2="21" y2="5" stroke="${c}" stroke-width="2.2" stroke-dasharray="${dash || "none"}" stroke-linecap="round"/></svg>`;
  const swDot = (c) => `<svg width="10" height="10" aria-hidden="true"><circle cx="5" cy="5" r="3.5" fill="${c}"/></svg>`;
  const swBar = (c) => `<svg width="12" height="10" aria-hidden="true"><rect x="1" y="1" width="10" height="8" rx="1.5" fill="${c}" fill-opacity="0.45"/></svg>`;
  const tipRow = (sw, label, value) => `<div class="tr"><span>${sw}${esc(label)}</span><b>${esc(value)}</b></div>`;

  function placeTip(tip, px, py, W) {
    tip.classList.add("on");
    const tw = tip.offsetWidth, th = tip.offsetHeight;
    let left = px + 14;
    if (left + tw > W) left = px - tw - 14;
    tip.style.left = Math.max(0, left) + "px";
    tip.style.top = (py - th - 10 < 0 ? py + 14 : py - th - 10) + "px";
  }

  // ---------------------------------------------------------------- line / scatter / histogram chart
  function lineChart(host, spec) {
    host.innerHTML = "";
    const legend = document.createElement("div");
    legend.className = "legend";
    host.appendChild(legend);
    const box = document.createElement("div");
    box.className = "chart";
    host.appendChild(box);
    const W = Math.max(260, Math.floor(box.clientWidth || host.clientWidth));
    const H = spec.height || (W < 520 ? 250 : 310);
    const m = { t: 10, r: 14, b: 46, l: 60 };
    const xType = spec.xType || "linear", yType = spec.yType || "linear";
    const toX = spec.time ? (v) => Date.parse(v) : (v) => v;
    const okX = (v) => v != null && isFinite(v) && (xType !== "log" || v > 0);
    const okY = (v) => v != null && isFinite(v) && (yType !== "log" || v > 0);

    const points = [];
    if (spec.points) spec.points[0].forEach((x, i) => {
      const px = toX(x), py = spec.points[1][i];
      if (okX(px) && okY(py)) points.push([px, py]);
    });
    const series = (spec.series || []).map((s) => Object.assign({}, s, { xy: s.x.map((x, i) => [toX(x), s.y[i]]) }));
    const bars = spec.bars || [];

    let xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    bars.forEach((b) => { xs.push(b.x0, b.x1); ys.push(0, b.d); });
    series.forEach((s) => s.xy.forEach(([x, y]) => {
      if (!okX(x)) return;
      if (!spec.xFromData) xs.push(x);
      if (okY(y) && !spec.yFromData) ys.push(y);
    }));
    xs = xs.filter(okX);
    ys = ys.filter(okY);
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    if (yType === "log") { y0 /= 1.25; y1 *= 1.25; } else { const p = (y1 - y0) * 0.07 || 1; y0 = bars.length ? 0 : y0 - p; y1 += p; }
    if (xType === "log") { x0 /= 1.08; x1 *= 1.08; } else if (!spec.time) { const p = (x1 - x0) * 0.02; x0 -= p; x1 += p; }

    const sx = mkScale(xType, x0, x1, m.l, W - m.r);
    const sy = mkScale(yType, y0, y1, H - m.b, m.t);
    const svg = S("svg", { width: W, height: H, role: "img", "aria-label": spec.aria || "" }, box);
    const clip = "c" + Math.random().toString(36).slice(2, 9);
    S("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b }, S("clipPath", { id: clip }, S("defs", null, svg)));
    const gridC = cssVar("--grid"), axisC = cssVar("--axis"), dataC = cssVar("--data");

    const yt = yType === "log" ? logTicks(y0, y1) : niceTicks(y0, y1, Math.max(3, Math.round((H - m.t - m.b) / 55)));
    yt.forEach((v) => {
      const y = sy(v);
      S("line", { x1: m.l, x2: W - m.r, y1: y, y2: y, stroke: gridC }, svg);
      S("text", { x: m.l - 8, y: y + 4, "text-anchor": "end" }, svg).textContent = fmtTick(v);
    });
    let xt;
    if (spec.time) {
      const ya = new Date(x0).getUTCFullYear(), yb = new Date(x1).getUTCFullYear();
      const every = Math.max(1, Math.ceil((yb - ya + 1) / Math.max(2, Math.floor((W - m.l - m.r) / 70))));
      xt = [];
      for (let yy = ya; yy <= yb + 1; yy += every) { const v = Date.UTC(yy, 0, 1); if (v >= x0 && v <= x1) xt.push(v); }
    } else {
      xt = xType === "log" ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(3, Math.round((W - m.l - m.r) / 85)));
    }
    xt.forEach((v) => {
      const x = sx(v);
      S("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: gridC }, svg);
      S("text", { x, y: H - m.b + 17, "text-anchor": "middle" }, svg).textContent = spec.time ? new Date(v).getUTCFullYear() : fmtTick(v);
    });
    S("line", { x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b, stroke: axisC }, svg);
    if (spec.xLabel) S("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, svg).textContent = spec.xLabel;
    if (spec.yLabel) S("text", { x: -(m.t + H - m.b) / 2, y: 14, transform: "rotate(-90)", "text-anchor": "middle", class: "axis-label" }, svg).textContent = spec.yLabel;

    const plot = S("g", { "clip-path": `url(#${clip})` }, svg);
    bars.forEach((b) => {
      const xa = sx(b.x0), xb = sx(b.x1);
      S("rect", { x: xa + 1, y: sy(b.d), width: Math.max(0, xb - xa - 2), height: Math.max(0, sy(0) - sy(b.d)), fill: dataC, "fill-opacity": 0.35, rx: 2 }, plot);
    });
    if (spec.rug) spec.rug.forEach((v) => {
      if (okX(v)) S("line", { x1: sx(v), x2: sx(v), y1: H - m.b - 8, y2: H - m.b, stroke: dataC, "stroke-opacity": 0.5 }, plot);
    });
    points.forEach(([x, y]) => S("circle", { cx: sx(x), cy: sy(y), r: spec.pointR || 3.2, fill: dataC, "fill-opacity": 0.8 }, plot));
    series.forEach((s) => {
      let d = "", pen = false;
      s.xy.forEach(([x, y]) => {
        if (okX(x) && okY(y)) { d += (pen ? "L" : "M") + sx(x).toFixed(1) + " " + sy(y).toFixed(1); pen = true; } else pen = false;
      });
      S("path", { d, fill: "none", stroke: s.color, "stroke-width": s.width || 2.2, "stroke-dasharray": s.dash || "none", "stroke-linejoin": "round", "stroke-linecap": "round" }, plot);
    });

    legend.innerHTML = (bars.length ? `<span><i>${swBar(dataC)}</i>${esc(spec.barsLabel || "")}</span>` : "")
      + (points.length ? `<span><i>${swDot(dataC)}</i>${esc(spec.pointsLabel || "")}</span>` : "")
      + series.map((s) => `<span><i>${swLine(s.color, s.dash)}</i>${esc(s.label)}</span>`).join("");

    const tip = document.createElement("div");
    tip.className = "tooltip";
    box.appendChild(tip);
    const cross = S("line", { y1: m.t, y2: H - m.b, stroke: axisC, "stroke-dasharray": "3 3", visibility: "hidden" }, svg);
    const surf = cssVar("--surface");
    const markers = series.map((s) => S("circle", { r: 4, fill: surf, stroke: s.color, "stroke-width": 2, visibility: "hidden" }, svg));
    const ring = S("circle", { r: 6.5, fill: "none", stroke: cssVar("--text"), "stroke-width": 1.5, visibility: "hidden" }, svg);
    const overlay = S("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
    const hide = () => { tip.classList.remove("on"); [cross, ring].concat(markers).forEach((e) => e.setAttribute("visibility", "hidden")); };
    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("pointermove", (ev) => {
      const rect = svg.getBoundingClientRect();
      const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
      const xv = sx.invert(px);
      cross.setAttribute("x1", px); cross.setAttribute("x2", px); cross.setAttribute("visibility", "visible");
      let rows = "", head = null;
      let best = null, bd = 22 * 22;
      points.forEach(([x, y]) => { const dx = sx(x) - px, dy = sy(y) - py, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = [x, y]; } });
      if (best) {
        ring.setAttribute("cx", sx(best[0])); ring.setAttribute("cy", sy(best[1])); ring.setAttribute("visibility", "visible");
        rows += tipRow(swDot(dataC), spec.pointsLabel || "", fmtNum(best[1], 4));
      } else ring.setAttribute("visibility", "hidden");
      const bar = bars.find((b) => xv >= b.x0 && xv < b.x1);
      if (bar) { rows += tipRow(swBar(dataC), spec.barsLabel || "", fmtNum(bar.d, 3)); head = `${fmtNum(bar.x0, 3)} … ${fmtNum(bar.x1, 3)}`; }
      series.forEach((s, i) => {
        const pt = s.xy[nearestIndex(s.xy, best ? best[0] : xv)];
        if (pt && okY(pt[1]) && pt[1] >= y0 && pt[1] <= y1) {
          markers[i].setAttribute("cx", sx(pt[0])); markers[i].setAttribute("cy", sy(pt[1])); markers[i].setAttribute("visibility", "visible");
          rows += tipRow(swLine(s.color, s.dash), s.label, fmtNum(pt[1], 4));
        } else markers[i].setAttribute("visibility", "hidden");
      });
      const xh = best ? best[0] : xv;
      if (!head) head = spec.time ? new Date(xh).toISOString().slice(0, 10) : `${spec.xShort || "x"} = ${fmtNum(xh, 4)}`;
      tip.innerHTML = `<div class="th">${esc(head)}</div>${rows}`;
      placeTip(tip, px, py, W);
    });
  }

  // ---------------------------------------------------------------- ranking dot plot (symlog)
  const FAMILY = { gaussian: ["additive", "加性"], lognormal: ["multiplicative", "乘性"], binomial: ["binomial", "二项"], poisson: ["Poisson", "Poisson"] };
  function rankChart(host, rows, opts) {
    opts = opts || {};
    host.innerHTML = "";
    const box = document.createElement("div");
    box.className = "chart";
    host.appendChild(box);
    const W = Math.max(260, Math.floor(box.clientWidth || host.clientWidth));
    const rowH = 27;
    const labelW = Math.min(260, Math.max(120, Math.round(W * 0.44)));
    const m = { t: 26, r: 16, b: 40, l: labelW };
    const H = m.t + rows.length * rowH + m.b;
    const c = 0.01;
    const f = (v) => Math.log10(1 + Math.max(v, 0) / c);
    const maxV = Math.max(0.3, ...rows.map((r) => (r.d || 0) + (r.se || 0)));
    const sx = (v) => m.l + (f(v) / f(maxV)) * (W - m.l - m.r);
    const svg = S("svg", { width: W, height: H, role: "img", "aria-label": opts.aria || "" }, box);
    const gridC = cssVar("--grid"), text2 = cssVar("--text-2"), muted = cssVar("--muted"), s1 = cssVar("--s1"), surf = cssVar("--surface"), elev = cssVar("--elev");
    [0, 0.01, 0.1, 1, 10].filter((v) => v <= maxV).forEach((v) => {
      const x = sx(v);
      S("line", { x1: x, x2: x, y1: m.t - 4, y2: H - m.b, stroke: gridC }, svg);
      S("text", { x, y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = v;
    });
    const gx = sx(0.1);
    S("line", { x1: gx, x2: gx, y1: m.t - 14, y2: H - m.b, stroke: cssVar("--warning"), "stroke-dasharray": "4 3", "stroke-opacity": 0.85 }, svg);
    S("text", { x: gx + 5, y: m.t - 12, "text-anchor": "start" }, svg).textContent = tr("0.1-nat gap", "0.1 差距");
    S("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, svg).textContent =
      opts.xLabel || tr("Δ held-out NLL per row vs best (symlog)", "与最优相比每行留出 NLL 之差（对称对数刻度）");

    const tip = document.createElement("div");
    tip.className = "tooltip";
    box.appendChild(tip);
    rows.forEach((r, i) => {
      const cy = m.t + i * rowH + rowH / 2;
      const band = S("rect", { x: 0, y: cy - rowH / 2 + 1, width: W, height: rowH - 2, rx: 6, fill: "transparent" }, svg);
      const color = r.rec ? s1 : r.tie ? text2 : muted;
      const fam = opts.showFamily && FAMILY[r.family] ? ` · ${tr(FAMILY[r.family][0], FAMILY[r.family][1])}` : "";
      const label = (r.rec ? "★ " : "") + L(r.label) + fam;
      const t = S("text", { x: m.l - 12, y: cy + 4, "text-anchor": "end", "pointer-events": "none" }, svg);
      t.textContent = fitText(label, m.l - 20);
      t.setAttribute("style", `font-family: var(--sans); font-size: 12px; fill: ${r.rec ? cssVar("--text") : text2}; font-weight: ${r.rec ? 600 : 400}`);
      if (r.se && r.d != null) {
        const a = sx(Math.max(r.d - r.se, 0)), b = sx(r.d + r.se);
        S("line", { x1: a, x2: b, y1: cy, y2: cy, stroke: color, "stroke-width": 1.5, "pointer-events": "none" }, svg);
        [a, b].forEach((x) => S("line", { x1: x, x2: x, y1: cy - 4, y2: cy + 4, stroke: color, "stroke-width": 1.5, "pointer-events": "none" }, svg));
      }
      const x = sx(r.d || 0);
      if (r.ref) S("path", { d: `M${x} ${cy - 6}L${x + 6} ${cy}L${x} ${cy + 6}L${x - 6} ${cy}Z`, fill: surf, stroke: color, "stroke-width": 1.8, "pointer-events": "none" }, svg);
      else S("circle", { cx: x, cy, r: r.rec ? 6 : 5, fill: color, stroke: surf, "stroke-width": 2, "pointer-events": "none" }, svg);
      band.addEventListener("pointerenter", () => band.setAttribute("fill", elev));
      band.addEventListener("pointerleave", () => { band.setAttribute("fill", "transparent"); tip.classList.remove("on"); });
      band.addEventListener("pointermove", (ev) => {
        const rect = svg.getBoundingClientRect();
        const status = r.ref ? tr("reference only — not eligible", "仅作参照，不参与推荐")
          : r.rec ? tr("recommended", "推荐") : r.tie ? tr("tied with best", "与最优并列") : tr("not tied", "不并列");
        tip.innerHTML = `<div class="th">#${r.rank || i + 1} ${esc(L(r.label) + fam)}</div>`
          + tipRow("", "Δ ± SE", `${fmtNum(r.d, 3)} ± ${fmtNum(r.se, 2)}`)
          + (r.k != null ? tipRow("", tr("parameters", "参数个数"), String(r.k)) : "")
          + tipRow("", tr("status", "状态"), status);
        placeTip(tip, ev.clientX - rect.left, ev.clientY - rect.top, W);
      });
    });
  }

  function rankTable(rows, showFamily) {
    const body = rows.map((r) => `<tr><td>${r.rec ? "★ " : ""}${esc(L(r.label))}${showFamily && FAMILY[r.family] ? `<div class="sub">${esc(tr(FAMILY[r.family][0], FAMILY[r.family][1]))}</div>` : ""}</td>`
      + `<td class="num">${r.k == null ? "–" : r.k}</td><td class="num">${fmtNum(r.d, 3)}</td><td class="num">${fmtNum(r.se, 2)}</td>`
      + `<td>${r.ref ? tr("reference", "参照") : r.tie ? tr("tied", "并列") : ""}</td></tr>`).join("");
    return `<details class="note"><summary>${tr("Show as table", "以表格查看")}</summary><div class="table-wrap" style="margin-top:8px"><table class="table mini">`
      + `<thead><tr><th>${tr("model", "模型")}</th><th class="num">k</th><th class="num">Δ</th><th class="num">SE</th><th></th></tr></thead><tbody>${body}</tbody></table></div></details>`;
  }

  // ---------------------------------------------------------------- hero
  let termTimer = null;
  function renderHeroTerm(animate) {
    const host = $("#heroTerm");
    clearTimeout(termTimer);
    const c = D.cases[0];
    const lines = [{ prompt: '>>> report = analyze(df["ln(K/F)"], df["implied_vol"])' }]
      .concat(c.log.map((l) => ({ tag: l.tag, text: L(l.text) })))
      .concat([{ prompt: ">>> report.recommended.candidate.key  # 'svi_smile'", cursor: true }]);
    host.innerHTML = "";
    const add = (line) => {
      const div = document.createElement("div");
      if (line.prompt) {
        div.className = "term-prompt";
        div.innerHTML = `<span class="p">❯</span> ${esc(line.prompt)}${line.cursor ? ' <span class="cursor"></span>' : ""}`;
      } else {
        div.className = "term-line" + (line.tag === "verdict" ? " verdict" : "");
        div.innerHTML = `<span class="tag">${esc(line.tag)}</span><span class="txt">${esc(line.text)}</span>`;
      }
      host.appendChild(div);
    };
    if (!animate || reduceMotion) { lines.forEach(add); return; }
    let i = 0;
    const step = () => { add(lines[i++]); if (i < lines.length) termTimer = setTimeout(step, i === 1 ? 650 : 420); };
    termTimer = setTimeout(step, 300);
  }

  function renderStats() {
    const hits = D.scoreboard.filter((r) => r.verdict === "hit").length;
    const items = [
      [D.stats.candidates, tr("candidate forms", "种候选形式")],
      [D.stats.likelihoods, tr("likelihoods", "种似然")],
      [`${hits}/${D.scoreboard.length}`, tr("known models recovered", "个已知模型被还原")],
      [D.stats.tests, tr("regression tests", "个回归测试")],
    ];
    $("#stats").innerHTML = items.map(([v, l]) => `<div class="stat"><b class="mono">${v}</b><span>${esc(l)}</span></div>`).join("");
  }

  // ---------------------------------------------------------------- cases
  const CASES = [
    { n: "01", tab: ["Volatility smile", "波动率微笑"], kicker: ["curve mode", "曲线模式"] },
    { n: "02", tab: ["Market impact", "市场冲击"], kicker: ["curve mode · noise model", "曲线模式 · 误差模型"] },
    { n: "03", tab: ["Loan defaults", "贷款违约"], kicker: ["multi-feature mode", "多特征模式"] },
    { n: "04", tab: ["Random-walk trap", "随机游走陷阱"], kicker: ["guardrail", "护栏"] },
  ];

  function renderTabs() {
    const host = $("#caseTabs");
    host.innerHTML = CASES.map((c, i) => `<button class="tab" role="tab" type="button" id="tab-${i}" aria-controls="casePanel" aria-selected="${i === state.caseIdx}" tabindex="${i === state.caseIdx ? 0 : -1}"><span class="n">${c.n}</span>${esc(tr(c.tab[0], c.tab[1]))}</button>`).join("");
    $$(".tab", host).forEach((b, i) => {
      b.addEventListener("click", () => selectCase(i));
      b.addEventListener("keydown", (ev) => {
        if (ev.key !== "ArrowRight" && ev.key !== "ArrowLeft") return;
        const j = (i + (ev.key === "ArrowRight" ? 1 : CASES.length - 1)) % CASES.length;
        selectCase(j);
        $(`#tab-${j}`).focus();
      });
    });
  }
  function selectCase(i) {
    state.caseIdx = i;
    try { history.replaceState(null, "", `${location.pathname}${location.search}#case-${D.cases[i].id}`); } catch (e) { /* file:// or sandboxed */ }
    $$(".tab").forEach((b, j) => { b.setAttribute("aria-selected", String(i === j)); b.tabIndex = i === j ? 0 : -1; });
    $("#casePanel").setAttribute("aria-labelledby", `tab-${i}`);
    renderCase();
  }

  function caseHead(c, meta) {
    return `<div class="case-head"><div><div class="card-kicker mono">case ${meta.n} · ${esc(tr(meta.kicker[0], meta.kicker[1]))}</div>`
      + `<h3>${esc(L(c.title))}</h3><p>${esc(L(c.claim))}</p></div>`
      + `<div class="badges"><div class="badge"><span>${tr("ground truth", "真实模型")}</span><b>${esc(L(c.truth))}</b></div>`
      + `<div class="badge ok"><span>${tr("identified", "识别结果")}</span><b>${esc(L(c.identified))}</b></div></div></div>`;
  }
  function logCard(c) {
    const lines = c.log.map((l) => `<div class="term-line${l.tag === "verdict" ? " verdict" : ""}"><span class="tag">${esc(l.tag)}</span><span class="txt">${esc(L(l.text))}</span></div>`).join("");
    return `<div class="log"><div class="term-bar"><span></span><span></span><span></span><div class="term-title mono">${tr("decision log", "决策日志")}</div></div><div class="term-body mono">${lines}</div></div>`;
  }
  function card(title, sub, inner, extraHead) {
    return `<div class="card"><div class="card-head"><div><h3>${title}</h3>${sub ? `<p>${sub}</p>` : ""}</div>${extraHead || ""}</div>${inner}</div>`;
  }
  const PALETTE = () => [cssVar("--s1"), cssVar("--s2"), cssVar("--s3")];
  const DASHES = ["none", "7 5", "2 4"];
  function styledSeries(list) {
    const pal = PALETTE();
    let k = 0;
    return list.map((s) => {
      if (s.truth) return { label: L(s.label), x: s.x, y: s.y, color: cssVar("--text"), dash: "1.5 5", width: 2.2 };
      const style = { label: L(s.label), x: s.x, y: s.y, color: pal[k], dash: DASHES[k] };
      k += 1;
      return style;
    });
  }
  const rankSub = () => tr("Held-out NLL per row relative to the best model, ±1 paired SE. ◆ = non-parametric reference. Dashed line = practical-equivalence gap.",
    "相对最优模型的每行留出 NLL 差，误差线为 ±1 个配对标准误。◆ = 非参数参照。虚线 = 实际等价差距。");

  function renderSmile(panel, c) {
    const params = c.params.map((p) => {
      const rel = p.truth ? (p.fitted - p.truth) / Math.abs(p.truth) : null;
      return `<tr><td class="mono">${esc(p.name)}</td><td class="num">${fmtNum(p.truth, 4)}</td><td class="num">${fmtNum(p.fitted, 4)}</td><td class="num">${rel == null ? "–" : (rel >= 0 ? "+" : "") + (rel * 100).toFixed(1) + "%"}</td></tr>`;
    }).join("");
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2">${card(tr("Data, top-3 forms and the ground truth", "数据、前三名形式与真实曲线"), tr("Hover to compare fitted values.", "悬停查看各曲线的拟合值。"), '<div id="chartMain"></div>')}${logCard(c)}</div>`
      + `<div class="grid-2">${card(tr("Cross-validated ranking", "交叉验证排名"), rankSub(), '<div id="chartRank"></div>' + rankTable(c.ranking))}`
      + card(tr("Parameter recovery", "参数还原"), tr("a′ = 10⁴·a/T and b′ = 10⁴·b/T put SVI in vol-% units.", "a′ = 10⁴·a/T、b′ = 10⁴·b/T，把 SVI 换算到波动率百分比单位。"),
        `<div class="table-wrap"><table class="table mini"><thead><tr><th>${tr("parameter", "参数")}</th><th class="num">${tr("truth", "真实")}</th><th class="num">${tr("fitted", "拟合")}</th><th class="num">${tr("error", "误差")}</th></tr></thead><tbody>${params}</tbody></table></div>`
        + `<p class="note">${tr("a′ and σ slide along a ridge — raising one and lowering the other barely changes the curve — so they are individually loose. Their combination at the smile's vertex is pinned down to about 2%.",
          "a′ 和 σ 位于一条“脊线”上：一个升、一个降，曲线几乎不变，所以单独看都不够准；但它们在微笑谷底的组合被确定到约 2%。")}</p>`) + `</div>`);
    const ch = c.chart;
    lineChart($("#chartMain"), { xLabel: L(ch.xLabel), yLabel: L(ch.yLabel), xShort: "ln(K/F)", points: ch.points, pointsLabel: tr("observed", "观测值"), series: styledSeries(ch.series), aria: L(c.title) });
    rankChart($("#chartRank"), c.ranking, { aria: tr("ranking", "排名") });
  }

  function renderImpact(panel, c) {
    const params = c.params.map((p) => `<tr><td>${esc(p.name)}</td><td class="num">${fmtNum(p.truth, 4)}</td><td class="num">${fmtNum(p.fitted, 4)}</td></tr>`).join("");
    const toggle = `<div class="seg" role="group" aria-label="axis scale"><button type="button" data-scale="lin" aria-pressed="${!state.impactLog}">linear</button><button type="button" data-scale="log" aria-pressed="${state.impactLog}">log–log</button></div>`;
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2">${card(tr("Impact vs participation", "冲击成本与参与率"), tr("Curves show the conditional median. On log–log axes a square-root law is a straight line with slope ½.", "曲线为条件中位数。在双对数坐标下，平方根律是一条斜率为 ½ 的直线。"), '<div id="chartMain"></div>', toggle)}${logCard(c)}</div>`
      + `<div class="grid-2">${card(tr("Ranking across both noise models", "两种误差模型一起排名"), rankSub(), '<div id="chartRank"></div>' + rankTable(c.ranking, true))}`
      + card(tr("Evidence", "证据"), tr("Noise model first, then the shape.", "先定误差模型，再定形状。"),
        `<div class="bigpair"><div><b>${fmtNum(c.families.multiplicative, 3)}</b><span>${tr("best multiplicative model, Δ", "最好的乘性误差模型 Δ")}</span></div><div><b>${fmtNum(c.families.additive, 3)} <span class="mono" style="font-size:13px">± ${fmtNum(c.families.additive_se, 2)}</span></b><span>${tr("best additive model, Δ", "最好的加性误差模型 Δ")}</span></div></div>`
        + `<div class="table-wrap"><table class="table mini"><thead><tr><th>${tr("quantity", "量")}</th><th class="num">${tr("truth", "真实")}</th><th class="num">${tr("fitted", "拟合")}</th></tr></thead><tbody>${params}</tbody></table></div>`
        + `<p class="note">${tr("The free power-law exponent is fitted independently of the √q form and lands on ½.", "自由幂律的指数与 √q 形式独立拟合，结果落在 ½ 附近。")}</p>`) + `</div>`);
    const draw = () => {
      const ch = c.chart;
      lineChart($("#chartMain"), {
        xLabel: L(ch.xLabel), yLabel: L(ch.yLabel), xShort: "q", points: ch.points, pointsLabel: tr("observed trades", "观测交易"),
        series: styledSeries(ch.series), xType: state.impactLog ? "log" : "linear", yType: state.impactLog ? "log" : "linear", aria: L(c.title),
      });
    };
    $$(".seg button", panel).forEach((b) => b.addEventListener("click", () => {
      state.impactLog = b.dataset.scale === "log";
      $$(".seg button", panel).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      draw();
    }));
    draw();
    rankChart($("#chartRank"), c.ranking, { showFamily: true, aria: tr("ranking", "排名") });
  }

  function renderLoans(panel, c) {
    const effects = c.panels.map((p, i) => card(`${esc(L(p.feature))}: ${esc(L(p.trend))}`,
      `${tr("importance", "重要性")} ${pct(p.importance)} · ${tr("fold-to-fold curve correlation", "各折曲线相关")} ${p.stability == null ? "–" : p.stability.toFixed(2)}`, `<div id="effect${i}"></div>`)).join("");
    const feats = c.features.map((f) => `<tr><td>${esc(L(f.name))}</td><td>${esc(L(f.trend))}</td><td class="num">${f.importance == null ? "–" : pct(f.importance)}</td><td class="num">${f.stability == null ? "–" : f.stability.toFixed(2)}</td></tr>`).join("");
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2 even">${effects}</div>`
      + `<div class="grid-2">${card(tr("Model comparison and features", "模型比较与特征"), rankSub(),
        '<div id="chartRank"></div>' + rankTable(c.ranking)
        + `<div class="table-wrap" style="margin-top:12px"><table class="table mini"><thead><tr><th>${tr("feature", "特征")}</th><th>${tr("shape", "形状")}</th><th class="num">${tr("importance", "重要性")}</th><th class="num">${tr("fold corr.", "各折相关")}</th></tr></thead><tbody>${feats}</tbody></table></div>`)}${logCard(c)}</div>`);
    c.panels.forEach((p, i) => lineChart($(`#effect${i}`), {
      xLabel: L(p.feature), yLabel: tr("contribution to logit(p)", "对 logit(p) 的贡献"), xShort: L(p.feature), height: 250,
      series: styledSeries([{ label: { en: "fitted", zh: "拟合" }, x: p.x, y: p.fitted }, { label: { en: "ground truth", zh: "真实" }, x: p.x, y: p.truth, truth: true }]),
      rug: p.rug, aria: L(p.feature),
    }));
    rankChart($("#chartRank"), c.ranking, { aria: tr("model comparison", "模型比较") });
  }

  function renderPath(panel, c) {
    const checks = c.checks.map((k) => `<tr><td>${esc(L(k.name))}</td><td class="num">${esc(typeof k.value === "number" ? fmtNum(k.value, 3) : k.value)}</td><td class="num">${esc(k.rule)}</td>`
      + `<td><span class="status ${k.pass ? "good" : "critical"}">${k.pass ? "✓ " + tr("pass", "满足") : "✗ " + tr("fail", "不满足")}</span></td></tr>`).join("");
    const r = c.rates;
    const rates = [
      [tr("smooth exponential curve + noise", "平滑指数曲线 + 噪声"), r.false_positive.smooth, tr("false alarm", "误报")],
      [tr("linear trend + noise", "线性趋势 + 噪声"), r.false_positive.linear, tr("false alarm", "误报")],
      [tr("bond price–yield curve", "债券价格–收益率曲线"), r.false_positive.bond, tr("false alarm", "误报")],
      [tr("GBM price path", "几何布朗运动路径"), r.detection.gbm, tr("detected", "识别")],
      [tr("AR(1), φ = 0.95", "AR(1)，φ = 0.95"), r.detection.ar, tr("detected", "识别")],
    ].map(([n, v, kind]) => `<tr><td>${esc(n)}</td><td class="num">${pct(v)}</td><td class="sub">${esc(kind)}</td></tr>`).join("");
    const dist = c.returns.ranking.map((d) => `<tr><td>${d.rec ? "★ " : ""}${esc(L(d.label))}</td><td class="num">${d.k}</td><td class="num">${fmtNum(d.d_aic, 3)}</td></tr>`).join("");
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2">${card(tr("A path that seems to have a shape", "一条看起来有形状的路径"), tr("The two best-scoring “curves” under rolling-origin CV. They fit — and mean nothing.", "滚动验证下得分最高的两条“曲线”：拟合得不错，但毫无意义。"), '<div id="chartMain"></div>')}${logCard(c)}</div>`
      + `<div class="grid-2 even">${card(tr("Random-walk test", "随机游走检验"), tr("All four conditions must hold. Error rates measured on 200 simulated series each.", "四个条件必须同时满足。误报率和识别率各在 200 条模拟序列上测得。"),
        `<div class="table-wrap"><table class="table mini"><thead><tr><th>${tr("statistic", "统计量")}</th><th class="num">${tr("value", "值")}</th><th class="num">${tr("rule", "规则")}</th><th></th></tr></thead><tbody>${checks}</tbody></table></div>`
        + `<div class="table-wrap" style="margin-top:12px"><table class="table mini"><thead><tr><th>${tr("simulated series", "模拟序列")}</th><th class="num">${tr("rate", "比例")}</th><th></th></tr></thead><tbody>${rates}</tbody></table></div>`)}`
      + card(tr("What it analyses instead: log returns", "转而分析：对数收益率"), tr("Histogram density with the two best distributions by AIC.", "直方图密度，叠加按 AIC 排名前二的分布。"),
        '<div id="chartReturns"></div>' + `<div class="table-wrap" style="margin-top:10px"><table class="table mini"><thead><tr><th>${tr("distribution", "分布")}</th><th class="num">k</th><th class="num">ΔAIC</th></tr></thead><tbody>${dist}</tbody></table></div>`) + `</div>`);
    const ch = c.chart;
    const pal = PALETTE();
    lineChart($("#chartMain"), {
      xLabel: L(ch.xLabel), yLabel: L(ch.yLabel), time: true, points: ch.points, pointsLabel: tr("monthly NAV", "月度净值"), yFromData: true,
      series: ch.series.map((s, i) => ({ label: L(s.label), x: s.x, y: s.y, color: pal[i + 1], dash: DASHES[i + 1] })), aria: L(c.title),
    });
    lineChart($("#chartReturns"), {
      xLabel: tr("monthly log return", "月度对数收益率"), yLabel: tr("density", "密度"), bars: c.returns.bins, barsLabel: tr("observed", "观测"), height: 240,
      series: c.returns.series.map((s, i) => ({ label: L(s.label), x: s.x, y: s.y, color: pal[i], dash: DASHES[i] })), xShort: "r", aria: tr("returns", "收益率"),
    });
  }

  function renderCase() {
    const c = D.cases[state.caseIdx], meta = CASES[state.caseIdx];
    const panel = $("#casePanel");
    panel.innerHTML = caseHead(c, meta);
    ({ smile: renderSmile, impact: renderImpact, loans: renderLoans, path: renderPath })[c.id](panel, c);
  }

  // ---------------------------------------------------------------- scoreboard
  const VERDICT = {
    hit: ["good", "✓", ["recovered", "识别正确"]],
    tie: ["warning", "≈", ["tied with truth", "与真实形式并列"]],
    miss: ["critical", "✗", ["missed", "未识别"]],
  };
  const MODE = { curve: ["curve", "曲线"], distribution: ["distribution", "分布"], guardrail: ["guardrail", "护栏"], additive: ["multi-feature", "多特征"] };
  function renderScoreboard() {
    const rows = D.scoreboard;
    const count = (v) => rows.filter((r) => r.verdict === v).length;
    $("#scoreSummary").innerHTML = ["hit", "tie", "miss"].map((v) => `<span class="status ${VERDICT[v][0]}">${VERDICT[v][1]} ${count(v)} ${esc(tr(VERDICT[v][2][0], VERDICT[v][2][1]))}</span>`).join("");
    $("#scoreTable").innerHTML = `<thead><tr><th>#</th><th>${tr("scenario", "场景")}</th><th>${tr("mode", "模式")}</th><th>${tr("ground truth", "真实模型")}</th><th>${tr("identified", "识别结果")}</th><th>${tr("verdict", "结论")}</th><th>${tr("detail", "说明")}</th></tr></thead><tbody>`
      + rows.map((r, i) => {
        const v = VERDICT[r.verdict];
        return `<tr><td class="num">${String(i + 1).padStart(2, "0")}</td><td>${esc(L(r.name))}</td><td><span class="mode">${esc(tr(MODE[r.mode][0], MODE[r.mode][1]))}</span></td>`
          + `<td>${esc(L(r.truth))}</td><td>${esc(L(r.identified))}</td><td><span class="status ${v[0]}">${v[1]} ${esc(tr(v[2][0], v[2][1]))}</span></td><td class="sub">${esc(L(r.detail))}</td></tr>`;
      }).join("") + "</tbody>";
  }

  // ---------------------------------------------------------------- library
  function renderLibrary() {
    const present = new Set(D.library.map((f) => f.group));
    const groups = [{ key: "all", label: { en: "All", zh: "全部" } }].concat(D.groups.filter((g) => present.has(g.key)));
    $("#libChips").innerHTML = groups.map((g) => `<button class="chipbtn" type="button" data-g="${g.key}" aria-pressed="${state.libGroup === g.key}">${esc(L(g.label))}</button>`).join("");
    $$("#libChips .chipbtn").forEach((b) => b.addEventListener("click", () => { state.libGroup = b.dataset.g; renderLibrary(); }));
    const q = state.libQuery.trim().toLowerCase();
    const items = D.library.filter((f) => (state.libGroup === "all" || f.group === state.libGroup)
      && (!q || [f.key, f.formula, f.label.en, f.label.zh, f.meaning.en, f.meaning.zh].join(" ").toLowerCase().includes(q)));
    $("#libGrid").innerHTML = items.length ? items.map((f) => `<div class="form"><div class="form-top"><b>${esc(L(f.label))}</b><span class="meta">${esc(L(f.groupLabel))}</span></div>`
      + `<code>${esc(f.formula)}</code><p>${esc(L(f.meaning))}</p><span class="meta">key: ${esc(f.key)} · k = ${f.k}</span></div>`).join("")
      : `<p class="note">${tr("No form matches.", "没有匹配的形式。")}</p>`;
  }

  // ---------------------------------------------------------------- wiring
  function renderAll(animateTerm) {
    applyStatic();
    renderStats();
    renderHeroTerm(animateTerm);
    renderTabs();
    renderCase();
    renderScoreboard();
    renderLibrary();
  }

  $("#version").textContent = "v" + D.version;
  $("#generated").textContent = D.generated;
  $("#langBtn").addEventListener("click", () => {
    state.lang = state.lang === "zh" ? "en" : "zh";
    store.set("fc-lang", state.lang);
    renderAll(false);
  });
  $("#themeBtn").addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    store.set("fc-theme", next);
    renderCase();
  });
  $("#libSearch").addEventListener("input", (ev) => { state.libQuery = ev.target.value; renderLibrary(); });
  $$(".copy").forEach((b) => b.addEventListener("click", async () => {
    const text = $("#" + b.dataset.target).innerText;
    try { await navigator.clipboard.writeText(text); b.textContent = tr("copied", "已复制"); }
    catch (e) { b.textContent = tr("select & copy", "请手动复制"); }
    setTimeout(() => { b.textContent = "copy"; }, 1400);
  }));

  let lastWidth = 0, resizeTimer = null;
  new ResizeObserver((entries) => {
    const w = Math.round(entries[0].contentRect.width);
    if (Math.abs(w - lastWidth) < 4) return;
    lastWidth = w;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderCase, 120);
  }).observe($("#casePanel"));

  const deep = /^#case-(\w+)$/.exec(location.hash);
  if (deep) {
    const i = D.cases.findIndex((c) => c.id === deep[1]);
    if (i >= 0) state.caseIdx = i;
  }
  renderAll(true);
  if (deep) $("#cases").scrollIntoView();
})();
