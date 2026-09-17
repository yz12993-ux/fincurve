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
  if (params.get("theme") === "light" || params.get("theme") === "dark") root.setAttribute("data-theme", params.get("theme"));
  const state = { caseIdx: 0, impactLog: false, libGroup: "all", libQuery: "" };

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
  const secs = (v) => (v == null ? "–" : v < 1 ? `${(v * 1000).toFixed(0)} ms` : `${v.toFixed(v < 10 ? 2 : 1)} s`);

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
      xs.push(x);
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
    overlay.addEventListener("pointerleave", () => {
      tip.classList.remove("on");
      [cross, ring].concat(markers).forEach((e) => e.setAttribute("visibility", "hidden"));
    });
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
  const FAMILY = { gaussian: "additive", lognormal: "multiplicative", binomial: "binomial", poisson: "Poisson" };
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
    S("text", { x: gx + 5, y: m.t - 12, "text-anchor": "start" }, svg).textContent = "0.1-nat gap";
    S("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, svg).textContent =
      opts.xLabel || "Δ held-out NLL per row vs best (symlog)";

    const tip = document.createElement("div");
    tip.className = "tooltip";
    box.appendChild(tip);
    rows.forEach((r, i) => {
      const cy = m.t + i * rowH + rowH / 2;
      const band = S("rect", { x: 0, y: cy - rowH / 2 + 1, width: W, height: rowH - 2, rx: 6, fill: "transparent" }, svg);
      const color = r.rec ? s1 : r.tie ? text2 : muted;
      const fam = opts.showFamily && FAMILY[r.family] ? ` · ${FAMILY[r.family]}` : "";
      const label = (r.rec ? "★ " : "") + r.label + fam;
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
        const status = r.ref ? "reference only — not eligible" : r.rec ? "recommended" : r.tie ? "tied with best" : "not tied";
        tip.innerHTML = `<div class="th">#${r.rank || i + 1} ${esc(r.label + fam)}</div>`
          + tipRow("", "Δ ± SE", `${fmtNum(r.d, 3)} ± ${fmtNum(r.se, 2)}`)
          + (r.k != null ? tipRow("", "parameters", String(r.k)) : "")
          + tipRow("", "status", status);
        placeTip(tip, ev.clientX - rect.left, ev.clientY - rect.top, W);
      });
    });
  }

  function rankTable(rows, showFamily) {
    const body = rows.map((r) => `<tr><td>${r.rec ? "★ " : ""}${esc(r.label)}${showFamily && FAMILY[r.family] ? `<div class="sub">${esc(FAMILY[r.family])}</div>` : ""}</td>`
      + `<td class="num">${r.k == null ? "–" : r.k}</td><td class="num">${fmtNum(r.d, 3)}</td><td class="num">${fmtNum(r.se, 2)}</td>`
      + `<td>${r.ref ? "reference" : r.tie ? "tied" : ""}</td></tr>`).join("");
    return `<details class="note"><summary>Show as table</summary><div class="table-wrap" style="margin-top:8px"><table class="table mini">`
      + `<thead><tr><th>model</th><th class="num">k</th><th class="num">Δ</th><th class="num">SE</th><th></th></tr></thead><tbody>${body}</tbody></table></div></details>`;
  }

  // ---------------------------------------------------------------- speed-up dumbbell chart (log seconds)
  function speedChart(host, rows) {
    host.innerHTML = "";
    const muted = cssVar("--muted"), s1 = cssVar("--s1"), text2 = cssVar("--text-2"), gridC = cssVar("--grid"), surf = cssVar("--surface"), elev = cssVar("--elev");
    const legend = document.createElement("div");
    legend.className = "legend";
    legend.innerHTML = `<span><i>${swDot(muted)}</i>v0.1.0: general least squares, NumPy loops</span><span><i>${swDot(s1)}</i>v0.2.0: variable projection, IRLS, C kernels</span>`;
    host.appendChild(legend);
    const box = document.createElement("div");
    box.className = "chart";
    host.appendChild(box);
    const W = Math.max(260, Math.floor(box.clientWidth || host.clientWidth));
    const rowH = 30;
    const labelW = Math.min(230, Math.max(120, Math.round(W * 0.34)));
    const m = { t: 8, r: 58, b: 40, l: labelW };
    const H = m.t + rows.length * rowH + m.b;
    const lo = Math.min(...rows.map((r) => r.after)) / 1.5, hi = Math.max(...rows.map((r) => r.before)) * 1.3;
    const sx = mkScale("log", lo, hi, m.l, W - m.r);
    const svg = S("svg", { width: W, height: H, role: "img", "aria-label": "benchmark before and after" }, box);
    logTicks(lo, hi).forEach((v) => {
      const x = sx(v);
      S("line", { x1: x, x2: x, y1: m.t, y2: H - m.b, stroke: gridC }, svg);
      S("text", { x, y: H - m.b + 16, "text-anchor": "middle" }, svg).textContent = v < 1 ? `${+(v * 1000).toPrecision(3)} ms` : `${v} s`;
    });
    S("text", { x: (m.l + W - m.r) / 2, y: H - 6, "text-anchor": "middle", class: "axis-label" }, svg).textContent = "wall-clock time per analyze() call (log scale)";
    const tip = document.createElement("div");
    tip.className = "tooltip";
    box.appendChild(tip);
    rows.forEach((r, i) => {
      const cy = m.t + i * rowH + rowH / 2;
      const band = S("rect", { x: 0, y: cy - rowH / 2 + 1, width: W, height: rowH - 2, rx: 6, fill: "transparent" }, svg);
      const t = S("text", { x: m.l - 12, y: cy + 4, "text-anchor": "end", "pointer-events": "none" }, svg);
      t.textContent = fitText(r.label, m.l - 20);
      t.setAttribute("style", `font-family: var(--sans); font-size: 12px; fill: ${text2}`);
      const xa = sx(r.before), xb = sx(r.after);
      S("line", { x1: xb, x2: xa, y1: cy, y2: cy, stroke: muted, "stroke-width": 2, "pointer-events": "none" }, svg);
      S("circle", { cx: xa, cy, r: 5, fill: muted, stroke: surf, "stroke-width": 2, "pointer-events": "none" }, svg);
      S("circle", { cx: xb, cy, r: 6, fill: s1, stroke: surf, "stroke-width": 2, "pointer-events": "none" }, svg);
      const speed = r.before / r.after;
      const label = S("text", { x: W - m.r + 10, y: cy + 4, "text-anchor": "start", "pointer-events": "none" }, svg);
      label.textContent = `${speed.toFixed(speed < 10 ? 1 : 0)}×`;
      label.setAttribute("style", `fill: ${speed >= 1.5 ? cssVar("--text") : muted}; font-weight: 600`);
      band.addEventListener("pointerenter", () => band.setAttribute("fill", elev));
      band.addEventListener("pointerleave", () => { band.setAttribute("fill", "transparent"); tip.classList.remove("on"); });
      band.addEventListener("pointermove", (ev) => {
        const rect = svg.getBoundingClientRect();
        tip.innerHTML = `<div class="th">${esc(r.label)}</div>` + tipRow(swDot(muted), "v0.1.0", secs(r.before))
          + tipRow(swDot(s1), "v0.2.0", secs(r.after)) + tipRow("", "speed-up", `${speed.toFixed(1)}×`)
          + tipRow("", "same recommendation", r.same ? "yes" : "no");
        placeTip(tip, ev.clientX - rect.left, ev.clientY - rect.top, W);
      });
    });
  }

  // ---------------------------------------------------------------- hero
  let termTimer = null;
  function renderHeroTerm() {
    const host = $("#heroTerm");
    clearTimeout(termTimer);
    const c = D.cases[0];
    const lines = [{ prompt: '>>> report = analyze(df["ln(K/F)"], df["implied_vol"])' }]
      .concat(c.log.map((l) => ({ tag: l.tag, text: l.text })))
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
    if (reduceMotion) { lines.forEach(add); return; }
    let i = 0;
    const step = () => { add(lines[i++]); if (i < lines.length) termTimer = setTimeout(step, i === 1 ? 650 : 420); };
    termTimer = setTimeout(step, 300);
  }

  function renderStats() {
    const hits = D.scoreboard.filter((r) => r.verdict === "hit").length;
    const b = D.benchmark;
    const items = [
      [D.stats.candidates, "candidate forms"],
      b ? [`${b.total_speedup.toFixed(1)}×`, "faster than v0.1.0"] : [D.stats.likelihoods, "likelihoods"],
      [`${hits}/${D.scoreboard.length}`, "known models recovered"],
      [D.stats.tests, "regression tests"],
    ];
    $("#stats").innerHTML = items.map(([v, l]) => `<div class="stat"><b class="mono">${v}</b><span>${esc(l)}</span></div>`).join("");
  }

  // ---------------------------------------------------------------- cases
  const CASES = [
    { n: "01", tab: "Volatility smile", kicker: "curve mode" },
    { n: "02", tab: "Market impact", kicker: "curve mode · noise model" },
    { n: "03", tab: "Loan defaults", kicker: "multi-feature mode" },
    { n: "04", tab: "Random-walk trap", kicker: "guardrail" },
  ];

  function renderTabs() {
    const host = $("#caseTabs");
    host.innerHTML = CASES.map((c, i) => `<button class="tab" role="tab" type="button" id="tab-${i}" aria-controls="casePanel" aria-selected="${i === state.caseIdx}" tabindex="${i === state.caseIdx ? 0 : -1}"><span class="n">${c.n}</span>${esc(c.tab)}</button>`).join("");
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
    $$(".tab").forEach((b, j) => { b.setAttribute("aria-selected", String(i === j)); b.tabIndex = i === j ? 0 : -1; });
    $("#casePanel").setAttribute("aria-labelledby", `tab-${i}`);
    try { history.replaceState(null, "", `${location.pathname}${location.search}#case-${D.cases[i].id}`); } catch (e) { /* file:// or sandboxed */ }
    renderCase();
  }

  function caseHead(c, meta) {
    return `<div class="case-head"><div><div class="card-kicker mono">case ${meta.n} · ${esc(meta.kicker)} · analyze() in ${esc(secs(c.seconds))}</div>`
      + `<h3>${esc(c.title)}</h3><p>${esc(c.claim)}</p></div>`
      + `<div class="badges"><div class="badge"><span>ground truth</span><b>${esc(c.truth)}</b></div>`
      + `<div class="badge ok"><span>identified</span><b>${esc(c.identified)}</b></div></div></div>`;
  }
  function logCard(c) {
    const lines = c.log.map((l) => `<div class="term-line${l.tag === "verdict" ? " verdict" : ""}"><span class="tag">${esc(l.tag)}</span><span class="txt">${esc(l.text)}</span></div>`).join("");
    return `<div class="log"><div class="term-bar"><span></span><span></span><span></span><div class="term-title mono">decision log</div></div><div class="term-body mono">${lines}</div></div>`;
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
      if (s.truth) return { label: s.label, x: s.x, y: s.y, color: cssVar("--text"), dash: "1.5 5", width: 2.2 };
      const style = { label: s.label, x: s.x, y: s.y, color: pal[k], dash: DASHES[k] };
      k += 1;
      return style;
    });
  }
  const RANK_SUB = "Held-out NLL per row relative to the best model, ±1 paired SE. ◆ = non-parametric reference. Dashed line = practical-equivalence gap.";

  function renderSmile(panel, c) {
    const params = c.params.map((p) => {
      const rel = p.truth ? (p.fitted - p.truth) / Math.abs(p.truth) : null;
      return `<tr><td class="mono">${esc(p.name)}</td><td class="num">${fmtNum(p.truth, 4)}</td><td class="num">${fmtNum(p.fitted, 4)}</td><td class="num">${rel == null ? "–" : (rel >= 0 ? "+" : "") + (rel * 100).toFixed(1) + "%"}</td></tr>`;
    }).join("");
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2">${card("Data, top-3 forms and the ground truth", "Hover to compare fitted values.", '<div id="chartMain"></div>')}${logCard(c)}</div>`
      + `<div class="grid-2">${card("Cross-validated ranking", RANK_SUB, '<div id="chartRank"></div>' + rankTable(c.ranking))}`
      + card("Parameter recovery", "a′ = 10⁴·a/T and b′ = 10⁴·b/T put SVI in vol-% units.",
        `<div class="table-wrap"><table class="table mini"><thead><tr><th>parameter</th><th class="num">truth</th><th class="num">fitted</th><th class="num">error</th></tr></thead><tbody>${params}</tbody></table></div>`
        + `<p class="note">a′ and σ slide along a ridge — raising one and lowering the other barely changes the curve — so each is loosely identified on its own. Their combination at the smile's vertex is pinned down to about 2%.</p>`) + `</div>`);
    const ch = c.chart;
    lineChart($("#chartMain"), { xLabel: ch.xLabel, yLabel: ch.yLabel, xShort: "ln(K/F)", points: ch.points, pointsLabel: "observed", series: styledSeries(ch.series), aria: c.title });
    rankChart($("#chartRank"), c.ranking, { aria: "ranking" });
  }

  function renderImpact(panel, c) {
    const params = c.params.map((p) => `<tr><td>${esc(p.name)}</td><td class="num">${fmtNum(p.truth, 4)}</td><td class="num">${fmtNum(p.fitted, 4)}</td></tr>`).join("");
    const toggle = `<div class="seg" role="group" aria-label="axis scale"><button type="button" data-scale="lin" aria-pressed="${!state.impactLog}">linear</button><button type="button" data-scale="log" aria-pressed="${state.impactLog}">log–log</button></div>`;
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2">${card("Impact vs participation", "Curves show the conditional median. On log–log axes a square-root law is a straight line with slope ½.", '<div id="chartMain"></div>', toggle)}${logCard(c)}</div>`
      + `<div class="grid-2">${card("Ranking across both noise models", RANK_SUB, '<div id="chartRank"></div>' + rankTable(c.ranking, true))}`
      + card("Evidence", "Noise model first, then the shape.",
        `<div class="bigpair"><div><b>${fmtNum(c.families.multiplicative, 3)}</b><span>best multiplicative model, Δ</span></div><div><b>${fmtNum(c.families.additive, 3)} <span class="mono" style="font-size:13px">± ${fmtNum(c.families.additive_se, 2)}</span></b><span>best additive model, Δ</span></div></div>`
        + `<div class="table-wrap"><table class="table mini"><thead><tr><th>quantity</th><th class="num">truth</th><th class="num">fitted</th></tr></thead><tbody>${params}</tbody></table></div>`
        + `<p class="note">The free power-law exponent is fitted independently of the √q form and lands on ½.</p>`) + `</div>`);
    const draw = () => {
      const ch = c.chart;
      lineChart($("#chartMain"), {
        xLabel: ch.xLabel, yLabel: ch.yLabel, xShort: "q", points: ch.points, pointsLabel: "observed trades",
        series: styledSeries(ch.series), xType: state.impactLog ? "log" : "linear", yType: state.impactLog ? "log" : "linear", aria: c.title,
      });
    };
    $$(".seg button", panel).forEach((b) => b.addEventListener("click", () => {
      state.impactLog = b.dataset.scale === "log";
      $$(".seg button", panel).forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      draw();
    }));
    draw();
    rankChart($("#chartRank"), c.ranking, { showFamily: true, aria: "ranking" });
  }

  function renderLoans(panel, c) {
    const effects = c.panels.map((p, i) => card(`${esc(p.feature)}: ${esc(p.trend)}`,
      `importance ${pct(p.importance)} · fold-to-fold curve correlation ${p.stability == null ? "–" : p.stability.toFixed(2)}`, `<div id="effect${i}"></div>`)).join("");
    const feats = c.features.map((f) => `<tr><td>${esc(f.name)}</td><td>${esc(f.trend)}</td><td class="num">${f.importance == null ? "–" : pct(f.importance)}</td><td class="num">${f.stability == null ? "–" : f.stability.toFixed(2)}</td></tr>`).join("");
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2 even">${effects}</div>`
      + `<div class="grid-2">${card("Model comparison and features", RANK_SUB,
        '<div id="chartRank"></div>' + rankTable(c.ranking)
        + `<div class="table-wrap" style="margin-top:12px"><table class="table mini"><thead><tr><th>feature</th><th>shape</th><th class="num">importance</th><th class="num">fold corr.</th></tr></thead><tbody>${feats}</tbody></table></div>`)}${logCard(c)}</div>`);
    c.panels.forEach((p, i) => lineChart($(`#effect${i}`), {
      xLabel: p.feature, yLabel: "contribution to logit(p)", xShort: p.feature, height: 250,
      series: styledSeries([{ label: "fitted", x: p.x, y: p.fitted }, { label: "ground truth", x: p.x, y: p.truth, truth: true }]),
      rug: p.rug, aria: p.feature,
    }));
    rankChart($("#chartRank"), c.ranking, { aria: "model comparison" });
  }

  function renderPath(panel, c) {
    const checks = c.checks.map((k) => `<tr><td>${esc(k.name)}</td><td class="num">${esc(typeof k.value === "number" ? fmtNum(k.value, 3) : k.value)}</td><td class="num">${esc(k.rule)}</td>`
      + `<td><span class="status ${k.pass ? "good" : "critical"}">${k.pass ? "✓ pass" : "✗ fail"}</span></td></tr>`).join("");
    const r = c.rates;
    const rates = [
      ["smooth exponential curve + noise", r.false_positive.smooth, "false alarm"],
      ["linear trend + noise", r.false_positive.linear, "false alarm"],
      ["bond price–yield curve", r.false_positive.bond, "false alarm"],
      ["GBM price path", r.detection.gbm, "detected"],
      ["AR(1), φ = 0.95", r.detection.ar, "detected"],
    ].map(([n, v, kind]) => `<tr><td>${esc(n)}</td><td class="num">${pct(v)}</td><td class="sub">${esc(kind)}</td></tr>`).join("");
    const dist = c.returns.ranking.map((d) => `<tr><td>${d.rec ? "★ " : ""}${esc(d.label)}</td><td class="num">${d.k}</td><td class="num">${fmtNum(d.d_aic, 3)}</td></tr>`).join("");
    panel.insertAdjacentHTML("beforeend",
      `<div class="grid-2">${card("A path that seems to have a shape", "The two best-scoring “curves” under rolling-origin CV. They fit — and mean nothing.", '<div id="chartMain"></div>')}${logCard(c)}</div>`
      + `<div class="grid-2 even">${card("Random-walk test", `All four conditions must hold. Error rates measured on ${r.n} simulated series each.`,
        `<div class="table-wrap"><table class="table mini"><thead><tr><th>statistic</th><th class="num">value</th><th class="num">rule</th><th></th></tr></thead><tbody>${checks}</tbody></table></div>`
        + `<div class="table-wrap" style="margin-top:12px"><table class="table mini"><thead><tr><th>simulated series</th><th class="num">rate</th><th></th></tr></thead><tbody>${rates}</tbody></table></div>`)}`
      + card("What it analyses instead: log returns", "Histogram density with the two best distributions by AIC.",
        '<div id="chartReturns"></div>' + `<div class="table-wrap" style="margin-top:10px"><table class="table mini"><thead><tr><th>distribution</th><th class="num">k</th><th class="num">ΔAIC</th></tr></thead><tbody>${dist}</tbody></table></div>`) + `</div>`);
    const ch = c.chart;
    const pal = PALETTE();
    lineChart($("#chartMain"), {
      xLabel: ch.xLabel, yLabel: ch.yLabel, time: true, points: ch.points, pointsLabel: "monthly NAV", yFromData: true,
      series: ch.series.map((s, i) => ({ label: s.label, x: s.x, y: s.y, color: pal[i + 1], dash: DASHES[i + 1] })), aria: c.title,
    });
    lineChart($("#chartReturns"), {
      xLabel: "monthly log return", yLabel: "density", bars: c.returns.bins, barsLabel: "observed", height: 240,
      series: c.returns.series.map((s, i) => ({ label: s.label, x: s.x, y: s.y, color: pal[i], dash: DASHES[i] })), xShort: "r", aria: "returns",
    });
  }

  function renderCase() {
    const c = D.cases[state.caseIdx], meta = CASES[state.caseIdx];
    const panel = $("#casePanel");
    panel.innerHTML = caseHead(c, meta);
    ({ smile: renderSmile, impact: renderImpact, loans: renderLoans, path: renderPath })[c.id](panel, c);
  }

  // ---------------------------------------------------------------- scoreboard
  const VERDICT = { hit: ["good", "✓", "recovered"], tie: ["warning", "≈", "tied with truth"], miss: ["critical", "✗", "missed"] };
  const MODE = { curve: "curve", distribution: "distribution", guardrail: "guardrail", additive: "multi-feature" };
  function renderScoreboard() {
    const rows = D.scoreboard;
    const count = (v) => rows.filter((r) => r.verdict === v).length;
    $("#scoreSummary").innerHTML = ["hit", "tie", "miss"].map((v) => `<span class="status ${VERDICT[v][0]}">${VERDICT[v][1]} ${count(v)} ${VERDICT[v][2]}</span>`).join("");
    $("#scoreTable").innerHTML = `<thead><tr><th>#</th><th>scenario</th><th>mode</th><th>ground truth</th><th>identified</th><th>verdict</th><th>detail</th></tr></thead><tbody>`
      + rows.map((r, i) => {
        const v = VERDICT[r.verdict];
        return `<tr><td class="num">${String(i + 1).padStart(2, "0")}</td><td>${esc(r.name)}</td><td><span class="mode">${esc(MODE[r.mode])}</span></td>`
          + `<td>${esc(r.truth)}</td><td>${esc(r.identified)}</td><td><span class="status ${v[0]}">${v[1]} ${v[2]}</span></td><td class="sub">${esc(r.detail)}</td></tr>`;
      }).join("") + "</tbody>";
  }

  // ---------------------------------------------------------------- performance
  function renderPerformance() {
    const b = D.benchmark;
    const section = $("#performance");
    if (!b) { section.hidden = true; return; }
    $("#perfStats").innerHTML = [
      [`${b.total_speedup.toFixed(1)}×`, "less total time over 11 workloads"],
      [`${b.max_speedup.toFixed(0)}×`, `largest gain: ${b.max_label}`],
      [`${b.kernel_speedup.toFixed(0)}×`, "C smoother vs NumPy (n = 3,000)"],
      [`${b.same}/${b.rows.length}`, "identical recommendations"],
    ].map(([v, l]) => `<div class="stat"><b class="mono">${v}</b><span>${esc(l)}</span></div>`).join("");
    speedChart($("#perfChart"), b.rows);
    $("#perfMachine").textContent = b.machine;
  }

  // ---------------------------------------------------------------- library
  function renderLibrary() {
    const present = new Set(D.library.map((f) => f.group));
    const groups = [{ key: "all", label: "all" }].concat(D.groups.filter((g) => present.has(g.key)));
    $("#libChips").innerHTML = groups.map((g) => `<button class="chipbtn" type="button" data-g="${g.key}" aria-pressed="${state.libGroup === g.key}">${esc(g.label)}</button>`).join("");
    $$("#libChips .chipbtn").forEach((b) => b.addEventListener("click", () => { state.libGroup = b.dataset.g; renderLibrary(); }));
    const q = state.libQuery.trim().toLowerCase();
    const items = D.library.filter((f) => (state.libGroup === "all" || f.group === state.libGroup)
      && (!q || [f.key, f.formula, f.label, f.meaning].join(" ").toLowerCase().includes(q)));
    $("#libGrid").innerHTML = items.length ? items.map((f) => `<div class="form"><div class="form-top"><b>${esc(f.label)}</b><span class="meta">${esc(f.groupLabel)}</span></div>`
      + `<code>${esc(f.formula)}</code><p>${esc(f.meaning)}</p><span class="meta">key: ${esc(f.key)} · k = ${f.k} · ${f.separable ? "variable projection" : "general least squares"}</span></div>`).join("")
      : `<p class="note">No form matches.</p>`;
  }

  // ---------------------------------------------------------------- wiring
  $("#version").textContent = "v" + D.version;
  $("#generated").textContent = D.generated;
  $("#themeBtn").addEventListener("click", () => {
    const next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    store.set("fc-theme", next);
    renderCase();
    renderPerformance();
  });
  $("#libSearch").addEventListener("input", (ev) => { state.libQuery = ev.target.value; renderLibrary(); });
  $$(".copy").forEach((b) => b.addEventListener("click", async () => {
    const text = $("#" + b.dataset.target).innerText;
    try { await navigator.clipboard.writeText(text); b.textContent = "copied"; }
    catch (e) { b.textContent = "select & copy"; }
    setTimeout(() => { b.textContent = "copy"; }, 1400);
  }));

  let lastWidth = 0, resizeTimer = null;
  new ResizeObserver((entries) => {
    const w = Math.round(entries[0].contentRect.width);
    if (Math.abs(w - lastWidth) < 4) return;
    lastWidth = w;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { renderCase(); renderPerformance(); }, 120);
  }).observe($("#casePanel"));

  const deep = /^#case-(\w+)$/.exec(location.hash);
  if (deep) {
    const i = D.cases.findIndex((c) => c.id === deep[1]);
    if (i >= 0) state.caseIdx = i;
  }
  renderStats();
  renderHeroTerm();
  renderTabs();
  renderCase();
  renderScoreboard();
  renderPerformance();
  renderLibrary();
  if (deep) $("#cases").scrollIntoView();
})();
