// Draws the chart strips as SVG. Nothing here knows about any particular measurement.

const SVG_NS = "http://www.w3.org/2000/svg";
const PAD_Y = 7;
const RIGHT_ROOM = 8;

export function svgNode(tag, attributes = {}, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, value);
  if (text !== undefined) node.textContent = text;
  return node;
}

export function timeToX(t, tMax, width) {
  return (t / tMax) * Math.max(width - RIGHT_ROOM, 1);
}

// Lines every so often, with labels that stay readable however long the session runs.
export function timeTicks(tMax) {
  const choices = [
    [0.5, 1], [1, 2], [3, 6], [6, 12], [12, 24], [24, 48], [24, 168], [168, 336], [168, 720],
  ];
  const [minor, major] = choices.find(([, m]) => tMax / m <= 8) || [720, 2160];
  const ticks = [];
  for (let t = 0; t <= tMax + 1e-6; t += minor) {
    ticks.push({ t, major: Math.abs(t % major) < 1e-6, label: tickLabel(t) });
  }
  return ticks;
}

function tickLabel(t) {
  if (t === 0) return "Start";
  if (t < 48) return `${t} h`;
  return `${+(t / 24).toFixed(1)} d`;
}

// A score has fixed limits. Anything else fits its data, but never zooms in far enough
// to make measurement noise look like a real change.
export function domainFor(series, { unit = "", noise = 0.05, threshold }) {
  if (unit.includes("0 to 10")) return [0, 10];
  if (unit.includes("0 to 1")) return [0, 1];
  const values = series.flatMap((p) => [p.p10, p.p90]).filter(Number.isFinite);
  if (!values.length) return [0, 1];
  let low = Math.min(...values);
  let high = Math.max(...values);
  const span = Math.max(high - low, 2 * noise);
  if (threshold !== undefined && threshold > low - span && threshold < high + span) {
    low = Math.min(low, threshold);
    high = Math.max(high, threshold);
  }
  const room = Math.max(high - low, span) * 0.1;
  const minimum = Math.min(...values) >= 0 ? 0 : -Infinity;
  const top = Math.max(high, low + span) + room;
  return [Math.max(low - room, minimum), top];
}

function drawGrid(svg, ticks, tMax, width, height) {
  for (const tick of ticks) {
    const x = timeToX(tick.t, tMax, width);
    svg.append(svgNode("line", {
      x1: x, x2: x, y1: 0, y2: height, class: tick.major ? "grid-major" : "grid-minor",
    }));
  }
}

function linePath(points, x, y, key) {
  return points.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p[key]).toFixed(1)}`).join("");
}

function bandPath(points, x, y) {
  const top = points.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)} ${y(p.p90).toFixed(1)}`).join("");
  const bottom = [...points].reverse().map((p) => `L${x(p.t).toFixed(1)} ${y(p.p10).toFixed(1)}`).join("");
  return `${top}${bottom}Z`;
}

function roundedScale(value) {
  const size = Math.abs(value);
  return String(size >= 100 ? Math.round(value) : +value.toPrecision(3));
}

export function drawStrip(svg, options) {
  const { width, height, tMax, ticks, points, ghost, domain, threshold, thresholdLabel, doseTimes } = options;
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const x = (t) => timeToX(t, tMax, width);
  const y = (value) => {
    const fraction = (value - domain[0]) / (domain[1] - domain[0] || 1);
    return height - PAD_Y - fraction * (height - 2 * PAD_Y);
  };

  drawGrid(svg, ticks, tMax, width, height);
  for (const t of doseTimes) {
    svg.append(svgNode("line", { x1: x(t), x2: x(t), y1: 0, y2: height, class: "dose-line" }));
  }
  if (threshold !== undefined && threshold >= domain[0] && threshold <= domain[1]) {
    svg.append(svgNode("line", { x1: 0, x2: width, y1: y(threshold), y2: y(threshold), class: "limit" }));
    svg.append(svgNode("text", {
      x: width - RIGHT_ROOM - 2, y: y(threshold) - 3, "text-anchor": "end", class: "limit-text",
    }, thresholdLabel));
  }
  if (points.length > 1) {
    svg.append(svgNode("path", { d: bandPath(points, x, y), class: "band-fill" }));
  }
  if (ghost && ghost.length > 1) {
    svg.append(svgNode("path", { d: linePath(ghost, x, y, "p50"), class: "ghost" }));
  }
  if (points.length > 1) {
    svg.append(svgNode("path", { d: linePath(points, x, y, "p50"), class: "trace" }));
  }
  const last = points[points.length - 1];
  if (last) svg.append(svgNode("circle", { cx: x(last.t), cy: y(last.p50), r: 3, class: "now-dot" }));

  svg.append(svgNode("text", { x: 4, y: 12, class: "scale-text" }, roundedScale(domain[1])));
  svg.append(svgNode("text", { x: 4, y: height - 3, class: "scale-text" }, roundedScale(domain[0])));
}

export function drawDoseRow(svg, { width, height, tMax, ticks, marks, ghostMarks = [] }) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  const x = (t) => timeToX(t, tMax, width);
  drawGrid(svg, ticks, tMax, width, height);
  for (const t of ghostMarks) {
    svg.append(svgNode("line", { x1: x(t), x2: x(t), y1: height - 8, y2: height - 3, class: "dose-mark ghosted" }));
  }
  for (const mark of marks) {
    if (mark.waiting) {
      svg.append(svgNode("circle", { cx: x(mark.t), cy: height / 2, r: 3.5, class: "dose-future" }));
    } else if (mark.exposure) {
      const cx = x(mark.t);
      svg.append(svgNode("path", { d: `M${cx - 5} ${height - 5}L${cx + 5} ${height - 5}L${cx} 5Z`, class: "exposure-mark" }));
    } else {
      svg.append(svgNode("line", { x1: x(mark.t), x2: x(mark.t), y1: 4, y2: height - 4, class: "dose-mark" }));
    }
  }
}

export function drawAxis(svg, { width, height, tMax, ticks }) {
  svg.replaceChildren();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  for (const tick of ticks) {
    if (!tick.major) continue;
    const x = timeToX(tick.t, tMax, width);
    const anchor = tick.t === 0 ? "start" : x > width - 24 ? "end" : "middle";
    svg.append(svgNode("text", { x, y: height - 6, "text-anchor": anchor, class: "axis-text" }, tick.label));
  }
}
