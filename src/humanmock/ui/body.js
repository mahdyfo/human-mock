// The body diagram, drawn like the outline on a medical chart. A measurement chooses where
// it appears with body_part in its organ file. To add a place, add it to PARTS.

import { svgNode } from "./chart.js";

const OUTLINE =
  "M91 62C91 72 86 76 76 79C64 82 54 86 50 98C45 118 42 140 38 160C34 182 30 200 27 218C25 228 24 238 27 246" +
  "C29 252 35 252 37 246C39 238 40 230 42 220C47 200 54 180 57 160C60 142 63 124 66 112C67 130 68 150 70 170" +
  "C71 186 67 200 66 216C65 240 68 280 71 320C72 344 72 368 74 392C74 402 72 410 76 414C82 417 94 417 95 411" +
  "C96 400 95 380 95 360C95 330 97 290 98 250Q100 244 102 250C103 290 105 330 105 360C105 380 104 400 105 411" +
  "C106 417 118 417 124 414C128 410 126 402 126 392C128 368 128 344 129 320C132 280 135 240 134 216" +
  "C133 200 129 186 130 170C132 150 133 130 134 112C137 124 140 142 143 160C146 180 153 200 158 220" +
  "C160 230 161 238 163 246C165 252 171 252 173 246C176 238 175 228 173 218C170 200 166 182 162 160" +
  "C158 140 155 118 150 98C146 86 136 82 124 79C114 76 109 72 109 62Z";

const VESSELS =
  "M100 92L100 214M100 214L85 300L85 396M100 214L115 300L115 396" +
  "M100 92Q72 92 60 118L46 172L33 232M100 92Q128 92 140 118L154 172L167 232M100 92L100 60";

const ellipse = (cx, cy, rx, ry) => ({ tag: "ellipse", cx, cy, rx, ry });

// In order from the top of the body down, which is also the order of the list beside it.
const PARTS = [
  ["head", "Head", [ellipse(100, 30, 14, 16)]],
  ["throat", "Throat", [ellipse(100, 68, 7.5, 8)]],
  ["lungs", "Lungs", [ellipse(83, 112, 11, 23), ellipse(117, 112, 11, 23)]],
  ["heart", "Heart", [ellipse(104, 118, 8, 9)]],
  ["liver", "Liver", [ellipse(87, 151, 14, 8.5)]],
  ["kidneys", "Kidneys", [ellipse(77, 171, 5, 8), ellipse(123, 171, 5, 8)]],
  ["gut", "Gut", [ellipse(100, 186, 19, 18)]],
  ["skin", "Whole body", []],
  ["blood", "Blood", []],
];

// Fixed places for germs, spread around the throat. The more germs, the more places fill up.
const GERM_PLACES = [
  [100, 68], [96, 64], [104, 65], [97, 72], [103, 71], [100, 62], [94, 68], [106, 69],
  [99, 75], [93, 62], [107, 62], [101, 58], [92, 73], [108, 74], [96, 57], [105, 56],
  [90, 66], [110, 66],
];

export function partLabel(name) {
  const known = PARTS.find(([id]) => id === name);
  return known ? known[1] : name.charAt(0).toUpperCase() + name.slice(1);
}

export function partOrder(name) {
  const position = PARTS.findIndex(([id]) => id === name);
  return position === -1 ? PARTS.length : position;
}

function shape({ tag, ...attributes }, extra) {
  return svgNode(tag, { ...attributes, ...extra });
}

export function drawBody(svg, { parts, germs, drug }) {
  const heatOf = (name) => (parts.get(name) ? parts.get(name).heat : 0);
  const nodes = [
    svgNode("ellipse", { cx: 100, cy: 34, rx: 22, ry: 27, class: "body-outline" }),
    svgNode("path", { d: OUTLINE, class: "body-outline" }),
  ];

  // Fever colours the whole body, faintly enough that the organs stay readable.
  const fever = heatOf("skin");
  if (fever > 0) {
    nodes.push(
      svgNode("ellipse", { cx: 100, cy: 34, rx: 21, ry: 26, class: "body-fever", "fill-opacity": fever * 0.3 }),
      svgNode("path", { d: OUTLINE, class: "body-fever", "fill-opacity": fever * 0.3 }),
    );
  }
  for (const [name, label, shapes] of PARTS) {
    for (const piece of shapes) {
      const node = shape(piece, { class: "body-part", "fill-opacity": 0.04 + heatOf(name) * 0.76 });
      node.append(svgNode("title", {}, label));
      nodes.push(node);
    }
  }

  const inflamed = heatOf("blood");
  if (inflamed > 0) {
    nodes.push(svgNode("path", { d: VESSELS, class: "body-vessel inflamed", "stroke-opacity": inflamed * 0.7 }));
  }
  if (drug > 0.002) {
    nodes.push(svgNode("path", { d: VESSELS, class: "body-vessel drug", "stroke-opacity": 0.15 + drug * 0.75 }));
  }

  const count = Math.round(germs * GERM_PLACES.length);
  for (const [cx, cy] of GERM_PLACES.slice(0, count)) {
    nodes.push(svgNode("circle", { cx, cy, r: 1.5, class: "germ" }));
  }
  svg.replaceChildren(...nodes);
}
