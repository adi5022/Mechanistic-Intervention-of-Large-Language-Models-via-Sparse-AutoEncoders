import { useEffect, useRef, useState } from "react";
import { TIPS } from "./tips.js";

// The bridge: a picture of what the experiment does.
//
// Two panels stand side by side, one per model. Each panel has the model's name, a box with the target word's rank, and a stack of its
// layers, numbered from 1 at the top. GPT-2 Small has 12 layers (thick slabs) and GPT-2 Medium has 24 (slabs half as thick), so you can
// see that Medium is twice as deep. The edit is made at ONE layer of Small, crosses the bridge (the translator, drawn as an arc), and is
// added to ONE layer of Medium. The pill on the arc says how big the injected change is.
//
// Layer numbers: the study's "layer 8" is the point after 8 blocks have run (blocks.8.hook_resid_pre), so it is drawn as the 8th slab.
//
// The picture changes with what is happening (props.phase):
//   idle      nothing is running: the panels sit still and the bridge is a slack, almost flat string.
//   tuning    the optimiser is adjusting the dials. Small's edit layer pulses, and a dashed arrow shows which model's answer the optimiser
//             is watching (Small's for the normal methods, Medium's for Direct).
//   crossing  the dials are tuned and the change is being carried across: a marker runs along the bridge and Medium's layer pulses.
//   done      the run is finished: the bridge stays lit, the pill shows the injected change, and each rank box shows before -> after.
//
// The bridge gets thicker with the dose, because the dose is the number that multiplies the change before it goes into Medium.
// Everything here is drawn from numbers the page already has; nothing is computed in this file.
//
// props: pair [small layer, medium layer], mapLabel, dose, phase, direct (true for the Direct/D6 method),
//        small / medium = { before, after } target ranks (after is null until a run has finished), injFrac (size of the injected change,
//        as a share of Medium's state), nSmall / nMedium = number of layers.

const W = 1000, H = 640;
const PANEL_W = 310, PANEL_Y = 60, PANEL_H = 566;       // each model panel
const SX = 66, MX = 624;                                  // left edge of the Small panel and of the Medium panel
const STACK_TOP = 200, STACK_BOTTOM = 500;                // both stacks are 300 units tall, so Medium's slabs are half as thick
const SLAB_X = 61, SLAB_W = 213;                          // slabs sit inside the panel: this far from its left edge, this wide

// Slab number l (0 = first layer, drawn at the top) of a stack with n layers.
function slab(l, n) {
  const unit = (STACK_BOTTOM - STACK_TOP) / n;
  const gap = n > 12 ? 3 : 4;
  return { y: STACK_TOP + l * unit + gap / 2, h: unit - gap, mid: STACK_TOP + (l + 0.5) * unit };
}

export default function Bridge({ pair, mapLabel, mapHelp = "", facts, dose, phase, direct, small, medium, injFrac, nSmall = 12, nMedium = 24 }) {
  const [sl, ml] = pair;
  const s = slab(sl - 1, nSmall);   // the edit layer in Small, and the layer that receives the change in Medium
  const m = slab(ml - 1, nMedium);
  const x0 = SX + SLAB_X + SLAB_W;  // where the bridge leaves Small (right end of its edit slab)
  const x1 = MX + SLAB_X;           // where it arrives in Medium (left end of its slab)
  // THE BRIDGE IS A VIBRATING STRING. It is a sine wave with 4 full cycles between Small's edit slab and Medium's slab. The wave travels
  // along the string (its phase moves), and how it moves depends on what the experiment is doing, like a string that goes slack or tight:
  //   idle      almost flat and still                    (slack)
  //   tuning    small slow ripple                        (the optimiser is feeling for the right dials)
  //   crossing  tall fast wave, taller with the dose     (the change is being carried across)
  //   done      calm steady wave, a little taller with the dose
  // The wave height and speed do not jump to their new values; they ease towards them over about a third of a second, so you see the
  // tension build up and release. An envelope (sin of pi*t) pins both ends of the string to the nodes so the wave always meets them.
  // The marker that rides the wave while crossing is placed in the same loop, on the current shape of the wave.
  const CYCLES = 4, POINTS = 80;
  const target = {
    idle: { amp: 2, speed: 0.4 },
    tuning: { amp: 9, speed: 2.5 },
    crossing: { amp: 14 + dose * 3, speed: 9 },
    done: { amp: 6 + dose * 1.5, speed: 1.6 },
  }[phase];
  const targetRef = useRef(target);
  targetRef.current = target;
  const phaseRef = useRef(phase);
  phaseRef.current = phase;
  const [wave, setWave] = useState({ ph: 0, amp: target.amp, u: 0 }); // ph: how far the wave has travelled, amp: current height, u: marker position 0..1
  useEffect(() => {
    // Respect "reduce motion": draw one still wave at the target height and do not animate.
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      setWave({ ph: 0, amp: targetRef.current.amp, u: 0.5 });
      return;
    }
    let raf, last = performance.now(), ph = 0, amp = targetRef.current.amp, u = 0;
    const tick = (now) => {
      const dt = Math.min((now - last) / 1000, 0.1); // seconds since the last frame (capped so a paused tab does not make a huge jump)
      last = now;
      const tg = targetRef.current;
      amp += (tg.amp - amp) * Math.min(1, dt * 3);           // ease the height towards its target
      speedNow += (tg.speed - speedNow) * Math.min(1, dt * 3); // and the speed
      ph += speedNow * dt;
      if (phaseRef.current === "crossing") u = (u + dt * 0.8) % 1;
      setWave({ ph, amp, u });
      raf = requestAnimationFrame(tick);
    };
    let speedNow = targetRef.current.speed;
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);
  const pts = Array.from({ length: POINTS + 1 }, (_, i) => {
    const t = i / POINTS;
    const env = Math.sin(Math.PI * t);
    return [x0 + t * (x1 - x0), s.mid + t * (m.mid - s.mid) - wave.amp * env * Math.sin(2 * Math.PI * CYCLES * t - wave.ph)];
  });
  const path = "M " + pts.map(([px, py]) => `${px.toFixed(1)} ${py.toFixed(1)}`).join(" L ");
  const marker = pts[Math.round(wave.u * POINTS)];
  const apex = (s.mid + m.mid) / 2 - 4;   // the middle of the wave: the pill's stem ends here (the wave is on its centre line at this point)
  const lit = phase === "crossing" || phase === "done";
  const width = 2 + dose * 1.2;
  const mid = (x0 + x1) / 2;

  // HOVER POP-UPS. Every labelled part of the picture explains itself when the mouse is over it. The box follows the mouse and is drawn
  // as normal page text on top of the picture (not inside the SVG), so it can wrap and be styled like the other pop-ups on the page.
  // T(entry) returns the handlers to spread onto an SVG element: `<g {...T(TIPS.rank)}>`. The wording is in tips.js.
  const wrapRef = useRef(null);
  const [tip, setTip] = useState(null); // { entry, x, y, w }: what to show and where the mouse is, measured inside the picture
  const place = (entry, e) => {
    const r = wrapRef.current.getBoundingClientRect();
    setTip({ entry, x: e.clientX - r.left, y: e.clientY - r.top, w: r.width });
  };
  const T = (entry) => ({
    onMouseEnter: (e) => place(entry, e),
    // stopPropagation: a part nested inside another (the "200 SAE features" line inside the footer box) must win over its parent
    onMouseMove: (e) => { e.stopPropagation(); place(entry, e); },
    onMouseLeave: () => setTip(null),
    style: { cursor: "help" },
  });
  const tipSmall = TIPS.small(facts), tipMedium = TIPS.medium(facts), tipSae = TIPS.saeFeatures(facts);
  const tipTranslator = TIPS.translator(mapLabel, mapHelp, dose);

  // One model's panel: name, rank box, stack of layers, footer, and the "sentence in -> prediction out" line.
  const panel = (x, name, sub, n, hot, rank, kind, footTitle, footSub, nameTip, footTip) => (
    <g>
      <rect x={x} y={PANEL_Y} width={PANEL_W} height={PANEL_H} rx="8" className="br-panel" />
      <g {...T(nameTip)}>
        <rect x={x + 10} y={PANEL_Y + 8} width={PANEL_W - 20} height="62" fill="transparent" />
        <text x={x + 20} y={PANEL_Y + 42} className="br-name">{name}</text>
        <text x={x + 20} y={PANEL_Y + 62} className="br-sub">{sub}</text>
      </g>
      <line x1={x + 20} x2={x + PANEL_W - 20} y1={PANEL_Y + 74} y2={PANEL_Y + 74} className="br-rule" />
      <g {...T(TIPS.rank)}>
      <rect x={x + 20} y={PANEL_Y + 82} width={PANEL_W - 40} height="47" rx="4" className="br-box" />
      <text x={x + 32} y={PANEL_Y + 111} className="br-label">TARGET RANK</text>
      {/* The box has room for about 130 units of text next to its label. A short rank ("2 -> 1") is drawn at full size; a long one
          ("1,278 -> 249") is drawn smaller so it never runs over the label. */}
      <text x={x + PANEL_W - 32} y={PANEL_Y + 114} textAnchor="end" className="br-rank"
            style={{ fontSize: (() => { const len = rank.before.toLocaleString("en-US").length + (rank.after != null ? 3 + rank.after.toLocaleString("en-US").length : 0); return len <= 6 ? 26 : len <= 8 ? 21 : len <= 10 ? 17 : 14; })() }}>
        {rank.before.toLocaleString("en-US")}
        {rank.after != null && (
          <>
            <tspan className="br-arrowtxt"> → </tspan>
            <tspan className="br-after">{rank.after.toLocaleString("en-US")}</tspan>
          </>
        )}
      </text>
      </g>
      {Array.from({ length: n }, (_, l) => {
        const p = slab(l, n);
        const isHot = l === hot - 1;
        const pulse = isHot && ((kind === "edit" && phase === "tuning") || (kind === "inject" && phase === "crossing"));
        return (
          <g key={l} {...T(isHot ? footTip : TIPS.layer)}>
            <text x={x + 40} y={p.mid + 3.5} textAnchor="end" className={`br-num${isHot ? " hot" : ""}${n > 12 ? " small" : ""}`}>{l + 1}</text>
            <rect x={x + SLAB_X} y={p.y} width={SLAB_W} height={p.h} rx={n > 12 ? 2 : 3}
                  className={`slab${isHot ? ` hot${pulse ? " pulse" : ""}${kind === "inject" && !lit ? " soft" : ""}` : ""}`} />
          </g>
        );
      })}
      <g {...T(footTip)}>
        <rect x={x + 20} y={PANEL_Y + 454} width={PANEL_W - 40} height="48" rx="4" className="br-box" />
        <text x={x + PANEL_W / 2} y={PANEL_Y + (footSub ? 475 : 483)} textAnchor="middle" className="br-foot">{footTitle}</text>
        {footSub && <text x={x + PANEL_W / 2} y={PANEL_Y + 493} textAnchor="middle" className="br-footsub" {...T(tipSae)}>{footSub}</text>}
      </g>
      <g {...T(TIPS.io)}>
      <rect x={x + 14} y={PANEL_Y + 524} width={PANEL_W - 28} height="24" fill="transparent" />
      <text x={x + 20} y={PANEL_Y + 541} className="br-io">SENTENCE IN</text>
      <line x1={x + 106} x2={x + 176} y1={PANEL_Y + 538} y2={PANEL_Y + 538} className="br-ioline" />
      <path d={`M ${x + 170} ${PANEL_Y + 534} L ${x + 177} ${PANEL_Y + 538} L ${x + 170} ${PANEL_Y + 542}`} className="br-ioline" fill="none" />
      <text x={x + PANEL_W - 20} y={PANEL_Y + 541} textAnchor="end" className="br-io">PREDICTION OUT</text>
      </g>
    </g>
  );

  return (
    <div className="tipwrap" ref={wrapRef}>
    <svg viewBox={`0 0 ${W} ${H}`} className="bridgesvg">
      <defs>
        <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto">
          <path d="M0,0 L8,4 L0,8 z" className="br-arrowfill" />
        </marker>
      </defs>

      {panel(SX, "GPT-2 Small", `${nSmall} layers · SAE available`, nSmall, sl, small, "edit", `INTERVENTION · LAYER ${sl}`, "200 SAE features", tipSmall, TIPS.intervention)}
      {panel(MX, "GPT-2 Medium", `${nMedium} layers · no SAE`, nMedium, ml, medium, "inject", `INJECTION · LAYER ${ml}`, null, tipMedium, TIPS.injection)}

      {/* ---- what the optimiser is watching (only while it tunes) ---- */}
      {phase === "tuning" && !direct && (
        <>
          <path d={`M ${SX + PANEL_W - 20} ${PANEL_Y + 105} C ${x0 + 90} ${PANEL_Y + 105}, ${x0 + 90} ${s.mid + 24}, ${x0 + 2} ${s.mid + 12}`} className="br-feedback" markerEnd="url(#arrowhead)" />
          <text x={x0 + 100} y={PANEL_Y + 150} className="br-cap warm">tuned on Small's answer</text>
        </>
      )}
      {phase === "tuning" && direct && (
        <>
          <path d={`M ${MX + 20} ${PANEL_Y + 130} C ${mid + 40} ${s.mid + 170}, ${x0 + 60} ${s.mid + 170}, ${x0 + 2} ${s.mid + 12}`} className="br-feedback" markerEnd="url(#arrowhead)" />
          <text x={mid} y={s.mid + 150} textAnchor="middle" className="br-cap warm">tuned on Medium's answer</text>
        </>
      )}

      {/* ---- the bridge (the translator) ---- */}
      <path d={path} className={`br-bridge${lit ? " lit" : ""}`} style={{ strokeWidth: width }} />
      {/* an invisible, much thicker copy of the string: the real one is thin and hard to hover */}
      <path d={path} fill="none" stroke="transparent" strokeWidth="28" {...T(tipTranslator)} />
      <circle cx={x0} cy={s.mid} r="5" className="br-node" {...T(tipTranslator)} />
      <circle cx={x1} cy={m.mid} r="5" className="br-node" {...T(tipTranslator)} />
      {phase === "crossing" && <circle cx={marker[0]} cy={marker[1]} r={5 + dose * 1.2} className="br-marker" />}
      {/* the caption is centred in the gap between the two panels, not on the arc, so it never touches either panel */}
      {/* two lines: the first (map name) can be long, e.g. "RIDGE MAP (TRAINED)", and one line would run past the panels' edges */}
      <text x={(SX + PANEL_W + MX) / 2} y={apex - 152} textAnchor="middle" className="br-cap" {...T(tipTranslator)}>
        <tspan x={(SX + PANEL_W + MX) / 2}>{`TRANSLATOR · ${mapLabel.replace(" trained on edits", " (trained)").toUpperCase()}`}</tspan>
        <tspan x={(SX + PANEL_W + MX) / 2} dy="15">{`DOSE ×${dose.toFixed(1)}`}</tspan>
      </text>
      {phase === "done" && injFrac != null && (
        <g {...T(TIPS.state((100 * injFrac).toFixed(1)))}>
          <line x1={mid} x2={mid} y1={apex - 66} y2={apex} className="br-rule" />
          <rect x={mid - 70} y={apex - 120} width="140" height="54" rx="14" className="br-pill" />
          <text x={mid} y={apex - 90} textAnchor="middle" className="br-pillnum">{(100 * injFrac).toFixed(1)}%</text>
          <text x={mid} y={apex - 73} textAnchor="middle" className="br-pillcap">STATE CHANGE</text>
        </g>
      )}
    </svg>
    {tip && (
      <div className="tipbox floating" role="tooltip" style={{ left: Math.max(0, Math.min(tip.x + 14, tip.w - 300)), top: tip.y + 18 }}>
        <b>{tip.entry.title}</b>
        {tip.entry.text}
      </div>
    )}
    </div>
  );
}
