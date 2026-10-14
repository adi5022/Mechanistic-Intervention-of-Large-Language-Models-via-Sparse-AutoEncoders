import { useEffect, useRef, useState } from "react";
import { TIPS } from "./tips.js";

// The rank panels (the file keeps its old name, Ladder).
//
// WHAT IT SHOWS
// Two cards side by side, one per model. Inside each card a vertical rank axis stands on the left (rank 1, the model's first choice for the
// next word, at the TOP; worse ranks lower down) and a table of the model's top 5 next words stands on the right. The target word is a marker
// that travels from its starting rank (a grey dot) to its final rank along a curved dashed path, so moving UP means the model has been pushed
// towards putting the target first.
// A model's marker reaches the top (rank 1) only when that model is the one the dials were tuned on: in a normal run Small's marker climbs to
// rank 1 while Medium, which only receives the translated change, moves partway; in a Direct run it is the other way round.
// When Small's edit is finished, one wave draws itself across the gap between the cards (the change crossing over) and a pill under it says
// how big that change is. The wave exists only from that moment on, so it always means "the change has crossed".
//
// HOW THE ALIGNMENT IS KEPT
// The cards are HTML in a three-column grid (card, gap, card), not one big drawing, so text wraps normally and both cards always have the same
// rows at the same heights: heading, axis label, then the axis next to the table. Both axes use the same drawing size, so rank 1 is at the
// same height in both cards, and the wave in the gap is placed at that height (see the stylesheet, .ld2-arcsvg).
//
// WHY A LOG SCALE
// Ranks run from 1 to the thousands. On a plain scale rank 1 and rank 2 would sit almost on top of each other and a move from 169 to 22
// would look like a move from 169 to 0. On a log scale each doubling is the same distance, so "2 to 1" and "169 to 22" are both visible.
// Both axes share one scale, so the heights of the two markers can be compared directly.
//
// THE TABLE
// It shows the top 5 words BEFORE the edit while the markers are travelling and switches to the list AFTER the edit when they arrive, so you
// can see what the target overtook (for example " French" going from rank 1 to rank 2). If the target is not in the top 5 afterwards, an
// extra highlighted row shows its own rank.
//
// WHAT IS REAL AND WHAT IS NOT (this matters if someone asks)
//  - Small's climb in a normal run is real: the target's rank after every gradient step was recorded (res.dials.ranks).
//  - Medium's move in a normal run is one jump: Medium is only given the finished change, so it has no steps.
//  - In a Direct run the rank after each tuning step is not recorded for either model, so each marker waits at its start and then moves to
//    its final rank. The movement is drawn smoothly for readability; the path between start and end is NOT data, and the caption says so.
//  - The curved dashed path and the arc between the cards are drawings. Only the heights on the axis (the ranks) and the
//    state-change number are data.
//
// props: res (one finished run), smallTop (Small's top-5 words before the edit: [[word, probability], ...]), facts (sizes of the two models)

const DURATION = 12000;                 // the replay takes 12 seconds, slow enough to follow
const V_W = 130, V_H = 290;             // drawing size of one vertical axis (fixed width, so it never squeezes the table)
const V_TOP = 16, V_BOT = 262;          // where rank 1 (top) and the worst rank (bottom) sit
const AXV = 54;                         // x position of the axis line; tick numbers stand to its left
const OFF = 30;                         // how far to the right of the axis the grey start dot sits
const COL = 106;                        // x of the column where the rank numbers are written, beyond the dashed path

export default function Ladder({ res, smallTop, facts }) {
  const direct = res.map === "direct";
  const trace = (res.dials && res.dials.ranks) || [];
  const sStart = res.small_before.rank, sEnd = res.small_after.rank;
  const mStart = res.medium_before.rank, mEnd = res.medium_after.rank;

  // f is the replay clock from 0 to 1. It starts by itself when the run appears; Play/Pause and the slider control it afterwards.
  const [f, setF] = useState(0);
  const [playing, setPlaying] = useState(true);
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setF((x) => {
        if (x >= 1) { setPlaying(false); return 1; }
        return Math.min(1, x + 40 / DURATION);
      });
    }, 40);
    return () => clearInterval(id);
  }, [playing]);

  // One scale for both axes: the worst rank anywhere in this run sets the bottom end (at least 10 so short runs are not squashed).
  const worst = Math.max(10, sStart, sEnd, mStart, mEnd, ...trace);
  const yOf = (rank) => V_TOP + (Math.log10(Math.max(rank, 1)) / Math.log10(worst)) * (V_BOT - V_TOP);
  const ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000].filter((t) => t <= worst);

  const lerp = (a, b, t) => 10 ** (Math.log10(a) + (Math.log10(b) - Math.log10(a)) * t); // move in log space, like the axis
  const ease = (t) => 1 - (1 - t) ** 3;                                                // fast start, gentle landing
  const clamp01 = (v) => Math.min(1, Math.max(0, v));

  // Where each marker is at clock f, as a rank, and how far along its journey it is (u: 0 = at its start, 1 = arrived).
  //   normal run, Small:  follows the recorded steps from f = 0.05 to 0.70, then holds.
  //   transfer:           the arc draws itself from f = 0.70 to 0.80.
  //   normal run, Medium: waits until the arc has landed (f = 0.80), then makes its one move by f = 1.
  //   Direct run:         both wait until f = 0.15 and then move together until f = 0.85; the arc draws from f = 0.10 to 0.15.
  function where(start, end, kind) {
    if (kind === "recorded") {
      const p = clamp01((f - 0.05) / 0.65);
      const idx = Math.min(trace.length - 1, Math.floor(p * (trace.length - 1) + 1e-9));
      const rank = trace[idx];
      const u = start === end ? 0 : clamp01((Math.log10(start) - Math.log10(rank)) / (Math.log10(start) - Math.log10(end)));
      return { rank, u: p >= 1 ? 1 : u };
    }
    const t = kind === "late" ? clamp01((f - 0.8) / 0.2) : clamp01((f - 0.15) / 0.7);
    const u = ease(t);
    return { rank: lerp(start, end, u), u };
  }
  const sKind = !direct && trace.length > 1 ? "recorded" : "direct";
  const sNow = where(sStart, sEnd, sKind);
  const mNow = where(mStart, mEnd, direct ? "direct" : "late");
  const arrived = f >= 0.99;
  const [w0, w1] = direct ? [0.1, 0.15] : [0.7, 0.8];
  const chanP = clamp01((f - w0) / (w1 - w0));     // 0 before the transfer, 1 after it
  const crossing = chanP > 0 && chanP < 1;

  // HOVER POP-UPS (same idea as the bridge): T(entry) gives the handlers for any element, and the box follows the mouse.
  const wrapRef = useRef(null);
  const [tip, setTip] = useState(null);
  const place = (entry, e) => {
    const r = wrapRef.current.getBoundingClientRect();
    setTip({ entry, x: e.clientX - r.left, y: e.clientY - r.top, w: r.width });
  };
  const T = (entry) => ({
    onMouseEnter: (e) => place(entry, e),
    onMouseMove: (e) => { e.stopPropagation(); place(entry, e); },
    onMouseLeave: () => setTip(null),
    style: { cursor: "help" },
  });

  // The marker begins OFF to the right of the axis (where the grey dot is) and slides onto the axis as it arrives, so its path is a curve.
  const xOff = (u) => AXV + OFF * (1 - u) ** 1.6;
  const pathPoints = (start, end, uTo) => {
    const n = 48, pts = [];
    for (let i = 0; i <= Math.round(n * uTo); i++) {
      const u = i / n;
      pts.push(`${xOff(u).toFixed(1)},${yOf(lerp(start, end, u)).toFixed(1)}`);
    }
    return pts.length > 1 ? pts.join(" ") : null;
  };

  // One card's axis: a vertical line with a tick and a number at each rank (rank 1 at the top), the grey start dot, the dashed path so far
  // and the target marker. The two rank numbers (where it started, where it is now) are written in a column to the right (COL), joined to
  // their dots by thin dotted lines, so the dashed path never runs through the text; if the two dots are close, the lower number is
  // moved down so they do not overlap.
  const axis = (start, end, now) => {
    const trail = pathPoints(start, end, now.u);
    const mx = xOff(now.u), my = yOf(now.rank);
    const moved = Math.round(now.rank) !== start;
    let sy = yOf(start) + 5, ny = my + 5;
    if (moved && Math.abs(sy - ny) < 18) { if (ny >= sy) ny = sy + 18; else sy = ny + 18; }
    return (
      <svg viewBox={`0 0 ${V_W} ${V_H}`} className="ld2-axis">
        <g {...T(TIPS.ladderScale)}>
          <rect x="0" y="0" width={AXV + 8} height={V_H} fill="transparent" />
          <line x1={AXV} x2={AXV} y1={V_TOP - 10} y2={V_BOT + 10} className="ld2-line" />
          {ticks.map((t) => (
            <g key={t}>
              <line x1={AXV - 8} x2={AXV} y1={yOf(t)} y2={yOf(t)} className="ld2-tick" />
              <text x={AXV - 14} y={yOf(t) + 4} textAnchor="end" className={`ld2-ticknum${t === 1 ? " goal" : ""}`}>{t.toLocaleString("en-US")}</text>
            </g>
          ))}
        </g>
        {trail && <polyline points={trail} className="ld2-trail" {...T(TIPS.ladderMarker)} />}
        <circle cx={AXV + OFF} cy={yOf(start)} r="8" className="ld2-start" {...T(TIPS.ladderStart)} />
        <circle cx={mx} cy={my} r="11" className="ld2-marker" {...T(TIPS.ladderMarker)} />
        {moved && (
          <g>
            <line x1={AXV + OFF + 11} x2={COL - 6} y1={yOf(start)} y2={yOf(start)} className="ld2-leader" />
            <text x={COL} y={sy} className="ld2-startnum">{start.toLocaleString("en-US")}</text>
          </g>
        )}
        <line x1={mx + 14} x2={COL - 6} y1={my} y2={my} className="ld2-leader" />
        <text x={COL} y={ny} className="ld2-target">{Math.round(now.rank).toLocaleString("en-US")}</text>
      </svg>
    );
  };

  // The top-5 table: words before the edit while the markers travel, words after the edit once they have arrived.
  const table = (before, after, now) => {
    const showAfter = f >= 0.97;
    const list = (showAfter ? after : before) || [];
    const targetIn = list.some(([w]) => w === res.target);
    return (
      <div {...T(showAfter ? TIPS.ladderAfter : TIPS.ladderBefore)}>
        <div className="ld2-tablehead">TOP 5 NEXT WORDS {showAfter ? "AFTER" : "BEFORE"} THE EDIT</div>
        <div className="ld2-table">
          {list.slice(0, 5).map(([word], i) => (
            <div key={i} className={`ld2-row${word === res.target ? " hit" : ""}`}>
              <span className="n">{i + 1}</span>
              <span className="w">{JSON.stringify(word)}</span>
            </div>
          ))}
          {showAfter && !targetIn && (
            <div className="ld2-row hit extra">
              <span className="n">{Math.round(now.rank)}</span>
              <span className="w">{JSON.stringify(res.target)}</span>
            </div>
          )}
        </div>
      </div>
    );
  };

  const panel = (idx, name, sub, start, end, now, before, after, note) => (
    <section className="ld2-panel">
      <div className="ld2-head">
        <div {...T(idx === 0 ? TIPS.small(facts) : TIPS.medium(facts))}>
          <div className="ld2-name">{name}</div>
          <div className="ld2-sub">{sub}</div>
        </div>
        <div className="ld2-rankbox" {...T(TIPS.ladderRank)}>
          <div className="ld2-boxhead">{arrived ? "FINAL RANK" : "RANK NOW"}</div>
          <div className="ld2-bignum">{Math.round(now.rank).toLocaleString("en-US")}</div>
          <div className="ld2-boxsub">{start.toLocaleString("en-US")} → {end.toLocaleString("en-US")}</div>
        </div>
      </div>
      <div className="ld2-axislabel"><b>RANK</b> (log scale, higher up is better)</div>
      <div className="ld2-body">
        {axis(start, end, now)}
        {table(before, after, now)}
      </div>
      <div className="ld2-note">{note}</div>
    </section>
  );

  // THE TRANSFER WAVE lives in the gap between the cards. It is a sine wave with 3 full cycles, drawn at the height of the two axes (see
  // .ld2-arcsvg in the stylesheet): it starts at the left card's edge and lands with an arrowhead on the right card's edge. It draws itself
  // while chanP goes 0 to 1 and it keeps travelling afterwards (its phase keeps moving), like a vibrating string between the two models.
  // It starts tall (17 units) and settles to a calm ripple (7 units) as it lands, the same idea as the string on the bridge picture:
  // tension while the change is crossing, calm once it has arrived. The two ends are pinned (sin of pi*t), so it always meets both cards.
  const WAVE_N = 60, CYCLES = 3;
  const [ph, setPh] = useState(0);                 // how far the wave has travelled along the string
  useEffect(() => {
    // keep the wave moving only after the transfer has started, and not at all if the computer asks for reduced motion
    if (chanP <= 0 || (window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches)) return;
    const id = setInterval(() => setPh((p) => p + 0.3), 40);
    return () => clearInterval(id);
  }, [chanP > 0]);
  const amp = 7 + 10 * (1 - chanP);
  const wavePt = (t) => [4 + t * 192, 8 - amp * Math.sin(Math.PI * t) * Math.sin(2 * Math.PI * CYCLES * t - ph)];
  const waveTo = (tEnd) => {
    const n = Math.max(2, Math.round(WAVE_N * tEnd));
    return Array.from({ length: n + 1 }, (_, i) => wavePt((i / n) * tEnd).map((v) => v.toFixed(1)).join(",")).join(" ");
  };
  const wavePoints = waveTo(Math.max(chanP, 0.02));   // only the part that has been drawn so far
  const [pulseX, pulseY] = wavePt(chanP);
  const pct = res.injected_size_frac != null ? (100 * res.injected_size_frac).toFixed(1) : null;

  const smallNote = direct ? "dials were tuned on Medium; this is what they do here" : "dials tuned on this model's answer";
  const mediumNote = direct ? "dials tuned on this model's answer" : "receives the translated change only";
  return (
    <div className="tipwrap" ref={wrapRef}>
      <div className="ld2-grid">
        {panel(0, "GPT-2 Small", "12 layers · SAE available", sStart, sEnd, sNow, smallTop, res.small_after.top, smallNote)}
        <div className="ld2-mid">
          {chanP > 0 && (
            <>
              <div className="ld2-chanlabel" {...T(TIPS.ladderChannel)}>Translated activation change</div>
              <svg viewBox="0 -24 200 64" className="ld2-arcsvg">
                <polyline points={wavePoints} className="ld2-arc" />
                <polyline points={wavePoints} fill="none" stroke="transparent" strokeWidth="22" {...T(TIPS.ladderChannel)} />
                {crossing && <circle cx={pulseX} cy={pulseY} r="4.5" className="ld2-pulse" />}
                {chanP >= 1 && <path d="M 189 2 L 199 8 L 189 14 z" className="ld2-arrowhead" />}
              </svg>
              {chanP >= 1 && pct != null && (
                <div className="ld2-pill" {...T(TIPS.state(pct))}>
                  <div className="num">{pct}%</div>
                  <div className="cap">change in Medium's<br />state norm</div>
                </div>
              )}
            </>
          )}
        </div>
        {panel(1, "GPT-2 Medium", "24 layers · no SAE", mStart, mEnd, mNow, res.medium_before.top, res.medium_after.top, mediumNote)}
      </div>
      {tip && (
        <div className="tipbox floating" role="tooltip" style={{ left: Math.max(0, Math.min(tip.x + 14, tip.w - 300)), top: tip.y + 18 }}>
          <b>{tip.entry.title}</b>
          {tip.entry.text}
        </div>
      )}
      <div className="replay">
        <button onClick={() => { if (f >= 1) setF(0); setPlaying(!playing); }}>{playing ? "Pause" : f >= 1 ? "Replay" : "Play"}</button>
        <input type="range" min="0" max="1000" value={Math.round(f * 1000)} onChange={(e) => { setPlaying(false); setF(Number(e.target.value) / 1000); }} />
      </div>
      <p className="muted">
        <span className="legend-dot teal" /> target word {JSON.stringify(res.target)} <span className="legend-dot grey" /> where it started <span className="legend-dash" /> illustrative path (only the position along the axis is data).{" "}
        {direct
          ? "Direct run: the rank after each tuning step is not recorded, so each marker waits and then moves to its final rank."
          : "Small's climb is the recorded rank after each gradient step. Medium only receives the finished change, so it moves once, after the change has crossed."}
      </p>
    </div>
  );
}
