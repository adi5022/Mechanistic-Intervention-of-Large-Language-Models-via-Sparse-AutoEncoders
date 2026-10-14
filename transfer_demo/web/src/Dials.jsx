import { useEffect, useMemo, useState } from "react";

// The dial visualiser.
//
// WHAT IT SHOWS
// An edit is 200 dials: one multiplier for each of the 200 SAE features that fire on the sentence. A multiplier of 1 means "leave this
// feature alone", above 1 turns it up (max 3), below 1 turns it down (min 0). Gradient descent starts with every dial at 1 and nudges
// them, step by step, until the target word becomes the model's first choice. This component draws those 200 dials and redraws them at
// every step, so you watch the optimiser move.
//
// TWO VIEWS OF THE SAME NUMBERS
//  - Ring: the 200 dials around a circle. The thin circle is "no change" (1x). A dial turned up grows outward from it, a dial turned
//    down shrinks inward. Easy to see at a glance whether the edit is a few big moves or a lot of small ones, and the middle of the ring
//    has room for the number that matters most (the target's rank).
//  - Bars: the same dials as bars on a straight baseline. Easier to read exact heights and to compare two dials.
// In both views the dials are sorted by how strongly the feature fires on this sentence (strongest first), because those are the ones
// that can change the output most.
//
// TWO MODES
//  - live: the run is going; frames are drawn at a steady pace (see LIVE PACING).
//  - replay: the run is over; the slider and Play button go back through every step.
//
// props.dials = { fids, strength, frames, ranks }   frames[t][k] is dial k's multiplier after step t; ranks[t] is the target's rank in Small
// (empty for Direct runs, which look at Medium instead).

import { Tip } from "./Tip.jsx";
import { TIPS } from "./tips.js";

const M_MAX = 3; // largest multiplier an edit can use

// ---- straight bars: drawing size in SVG units (the SVG scales to the width of its box) ----
const W = 1000, PLOT_TOP = 14, PLOT_H = 200, PAD_L = 46, PAD_R = 8;

// ---- ring: a 460 x 460 square, centre at (230, 230). A multiplier m sits at radius R_IN + m * (R_OUT - R_IN) / 3,
//      so 0x is the small inner circle, 1x ("no change") the middle circle, 3x the outer edge. ----
const S = 460, C = S / 2, R_IN = 70, R_OUT = 214;
const rOf = (m) => R_IN + (m / M_MAX) * (R_OUT - R_IN);

export default function Dials({ dials, live = false, done = false, onCaughtUp, layer = 8 }) {
  const { fids = [], strength = [], frames = [], ranks = [] } = dials || {};
  const [view, setView] = useState("ring"); // "ring" or "bars"
  const [pos, setPos] = useState(null); // step shown in replay mode (null = the last one)
  const [playing, setPlaying] = useState(false);
  const [hover, setHover] = useState(null); // index of the dial under the mouse
  const [shown, setShown] = useState(0); // step shown in live mode (see the note below)

  const last = frames.length - 1;
  const t = live ? Math.min(shown, last) : Math.min(pos ?? last, last); // the step being drawn

  // LIVE PACING. On a fast GPU all 100 steps arrive within a couple of seconds, and the early steps swing the dials hard, so drawing every
  // step the moment it arrives looks like flicker. Instead the frames are queued and drawn at a steady 20 a second. If the drawing falls
  // behind the run it speeds up a little (2 or 3 steps per tick) so it never lags far. When the run is over and the last frame has been
  // drawn, we tell the page (onCaughtUp) so it can hand over to the replay panel in the results.
  useEffect(() => {
    if (!live) return;
    const id = setInterval(() => {
      setShown((s) => {
        const backlog = last - s;
        return backlog <= 0 ? s : s + (backlog > 40 ? 3 : backlog > 15 ? 2 : 1);
      });
    }, 50);
    return () => clearInterval(id);
  }, [live, last]);
  useEffect(() => {
    if (live && done && last >= 0 && shown >= last && onCaughtUp) onCaughtUp();
  }, [live, done, shown, last, onCaughtUp]);

  // Left-to-right (or clockwise) order: strongest-firing feature first. Computed once per run, not per frame.
  const order = useMemo(() => fids.map((_, i) => i).sort((a, b) => strength[b] - strength[a]), [fids, strength]);

  // Play: step forward every 40 ms (about 25 steps a second) and stop at the end. Pressing Play again at the end starts over.
  useEffect(() => {
    if (!playing) return;
    const id = setInterval(() => {
      setPos((p) => {
        const next = (p ?? last) + 1;
        if (next >= last) setPlaying(false);
        return Math.min(next, last);
      });
    }, 40);
    return () => clearInterval(id);
  }, [playing, last]);

  if (!fids.length) return <p className="muted">Waiting for the first step…</p>;
  if (t < 0) return <p className="muted">Choosing the dials…</p>;

  const m = frames[t];

  // The numbers in the caption: how many dials are up, how many down, and the biggest mover so far.
  let up = 0, down = 0, big = 0;
  m.forEach((v, k) => {
    if (v > 1.005) up++;
    else if (v < 0.995) down++;
    if (Math.abs(v - 1) > Math.abs(m[big] - 1)) big = k;
  });
  // The six dials that have moved furthest from 1 (links go to the feature's page on Neuronpedia, which needs internet).
  const movers = m.map((v, k) => [k, v]).sort((a, b) => Math.abs(b[1] - 1) - Math.abs(a[1] - 1)).slice(0, 6);
  const rank = ranks[t];

  const caption = (
    <p className="dialcap">
      {hover !== null ? (
        <>feature #{fids[hover]} · fires at {strength[hover].toFixed(2)} on this sentence · multiplier <b>{m[hover].toFixed(2)}×</b></>
      ) : (
        <>
          step <b>{t}</b> of {Math.max(last, t)}
          {rank !== undefined && <> · target rank in Small <b>{rank.toLocaleString("en-US")}</b></>}
          {" · "}
          <Tip tip={TIPS.turnedUp}><span className="up-text">{up} turned up</span></Tip>, <Tip tip={TIPS.turnedDown}><span className="down-text">{down} turned down</span></Tip>, {fids.length - up - down} unchanged
          {" · "}biggest change: feature #{fids[big]} at <b>{m[big].toFixed(2)}×</b>
        </>
      )}
    </p>
  );

  const side = (
    <>
      {caption}
      {/* the target's rank in Small at every step, so you can see WHEN the dials made the difference */}
      {ranks.length > 1 && <RankCurve ranks={ranks} t={t} />}
      {/* replay controls (hidden while the run is still going) */}
      {!live && last > 0 && (
        <div className="replay">
          <button onClick={() => { if (t >= last) setPos(0); setPlaying(!playing); }}>{playing ? "Pause" : "Play"}</button>
          <input type="range" min="0" max={last} value={t} onChange={(e) => { setPlaying(false); setPos(Number(e.target.value)); }} />
        </div>
      )}
      <div className="movers">
        <Tip tip={TIPS.movers}><span className="muted">Biggest movers:</span></Tip>
        {movers.map(([k, v]) => (
          <a key={k} href={`https://www.neuronpedia.org/gpt2-small/${layer}-res-jb/${fids[k]}`} target="_blank" rel="noreferrer"
             className={v > 1 ? "up-text" : "down-text"} onMouseEnter={() => setHover(k)} onMouseLeave={() => setHover(null)}>
            #{fids[k]} {v.toFixed(2)}×
          </a>
        ))}
      </div>
    </>
  );

  return (
    <div className={`dials ${view}`}>
      <div className="viewtoggle">
        {["ring", "bars"].map((v) => (
          <button key={v} className={view === v ? "on" : ""} onClick={() => setView(v)}>{v === "ring" ? "Ring" : "Bars"}</button>
        ))}
      </div>
      {view === "ring" ? (
        <div className="ringgrid">
          <RingView m={m} order={order} hover={hover} setHover={setHover} t={t} rank={rank} />
          <div>{side}</div>
        </div>
      ) : (
        <>
          <BarsView m={m} order={order} hover={hover} setHover={setHover} />
          {side}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------------------------------------------------------------
// RING. Dial number i (in strength order) sits at angle i * 360 / 200 degrees, clockwise from the top. Each dial is a thin bar drawn
// pointing straight up from the centre and then rotated to its angle. Up-dials start at the "no change" circle and grow outward;
// down-dials start there and shrink inward. y and height are animated with CSS, same as the straight bars.
// ---------------------------------------------------------------------------------------------------------------------------------
function RingView({ m, order, hover, setHover, t, rank }) {
  const n = order.length;
  const r1 = rOf(1);
  const w = ((2 * Math.PI * r1) / n) * 0.72; // bar thickness: 72% of the gap between neighbours, so they do not touch
  return (
    <svg viewBox={`0 0 ${S} ${S}`} className="ringsvg" onMouseLeave={() => setHover(null)}>
      {[0, 2, 3].map((g) => <circle key={g} cx={C} cy={C} r={rOf(g)} className="dial-grid" fill="none" />)}
      <circle cx={C} cy={C} r={r1} className="dial-base" fill="none" />
      {[1, 2, 3].map((g) => <text key={g} x={C + 4} y={C - rOf(g) + 11} className="dial-axis">{g}×</text>)}
      {order.map((k, i) => {
        const v = m[k];
        const kind = v > 1.005 ? "up" : v < 0.995 ? "down" : "rest";
        const rv = rOf(v);
        const y = C - Math.max(rv, r1); // the outer end of the bar, measured upward from the centre
        const h = Math.max(Math.abs(rv - r1), 1.2);
        return (
          <g key={k} transform={`rotate(${(i * 360) / n} ${C} ${C})`}>
            <rect x={C - w / 2} width={w} y={y} height={h} className={`dial ${kind}${hover === k ? " hot" : ""}`} onMouseEnter={() => setHover(k)} />
          </g>
        );
      })}
      {/* centre: the number that matters (the target's rank in Small), or the step when there is no rank (Direct runs) */}
      <text x={C} y={C - 2} textAnchor="middle" className="ring-big">{rank !== undefined ? rank.toLocaleString("en-US") : t}</text>
      <text x={C} y={C + 20} textAnchor="middle" className="dial-axis">{rank !== undefined ? "target rank in Small" : "step"}</text>
    </svg>
  );
}

// ---------------------------------------------------------------------------------------------------------------------------------
// BARS. Same dials on a straight baseline: x is the dial's position in strength order, height is how far it is from 1.
// ---------------------------------------------------------------------------------------------------------------------------------
function BarsView({ m, order, hover, setHover }) {
  const barW = (W - PAD_L - PAD_R) / order.length;
  const yOf = (v) => PLOT_TOP + (1 - v / M_MAX) * PLOT_H; // multiplier -> vertical position (3 at the top, 0 at the bottom)
  const y1 = yOf(1);
  return (
    <svg viewBox={`0 0 ${W} ${PLOT_TOP + PLOT_H + 18}`} className="dialsvg" onMouseLeave={() => setHover(null)}>
      {[0, 1, 2, 3].map((g) => (
        <g key={g}>
          <line x1={PAD_L} x2={W - PAD_R} y1={yOf(g)} y2={yOf(g)} className={g === 1 ? "dial-base" : "dial-grid"} />
          <text x={PAD_L - 6} y={yOf(g) + 4} textAnchor="end" className="dial-axis">{g}×</text>
        </g>
      ))}
      {order.map((k, i) => {
        const v = m[k];
        const top = Math.min(yOf(v), y1);
        const h = Math.max(Math.abs(yOf(v) - y1), 1.2); // unmoved dials still show as a thin tick on the baseline
        const kind = v > 1.005 ? "up" : v < 0.995 ? "down" : "rest";
        return (
          <rect key={k} x={PAD_L + i * barW + barW * 0.14} width={barW * 0.72} y={kind === "down" ? y1 : top} height={h}
                className={`dial ${kind}${hover === k ? " hot" : ""}`} onMouseEnter={() => setHover(k)} />
        );
      })}
      <text x={PAD_L} y={PLOT_TOP + PLOT_H + 14} className="dial-axis">strongest-firing feature</text>
      <text x={W - PAD_R} y={PLOT_TOP + PLOT_H + 14} textAnchor="end" className="dial-axis">weakest of the 200</text>
    </svg>
  );
}

// The target's rank in Small after each step. Drawn on a log scale because the rank can start in the hundreds and end at 1,
// and on a plain scale the last few steps (the interesting ones) would be squashed flat. Rank 1 is at the top: higher is better.
function RankCurve({ ranks, t }) {
  const W2 = 1000, H2 = 70, L = 46, R = 8, T = 6, B = 16;
  const lr = ranks.map((r) => Math.log10(Math.max(r, 1)));
  const hi = Math.max(...lr, 0.3); // log10 of the worst rank seen
  const x = (i) => L + (i / Math.max(ranks.length - 1, 1)) * (W2 - L - R);
  const y = (v) => T + (v / hi) * (H2 - T - B); // rank 1 -> v = 0 -> top
  const upto = ranks.slice(0, t + 1).map((_, i) => `${x(i)},${y(lr[i])}`).join(" ");
  const all = ranks.map((_, i) => `${x(i)},${y(lr[i])}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W2} ${H2}`} className="ranksvg">
      <text x={L - 6} y={y(0) + 4} textAnchor="end" className="dial-axis">1</text>
      <text x={L - 6} y={y(hi) + 4} textAnchor="end" className="dial-axis">{Math.round(10 ** hi)}</text>
      <line x1={L} x2={W2 - R} y1={y(0)} y2={y(0)} className="dial-grid" />
      <polyline points={all} className="rank-ghost" />
      <polyline points={upto} className="rank-line" />
      <circle cx={x(Math.min(t, ranks.length - 1))} cy={y(lr[Math.min(t, ranks.length - 1)])} r="4" className="rank-dot" />
      <text x={L} y={H2 - 2} className="dial-axis">target's rank in Small, by step (log scale, 1 = first choice)</text>
    </svg>
  );
}
