import { useEffect, useRef, useState } from "react";
import { getConfig, getReadout, getStatus, runEdit } from "./api.js";
import { BeforeAfter, Readout, diffText, num } from "./pieces.jsx";
import Dials from "./Dials.jsx";
import Bridge from "./Bridge.jsx";
import { Tip } from "./Tip.jsx";
import { TIPS } from "./tips.js";

// This page is the Streamlit "Cross-model transfer" tab, rebuilt. Same order of things:
//   1. prompt and target, 2. Small | Medium side by side (unedited), 3. the edit settings, 4. the runs, newest last.
// All the maths happens on the Python server; this file only collects the choices, asks the server, and shows the answers.

export default function App() {
  // ---- what the server tells us ---------------------------------------------------------------------------------------------
  const [status, setStatus] = useState(null); // models loaded? device? facts of both models
  const [config, setConfig] = useState(null); // texts, layer pairs, which maps exist for each pair

  // ---- what the user chooses (defaults are the study's: layers 8 to 16, dose 2.0, 100 steps, 200 candidate features) -----------
  const [prompt, setPrompt] = useState("The mother tongue of Danielle Darrieux is");
  const [target, setTarget] = useState(" English");
  const [pair, setPair] = useState([8, 16]);
  const [mechanism, setMechanism] = useState("Gradient descent (default)");
  const [mapKind, setMapKind] = useState("ridge");
  const [dose, setDose] = useState(2.0);
  const [steps, setSteps] = useState(100);
  const [topN, setTopN] = useState(200);
  const [addCap, setAddCap] = useState(1.0);

  // ---- results ----------------------------------------------------------------------------------------------------------------
  const [clean, setClean] = useState(null); // unedited ranking in both models for the current prompt and target
  const [runs, setRuns] = useState([]); // every finished run, in order
  const [note, setNote] = useState(null); // e.g. "the target is already rank 1"
  const [error, setError] = useState(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState({ frac: 0, text: "" });
  const [liveRank, setLiveRank] = useState(null); // target's rank in Small after the latest gradient step
  const [liveDials, setLiveDials] = useState(null); // the 200 dials as they move during a run: { fids, strength, frames, ranks }
  const [collapseOld, setCollapseOld] = useState(false);

  // The server loads the models in the background when it starts (about a minute). Until it is done, ask again every 2 seconds.
  useEffect(() => {
    let stop = false;
    const poll = async () => {
      try {
        const s = await getStatus();
        if (stop) return;
        setStatus(s);
        if (s.loading) setTimeout(poll, 2000);
        else getConfig().then((c) => !stop && setConfig(c));
      } catch {
        if (!stop) setTimeout(poll, 2000); // the server is not up yet
      }
    };
    poll();
    return () => {
      stop = true;
    };
  }, []);

  const ready = status && !status.loading && !status.error && config;

  // Re-read the unedited ranking whenever the prompt or target changes, but wait 400 ms after the last keystroke so we do not
  // run the models on every letter. `latest` makes sure a slow old answer cannot overwrite a newer one.
  const latest = useRef(0);
  useEffect(() => {
    if (!ready || !prompt.trim()) {
      setClean(null);
      return;
    }
    const id = ++latest.current;
    const t = setTimeout(() => {
      getReadout(prompt, target)
        .then((r) => id === latest.current && setClean(r))
        .catch((e) => id === latest.current && setError(String(e.message || e)));
    }, 400);
    return () => clearTimeout(t);
  }, [ready, prompt, target]);

  // Maps offered for the chosen pair; if the chosen map does not exist for a newly picked pair, fall back to the ridge map.
  const pairInfo = config?.pairs.find((p) => p.small === pair[0] && p.medium === pair[1]);
  const mapsHere = pairInfo?.maps ?? ["ridge"];
  useEffect(() => {
    if (!mapsHere.includes(mapKind)) setMapKind("ridge");
  }, [mapsHere.join(","), mapKind]);

  const isDirect = mapKind === "direct";
  const smallLayers = config ? [...new Set(config.pairs.map((p) => p.small))] : [];
  const mediumLayers = config ? [...new Set(config.pairs.map((p) => p.medium))] : [];

  // Why the Run button is off (the same checks as the Streamlit tab, minus the ones the server handles itself).
  let whyNot = "";
  if (!prompt.trim() || !target.trim()) whyNot = "Enter a prompt and a target.";
  else if (config && pair[0] !== config.sae_layer) whyNot = `Small layer ${pair[0]} needs its own SAE; this server holds the SAE for layer ${config.sae_layer}.`;
  else if (status && !status.maps_file_found) whyNot = `Translator file missing: ${status.maps_file}`;

  async function run() {
    setRunning(true);
    setError(null);
    setNote(null);
    setLiveRank(null);
    setLiveDials(null);
    setProgress({ frac: 0, text: "Starting" });
    try {
      await runEdit(
        { prompt, target, pair, mechanism, map: mapKind, dose, steps, top_n: topN, add_cap: addCap },
        (ev) => {
          if (ev.type === "progress") setProgress({ frac: ev.frac, text: ev.text });
          else if (ev.type === "dial_setup") setLiveDials({ fids: ev.fids, strength: ev.strength, frames: [], ranks: [] }); // which 200 dials there are
          else if (ev.type === "dials") setLiveDials((d) => d && { ...d, frames: [...d.frames, ev.m] }); // their values after one more step
          else if (ev.type === "step") {
            setLiveRank(ev.rank);
            setLiveDials((d) => d && { ...d, ranks: [...d.ranks, ev.rank] });
          } else if (ev.type === "result") {
            // the finished run carries the full record of the dials, so each run can be replayed later
            if (ev.result.note) setNote(ev.result.note);
            else setRuns((rs) => [...rs, ev.result]);
          } else if (ev.type === "error") setError(ev.text);
        }
      );
    } catch (e) {
      setError(String(e.message || e));
    }
    setRunning(false);
  }

  // Runs belong to the prompt and target they were made with; the list shows only those matching what is typed now.
  const myRuns = runs.filter((r) => r.prompt === prompt && r.target === " " + target.trim());
  const smallBefore = clean?.small && "target_rank" in clean.small ? { rank: clean.small.target_rank, prob: clean.small.target_prob } : null;

  // What the bridge picture should show. While a run is going it is "tuning" until the server reports it has reached the translate step
  // (progress 0.85), then "crossing". When a run has finished the picture shows that run's own settings and ranks, so it always matches
  // the result below it, even if the controls on the left have been changed since.
  const lastRun = myRuns[myRuns.length - 1];
  const phase = running ? (progress.frac < 0.85 ? "tuning" : "crossing") : lastRun ? "done" : "idle";
  const showRun = phase === "done" ? lastRun : null;
  const bridgeMap = showRun ? showRun.map : mapKind;
  const bridgeReady = clean?.small && "target_rank" in clean.small && clean?.medium && "target_rank" in clean.medium;

  // ---- loading / failure screens ----------------------------------------------------------------------------------------------
  if (!status) return <Page><main className="plain"><p>Waiting for the server (python transfer_demo/server.py)…</p></main></Page>;
  if (status.error) return <Page><main className="plain"><p className="warn">The server could not load the models: {status.error}</p></main></Page>;
  if (status.loading) return <Page><main className="plain"><p>Loading GPT-2 Small, its SAE and GPT-2 Medium (about a minute the first time)…</p></main></Page>;
  if (!config) return <Page><main className="plain"><p>Loading settings…</p></main></Page>;

  // facts about the two models (layers, size, SAE features), used by the hover pop-ups
  const facts = { small: status.small, medium: status.medium };

  // The page has two columns: the RAIL on the left holds every choice, in the order you make them (numbered like steps of a protocol);
  // the STAGE on the right shows what the models do. The rail stays in view while you scroll through the results, so you can change a
  // setting and run again without scrolling back up.
  return (
    <Page status={status}>
      <div className="layout">
        <aside className="rail">
          <div className="step">
            <div className="label" data-n="01"><Tip tip={TIPS.sentence}>Sentence</Tip></div>
            <label>
              Prompt
              <input value={prompt} onChange={(e) => setPrompt(e.target.value)} />
            </label>
            <label>
              Target completion
              <input value={target} onChange={(e) => setTarget(e.target.value)} />
              <span className="hint">One word; the leading space is added for you. Both models read the same prompt and target.</span>
            </label>
          </div>

          <div className="step">
            <div className="label" data-n="02"><Tip tip={TIPS.layers}>Translation layers</Tip></div>
            <p className="muted">
              The change an edit makes in Small's residual stream at one layer is translated by a linear map and added to Medium's at another layer.
              All 25 fitted pairs are offered; 8 to 16 was the primary pair of the study.
            </p>
            <div className="row">
              <label className="grow">
                Small layer
                <select value={pair[0]} onChange={(e) => setPair([Number(e.target.value), pair[1]])}>
                  {smallLayers.map((l) => <option key={l}>{l}</option>)}
                </select>
              </label>
              <label className="grow">
                Medium layer
                <select value={pair[1]} onChange={(e) => setPair([pair[0], Number(e.target.value)])}>
                  {mediumLayers.map((l) => <option key={l}>{l}</option>)}
                </select>
              </label>
            </div>
            {pairInfo && (
              <p className="muted">
                Fit (R2) <b>{pairInfo.r2.toFixed(2)}</b> · cosine <b>{pairInfo.cos.toFixed(2)}</b> · map file {status.maps_file_found ? "found" : "missing"} · {pairInfo.note}
              </p>
            )}
          </div>

          <div className="step">
            <div className="label" data-n="03"><Tip tip={TIPS.mechanism}>Edit mechanism (in Small)</Tip></div>
            {Object.entries(config.edit_methods).map(([name]) => (
              <label key={name} className="radio">
                {/* The Direct method makes its own edit (gradient descent on Medium's output), so this choice does nothing there; it is greyed out to say so. */}
                <input type="radio" name="mech" disabled={isDirect} checked={mechanism === name} onChange={() => setMechanism(name)} />
                {name}
              </label>
            ))}
            <p className="muted">{isDirect ? "Not used by Direct: the dials are tuned on Medium's output, whatever is chosen here." : config.edit_methods[mechanism]}</p>
          </div>

          <div className="step">
            <div className="label" data-n="04"><Tip tip={TIPS.map}>Translation map</Tip></div>
            {mapsHere.map((k) => (
              <label key={k} className="radio">
                <input type="radio" name="map" checked={mapKind === k} onChange={() => setMapKind(k)} />
                {config.map_choices[k]}
              </label>
            ))}
            <p className="muted">
              {config.map_help[mapKind]}
              {mapsHere.length < 4 && ` Maps offered for this layer pair: ${mapsHere.map((k) => config.map_choices[k].split(" (")[0]).join(", ")}.`}
            </p>
          </div>

          <div className="step">
            <div className="label" data-n="05"><Tip tip={TIPS.dose}>Dose</Tip></div>
            <label>
              Strength of the translated change added to Medium: <b className="mono">{dose.toFixed(2)}</b>
              <input type="range" min="0.5" max="6" step="0.5" value={dose} onChange={(e) => setDose(Number(e.target.value))} />
              <span className="hint">2.0 was chosen on development records for the 8 to 16 pair. Other pairs were not tuned.</span>
            </label>
            <details>
              <summary>Advanced settings</summary>
              <label>
                Gradient steps
                <input type="number" min="10" max="300" step="10" value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
              </label>
              <label>
                Candidate features (top N)
                <input type="number" min="20" max="400" step="20" value={topN} onChange={(e) => setTopN(Number(e.target.value))} />
              </label>
              <label>
                Additive cap factor (additive only)
                <input type="number" min="0.05" max="3" step="0.05" value={addCap} onChange={(e) => setAddCap(Number(e.target.value))} />
              </label>
            </details>

            {whyNot && <p className="warn">{whyNot}</p>}
            <button className="primary" disabled={running || !!whyNot} onClick={run}>
              Run edit and inject into Medium
            </button>
            {running && (
              <div className="progress">
                <div className="bar"><div style={{ width: `${100 * progress.frac}%` }} /></div>
                <span>{progress.text}</span>
              </div>
            )}
            {error && <p className="warn">{error}</p>}
            {note && <p className="info">{note}</p>}
          </div>
        </aside>

        <div className="stage">
          {/* ---- the sentence and the plan ------------------------------------------------------------------------------------- */}
          <section className="card">
            <span className="kicker">Sentence to edit</span>
            <div className="sentence">
              {prompt.trim() || "(empty prompt)"}
              <span className="chip green">{target.trim() ? " " + target.trim() : "?"}</span>
            </div>
            <p>
              Make <code>{target.trim() ? " " + target.trim() : ""}</code> the <b>first</b> next word. GPT-2 Small now:{" "}
              <b>{clean?.small && "target_rank" in clean.small ? `rank ${num(clean.small.target_rank)}` : "…"}</b>. GPT-2 Medium now:{" "}
              <b>{clean?.medium && "target_rank" in clean.medium ? `rank ${num(clean.medium.target_rank)}` : "…"}</b>.
            </p>
            <p className="muted">
              Plan: {isDirect ? "tune the dials against Medium's output (Direct, D6)" : mechanism} in Small at layer {pair[0]}; translate the change and add it to Medium at layer{" "}
              {pair[1]}.
            </p>
          </section>

          {/* ---- the bridge: where the edit is made, how it crosses, where it lands ---------------------------------------------- */}
          {bridgeReady && (
            <section className="card">
              <span className="kicker">The experiment</span>
              <Bridge
                pair={showRun ? showRun.pair : pair}
                mapLabel={config.map_choices[bridgeMap].split(" (")[0].split(":")[0]}
                mapHelp={config.map_help[bridgeMap]}
                facts={facts}
                dose={showRun ? showRun.dose : dose}
                phase={phase}
                direct={bridgeMap === "direct"}
                small={{ before: clean.small.target_rank, after: showRun ? showRun.small_after.rank : null }}
                medium={{ before: clean.medium.target_rank, after: showRun ? showRun.medium_after.rank : null }}
                injFrac={showRun ? showRun.injected_size_frac : null}
              />
            </section>
          )}

          {/* ---- the two models, side by side, unedited ----------------------------------------------------------------------- */}
          <div className="row">
            <section className="card grow model">
              <span className="kicker">Source model</span>
              <h2><Tip tip={TIPS.small(facts)}>GPT-2 Small</Tip></h2>
              <p className="muted">Has the SAE. The edit (one multiplier per SAE feature) is made here.</p>
              <Readout res={clean?.small} target={target} />
              <Facts facts={status.small} rows={{ SAE: status.small.sae, "SAE features": num(status.small.sae_features), "Edit layer / hook": `${status.small.edit_layer} / ${status.small.hook}`, Device: status.device }} />
            </section>
            <section className="card grow model">
              <span className="kicker">Receiving model</span>
              <h2><Tip tip={TIPS.medium(facts)}>GPT-2 Medium</Tip></h2>
              <p className="muted">No SAE. It receives the change an edit makes in Small, translated by a linear map into its own coordinates.</p>
              <Readout res={clean?.medium} target={target} />
              <Facts facts={status.medium} rows={{ SAE: "none", "Injection layer (paired with Small layer 8)": status.medium.injection_layer, Device: status.device }} />
            </section>
          </div>

          {/* Stays on screen after the run ends until the drawing has caught up with the last step, then gives way to the replay in the results. */}
          {liveDials && (
            <section className="card">
              <span className="kicker">Live</span>
              <h2><Tip tip={TIPS.dial}>The 200 dials</Tip></h2>
              <Dials dials={liveDials} live done={!running} onCaughtUp={() => setLiveDials(null)} layer={pair[0]} />
            </section>
          )}

          {/* ---- the runs ----------------------------------------------------------------------------------------------------------- */}
          <div className="row between">
            <h2>Results: before and after</h2>
            {myRuns.length > 0 && <button onClick={() => setRuns([])}>Clear runs</button>}
          </div>
          {myRuns.length === 0 ? (
            <p className="muted">No runs yet for this prompt and target. Press Run, then change the map or the dose and run again; each run adds a block below.</p>
          ) : (
            <>
              <label className="radio">
                <input type="checkbox" checked={collapseOld} onChange={(e) => setCollapseOld(e.target.checked)} />
                Collapse older runs (keep only the newest open)
              </label>
              {myRuns.map((res, i) => (
                <RunCard key={i} i={i} res={res} runs={myRuns} smallBefore={smallBefore} target={target} config={config} open={i === myRuns.length - 1 || !collapseOld} />
              ))}
            </>
          )}
        </div>
      </div>
    </Page>
  );
}

// Page frame: a header bar on top (title on the left, what is loaded on the right), then whatever the page puts below it.
// Kept as one component so the loading screens look the same as the real page.
function Page({ status, children }) {
  return (
    <>
      <header className="topbar">
        <div className="brand">
          <h1>Cross-Model-Transfer</h1>
          <span className="tag">GPT-2 Small → GPT-2 Medium</span>
        </div>
        {status && !status.loading && !status.error && (
          <div className="status">
            <i /> Small + SAE layer {status.small.edit_layer} · Medium · {status.device}
          </div>
        )}
      </header>
      {children}
    </>
  );
}

// "Model details" box under each model.
function Facts({ facts, rows }) {
  const all = { Layers: facts.n_layers, "Width (d_model)": facts.d_model, "Parameters (millions)": facts.params_millions, Vocabulary: num(facts.d_vocab), "Context length": facts.n_ctx, ...rows };
  return (
    <details>
      <summary>Model details</summary>
      <table className="facts">
        <tbody>
          {Object.entries(all).map(([k, v]) => (
            <tr key={k}><td>{k}</td><td>{String(v)}</td></tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}

// Short name that tells runs apart: the edit method and the translation map.
// A Direct run says so plainly, because its dials are tuned on Medium's output and the "gradient descent" label alone made it look like the other runs.
function runName(res, config) {
  if (res.map === "direct") return "Direct (D6)";
  const mech = res.mechanism.startsWith("Gradient descent + additive") ? "GD + additive" : "GD";
  return `${mech} | ${config.map_choices[res.map].split(" (")[0]}`;
}

// One finished run: tag, settings line, Small and Medium before/after, and how it compares with every earlier run.
function RunCard({ i, res, runs, smallBefore, target, config, open }) {
  const additive = res.mechanism.startsWith("Gradient descent + additive");
  const direct = res.map === "direct";
  return (
    <details className="run" open={open}>
      <summary>
        Run {i + 1}: {runName(res, config)} | dose {res.dose}
      </summary>
      <span className={`chip ${direct ? "violet" : additive ? "orange" : "blue"}`}>
        {direct ? "Direct (D6): dials tuned on Medium's output" : `Edit: ${res.mechanism}`}
      </span>
      <p className="muted">
        {direct ? "Edit: dials tuned on Medium's output" : `Mechanism: ${res.mechanism}`} | map: {config.map_choices[res.map]} | layers {res.pair[0]} to {res.pair[1]} | dose {res.dose} |{" "}
        {res.seconds.toFixed(1)} s | injected change {(100 * res.injected_size_frac).toFixed(1)}% of Medium's state norm at the last position
      </p>
      {res.dials && (
        <details className="card" open={open}>
          <summary>Watch the 200 dials move (replay)</summary>
          <Dials dials={res.dials} layer={res.pair[0]} />
          {res.mechanism.startsWith("Gradient descent + additive") && <p className="muted">Only the multipliers are shown; the additive amounts on silent features are not drawn.</p>}
        </details>
      )}
      <div className="row">
        <div className="card grow">
          {res.small_edit && smallBefore ? (
            <>
              <BeforeAfter title="GPT-2 Small (edited)" before={smallBefore} after={res.small_after} target={target} />
              {res.small_note && <p className="muted">{res.small_note}</p>}
            </>
          ) : (
            <p className="muted">Small's clean numbers are not available for this prompt.</p>
          )}
        </div>
        <div className="card grow">
          <BeforeAfter title="GPT-2 Medium (receives the translated change)" before={res.medium_before} after={res.medium_after} target={target} />
        </div>
      </div>
      {i > 0 && (
        <div className="info">
          {runs.slice(0, i).map((prev, j) => {
            const parts = [diffText("Medium", res.medium_after, prev.medium_after, j + 1)];
            if (res.small_edit && prev.small_edit) parts.push(diffText("Small", res.small_after, prev.small_after, j + 1));
            return (
              <div key={j}>
                <b>Compared with run {j + 1}</b> ({runName(prev, config)}): {parts.join("; ")}
              </div>
            );
          })}
        </div>
      )}
    </details>
  );
}
