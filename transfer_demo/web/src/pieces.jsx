// Small display pieces used by App.jsx. They only show numbers; no study logic lives here.

// 0.12345 -> "12.345%"  (three decimals, as in the Streamlit tab, because target probabilities can be tiny)
import { Tip } from "./Tip.jsx";
import { TIPS } from "./tips.js";

export const pct = (p) => `${(100 * p).toFixed(3)}%`;
// 1234 -> "1,234"
export const num = (n) => Number(n).toLocaleString("en-US");

// The next-word table: word and probability, with the target's row marked.
// A word looks like " English" (with the leading space the model uses), so we compare against " " + target.
export function TopTable({ top, target, heading }) {
  const tgt = target.trim() ? " " + target.trim() : null;
  return (
    <table className="top">
      <thead>
        <tr>
          <th>Word</th>
          <th>{heading}</th>
        </tr>
      </thead>
      <tbody>
        {top.map(([word, p]) => (
          <tr key={word} className={word === tgt ? "hit" : ""}>
            <td>
              {JSON.stringify(word)}
              {word === tgt ? " ◀ target" : ""}
            </td>
            <td>{(100 * p).toFixed(2)}%</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// The unedited result of one model: the target's probability and rank first, then the top 5 words.
export function Readout({ res, target }) {
  if (!res) return <p className="muted">Loading…</p>;
  return (
    <div>
      {"target_rank" in res && (
        <div className="bigbox green">
          <div className="small"><Tip tip={TIPS.probability}>Probability</Tip> of the token {JSON.stringify(res.target_str)}</div>
          <div className="big">{pct(res.target_prob)}</div>
          <div>
            <Tip tip={TIPS.rank}>Rank</Tip> <b>{num(res.target_rank)}</b>
          </div>
        </div>
      )}
      {res.target_multi_token && (
        <p className="warn">The target is several tokens long; only its last token is scored (the same rule as the rest of the app).</p>
      )}
      <h4>Top 5 next words</h4>
      <TopTable top={res.top} target={target} heading="Probability" />
    </div>
  );
}

// "Medium: 120 places better, probability +2.1 points (rank 142, 0.4% in run 1)" — how this run differs from an earlier one.
export function diffText(model, now, other, otherNo) {
  const dr = now.rank - other.rank;
  const dp = 100 * (now.prob - other.prob);
  const unit = Math.abs(dr) === 1 ? "place" : "places";
  const rankTxt = dr === 0 ? "same rank" : dr < 0 ? `${num(Math.abs(dr))} ${unit} better` : `${num(dr)} ${unit} worse`;
  return `${model}: ${rankTxt}, probability ${dp >= 0 ? "+" : ""}${dp.toFixed(3)} points (rank ${num(other.rank)}, ${pct(other.prob)} in run ${otherNo})`;
}
