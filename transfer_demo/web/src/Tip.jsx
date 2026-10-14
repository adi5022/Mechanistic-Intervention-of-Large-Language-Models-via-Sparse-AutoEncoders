// Pop-up explanations that appear when you hover (or tab to) a piece of text.
//
// <Tip title="Dose" text="..."> the words </Tip>   puts a dotted underline under the words and shows the box above them on hover.
// It is pure CSS: no JavaScript runs on hover, so it costs nothing and cannot get stuck open. The wording itself lives in tips.js.
//
// The pictures drawn in SVG (the bridge) cannot use this, because the box has to follow the mouse; Bridge.jsx has its own version.

export function Tip({ title, text, tip, children, side = "left" }) {
  // `tip` lets callers pass a whole { title, text } entry from tips.js at once.
  const t = tip || { title, text };
  return (
    <span className="tip-anchor" tabIndex={0}>
      {children}
      <span className={`tipbox ${side}`} role="tooltip">
        {t.title && <b>{t.title}</b>}
        {t.text}
      </span>
    </span>
  );
}
