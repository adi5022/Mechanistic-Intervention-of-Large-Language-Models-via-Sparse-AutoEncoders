import { useRef, useState } from "react";

// Pop-up explanations that appear when you hover (or tab to) a piece of text.
//
// <Tip title="Dose" text="..."> the words </Tip>   puts a dotted underline under the words and shows the box when the mouse is over them.
// The wording itself lives in tips.js.
//
// WHY THE BOX IS POSITIONED WITH JAVASCRIPT
// An earlier version showed the box with plain CSS, attached to the words. Near the right edge of the window the box ran off the screen
// (and made the page scroll sideways). Now the box is drawn with "fixed" positioning (relative to the window, so it can never widen the page)
// and its left edge is clamped so the whole box stays inside the window. It appears above the words if there is room, otherwise below.
//
// The pictures drawn in SVG (the bridge, the rank panels) have their own pop-up that follows the mouse; see Bridge.jsx and Ladder.jsx.

const BOX_W = 290;   // the box's width (the stylesheet's max-width is the same)
const MARGIN = 12;   // the least gap kept between the box and the window's edge

export function Tip({ title, text, tip, children }) {
  // `tip` lets callers pass a whole { title, text } entry from tips.js at once.
  const t = tip || { title, text };
  const ref = useRef(null);
  const [pos, setPos] = useState(null);

  const show = () => {
    const r = ref.current.getBoundingClientRect();
    const width = Math.min(BOX_W, window.innerWidth - 2 * MARGIN);
    const left = Math.min(Math.max(r.left, MARGIN), window.innerWidth - width - MARGIN);
    const above = r.top > 160;               // enough room above the words? (the box is about 120 to 150 units tall)
    setPos({ left, top: above ? r.top - 8 : r.bottom + 8, above });
  };
  const hide = () => setPos(null);

  return (
    <span className="tip-anchor" tabIndex={0} ref={ref} onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      {pos && (
        <span className="tipbox fixed" role="tooltip" style={{ left: pos.left, top: pos.top, transform: pos.above ? "translateY(-100%)" : "none" }}>
          {t.title && <b>{t.title}</b>}
          {t.text}
        </span>
      )}
    </span>
  );
}
