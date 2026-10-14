// Everything the page says to the Python server (transfer_demo/server.py) goes through this file,
// so that if a route ever changes there is one place to fix.

async function getJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function postJson(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || `${url}: ${r.status}`);
  return data;
}

// Are the models loaded yet, which files exist, facts about both models.
export const getStatus = () => getJson("/api/status");

// Texts, layer pairs and which translation maps exist for each pair (all decided on the server side).
export const getConfig = () => getJson("/api/config");

// The unedited ranking of the prompt in both models.
export const getReadout = (prompt, target) => postJson("/api/readout", { prompt, target });

// Run one edit. The server sends one JSON object per line while it works (progress, one event per
// gradient step, then the finished result), so we read the reply piece by piece and call onEvent for each line.
export async function runEdit(body, onEvent) {
  const r = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new Error(data.error || `run failed: ${r.status}`);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  // A line can arrive split across two chunks, so keep the unfinished tail in `buffer` until its newline shows up.
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
  }
}
