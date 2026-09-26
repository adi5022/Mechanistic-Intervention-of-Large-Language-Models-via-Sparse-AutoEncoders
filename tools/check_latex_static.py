"""
Static sanity checks for a LaTeX file when no compiler is available.

    python tools/check_latex_static.py research/paper/draft_v1/main.tex

Checks: brace balance, \\begin/\\end pairing, \\label uniqueness, \\ref/\\cite resolution, \\includegraphics files exist,
non-ASCII characters, em/en dashes, odd number of unescaped dollar signs per paragraph, tabular column counts.
This is NOT a compile; it catches the most common errors only.
"""
import os
import re
import sys


def strip_comments(s):
    out = []
    for line in s.split("\n"):
        i = 0
        while i < len(line):
            if line[i] == "%" and (i == 0 or line[i - 1] != "\\"):
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


def main(path):
    raw = open(path, encoding="utf-8").read()
    s = strip_comments(raw)
    base = os.path.dirname(path)
    problems = []

    # characters
    for n, line in enumerate(raw.split("\n"), 1):
        for ch in line:
            if ord(ch) > 127:
                problems.append(f"line {n}: non-ASCII character {ch!r} (U+{ord(ch):04X})")
                break

    for n, line in enumerate(strip_comments(raw).split("\n"), 1):
        if "\u2014" in line or "\u2013" in line or "---" in line:
            problems.append(f"line {n}: em/en dash")

    # braces (ignore escaped and \detokenize content is still balanced)
    depth = 0
    for n, line in enumerate(s.split("\n"), 1):
        for i, ch in enumerate(line):
            if ch == "{" and (i == 0 or line[i - 1] != "\\"):
                depth += 1
            elif ch == "}" and (i == 0 or line[i - 1] != "\\"):
                depth -= 1
                if depth < 0:
                    problems.append(f"line {n}: unmatched closing brace")
                    depth = 0
    if depth != 0:
        problems.append(f"unbalanced braces: {depth} unclosed")

    # environments
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", s):
        kind, env = m.group(1), m.group(2)
        if kind == "begin":
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                problems.append(f"environment mismatch near \\end{{{env}}} (open: {stack[-3:]})")
            else:
                stack.pop()
    if stack:
        problems.append(f"unclosed environments: {stack}")

    # labels, refs, cites
    labels = re.findall(r"\\label\{([^}]+)\}", s)
    dup = {l for l in labels if labels.count(l) > 1}
    if dup:
        problems.append(f"duplicate labels: {sorted(dup)}")
    refs = set()
    for m in re.finditer(r"\\(?:ref|eqref)\{([^}]+)\}", s):
        refs.add(m.group(1))
    for r in sorted(refs - set(labels)):
        problems.append(f"\\ref to undefined label: {r}")
    for l in sorted(set(labels) - refs):
        print(f"note: label never referenced: {l}")
    bibkeys = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
    cites = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", s):
        cites.update(k.strip() for k in m.group(1).split(","))
    for c in sorted(cites - bibkeys):
        problems.append(f"\\cite to missing bibitem: {c}")
    for b in sorted(bibkeys - cites):
        print(f"note: bibitem never cited: {b}")

    # graphics
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s):
        f = m.group(1)
        if not os.path.exists(os.path.join(base, f)):
            problems.append(f"missing figure file: {f}")

    # tabular column counts
    for m in re.finditer(r"\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}(.*?)\\end\{tabular\}", s, flags=re.S):
        spec = re.sub(r"@\{[^}]*\}", "", m.group(1))
        ncols = len(re.findall(r"[lcrp]", spec))
        body = m.group(2)
        for row in re.split(r"\\\\", body):
            row = re.sub(r"\\(toprule|midrule|bottomrule)", "", row).strip()
            if not row or "&" not in row:
                continue
            cells = len(row.split("&"))
            if cells != ncols:
                problems.append(f"tabular with {ncols} columns has a row with {cells} cells: {row[:60]!r}")

    # dollars per paragraph
    for k, para in enumerate(re.split(r"\n\s*\n", s)):
        p = para.replace("\\$", "")
        p = re.sub(r"\$\$", "", p)
        if p.count("$") % 2:
            problems.append(f"odd number of $ in paragraph starting: {para.strip()[:60]!r}")

    print(f"file: {path} | labels {len(labels)} | refs {len(refs)} | cites {len(cites)} | bibitems {len(bibkeys)}")
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("static checks passed (this is not a compile)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "research/paper/draft_v1/main.tex")
