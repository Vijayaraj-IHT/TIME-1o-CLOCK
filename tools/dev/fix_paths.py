"""One-shot Step-0 transformer: D:/SIH_Model -> portable _REPO_ROOT. Idempotent; safe to re-run."""
import os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
Q = chr(34)
STRPAT = re.compile("r?(" + Q + "|')D:[\\\\/]SIH_Model((?:[\\\\/][^" + Q + "'\\\\/\n]+)*)(" + Q + "|')")
SYSPAT = re.compile("^(\\s*)sys\\.path\\.insert\\(0,\\s*r?(" + Q + "|')D:[\\\\/]SIH_Model\\2\\)[ \\t]*$")
def def_block(depth):
    ups = ", ".join([Q + ".." + Q] * depth)
    return ("# Repo root from this file location (portable; was a hardcoded Windows path).\n"
            "_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), " + ups + "))")
def find_inject_point(lines):
    first_def = len(lines)
    for i, ln in enumerate(lines):
        if ln.startswith("def ") or ln.startswith("class ") or ln.startswith("async def "):
            first_def = i
            break
    last_import = -1
    for i in range(min(first_def, len(lines))):
        if lines[i].startswith("import ") or lines[i].startswith("from "):
            last_import = i
    if last_import >= 0:
        return last_import + 1
    if lines and lines[0].strip().startswith(Q*3):
        for i in range(1, len(lines)):
            if Q*3 in lines[i]:
                return i + 1
    return 0
def has_os(lines):
    return any(re.match(r"^import os(\s|$)|^from os(\s|$)", ln) for ln in lines)
changed = []
for dirpath, _, files in os.walk(ROOT):
    if ".git" in dirpath or "tools/dev" in dirpath:
        continue
    for fn in sorted(files):
        if not fn.endswith(".py"):
            continue
        p = os.path.join(dirpath, fn)
        with open(p, encoding="utf-8") as f:
            src = f.read()
        if "SIH_Model" not in src or "D:" not in src:
            continue
        if not (STRPAT.search(src) or SYSPAT.search(src)):
            continue
        rel = os.path.relpath(p, ROOT)
        parent = os.path.dirname(rel)
        depth = len(parent.split(os.sep)) if parent else 1
        lines = src.split("\n")
        idx = [i for i, ln in enumerate(lines) if SYSPAT.match(ln)]
        if idx:
            i0 = idx[0]
            indent = SYSPAT.match(lines[i0]).group(1)
            newb = def_block(depth).split("\n") + ["sys.path.insert(0, _REPO_ROOT)"]
            block = [((indent + l) if l else l) for l in newb]
            shift = len(block) - 1
            lines[i0:i0 + 1] = block
            for j in idx[1:]:
                m = SYSPAT.match(lines[j + shift])
                lines[j + shift] = (m.group(1) if m else "") + "sys.path.insert(0, _REPO_ROOT)"
            if not has_os(lines[:i0]):
                lines.insert(i0, "import os")
            src = "\n".join(lines)
        else:
            at = find_inject_point(lines)
            if "_REPO_ROOT" not in src:
                if not has_os(lines):
                    lines.insert(at, "import os")
                    at += 1
                for j, bl in enumerate(def_block(depth).split("\n")):
                    lines.insert(at + j, bl)
            src = "\n".join(lines)
        def rep(m):
            tail = m.group(2)
            if not tail:
                return "_REPO_ROOT"
            parts = [t for t in re.split(r"[\\/]", tail) if t]
            inner = ", ".join([Q + t + Q for t in parts])
            return "os.path.join(_REPO_ROOT, " + inner + ")"
        src2 = STRPAT.sub(rep, src)
        with open(p, "w", encoding="utf-8") as f:
            f.write(src2)
        changed.append(rel)
print("CHANGED %d files:" % len(changed))
for c in changed:
    print("  " + c)
