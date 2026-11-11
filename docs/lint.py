"""Fail the documentation build on local prose and execution-policy violations."""
from pathlib import Path
import re

root = Path(__file__).resolve().parent
expected = {
    "index.md", "getting-started.md", "reductions.md", "ragged.md",
    "streaming.md", "tables.md", "trajectories.md", "api.md",
}
errors = []
pages = {p.name for p in (root / "src").glob("*.md")}
if pages != expected:
    errors.append(f"Page list differs from docs/README.md: {pages ^ expected}")
make = (root / "make.jl").read_text()
groups = re.findall(r'^        "(\w+)" => \[', make, re.M)
if groups != ["Start", "Guides", "Examples", "Reference"]:
    errors.append(f"Navigation groups: {groups}")
banned = re.compile(r"\b(seamless|robust|powerful|leverage|delve|crucial|comprehensive|unlock|simply)\b|it's worth noting", re.I)
for name in sorted(pages):
    text = (root / "src" / name).read_text()
    prose = re.sub(r'```.*?```', '', text, flags=re.S)
    runnable = re.findall(r'^```(?:julia|julia-repl|python|javascript)\b', text, re.M)
    executed = len(re.findall(r'^```@(example|repl|eval)\b', text, re.M))
    words = len(text.split())
    print(f"{name}: {words} source words; {len(prose.split())} prose words; {executed} executed blocks")
    if runnable:
        errors.append(f"{name}: static runnable fences")
    if banned.search(prose):
        errors.append(f"{name}: banned prose: {banned.search(prose).group()}")
    if '**' in prose:
        errors.append(f"{name}: bold prose")
    if name not in {"api.md", "index.md", "trajectories.md"} and words >= 1500:
        errors.append(f"{name}: guide exceeds 1500 source words")
    if name == 'trajectories.md':
        if not 800 <= words <= 1500:
            errors.append(f"{name}: example needs 800–1500 source words")
        if not executed or 'vdraw(' not in text or '::: details Reproduce' not in text:
            errors.append(f"{name}: missing executed example, figure, or Reproduce footer")
for path in (root.parent / 'src').glob('*.jl'):
    for doc in re.findall(r'"""(.*?)"""', path.read_text(), re.S):
        if re.search(r'^```julia', doc, re.M):
            errors.append(f"{path.name}: static runnable docstring example")
        if banned.search(doc):
            errors.append(f"{path.name}: banned docstring prose")
if errors:
    raise SystemExit('\n'.join(errors))
print('Documentation lint passed')
