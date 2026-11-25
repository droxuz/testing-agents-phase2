"""Split a project into small units of code that one short model conversation can test.

This is plain Python, no model calls. It parses every module in the package with `ast` and makes
one unit per class and one per group of module-level functions. Units longer than MAX_LINES are
split into consecutive chunks of methods, each one repeating the class header so the model knows
what it is looking at.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path

MAX_LINES = 220     # source lines shown to the model in one unit
HEADER_LINES = 40   # lines of the class header (signature, docstring) repeated in every chunk


@dataclass
class Unit:
    name: str                 # e.g. "SortedList (part 2/4)"
    slug: str                 # used in the test file name
    module: str               # dotted import path, e.g. "sortedcontainers.sortedlist"
    file: str                 # path relative to the workspace
    first_line: int
    last_line: int
    members: list[str] = field(default_factory=list)   # functions or methods covered
    source: str = ""          # numbered source shown to the model


def _numbered(lines: list[str], start: int, end: int) -> str:
    return "\n".join(f"{n:5d}  {lines[n - 1]}" for n in range(start, end + 1))


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno] + [d.lineno for d in decorators])


def _chunks(items: list[tuple[str, int, int]], max_lines: int) -> list[list[tuple[str, int, int]]]:
    """Group consecutive (name, start, end) items so each group spans at most max_lines lines."""
    groups, current = [], []
    for item in items:
        if current and item[2] - current[0][1] + 1 > max_lines:
            groups.append(current)
            current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def plan(workspace: Path, package: str) -> list[Unit]:
    units: list[Unit] = []
    root = Path(workspace)
    for path in sorted((root / package).rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root)
        module = ".".join(rel.with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[: -len(".__init__")]
        text = path.read_text(errors="replace")
        lines = text.splitlines()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        functions = [(n.name, _start(n), n.end_lineno) for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for i, group in enumerate(_chunks(functions, MAX_LINES)):
            label = f"{module} functions" + (f" (group {i + 1})" if len(functions) > len(group) else "")
            first, last = group[0][1], group[-1][2]
            units.append(Unit(label, _slug(label), module, str(rel), first, last,
                              [g[0] for g in group], _numbered(lines, first, last)))

        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            start, end = _start(node), node.end_lineno
            methods = [(f"{node.name}.{m.name}", _start(m), m.end_lineno) for m in node.body
                       if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            if end - start + 1 <= MAX_LINES or not methods:
                units.append(Unit(node.name, _slug(node.name), module, str(rel), start, end,
                                  [m[0] for m in methods], _numbered(lines, start, end)))
                continue
            header_end = min(methods[0][1] - 1, start + HEADER_LINES - 1)
            groups = _chunks(methods, MAX_LINES - (header_end - start + 1))
            for i, group in enumerate(groups):
                label = f"{node.name} (part {i + 1}/{len(groups)})"
                first, last = group[0][1], group[-1][2]
                source = _numbered(lines, start, header_end) + "\n    ...\n" + _numbered(lines, first, last)
                units.append(Unit(label, _slug(label), module, str(rel), first, last,
                                  [g[0] for g in group], source))
    return _unique_slugs(_merge_small(units))


def _unique_slugs(units: list[Unit]) -> list[Unit]:
    """Slugs key per-unit state and file names, so two units (e.g. classes with the same name in
    different modules) must not share one."""
    seen = {}
    for unit in units:
        base = unit.slug or "unit"
        seen[base] = seen.get(base, 0) + 1
        unit.slug = base if seen[base] == 1 else f"{base}_{seen[base]}"
    return units


def _merge_small(units: list[Unit]) -> list[Unit]:
    """Merge neighbouring units of the same file while the combined source stays under MAX_LINES."""
    merged: list[Unit] = []
    for unit in sorted(units, key=lambda u: (u.file, u.first_line)):
        prev = merged[-1] if merged else None
        if (prev and prev.file == unit.file
                and len(prev.source.splitlines()) + len(unit.source.splitlines()) <= MAX_LINES):
            prev.name = f"{prev.name} + {unit.name}"
            prev.slug = _slug(prev.name)[:60]
            prev.last_line = unit.last_line
            prev.members += unit.members
            prev.source += "\n\n" + unit.source
        else:
            merged.append(unit)
    return merged


DOC_SUFFIXES = {".rst", ".md", ".txt", ".markdown"}


def doc_files(root: Path) -> list[Path]:
    """The project's documentation: README* at the top, and every text document under docs/ or doc/."""
    root = Path(root)
    files = sorted(p for p in root.glob("README*") if p.is_file())
    for folder in ("docs", "doc"):
        if (root / folder).is_dir():
            files += sorted(p for p in (root / folder).rglob("*") if p.suffix.lower() in DOC_SUFFIXES and p.is_file())
    return files


def doc_snippets(workspace: Path, unit: Unit, max_lines: int = 80) -> str:
    """Paragraphs of README.rst and docs/ that mention the unit's classes or functions.

    Gives the model the documented promises for this unit without spending steps on searching.
    """
    root = Path(workspace)
    names = {m.split(".")[0] for m in unit.members} | {m.split(".")[-1] for m in unit.members}
    names = {n for n in names if not n.startswith("_") and len(n) > 3}
    if not names:
        return ""
    pattern = re.compile(r"\b(" + "|".join(sorted(map(re.escape, names))) + r")\b")
    docs = doc_files(root)
    scored, seen = [], set()
    for order, path in enumerate(docs):
        for para in re.split(r"\n\s*\n", path.read_text(errors="replace")):
            hits = set(pattern.findall(para))
            if hits and para not in seen and len(para) < 2500:
                seen.add(para)
                directive = 2 if re.search(r"^\.\. \w+:: \w*(" + "|".join(map(re.escape, hits)) + ")", para, re.M) else 0
                scored.append((-(len(hits) + directive), order, len(scored), para.strip("\n")))
    picked = [para for *_, para in sorted(scored)]
    lines = "\n\n".join(picked).splitlines()[:max_lines]
    return "\n".join(lines)


def spec_view(workspace: Path, unit: Unit) -> str:
    """The unit's signatures and docstrings without the function bodies.

    The reviewer sees what the code promises, not how the (possibly buggy) body computes it, so it
    is not pulled towards the value the buggy code returned.
    """
    tree = ast.parse((Path(workspace) / unit.file).read_text())
    wanted = set(unit.members)
    out = []

    def describe(node, prefix=""):
        args = ast.unparse(node.args)
        doc = ast.get_docstring(node) or ""
        line = f"{prefix}def {node.name}({args})"
        out.append(line + (f':\n{prefix}    """{doc}"""' if doc else ""))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            describe(node)
        elif isinstance(node, ast.ClassDef):
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and f"{node.name}.{m.name}" in wanted]
            if methods:
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                doc = ast.get_docstring(node) or ""
                out.append(f"class {node.name}({bases}):" + (f'\n    """{doc}"""' if doc else ""))
                for m in methods:
                    describe(m, "    ")
    return "\n".join(out)


def _identifiers(tree: ast.AST) -> set[str]:
    names = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            names.add(n.attr)
    return {n for n in names if len(n) > 2 and not n.startswith("_")}


def existing_tests(workspace: Path, test_sources: list[str], max_lines: int = 90) -> str:
    """The project's own tests (tests/, they all pass) that use the same API as test_sources.

    They pass on the current code, so they show real behaviour wherever they check it: a new test
    that contradicts them is wrong. (The bugs are in places these tests do not check.)
    """
    root = Path(workspace) / "tests"
    wanted = set()
    for source in test_sources:
        try:
            wanted |= _identifiers(ast.parse(source))
        except SyntaxError:
            pass
    wanted -= {"assert", "len", "list", "dict", "set", "tuple", "int", "str", "range", "self", "pytest",
               "raises", "assertEqual", "assertTrue", "assertFalse", "assertIn", "assertNotIn", "assertRaises",
               "True", "False", "None", "value", "key", "keys", "values", "items", "test"}
    if not wanted or not root.is_dir():
        return ""
    scored = []
    for path in sorted(root.rglob("test*.py")):
        text = path.read_text(errors="replace")
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                hits = _identifiers(node) & wanted
                if hits:
                    weight = sum(3 if h[:1].isupper() else 1 for h in hits)
                    source = ast.get_source_segment(text, node) or ""
                    scored.append((-weight, len(source), f"# {path.relative_to(root.parent)}\n{source}"))
    picked, used = [], 0
    for _, _, source in sorted(scored):
        n = source.count("\n") + 1
        if used + n > max_lines:
            continue
        picked.append(source)
        used += n
    return "\n\n".join(picked)


def risk(workspace: Path, unit: Unit) -> float:
    """How many places a small bug could be in, per unit: comparisons, arithmetic, boolean logic,
    negations and integer constants in the unit's code, the kinds of mistakes the bugs are.

    Private modules and classes (leading underscore) count less: they are reached only through
    the public API, and their behaviour is less often documented."""
    try:
        tree = ast.parse((Path(workspace) / unit.file).read_text(errors="replace"))
    except (SyntaxError, OSError):
        return 0.0
    wanted, bodies = set(unit.members), []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            bodies.append(node)
        elif isinstance(node, ast.ClassDef):
            bodies += [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
                       and f"{node.name}.{m.name}" in wanted]
    points = 0
    for node in (n for body in bodies for n in ast.walk(body)):
        if isinstance(node, ast.Compare):
            points += len(node.ops)
        elif isinstance(node, (ast.BinOp, ast.AugAssign, ast.BoolOp)):
            points += 1
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.USub)):
            points += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            points += 1
    private = any(part.startswith("_") and part != "__init__" for part in unit.module.split(".")) or \
        all(m.split(".")[0].startswith("_") for m in unit.members)
    return points * (0.3 if private else 1.0)


def by_risk(workspace: Path, units: list[Unit]) -> list[Unit]:
    """Units in the order to test them when the budget may not reach all of them (risk-based test
    prioritisation): the most bug-prone first."""
    return sorted(units, key=lambda u: -risk(workspace, u))
