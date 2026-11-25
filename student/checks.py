"""Checks the agent runs in Python on each test file after the model writes it (no model calls).

- rejection_reason: would the evaluator reject the whole file? (mirrors the README's rules)
- add_missing_imports: add `import pytest` / `import math` etc. when the file uses them without importing.
- run_tests: run one file with pytest and get every test's outcome and the kind of failure.
- remove_tests: delete test functions from a file by name, using the AST.

The key rule (triage): the workspace code is buggy, so a failing test is either a found bug or a
wrong test. A test that fails on an *assertion* states an expectation and may have found a bug.
A test that crashes with any other error (AttributeError, NameError, TypeError, ...) almost always
misuses the API, so it is removed. So is a test that gives different results on two runs.
"""

import ast
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

FORBIDDEN_MODULES = {"inspect", "os", "io", "pathlib", "importlib", "linecache", "dis", "subprocess", "shutil",
                     "glob", "marshal", "ast", "tokenize", "pkgutil", "runpy", "sys", "gc", "ctypes", "mmap"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "globals", "vars", "__import__", "breakpoint"}
FORBIDDEN_ATTRS = {"__code__", "__file__", "__globals__", "__loader__", "__spec__", "__path__", "modules",
                   "getsource", "getsourcelines", "read_text", "read_bytes", "co_code", "f_code"}
AUTO_IMPORTS = {"pytest", "math", "itertools", "functools", "collections", "operator", "random", "string"}
ASSERTION_PREFIXES = ("AssertionError", "assert ", "Failed: DID NOT RAISE")


def rejection_reason(path: Path) -> str:
    """Why the evaluator would reject this file, or '' if it looks fine."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as error:
        return f"syntax error on line {error.lineno}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names = {(node.module or "").split(".")[0]}
        else:
            names = set()
        if names & FORBIDDEN_MODULES:
            return f"imports {sorted(names & FORBIDDEN_MODULES)[0]}"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return f"uses {node.id}"
        if isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            return f"uses .{node.attr}"
    return ""


TOP_LEVEL = re.compile(r"^(?:def |async def |class |@)", re.M)


def drop_unparsable_tests(path: Path) -> int:
    """Remove top-level functions or classes with a syntax error, keeping the rest of the file.

    One bad line (e.g. a positional argument after a keyword) would otherwise lose every test in it.
    Returns how many blocks were removed, or -1 if the imports at the top do not parse either.
    """
    text = path.read_text()
    try:
        ast.parse(text)
        return 0
    except SyntaxError:
        pass
    starts = [m.start() for m in TOP_LEVEL.finditer(text)]
    if not starts:
        return -1
    header, blocks = text[:starts[0]], [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    # a decorator line belongs to the block that follows it
    merged = []
    for block in blocks:
        if merged and not re.search(r"^(?:def |async def |class )", merged[-1], re.M):
            merged[-1] += block
        else:
            merged.append(block)
    try:
        ast.parse(header)
    except SyntaxError:
        return -1
    good = []
    for block in merged:
        try:
            ast.parse(header + block)
            good.append(block)
        except SyntaxError:
            pass
    try:
        ast.parse(header + "".join(good))
    except SyntaxError:
        return -1
    path.write_text(header + "".join(good))
    return len(merged) - len(good)


class _ChainFixer(ast.NodeTransformer):
    """`x in s is True` means `x in s and s is True` in Python, so it is always False.

    The writer means `(x in s) is True`; rewrite it to `x in s` (or `not (x in s)` for `is False`).
    """

    def __init__(self):
        self.fixed = 0

    def visit_Compare(self, node):
        self.generic_visit(node)
        last, value = node.ops[-1], node.comparators[-1]
        if (len(node.ops) >= 2 and isinstance(last, (ast.Is, ast.IsNot))
                and isinstance(value, ast.Constant) and isinstance(value.value, bool)):
            inner = ast.Compare(left=node.left, ops=node.ops[:-1], comparators=node.comparators[:-1])
            self.fixed += 1
            if value.value == isinstance(last, ast.Is):
                return ast.copy_location(inner, node)
            return ast.copy_location(ast.UnaryOp(op=ast.Not(), operand=inner), node)
        return node


def fix_chained_comparisons(path: Path) -> int:
    """Rewrite `a OP b is True/False` (always False in Python) to what the writer meant."""
    tree = ast.parse(path.read_text())
    fixer = _ChainFixer()
    tree = fixer.visit(tree)
    if fixer.fixed:
        path.write_text(ast.unparse(ast.fix_missing_locations(tree)) + "\n")
    return fixer.fixed


def _package_names(workspace: Path, package: str) -> dict[str, dict[str, set[str]]]:
    """{module: {"defined": top-level names it defines or imports}} for every module of the package."""
    names = {}
    for file in sorted((Path(workspace) / package).rglob("*.py")):
        module = ".".join(file.relative_to(workspace).with_suffix("").parts)
        module = module[: -len(".__init__")] if module.endswith(".__init__") else module
        try:
            tree = ast.parse(file.read_text(errors="replace"))
        except SyntaxError:
            continue
        defined = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                defined |= {t.id for t in targets if isinstance(t, ast.Name)}
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                defined |= {(a.asname or a.name).split(".")[0] for a in node.names}
        names[module] = defined
    for module in list(names):                                            # `from pkg import sub` works too
        parent = module.rsplit(".", 1)[0]
        if parent in names and parent != module:
            names[parent].add(module.rsplit(".", 1)[-1])
    return names


def fix_imports(workspace: Path, path: Path, package: str) -> list[str]:
    """Move `from pkg.a import X` to the one module of the package that defines X, if a does not,
    and drop X from the import if no module defines it.

    One import of a name from the wrong module stops the whole file from loading (every test in it
    is lost), and the model often guesses the module wrong. Returns the names that were moved.
    """
    text = path.read_text()
    tree = ast.parse(text)
    modules = _package_names(workspace, package)
    lines = text.splitlines(keepends=True)
    moved, edits = [], []
    for node in tree.body:
        if not (isinstance(node, ast.ImportFrom) and node.level == 0
                and (node.module or "").split(".")[0] == package and node.module in modules):
            continue
        keep, move, gone = [], [], []
        for alias in node.names:
            if alias.name == "*" or alias.name in modules[node.module]:
                keep.append(alias)
                continue
            homes = [m for m, defined in modules.items() if alias.name in defined and m != node.module]
            if len(homes) == 1:
                move.append((homes[0], alias))
            elif not homes:
                gone.append(alias.name)    # defined nowhere: only the tests that use it should fail
            else:
                keep.append(alias)
        if not move and not gone:
            continue

        def clause(aliases):
            return ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in aliases)

        new = [f"from {node.module} import {clause(keep)}\n"] if keep else []
        new += [f"from {home} import {clause([a])}\n" for home, a in move]
        edits.append((node.lineno - 1, node.end_lineno, "".join(new)))
        moved += [a.name for _, a in move] + [f"{name} (removed)" for name in gone]
    for start, end, new in sorted(edits, reverse=True):
        lines[start:end] = [new]
    if edits:
        path.write_text("".join(lines))
    return moved


def add_missing_imports(path: Path) -> list[str]:
    """Add imports for well-known modules the file uses but never imports. Returns what was added."""
    text = path.read_text()
    tree = ast.parse(text)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported |= {a.asname or a.name for a in node.names}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    missing = sorted((used & AUTO_IMPORTS) - imported)
    if missing:
        path.write_text("".join(f"import {m}\n" for m in missing) + text)
    return missing


def run_tests(workspace: Path, target: str, timeout: int = 110, lines: dict | None = None) -> dict[str, str]:
    """Run pytest on target; return {"Class::test" or "test": outcome}.

    Outcome is "passed", "skipped", "assertion: <message>" (failed on an assert or a pytest.raises
    that did not raise), or "error: <message>". {"<timeout>": ...} if the whole run timed out.
    If lines is given, it gets {key: line number in target where the test failed}.
    """
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "report.xml"
        try:
            subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly",
                            f"--junitxml={report}", target], cwd=workspace, capture_output=True, text=True,
                           timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"<timeout>": "error: the file took too long"}
        if not report.exists():
            return {"<crash>": "error: pytest did not run"}
        root = ET.parse(report).getroot()
    outcomes = {}
    for case in root.iter("testcase"):
        cls = case.get("classname", "").split(".")[-1]
        name = re.sub(r"\[.*\]$", "", case.get("name", ""))
        key = f"{cls}::{name}" if cls[:1].isupper() else name
        failure, error, skipped = case.find("failure"), case.find("error"), case.find("skipped")
        if error is not None:
            outcome = "error: " + (error.get("message") or "error")[:200]
        elif failure is not None:
            message = (failure.get("message") or "").strip()
            where = re.findall(r"^(\S*?)([^/\s]+\.py):(\d+): ", failure.text or "", re.M)
            where = [int(n) for _, base, n in where if base == Path(target).name]
            if lines is not None and where:
                lines[key] = where[-1]
            kind = "assertion: " if message.startswith(ASSERTION_PREFIXES) else "error: "
            outcome = kind + message[:300]
        elif skipped is not None:
            outcome = "skipped"
        else:
            outcome = "passed"
        # a parametrized test counts as its worst case
        if outcome != "passed" and not outcomes.get(key, "passed").startswith("error"):
            outcomes[key] = outcome
        outcomes.setdefault(key, outcome)
    return outcomes


def remove_tests(path: Path, keys: set[str]) -> int:
    """Delete the test functions named by keys ("Class::test" or "test") from the file."""
    if not keys:
        return 0
    text = path.read_text()
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    spans = []

    def span(node):
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        return start - 1, node.end_lineno

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in keys:
            spans.append(span(node))
        elif isinstance(node, ast.ClassDef):
            methods = [m for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
            doomed = [m for m in methods if f"{node.name}::{m.name}" in keys]
            tests = [m for m in methods if m.name.startswith("test")]
            if doomed and len(doomed) == len(tests):
                spans.append(span(node))          # every test in the class goes: drop the class
            else:
                spans.extend(span(m) for m in doomed)
    for start, end in sorted(spans, reverse=True):
        del lines[start:end]
    path.write_text("".join(lines))
    return len(spans)


def test_sources(path: Path) -> dict[str, str]:
    """{"Class::test" or "test": source of that test function} for a test file."""
    text = path.read_text()
    tree = ast.parse(text)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[node.name] = ast.get_source_segment(text, node)
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"{node.name}::{m.name}"] = ast.get_source_segment(text, m)
    return found


def private_name_tests(path: Path, package: str) -> set[str]:
    """Tests that use a private name of the package (a name or attribute starting with one "_").

    Private names are not documented, so a test of them has no source for its expected value.
    """
    text = path.read_text()
    tree = ast.parse(text)
    private = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == package:
            private |= {a.asname or a.name for a in node.names if a.name.startswith("_")}

    def uses_private(func) -> bool:
        for n in ast.walk(func):
            if isinstance(n, ast.Name) and n.id in private:
                return True
            if isinstance(n, ast.Attribute) and n.attr.startswith("_") and not n.attr.startswith("__"):
                return True
        return False

    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and uses_private(node):
            found.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and uses_private(m):
                    found.add(f"{node.name}::{m.name}")
    return found


def uncollectable_files(workspace: Path, folder: str, timeout: int = 110) -> list[str]:
    """Files in folder that pytest cannot collect. One such file stops the whole measured run."""
    try:
        result = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
                                 folder], cwd=workspace, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return []
    return sorted(set(re.findall(r"^ERROR (\S+\.py)", result.stdout, re.M)))


CLOCK_PARAMS = {"timer", "clock", "time_func", "timefunc", "now"}


def clock_params(workspace: Path, package: str) -> dict[str, str]:
    """{name: clock parameter} for package classes (via __init__) and functions that take a clock."""
    found = {}
    for file in sorted((Path(workspace) / package).rglob("*.py")):
        try:
            tree = ast.parse(file.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            funcs = [node] if isinstance(node, ast.FunctionDef) else [
                m for m in getattr(node, "body", []) if isinstance(m, ast.FunctionDef) and m.name == "__init__"]
            for f in funcs:
                params = {a.arg for a in f.args.args + f.args.kwonlyargs} & CLOCK_PARAMS
                if params and isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    found[node.name] = sorted(params)[0]
    return found


def real_clock_tests(path: Path, clocks: dict[str, str]) -> set[str]:
    """Tests that call something taking a clock parameter without passing one: they depend on the
    real time, so their result is not determined by the docs (and may change between runs)."""
    tree = ast.parse(path.read_text())

    def uses_real_clock(func) -> bool:
        for n in ast.walk(func):
            if isinstance(n, ast.Call):
                name = n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None)
                if name in clocks:
                    given = {k.arg for k in n.keywords}
                    if clocks[name] not in given and None not in given and len(n.args) < 3:
                        return True
        return False

    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and uses_real_clock(node):
            found.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and uses_real_clock(m):
                    found.add(f"{node.name}::{m.name}")
    return found


def documented_names(workspace: Path, package: str) -> set[str] | None:
    """Public top-level names of the package that the documentation mentions, or None if the docs
    mention too few of them (under 30%) to judge by: then every name counts as documented."""
    from planner import doc_files
    public = set()
    for module in _package_names(workspace, package).values():
        public |= {n for n in module if not n.startswith("_")}
    tree_names = set()
    for file in sorted((Path(workspace) / package).rglob("*.py")):
        try:
            tree = ast.parse(file.read_text(errors="replace"))
        except SyntaxError:
            continue
        tree_names |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                       and not n.name.startswith("_")}
    public &= tree_names                         # defined here, not names the package imports
    text = "\n".join(p.read_text(errors="replace") for p in doc_files(workspace))
    mentioned = {n for n in public if re.search(r"\b" + re.escape(n) + r"\b", text)}
    if not public or len(mentioned) < 0.3 * len(public):
        return None
    return mentioned


def undocumented_tests(path: Path, package: str, documented: set[str] | None, defined: set[str]) -> set[str]:
    """Tests that use a public package name the documentation never mentions. Nothing can decide
    what such code should return (the test oracle problem), so a failure there is no evidence of a bug."""
    if documented is None:
        return set()
    tree = ast.parse(path.read_text())
    from_package = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == package:
            from_package |= {a.asname or a.name for a in node.names}
        elif isinstance(node, ast.Import):
            from_package |= {a.asname or a.name for a in node.names if a.name.split(".")[0] == package}

    def uses_undocumented(func) -> bool:
        for n in ast.walk(func):
            name = n.id if isinstance(n, ast.Name) and n.id in from_package else (
                n.attr if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                and n.value.id in from_package and n.value.id not in defined else None)
            if name and name in defined and not name.startswith("_") and name not in documented:
                return True
        return False

    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and uses_undocumented(node):
            found.add(node.name)
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and uses_undocumented(m):
                    found.add(f"{node.name}::{m.name}")
    return found


def defined_names(workspace: Path, package: str) -> set[str]:
    """Public classes and functions defined (not just imported) in the package's modules."""
    names = set()
    for file in sorted((Path(workspace) / package).rglob("*.py")):
        try:
            tree = ast.parse(file.read_text(errors="replace"))
        except SyntaxError:
            continue
        names |= {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    return names


_DOC_CACHE: dict = {}


def undocumented_claims(workspace: Path, path: Path, package: str) -> set[str]:
    """undocumented_tests with the project-level part (docs, defined names) computed once."""
    key = (str(workspace), package)
    if key not in _DOC_CACHE:
        _DOC_CACHE[key] = (documented_names(workspace, package), defined_names(workspace, package))
    documented, defined = _DOC_CACHE[key]
    return undocumented_tests(path, package, documented, defined)
