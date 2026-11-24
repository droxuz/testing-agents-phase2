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


def run_tests(workspace: Path, target: str, timeout: int = 110) -> dict[str, str]:
    """Run pytest on target; return {"Class::test" or "test": outcome}.

    Outcome is "passed", "skipped", "assertion: <message>" (failed on an assert or a pytest.raises
    that did not raise), or "error: <message>". {"<timeout>": ...} if the whole run timed out.
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
