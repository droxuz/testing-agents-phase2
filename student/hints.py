"""Test design hints computed from the code and the docs (no model calls).

Two lecture techniques, done in Python and given to the test writer as concrete inputs to try:

- Boundary value analysis: integer constants the unit's code depends on (named class or module
  constants such as a load factor, and parameter defaults such as a maxsize) mark places where
  behaviour changes, so tests should use sizes and values just below, at and above them.
- Equivalence partitioning: (a) from the code's own branches on a parameter (`x is None`,
  `callable(x)`, `isinstance(x, T)`, `x < 0`, ...), each branch is an input class worth a test;
  (b) from each parameter's name and default, the input classes worth one test each (negative
  indexes, equal keys for a key function, times well before and after an expiry, ...).

And one form of specification mining / differential testing:

- Docs vs code: signatures written in the documentation (for example
  `lru_cache(maxsize=128, typed=False)`) are compared with the defaults in the code. A mismatch
  means the docs or the code is wrong, and a test of the documented default decides which.
"""

import ast
import re
from pathlib import Path

from planner import doc_files

SIGNATURE_LINE = re.compile(r"^\s*(?:\.\. [\w:]+:: )?(?:[\w]+\.)*(\w+)\((.*)\)\s*$")

ROLES = [   # (parameter-name pattern, input classes to cover)
    (r"^(index|idx|pos|position|i|start|stop|begin|end)$",
     "0, a positive index, -1 and other negative indexes, len-1, len and beyond (both ends), "
     "leaving it out if it has a default, and (for a start/stop range) a value that occurs only "
     "before start or only at or after stop, which must not be found"),
    (r"^key=None$", "a key function where different values have equal keys (for example key=abs with 3 and -3, "
     "or key=lambda x: x // 10), duplicates of the same key, and key=None"),
    (r"^(maxsize|size|capacity|max_size|limit|n|count|load)$", "0, 1, a small number, and the default value"),
    (r"^(ttl|ttu|timer|time|now|timeout|expires?)$",
     "a fake timer you set explicitly; check well before and well after an expiry time (not the exact "
     "instant, which the docs rarely define)"),
    (r"^(reverse|inclusive|typed|strict)$", "both True and False"),
    (r"^(default|fill|fillvalue)$", "leaving it out, and passing a value"),
    (r"^(minimum|maximum|min|max|lo|hi|low|high|lower|upper)$",
     "values below, at and above each bound, and None if allowed"),
]


def _literal(node):
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _defaults(args: ast.arguments) -> dict[str, object]:
    out = {}
    positional = args.posonlyargs + args.args
    for arg, default in zip(positional[len(positional) - len(args.defaults):], args.defaults):
        ok, value = _literal(default)
        if ok:
            out[arg.arg] = value
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        if default is not None:
            ok, value = _literal(default)
            if ok:
                out[arg.arg] = value
    return out


def _unit_functions(tree: ast.Module, members: list[str]):
    """(display name, doc name, FunctionDef) for the unit's functions and methods."""
    wanted = set(members)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            yield node.name, node.name, node
        elif isinstance(node, ast.ClassDef):
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and f"{node.name}.{m.name}" in wanted:
                    doc_name = node.name if m.name == "__init__" else m.name
                    yield f"{node.name}.{m.name}", doc_name, m


def _with_bases(tree: ast.Module, classes: set[str]) -> set[str]:
    """The classes plus their base classes defined in the same module."""
    defs = {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}
    todo, seen = list(classes), set()
    while todo:
        name = todo.pop()
        if name in seen or name not in defs:
            continue
        seen.add(name)
        todo += [b.id for b in defs[name].bases if isinstance(b, ast.Name)]
    return seen


def _branches(func) -> list[str]:
    """Conditions in func's body that test one of its parameters directly."""
    params = {a.arg for a in func.args.posonlyargs + func.args.args + func.args.kwonlyargs} - {"self", "cls"}
    found = []
    for node in ast.walk(func):
        test = getattr(node, "test", None) if isinstance(node, (ast.If, ast.IfExp, ast.While)) else None
        parts = test.values if isinstance(test, ast.BoolOp) else [test]
        for part in parts:
            if isinstance(part, ast.UnaryOp) and isinstance(part.op, ast.Not):
                part = part.operand
            names = {n.id for n in ast.walk(part) if isinstance(n, ast.Name)} if part is not None else set()
            simple = isinstance(part, (ast.Compare, ast.Call, ast.Name)) and len(ast.unparse(part)) <= 50
            if simple and names & params:
                found.append(ast.unparse(part))
    return list(dict.fromkeys(found))


def _constants(tree: ast.Module, classes: set[str]) -> list[str]:
    """Named integer constants (>= 2) at module level or in the unit's classes."""
    found = []

    def scan(body, owner):
        for node in body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                ok, value = _literal(node.value)
                if ok and isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 2:
                    found.append(f"{owner}{node.targets[0].id} = {value}")

    scan(tree.body, "")
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in classes:
            scan(node.body, f"{node.name}.")
    return found


def _documented_signatures(workspace: Path) -> dict[str, list[str]]:
    """{function name: [argument text of each signature line in the docs]}."""
    root = Path(workspace)
    paths = doc_files(root)
    found: dict[str, list[str]] = {}
    for path in paths:
        for line in path.read_text(errors="replace").splitlines():
            if line.lstrip().startswith((">>>", "...", "@", "#")):
                continue
            m = SIGNATURE_LINE.match(line)
            if m and "=" in m.group(2):
                found.setdefault(m.group(1), []).append(m.group(2))
    return found


def _doc_defaults(args_text: str) -> dict[str, object]:
    try:
        func = ast.parse(f"def _({args_text}): pass").body[0]
    except SyntaxError:
        return {}
    return _defaults(func.args)


def hints(workspace: Path, unit) -> str:
    """The hints section for one unit (planner.Unit), or a short note if there are none."""
    tree = ast.parse((Path(workspace) / unit.file).read_text())
    classes = _with_bases(tree, {m.split(".")[0] for m in unit.members if "." in m})
    lines = []

    documented = _documented_signatures(workspace)
    mismatches = []
    for shown, doc_name, func in _unit_functions(tree, unit.members):
        code = _defaults(func.args)
        for args_text in documented.get(doc_name, []):
            for name, doc_value in _doc_defaults(args_text).items():
                if name in code and code[name] != doc_value:
                    mismatches.append(f"- `{shown}`: the documentation gives `{doc_name}({args_text})`, but in the "
                                      f"code the default is `{name}={code[name]!r}`.")
    if mismatches:
        lines += ["Documentation and code disagree (one of them is wrong; test the documented value through "
                  "behaviour you can observe, for example by calling without the argument):"]
        lines += sorted(set(mismatches))

    constants = _constants(tree, classes)
    defaults = []
    roles = {}
    branches = []
    constructors = [m for n in tree.body if isinstance(n, ast.ClassDef) and n.name in classes
                    for m in n.body if isinstance(m, ast.FunctionDef) and m.name in ("__init__", "__new__")]
    for func in constructors:      # a chunk of a class without its __init__ still needs its key= classes
        if _defaults(func.args).get("key", 0) is None:
            roles.setdefault(("key", ROLES[1][1]), set()).add("the constructor")
    for shown, _, func in _unit_functions(tree, unit.members):
        tests = _branches(func)
        if tests and not shown.split(".")[-1].startswith("_") or tests and shown.endswith("__init__"):
            branches.append(f"- `{shown}`: " + ", ".join(f"`{t}`" for t in tests[:6]))
        for name, value in _defaults(func.args).items():
            if isinstance(value, int) and not isinstance(value, bool) and abs(value) >= 2:
                defaults.append(f"{shown}({name}={value})")
        for arg in func.args.posonlyargs + func.args.args + func.args.kwonlyargs:
            label = f"{arg.arg}=None" if _defaults(func.args).get(arg.arg, 0) is None else arg.arg
            for pattern, classes_text in ROLES:
                if re.match(pattern, label):
                    roles.setdefault((arg.arg, classes_text), set()).add(shown.split(".")[-1])
    if constants or defaults:
        lines += ["Boundary values: behaviour may change around these numbers, so also use sizes or values just "
                  "below, at and above them (for a container, more elements than the number, e.g. 2.5x):"]
        lines += [f"- {c}" for c in constants + sorted(set(defaults))][:12]
    if branches:
        lines += ["Input classes from the code's own branches (each condition true and false is a case worth "
                  "a test; check the documented result for each, not what the code does):"]
        lines += branches[:10]
    if roles:
        lines += ["Input classes to cover, from the parameters:"]
        for (name, classes_text), funcs in sorted(roles.items()):
            lines.append(f"- `{name}` (in {', '.join(sorted(funcs)[:6])}): {classes_text}")
    return "\n".join(lines) if lines else "(no computed hints for this unit)"
