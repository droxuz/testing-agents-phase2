"""Blind oracle: predict the value at a failing assert without seeing what the test expects.

Why: our reviewer (reviewer.py) sees each failing test with its expected value, and the small model
tends to agree with whatever the test claims ("anchoring"), even when the project's own tests show
the claim is wrong. So before the review, every failing test is cut off just before the assert that
failed, and a fresh model call, which sees only the documentation, the signatures and the project's
own tests, predicts what the asserted expression evaluates to. This is N-version testing with the
answer hidden: a claim survives only if an independent version (the blind prediction) agrees with
the test's expected value. A test that guessed wrongly about an undocumented detail rarely gets
the same guess from an independent reader.

Three kinds of failing assert are handled:
- value:  `assert X == LITERAL` (or LITERAL == X)  -> predict X, compare with LITERAL
- truth:  any other `assert E`                     -> predict bool(E), expected True
- raises: `with pytest.raises(T):` that did not raise -> predict RAISES/NO, expected RAISES
Anything else (no failing line found, assert in a helper) is passed to the reviewer unchanged.
"""

import ast
import re
import textwrap
from pathlib import Path

from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError

ANSWER = re.compile(r"^\s*[-*]?\s*`?(snippet_\d+)`?\s*[:=]\s*(.+?)\s*$", re.M)


def _literal(node):
    try:
        return True, ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False, None


def _function(tree: ast.Module, key: str):
    cls, _, name = key.rpartition("::")
    for node in tree.body:
        if cls and isinstance(node, ast.ClassDef) and node.name == cls:
            for m in node.body:
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name == name:
                    return m
        if not cls and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _statement_at(func, line: int):
    """The innermost assert or `with pytest.raises` statement of func on that line."""
    best = None
    for node in ast.walk(func):
        if isinstance(node, (ast.Assert, ast.With)) and node.lineno <= line <= node.end_lineno:
            if isinstance(node, ast.With) and node.lineno != line:
                continue
            if best is None or node.lineno >= best.lineno:
                best = node
    return best


def question(text: str, key: str, line: int) -> dict | None:
    """Build the blind question for one failing test, or None if it cannot be built."""
    tree = ast.parse(text)
    func = _function(tree, key)
    if func is None or not (func.lineno <= line <= func.end_lineno):
        return None
    stmt = _statement_at(func, line)
    if stmt is None:
        return None
    lines = text.splitlines()
    indent = " " * stmt.col_offset
    before = lines[func.lineno - 1:stmt.lineno - 1]
    if isinstance(stmt, ast.Assert):
        test = stmt.test
        kind, expected, expr = "truth", True, f"bool({ast.unparse(test)})"
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            left_ok, left = _literal(test.left)
            right_ok, right = _literal(test.comparators[0])
            if right_ok and not left_ok:
                kind, expected, expr = "value", right, ast.unparse(test.left)
            elif left_ok and not right_ok:
                kind, expected, expr = "value", left, ast.unparse(test.comparators[0])
        ask = before + [f"{indent}result = {expr}", f"{indent}# What is `result` here?"]
    else:
        items = stmt.items
        if not (items and isinstance(items[0].context_expr, ast.Call)
                and ast.unparse(items[0].context_expr.func).endswith("raises") and items[0].context_expr.args):
            return None
        kind, expected = "raises", "RAISES"
        exc = ast.unparse(items[0].context_expr.args[0])
        body = "\n".join(ast.get_source_segment(text, s) or "" for s in stmt.body)
        ask = before + [textwrap.indent(body, indent),
                        f"{indent}# Does the code just above raise {exc}? Answer RAISES or NO."]
    imports = [ast.get_source_segment(text, n) for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    return {"kind": kind, "expected": expected, "snippet": "\n".join(imports + [""] + [_blind(ask)])}


def _blind(lines: list[str]) -> str:
    """Drop the writer's comments and docstring: they state what the test expects."""
    question_line = lines[-1].strip()
    source = textwrap.dedent("\n".join(lines[:-1]))
    try:
        func = ast.parse(source).body[0]
    except (SyntaxError, IndexError):
        return "\n".join(lines)
    if (func.body and isinstance(func.body[0], ast.Expr) and isinstance(func.body[0].value, ast.Constant)
            and isinstance(func.body[0].value.value, str) and len(func.body) > 1):
        func.body = func.body[1:]
    func.name = "snippet"
    func.args = ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])
    func.decorator_list = []
    return ast.unparse(func) + "\n    " + question_line


def _parse(value: str):
    value = value.strip().strip("`").strip()
    try:
        return ast.literal_eval(value)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return value


def agrees(kind: str, expected, answer: str) -> bool:
    """Does the blind answer agree with the value the test expects?"""
    if kind == "raises":
        return answer.strip().strip("`'\"").upper().startswith("RAISE")
    predicted = _parse(answer)
    if kind == "truth":
        said = True if predicted is True or str(predicted).lower() == "true" else (
            False if predicted is False or str(predicted).lower() == "false" else None)
        return said is not None and said == bool(expected)
    if predicted == expected:
        return True
    return isinstance(expected, str) and str(predicted).strip("'\"") == expected


def _reply_text(model, messages) -> tuple[str, dict]:
    try:
        message = model.query(messages)
        commands = [a.get("command", "") for a in message.get("extra", {}).get("actions", [])]
        return "\n".join(commands + [message.get("content") or ""]), message
    except FormatError as error:           # the model answered in plain text: read it anyway
        message = error.messages[0]
        response = message.get("extra", {}).get("response") or {}
        try:
            return response["choices"][0]["message"].get("content") or "", message
        except (KeyError, IndexError, TypeError):
            return "", message


def predict(model, prompts: dict, base: dict, text: str, failing: dict[str, str],
            lines: dict[str, int], control_questions: list[dict] | None = None) -> tuple[set[str], set[str], dict]:
    """Return (keys whose blind prediction agrees, keys that could not be asked, log).

    control_questions (from controls()) are asked in the same call, mixed in with the claims; if
    the oracle gets any of them wrong, no claim of this call is accepted."""
    questions, unasked = {}, set()
    for key in sorted(failing):
        q = question(text, key, lines[key]) if key in lines else None
        if q:
            questions[key] = q
        else:
            unasked.add(key)
    log = {"questions": {k: {"kind": q["kind"], "expected": repr(q["expected"]), "snippet": q["snippet"]}
                         for k, q in questions.items()}}
    if not questions:
        return set(), unasked, log
    asked = dict(questions)
    for i, q in enumerate(control_questions or []):
        asked[f"<control {i}>"] = q
    order = sorted(asked)
    random.Random(len(text)).shuffle(order)             # controls are not always last
    names = {f"snippet_{i}": key for i, key in enumerate(order, 1)}   # test names would leak the claim
    snippets = "\n\n".join(f"### {name}\n```python\n{asked[key]['snippet']}\n```" for name, key in names.items())
    messages = [
        {"role": "system", "content": prompts["oracle_system_template"]},
        {"role": "user", "content": Template(prompts["oracle_template"], undefined=StrictUndefined).render(
            **base, snippets=snippets, names=", ".join(names))},
    ]
    reply, message = _reply_text(model, messages)
    messages.append(message)
    answers = {}
    for name, value in ANSWER.findall(reply.replace("\\n", "\n")):
        if name in names:
            answers[names[name]] = value             # the last answer for a name wins
    agreed = {k for k, q in questions.items() if k in answers and agrees(q["kind"], q["expected"], answers[k])}
    failed_controls = [k for k in asked if k.startswith("<control")
                       and not (k in answers and agrees(asked[k]["kind"], asked[k]["expected"], answers[k]))]
    log |= {"controls": [{"snippet": q["snippet"], "expected": repr(q["expected"]),
                          "answer": answers.get(f"<control {i}>")} for i, q in enumerate(control_questions or [])],
            "failed_controls": failed_controls, "agreed_before_controls": sorted(agreed)}
    if failed_controls:
        agreed = set()                                   # the oracle does not understand this area
    log |= {"answers": answers, "agreed": sorted(agreed), "messages": messages}
    return agreed, unasked, log


# ---------------------------------------------------------------------------------------------
# Control questions: known answers from the project's own tests.
#
# The project's tests pass on the current code, so each of their asserts has a known answer.
# A few of them, chosen for using the same names as the claims, are blinded the same way and
# mixed in with the claims. If the oracle gets a control wrong, it does not understand this part
# of the project well enough to be trusted, and none of the claims of that call is kept. (Like
# calibrating an instrument on samples with a known value before trusting its readings.)
# ---------------------------------------------------------------------------------------------

import builtins
import random

_UNITTEST = {   # self.assertX(a, b) -> the plain assert it means
    "assertEqual": lambda a: ast.Compare(a[0], [ast.Eq()], [a[1]]),
    "assertNotEqual": lambda a: ast.Compare(a[0], [ast.NotEq()], [a[1]]),
    "assertIn": lambda a: ast.Compare(a[0], [ast.In()], [a[1]]),
    "assertNotIn": lambda a: ast.Compare(a[0], [ast.NotIn()], [a[1]]),
    "assertIs": lambda a: ast.Compare(a[0], [ast.Is()], [a[1]]),
    "assertIsNot": lambda a: ast.Compare(a[0], [ast.IsNot()], [a[1]]),
    "assertIsNone": lambda a: ast.Compare(a[0], [ast.Is()], [ast.Constant(None)]),
    "assertIsNotNone": lambda a: ast.Compare(a[0], [ast.IsNot()], [ast.Constant(None)]),
    "assertTrue": lambda a: a[0],
    "assertFalse": lambda a: ast.UnaryOp(ast.Not(), a[0]),
    "assertLess": lambda a: ast.Compare(a[0], [ast.Lt()], [a[1]]),
    "assertGreater": lambda a: ast.Compare(a[0], [ast.Gt()], [a[1]]),
}


class _ToPlainAsserts(ast.NodeTransformer):
    def visit_Expr(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "self"
                and call.func.attr in _UNITTEST and len(call.args) >= (1 if call.func.attr in
                                                                       ("assertTrue", "assertFalse", "assertIsNone",
                                                                        "assertIsNotNone") else 2)):
            return ast.copy_location(ast.Assert(test=_UNITTEST[call.func.attr](call.args), msg=None), node)
        return node


def _blind_assert(test):
    """(kind, expected, expression to ask about) for one plain assert of a known-good test."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        op, left, right = test.ops[0], test.left, test.comparators[0]
        left_ok, left_value = _literal(left)
        right_ok, right_value = _literal(right)
        if isinstance(op, ast.Eq) and right_ok and not left_ok:
            return "value", right_value, left
        if isinstance(op, ast.Eq) and left_ok and not right_ok:
            return "value", left_value, right
        if isinstance(op, (ast.NotIn, ast.IsNot, ast.NotEq)):      # ask the positive form: answer False
            positive = {ast.NotIn: ast.In, ast.IsNot: ast.Is, ast.NotEq: ast.Eq}[type(op)]()
            return "truth", False, ast.Compare(left, [positive], [right])
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return "truth", False, test.operand
    return "truth", True, test


class _SelfAttrs(ast.NodeTransformer):
    """self.Cache -> MRUCache, where the concrete TestCase sets `Cache = MRUCache` (mixin pattern)."""

    def __init__(self, attrs: dict[str, ast.expr]):
        self.attrs = attrs

    def visit_Attribute(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self" and node.attr in self.attrs:
            return ast.copy_location(self.attrs[node.attr], node)
        return node


def _control(func, imports: list[str], imported: set[str], helpers: dict[str, str] | None = None,
             attrs: dict[str, ast.expr] | None = None) -> dict | None:
    """Blind question from an assert of an existing test function, or None if it is not self-contained."""
    func = ast.parse(ast.unparse(func)).body[0]
    if attrs:
        func = _SelfAttrs(attrs).visit(func)
    func = _ToPlainAsserts().visit(func)
    if any(isinstance(n, ast.Name) and n.id == "self" for n in ast.walk(func)):
        return None                    # uses fixtures or helpers of a TestCase: not self-contained
    if any(isinstance(n, ast.Attribute) and n.attr.startswith("_") and not n.attr.startswith("__")
           for n in ast.walk(func)):
        return None                    # private internals: the docs do not say how they behave
    assigned = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    assigned |= {a.arg for a in func.args.args} | {n.name for n in ast.walk(func)
                                                  if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    loaded = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    params = [a.arg for a in func.args.args]
    helpers = helpers or {}
    missing = loaded - assigned - imported - set(dir(builtins))
    if missing - set(helpers) or params not in ([], ["self"]):
        return None                    # needs names from elsewhere (fixtures, unknown helpers)
    needed = [helpers[name] for name in sorted(missing)]   # helper classes/functions of the test module
    asserts = [i for i, s in enumerate(func.body) if isinstance(s, ast.Assert)]
    if not asserts:
        return None
    options = []
    for i in asserts:
        kind, expected, expr = _blind_assert(func.body[i].test)
        trivial = (expected is True) + (kind == "value" and ast.unparse(expr).startswith("len("))
        steps = sum(not isinstance(st, ast.Assert) for st in func.body[:i])   # state to trace before it
        options.append((trivial, -steps, i, kind, expected, expr))
    _, _, i, kind, expected, expr = min(options)   # a question that needs the semantics, not a guess of True
    keep = [st for st in func.body[:i] if not isinstance(st, ast.Assert)]   # other asserts give answers away
    body = keep + [ast.parse(f"result = {'bool(' + ast.unparse(expr) + ')' if kind == 'truth' else ast.unparse(expr)}").body[0]]
    snippet = ast.FunctionDef(name="snippet", args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[],
                              kw_defaults=[], defaults=[]), body=body, decorator_list=[], returns=None,
                              type_params=[])
    text = ast.unparse(ast.fix_missing_locations(snippet)) + "\n    # What is `result` here?"
    if text.count("\n") > 40 or re.search(r"result = (repr|str|format)\(", text):
        return None                    # too long, or about text formatting (rarely documented)
    return {"kind": kind, "expected": expected, "snippet": "\n".join(imports + [""] + needed + ([""] if needed else []) + [text])}


def controls(workspace, claim_sources: list[str], package: str, count: int = 2) -> list[dict]:
    """Up to count control questions from the project's tests that use the claims' names."""
    from planner import _identifiers      # same name matching as the evidence for the reviewer
    root = Path(workspace) / "tests"
    wanted = set()
    for source in claim_sources:
        try:
            wanted |= _identifiers(ast.parse(textwrap.dedent(source)))
        except SyntaxError:
            pass
    if not wanted or not root.is_dir():
        return []
    files = sorted(root.rglob("test*.py")) + sorted(p for p in root.rglob("*.py") if p.name in ("__init__.py", "conftest.py"))
    classes, helpers, file_imports = {}, {}, {}
    for path in files:
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        imports, imported = [], set()
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = (node.module or "") if isinstance(node, ast.ImportFrom) else node.names[0].name
                if module.split(".")[0] in (package, "warnings", "math", "functools", "itertools",
                                            "collections", "operator", "string", "random"):
                    imports.append(ast.unparse(node))
                    imported |= {(a.asname or a.name).split(".")[0] for a in node.names}
            elif isinstance(node, ast.ClassDef):
                classes[node.name] = (node, path)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and not node.name.lower().startswith("test"):
                has_tests = isinstance(node, ast.ClassDef) and any(
                    isinstance(m, ast.FunctionDef) and m.name.startswith("test") for m in node.body)
                source = ast.unparse(node)
                if not has_tests and source.count("\n") < 25:
                    helpers[node.name] = source            # e.g. a fake timer class
        file_imports[path] = (imports, imported)

    def lineage(name, depth=0):
        """The class and its bases defined in the tests, most derived first."""
        if name not in classes or depth > 5:
            return []
        node, _ = classes[name]
        return [node] + [c for b in node.bases if isinstance(b, ast.Name) for c in lineage(b.id, depth + 1)]

    scored = []

    def consider(func, path, attrs, extra_imports=None):
        imports, imported = file_imports.get(path, ([], set()))
        if extra_imports:
            imports, imported = list(dict.fromkeys(imports + extra_imports[0])), imported | extra_imports[1]
        probe = _SelfAttrs(attrs).visit(ast.parse(ast.unparse(func)).body[0]) if attrs else func
        hits = _identifiers(probe) & wanted
        weight = sum(3 if h[:1].isupper() or len(h) >= 8 else 1 for h in hits)   # distinctive names
        if weight >= 3:
            scored.append((-weight, path.name, func.lineno, func, imports, imported, attrs))

    for path in files:
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                consider(node, path, {})
            elif isinstance(node, ast.ClassDef):
                chain = lineage(node.name)
                attrs = {}
                for cls in reversed(chain):               # derived classes override their bases
                    for st in cls.body:
                        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                            value = st.value
                            if isinstance(value, ast.Call) and ast.unparse(value.func) in ("staticmethod", "classmethod") and value.args:
                                value = value.args[0]
                            if isinstance(value, (ast.Name, ast.Attribute)):
                                attrs[st.targets[0].id] = value
                methods = {}
                for cls in reversed(chain):
                    methods.update({m.name: (m, classes[cls.name][1]) for m in cls.body
                                    if isinstance(m, ast.FunctionDef) and m.name.startswith("test")})
                for name, (method, origin) in methods.items():
                    consider(method, path, attrs, file_imports.get(origin) if origin != path else None)
    candidates, seen = [], set()
    for weight, _, _, node, imports, imported, attrs in sorted(scored, key=lambda s: s[:3])[:80]:
        question = _control(node, imports, imported, helpers, attrs)
        if question and question["snippet"] not in seen:
            seen.add(question["snippet"])
            body = question["snippet"].rsplit("result = ", 1)[0]
            echoes = question["kind"] == "value" and not isinstance(question["expected"], bool) and \
                re.search(r"(?<![\w.])" + re.escape(repr(question["expected"])) + r"(?![\w.])", body)
            question["trivial"] = (question["expected"] is True) + question["snippet"].count("result = len(") + \
                bool(echoes)                               # the answer is written in the code already
            candidates.append((weight, question))
    # Most relevant first (same names as the claims); among equally relevant ones, a question that
    # needs the semantics, not one that the answer True (or a length) passes.
    # A control whose answer is True tells nothing when passed (guessing True passes it), and on
    # controls-r2 it blocked only real claims when failed, so only harder controls are used.
    candidates = [c for c in candidates if c[1]["expected"] is not True]
    candidates.sort(key=lambda c: (c[0], c[1]["trivial"]))
    return [q for _, q in candidates[:count]]
