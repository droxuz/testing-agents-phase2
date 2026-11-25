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
        return predicted is True or str(predicted).lower() == "true"
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
            lines: dict[str, int]) -> tuple[set[str], set[str], dict]:
    """Return (keys whose blind prediction agrees, keys that could not be asked, log)."""
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
    names = {f"snippet_{i}": key for i, key in enumerate(questions, 1)}   # test names would leak the claim
    snippets = "\n\n".join(f"### {name}\n```python\n{questions[key]['snippet']}\n```" for name, key in names.items())
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
    log |= {"answers": answers, "agreed": sorted(agreed), "messages": messages}
    return agreed, unasked, log
