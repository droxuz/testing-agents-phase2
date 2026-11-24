"""Reviewer: decide, for each test that fails on the buggy code, whether the bug is in the code or in the test.

A failing test is worth keeping only if it is likely to be a real bug: a kept real bug scores +1,
a kept false alarm -0.5. On our development runs most failing tests were mistakes in the test
(for example, misreading how a fake timer works). So every failing test goes to a reviewer that

- sees the documentation and the unit's signatures and docstrings, but not the function bodies,
  so it judges what the code promises rather than what the buggy code does;
- sees the test and what the code actually returned;
- traces the test by hand and answers KEEP (the docs support the test, so the code is wrong) or
  DROP (the test is wrong, or the docs do not clearly decide it).

A test is kept only if VOTES independent reviews all answer KEEP. The model is sampled at
temperature 0.6, so the reviews are independent draws.
"""

import re

from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError

VOTES = 2
MAX_REVIEWED_PER_FILE = 8
VERDICT = re.compile(r"\b(test\w*)\s*[:=-]\s*\**\s*(KEEP|DROP)\b", re.I)


def _reply_text(message: dict) -> str:
    commands = [a.get("command", "") for a in message.get("extra", {}).get("actions", [])]
    return "\n".join(commands + [message.get("content") or ""])


def _ask(model, prompts: dict, variables: dict) -> tuple[dict[str, str], list[dict]]:
    """One review call. Returns ({test function name: KEEP/DROP}, messages)."""
    messages = [
        {"role": "system", "content": prompts["review_system_template"]},
        {"role": "user", "content": Template(prompts["review_template"], undefined=StrictUndefined).render(**variables)},
    ]
    try:
        message = model.query(messages)
        text = _reply_text(message)
    except FormatError as error:           # the model answered in plain text: read it anyway
        message = error.messages[0]
        response = message.get("extra", {}).get("response") or {}
        try:
            text = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            text = ""
    messages.append(message)
    verdicts = {}
    text = text.replace("\\n", "\n")    # printf 'a: KEEP\nb: DROP' arrives with a literal backslash-n
    for name, verdict in VERDICT.findall(text):
        verdicts[name] = verdict.upper()   # the last verdict for a name wins
    return verdicts, messages


def review(model, prompts: dict, base: dict, failing: dict[str, str], sources: dict[str, str]) -> tuple[set[str], list]:
    """Return the keys of failing tests to keep, and the review conversations.

    failing: {"Class::test" or "test": "assertion: <message>"}; sources: the same keys -> test source.
    """
    candidates = sorted(failing)[:MAX_REVIEWED_PER_FILE]
    conversations = []
    for _ in range(VOTES):
        if not candidates:
            break
        tests = "\n\n".join(
            f"### {key.split('::')[-1]}\n```python\n{sources[key]}\n```\n"
            f"Result on the current code: {failing[key].split(': ', 1)[-1]}" for key in candidates)
        verdicts, messages = _ask(model, prompts, base | {"tests": tests,
                                                          "names": ", ".join(k.split("::")[-1] for k in candidates)})
        conversations.append(messages)
        candidates = [k for k in candidates if verdicts.get(k.split("::")[-1]) == "KEEP"]
    return set(candidates), conversations
