"""Our agent: a planner loop in Python around many short model conversations.

The starter agent kept one conversation for the whole project. Every call resent every earlier output,
so the prompt grew to ~30k tokens and the 400k budget was gone after ~20 calls, mostly spent reading.

This agent instead:
1. Plans without the model: runs extract_spec and sbst --list, and splits the package into work
   items (a class or a few of its methods, or a few functions), ordered so every class gets a turn early.
2. For each item, starts a FRESH conversation with a short brief that Python prepares: the spec claims
   (step 1), and the sbst cases (inputs that reach branches and boundaries). The model writes one test
   file, runs it, fixes it, and submits. Each conversation has a step limit and a token cap, keeps only
   the last few command outputs in full, and cuts long outputs.
3. Runs mutants for each class it tested. Where a surviving mutant has an input on which current code
   and mutant differ, a short conversation adds tests asserting what the spec promises.
4. Checks every test that fails on the current code: `mutants --check-failing` looks for one small
   change to the code that makes the test pass while all other tests stay green (the bugs are small
   changes, so a right test can be "repaired" this way). Tests with no such change are probably wrong
   expectations and are disabled, since each false alarm costs half a found bug.
5. Removes test files the evaluator would reject or that do not even import.

Settings (which techniques are on, limits) are in config.yaml next to this file.
"""

import copy
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template
from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import LimitsExceeded

ELIDED = "(older output removed to save tokens; rerun the command if you need it again)"
SOURCE_BLOCKED = ("Reading the package source is switched off for this task: it shows the buggy behaviour "
                  "and uses up your budget. Take expected values from the spec in the task, README.rst or docs/.")
READERS = r"\b(cat|head|tail|sed|less|more|awk|grep|nl|python3?\s+-c\s+.*getsource)\b"


# --------------------------------------------------------------------------------- conversations
class Episode(DefaultAgent):
    """One short conversation: a step limit, a token cap, and a trimmed history."""

    def __init__(self, model, env, *, token_cap: int, keep_outputs: int, max_output_chars: int,
                 package: str = "", **kwargs):
        super().__init__(model, env, **kwargs)
        self.source = re.compile(READERS + r"[^|;&]*" + re.escape(package) + r"/") if package else None
        self.token_cap = token_cap
        self.keep_outputs = keep_outputs
        self.max_output_chars = max_output_chars
        self.start_tokens = model.total_tokens

    def query(self) -> dict:
        used = self.model.total_tokens - self.start_tokens
        if 0 < self.config.step_limit <= self.n_calls or used >= self.token_cap:
            raise LimitsExceeded({"role": "exit", "content": "LimitsExceeded",
                                  "extra": {"exit_status": "LimitsExceeded", "submission": ""}})
        self.n_calls += 1
        message = self.model.query(self.trimmed())
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self.add_messages(message)
        return message

    def trimmed(self) -> list[dict]:
        """The history sent to the model: old outputs and old reasoning replaced by a short note."""
        messages = copy.copy(self.messages)
        tools = [i for i, m in enumerate(messages) if m.get("role") == "tool"]
        for i in tools[:-self.keep_outputs] if self.keep_outputs else tools:
            messages[i] = {**messages[i], "content": ELIDED}
        assistants = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        for i in assistants[:-1]:
            messages[i] = {k: v for k, v in messages[i].items() if k not in ("reasoning", "reasoning_details")}
        return messages

    def execute_actions(self, message: dict) -> list[dict]:
        outputs = []
        for action in message.get("extra", {}).get("actions", []):
            if self.source and self.source.search(action.get("command", "")):
                outputs.append({"output": SOURCE_BLOCKED, "returncode": 1, "exception_info": ""})
            else:
                outputs.append(self.env.execute(action))
        for output in outputs:
            output["output"] = cut(output.get("output", ""), self.max_output_chars)
        return self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))


def cut(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return (text[:head] + f"\n... [{len(text) - limit} characters cut; use grep, head or a narrower command] ...\n"
            + text[-(limit - head):])


# ---------------------------------------------------------------------------------------- planning
@dataclass
class Item:
    """One unit of work: a class (or some of its methods) or a few functions."""
    title: str                    # e.g. cachetools.LFUCache  or  cachetools.LFUCache (part 2)
    owner: str                    # the class or module the item belongs to
    entities: list = field(default_factory=list)   # extract_spec entities in this item
    sbst_target: str = ""         # what to run sbst on for the brief
    slug: str = ""
    score: float = 0.0
    part: int = 1
    is_class: bool = True


def sh(env, command: str, timeout: int | None = None) -> str:
    try:
        result = env.execute({"command": command}, timeout=timeout)
    except Exception as error:  # a tool failing must never stop the agent
        return f"(command failed: {error})"
    return result.get("output", "") or ""


def load_json(text: str) -> dict:
    start = text.find("{")
    try:
        return json.loads(text[start:]) if start >= 0 else {}
    except json.JSONDecodeError:
        return {}


def qualname(entity: dict) -> str:
    """The name inside its module, as sbst and mutants take it: LFUCache, LFUCache.popitem, hashkey."""
    parts = entity["name"].split(".")
    keep = 1 if entity["kind"] in ("class", "function") else 2
    return ".".join(parts[-keep:])


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "x"


def plan(env, task, cfg) -> tuple[list[Item], dict]:
    spec = load_json(sh(env, "extract_spec --compact", timeout=120))
    entities = [e for e in spec.get("entities", []) if e.get("kind") != "module"]
    branches = {}
    if cfg["use_sbst"]:
        for line in sh(env, "sbst --list", timeout=120).splitlines():
            m = re.match(r"(\S+)\s+\[(\w+)\]\s+(\d+) branch goals", line)
            if m:
                branches[m.group(1)] = int(m.group(3))

    classes = [e for e in entities if e.get("kind") == "class"]
    class_names = {e["name"] for e in classes}
    members = {c["name"]: [] for c in classes}
    loose = []
    for e in entities:
        if e["kind"] == "class":
            continue
        owner = next((c for c in sorted(class_names, key=len, reverse=True) if e["name"].startswith(c + ".")), None)
        if owner:
            members[owner].append(e)
        elif e["kind"] in ("function",):
            loose.append(e)

    size = cfg["methods_per_item"]
    rounds: list[list[Item]] = []
    for c in classes:
        methods = sorted(members[c["name"]], key=lambda e: -len(e.get("claims", [])))
        methods = [m for m in methods if m.get("claims") or not m["name"].rsplit(".", 1)[-1].startswith("_")
                   or m["name"].rsplit(".", 1)[-1] in ("__getitem__", "__setitem__", "__delitem__", "__contains__")]
        if not c.get("claims") and not methods:
            continue
        score = len(c.get("claims", [])) + sum(len(m.get("claims", [])) for m in methods) \
            + branches.get(c["name"], 0) / 5
        chunks = [methods[i:i + size] for i in range(0, len(methods), size)] or [[]]
        short = qualname(c)
        for n, chunk in enumerate(chunks, 1):
            target = short if n == 1 or not chunk else qualname(chunk[0])
            item = Item(f"{short}" + (f" (part {n})" if n > 1 else ""), c["name"], ([c] if n == 1 else []) + chunk,
                        target, slugify(short + (f"_{n}" if n > 1 else "")), score, n)
            while len(rounds) < n:
                rounds.append([])
            rounds[n - 1].append(item)
    modules = {}
    for f in loose:
        modules.setdefault(f["name"].rsplit(".", 1)[0], []).append(f)
    for module, funcs in modules.items():
        funcs.sort(key=lambda e: -len(e.get("claims", [])))
        for n, chunk in enumerate([funcs[i:i + size] for i in range(0, len(funcs), size)], 1):
            score = sum(len(f.get("claims", [])) for f in chunk) + branches.get(chunk[0]["name"], 0) / 5
            item = Item(f"functions in {module}" + (f" (part {n})" if n > 1 else ""), module, chunk,
                        qualname(chunk[0]), slugify(module.split(".")[-1] + f"_funcs_{n}"), score, n, False)
            while len(rounds) < n:
                rounds.append([])
            rounds[n - 1].append(item)
    items = [i for r in rounds for i in sorted(r, key=lambda i: -i.score)]
    return items, spec


def import_line(fullname: str) -> str:
    module, name = fullname.rsplit(".", 1)
    return f"from {module} import {name}"


def brief(item: Item, env, cfg) -> str:
    """What the model needs for one item, prepared by Python: imports, signatures, claims, sbst cases."""
    lines = []
    if item.is_class:
        lines.append(f"Import: {import_line(item.owner)}")
    else:
        lines.append("Imports: " + "; ".join(import_line(e["name"]) for e in item.entities))
    lines.append("\n## Spec (extract_spec claims: the ONLY source of expected values)")
    budget = cfg["spec_chars"]
    for e in item.entities:
        sig = e.get("signature", "")
        head = f"\n### {e['name']}{sig if e['kind'] not in ('property',) else ''}  [{e['kind']}]"
        claims = []
        for c in e.get("claims", []):
            text = " ".join(str(c.get("text", "")).split())
            if c.get("kind") == "example" and c.get("code"):
                setup = c.get("setup") or []
                setup = "; ".join(" ".join(str(x).split()) for x in (setup if isinstance(setup, list) else [setup]))
                text = (f"example: {setup[-200:] + '; ' if setup else ''}{' '.join(c['code'].split())}  ->  "
                        f"{' '.join(str(c.get('expected', '')).split())}")
            else:
                text = f"{c.get('kind', 'claim')}: {text}"
            if text not in claims:
                claims.append(text[:300])
        block = head + ("\n" + "\n".join(f"- {c}" for c in claims[:8]) if claims else "\n- (no claims of its own)")
        if budget - len(block) < 0 and len(lines) > 3:
            lines.append(f"\n(more methods in this item: run extract_spec --name '{qualname(e)}$')")
            break
        budget -= len(block)
        lines.append(block)
    if cfg["use_sbst"] and item.sbst_target:
        out = sh(env, f"sbst {item.sbst_target} --cases {cfg['sbst_cases']} --budget {cfg['sbst_budget']}", timeout=110)
        out = out.split("\nSpec from extract_spec")[0].strip()
        if "Case 1" in out:
            lines.append("\n## sbst cases: inputs that reach branches and boundaries of the code\n"
                         "`current:` is what the CURRENT, possibly buggy, code returns. It is NOT the expected value.")
            lines.append(cut(out, cfg["sbst_chars"]))
    return "\n".join(lines)


# -------------------------------------------------------------------------------------------- main
def solve(task, model, env) -> None:
    student = Path(task.student_dir)
    cfg = yaml.safe_load((student / "config.yaml").read_text())
    prompts = yaml.safe_load((student / "prompts" / "agent.yaml").read_text())
    log = {"config": cfg, "episodes": []}
    started = time.time()

    def save():
        log["usage"] = model.usage()
        log["seconds"] = round(time.time() - started)
        Path(task.trajectory_path).parent.mkdir(parents=True, exist_ok=True)
        Path(task.trajectory_path).write_text(json.dumps(log, indent=1, default=str))

    def episode(kind: str, title: str, file: str, body: str, cap: int) -> None:
        agent = Episode(model, env, token_cap=cap, keep_outputs=cfg["keep_outputs"],
                        max_output_chars=cfg["max_output_chars"], package=task.package,
                        system_template=prompts["system_template"],
                        instance_template=prompts[kind + "_template"], step_limit=cfg["steps_per_episode"],
                        cost_limit=0, output_path=None)
        before = model.total_tokens
        try:
            result = agent.run(project=task.project, package=task.package, summary=task.summary, title=title,
                               file=file, brief=body, new_tests_dir=task.new_tests_dir)
            status = result.get("exit_status", "")
        except Exception as error:  # one broken conversation must not end the project
            status = f"error: {type(error).__name__}: {error}"
        tidy(env, task, file)
        log["episodes"].append({"kind": kind, "title": title, "file": file, "status": status,
                                "calls": agent.n_calls, "tokens": model.total_tokens - before,
                                "messages": agent.messages})
        save()

    items, _ = plan(env, task, cfg)
    log["plan"] = [i.title for i in items]
    save()

    # Phase 1: one fresh conversation per item, with the spec and sbst cases already in the prompt.
    reserve = int(task.token_limit * cfg["mutants_share"]) if cfg["use_mutants"] else 0
    done_owners = []
    for n, item in enumerate(items):
        left = model.tokens_left - reserve
        if left < cfg["min_episode_tokens"]:
            break
        cap = max(cfg["min_episode_tokens"], min(cfg["max_episode_tokens"], left // max(1, len(items) - n)))
        file = f"{task.new_tests_dir}/test_{item.slug}.py"
        episode("test", item.title, file, brief(item, env, cfg), cap)
        if item.is_class and item.owner not in done_owners:
            done_owners.append(item.owner)

    # Phase 2: surviving mutants of the classes we tested.
    if cfg["use_mutants"]:
        for owner in done_owners:
            if model.tokens_left < cfg["min_episode_tokens"]:
                break
            name = owner.rsplit(".", 1)[-1]
            out = sh(env, f"mutants --target '{name}' --show {cfg['mutants_show']} --budget {cfg['mutants_budget']}",
                     timeout=118)
            log["episodes"].append({"kind": "mutants-run", "title": name, "output": out})
            if "differ on:" not in out:
                continue   # nothing concrete to test: do not spend tokens
            cap = min(cfg["max_episode_tokens"], model.tokens_left)
            episode("mutant", name, f"{task.new_tests_dir}/test_{slugify(name)}_mutants.py",
                    cut(out, cfg["mutants_chars"]), cap)

    # Phase 3: keep a failing test only if one small change to the code makes it pass (no model calls).
    if cfg["confirm_failures"]:
        out = sh(env, f"mutants --check-failing --budget {cfg['confirm_budget']}", timeout=cfg["confirm_budget"] + 90)
        log["check_failing"] = out
        disabled = disable_tests(Path(task.workspace), [line.split()[1] for line in out.splitlines()
                                                        if line.startswith("UNCONFIRMED ")])
        log["disabled"] = disabled
    save()


def disable_tests(workspace: Path, nodeids: list[str]) -> list[str]:
    """Rename test functions so pytest no longer collects them (test_x -> unconfirmed_test_x)."""
    done = []
    by_file = {}
    for nodeid in nodeids:
        path, *rest = nodeid.split("::")
        if rest:
            by_file.setdefault(path, set()).add(rest[-1].split("[")[0])
    for path, names in by_file.items():
        file = workspace / path
        if not file.is_file():
            continue
        text = file.read_text()
        for name in names:
            new = re.sub(rf"(\bdef\s+){re.escape(name)}\s*\(", rf"\1unconfirmed_{name}(", text)
            if new != text:
                done.append(f"{path}::{name}")
                text = new
        file.write_text(text)
    return done


def tidy(env, task, file: str) -> None:
    """Drop a test file that the evaluator would reject or that cannot be collected: it only costs."""
    if "yes" not in sh(env, f"test -f {file} && echo yes"):
        return
    check = sh(env, "check_tests", timeout=110)
    rejected = [line for line in check.splitlines() if line.startswith(file) and "reject this whole file" in line]
    collect = sh(env, f"python -m pytest -q --co -p no:cacheprovider {file} 2>&1 | tail -3", timeout=110)
    if rejected or "error" in collect.lower() and "no tests ran" not in collect.lower():
        sh(env, f"mv {file} {file}.broken")   # *.broken is not test_*.py, so it is not collected
