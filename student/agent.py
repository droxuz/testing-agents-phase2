"""Our agent: many short, focused model calls, with Python doing the reading, running and checking.

What we learned from the starter agent and from our first versions (runs units-r1, units-r2):
- one long conversation spends the 400k budget re-sending its history while it reads code;
- the model re-reads files it was already shown, and "fixes" failing tests until they pass, which
  also erases any bug those tests found;
- most false alarms were broken tests (wrong API, missing imports) or wrong expectations;
- a test that passes on the buggy code can (almost) never find a bug, since it already passed
  with every bug in place, but it can still be a false alarm if it encodes a bug's behaviour.

So the agent works like this, for each unit of code and in several rounds with different focuses:
1. Plan (Python, planner.py): split the package into units of at most ~220 source lines and find
   the documentation paragraphs about each unit.
2. Write (model): one short conversation that is shown the unit's code and docs and writes one
   test file in a single command.
3. Triage (Python, checks.py): add missing imports, run the file twice, and remove tests that
   crash with an error, change outcome between runs, or pass. Only tests that fail on an assertion
   remain: each one claims to have found a bug.
4. Review (model, reviewer.py): independent reviewers who see the docs and signatures but not the
   function bodies decide for each claim whether the code or the test is wrong. Only unanimous
   KEEP verdicts stay.

Every model call goes through `model`, the harness's budgeted model.
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the harness loads this file by path

import yaml
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError, LimitsExceeded

from checks import (add_missing_imports, private_name_tests, rejection_reason, remove_tests, run_tests,
                    test_sources, uncollectable_files)
from planner import doc_snippets, plan, spec_view
from reviewer import review

MAX_QUERIES_PER_UNIT = 3      # model calls allowed to get the file written
MIN_TOKENS_FOR_UNIT = 15_000  # do not start a unit (write + review) with less budget than this
MAX_ROUNDS = 6
TEST_HEADROOM = 20            # stop a little before the evaluator's test limit
WORKERS = 4                   # units worked on at the same time (model calls are slow)
MAX_KEPT_PER_UNIT = 2         # bug-claiming tests kept per unit over all rounds (about 1 bug per 200 lines)
MAX_ERRORS = 6                # stop the project after this many failed units in a row
FOCUSES = [
    "the documented behaviour: what the docstrings and docs above promise, and any `>>>` examples in them",
    "boundaries: empty and single-element inputs, values exactly at a limit and just past it, "
    "negative and out-of-range indexes, and the first and last element",
    "consistency between methods: after a sequence of operations, check that related methods agree "
    "(length, iteration order, membership, lookups, and what a removal returns)",
]


def write_unit(model, env, prompts: dict, variables: dict, test_file: Path) -> list[dict]:
    """One short conversation that writes test_file. Returns the messages for the trajectory."""
    messages = [
        {"role": "system", "content": prompts["system_template"]},
        {"role": "user", "content": Template(prompts["unit_template"], undefined=StrictUndefined).render(**variables)},
    ]
    for _ in range(MAX_QUERIES_PER_UNIT):
        try:
            message = model.query(messages)
        except FormatError as error:
            messages.extend(error.messages)
            continue
        messages.append(message)
        outputs = [env.execute(action) for action in message.get("extra", {}).get("actions", [])]
        messages.extend(model.format_observation_messages(message, outputs, {}))
        if test_file.exists():
            break
        messages.append({"role": "user", "content": f"The file {variables['test_file']} does not exist yet. "
                         "Write it now with a single `cat > ... <<'EOF'` command."})
    return messages


def triage(workspace: Path, test_file: Path, rel: str, package: str) -> tuple[dict, dict]:
    """Remove tests that error, are flaky, pass or use private names.

    Returns (info, {key: failure} of the tests left, which all fail on an assertion)."""
    reason = rejection_reason(test_file)
    if reason:
        test_file.unlink()
        return {"dropped_file": reason}, {}
    info = {"added_imports": add_missing_imports(test_file)}
    first = run_tests(workspace, rel)
    if not first or any(k.startswith("<") for k in first):
        test_file.unlink()
        return info | {"dropped_file": next(iter(first.values()), "no tests collected")}, {}
    second = run_tests(workspace, rel)
    info["written"] = len(first)
    info["passed"] = sum(v == "passed" for v in first.values())
    info["errors"] = sum(v.startswith("error") for v in first.values())
    info["flaky"] = sum(second.get(k) != v for k, v in first.items())
    private = private_name_tests(test_file, package)
    info["private"] = len(private)
    failing = {k: v for k, v in first.items()
               if v.startswith("assertion") and second.get(k) == v and k not in private}
    return info, failing


def keep_only(workspace: Path, test_file: Path, rel: str, keep: set[str]) -> int:
    """Leave only the tests in keep; check they still fail the same way. Returns how many are left."""
    if not keep:
        test_file.unlink()
        return 0
    remove_tests(test_file, set(test_sources(test_file)) - keep)
    final = run_tests(workspace, rel)
    still = {k for k, v in final.items() if v.startswith("assertion")}
    if still != set(final) or not still:
        remove_tests(test_file, set(final) - still)
        final = run_tests(workspace, rel)
        still = {k for k, v in final.items() if v.startswith("assertion")}
        if still != set(final) or not still:
            test_file.unlink()
            return 0
    return len(still)


class UsageCounter:
    """Counts the tokens of one unit's calls, so per-unit numbers stay right when units run in parallel."""

    def __init__(self, model):
        self._model = model
        self.tokens = 0

    def query(self, messages, **kwargs):
        message = self._model.query(messages, **kwargs)
        usage = (message.get("extra", {}).get("response") or {}).get("usage") or {}
        if not isinstance(usage, dict):
            usage = {k: getattr(usage, k, 0) for k in ("prompt_tokens", "completion_tokens")}
        self.tokens += int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
        return message

    def __getattr__(self, name):
        return getattr(self._model, name)


def solve(task, model, env) -> None:
    student = Path(task.student_dir)
    prompts = yaml.safe_load((student / "prompts" / "agent.yaml").read_text())
    workspace = Path(task.workspace)
    traj_dir = Path(task.trajectory_path).parent / "units"
    traj_dir.mkdir(parents=True, exist_ok=True)

    units = plan(workspace, task.package)
    docs = {u.slug: doc_snippets(workspace, u) or "(no documentation found for this unit)" for u in units}
    specs = {u.slug: spec_view(workspace, u) for u in units}
    log = {"project": task.project, "units": len(units), "steps": []}
    state = {"tests": 0, "stop": False, "errors": 0}
    kept_per_unit = {u.slug: 0 for u in units}
    lock = threading.Lock()

    def do_unit(round_no: int, i: int, unit) -> None:
        with lock:
            if (state["stop"] or model.tokens_left < MIN_TOKENS_FOR_UNIT
                    or state["tests"] >= task.max_tests - TEST_HEADROOM):
                state["stop"] = True
                return
            if kept_per_unit[unit.slug] >= MAX_KEPT_PER_UNIT:
                return                 # this unit already has its share; spend the budget elsewhere
        focus = FOCUSES[(round_no - 1) % len(FOCUSES)]
        rel = f"{task.new_tests_dir}/test_r{round_no}_{i:02d}_{unit.slug}.py"
        path = workspace / rel
        counter = UsageCounter(model)
        base = dict(project=task.project, package=task.package, summary=task.summary, unit=unit.name,
                    module=unit.module, file=unit.file, first_line=unit.first_line, last_line=unit.last_line,
                    source=unit.source, docs=docs[unit.slug], spec=specs[unit.slug])
        entry = {"round": round_no, "unit": unit.name, "file": rel}
        trajectory = {}
        try:
            trajectory["write"] = write_unit(counter, env, prompts, base | {"focus": focus, "test_file": rel}, path)
            if path.exists():
                info, failing = triage(workspace, path, rel, task.package)
                entry |= info
                entry["claims"] = sorted(failing)
                kept = set()
                if failing:
                    kept, trajectory["review"] = review(counter, prompts, base, failing, test_sources(path))
                with lock:             # reserve this unit's remaining share
                    room = max(0, MAX_KEPT_PER_UNIT - kept_per_unit[unit.slug])
                    kept = set(sorted(kept)[:room])
                    kept_per_unit[unit.slug] += len(kept)
                if path.exists():
                    entry["kept"] = keep_only(workspace, path, rel, kept)
                    entry["kept_tests"] = sorted(kept)
            else:
                entry["dropped_file"] = "never written"
        except Exception as error:     # budget used up, or a bad response from the API
            entry["dropped_file"] = f"{type(error).__name__}: {str(error)[:120]}"
            if path.exists():
                path.unlink()          # never leave a file that was not triaged and reviewed
            with lock:
                state["errors"] += 1
                if isinstance(error, LimitsExceeded) or state["errors"] >= MAX_ERRORS:
                    state["stop"] = True
        entry["tokens"] = counter.tokens
        with lock:
            if "dropped_file" not in entry or "Error" not in entry["dropped_file"]:
                state["errors"] = 0    # count failed units in a row only
            state["tests"] += entry.get("kept", 0)
            log["steps"].append(entry)
            total = state["tests"]
        (traj_dir / f"r{round_no}_{i:02d}_{unit.slug}.json").write_text(json.dumps(trajectory, indent=1, default=str))
        print(f"[{task.project}] round {round_no} unit {i + 1}/{len(units)} {unit.name}: {entry['tokens']} tokens, "
              + (f"dropped ({entry['dropped_file']})" if "dropped_file" in entry else
                 f"{entry.get('written', 0)} written, {len(entry.get('claims', []))} claim a bug, "
                 f"{entry.get('kept', 0)} kept after review") + f"; total kept {total}", flush=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for round_no in range(1, MAX_ROUNDS + 1):
            list(pool.map(lambda item: do_unit(round_no, *item), enumerate(units)))
            if state["stop"]:
                break

    # One file that pytest cannot collect stops the whole measured run, so check the folder as a whole.
    for bad in uncollectable_files(workspace, task.new_tests_dir):
        (workspace / bad).unlink(missing_ok=True)
        log.setdefault("removed_uncollectable", []).append(bad)
    log["tokens_used"] = model.total_tokens
    log["total_tests"] = state["tests"]
    Path(task.trajectory_path).write_text(json.dumps(log, indent=2))
