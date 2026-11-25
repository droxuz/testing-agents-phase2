"""Our agent: many short, focused model calls, with Python doing the reading, running and checking.

What we learned from the starter agent and from our first versions (runs units-r1, units-r2):
- one long conversation spends the 400k budget re-sending its history while it reads code;
- the model re-reads files it was already shown, and "fixes" failing tests until they pass, which
  also erases any bug those tests found;
- most false alarms were broken tests (wrong API, missing imports) or wrong expectations;
- a test that passes on the buggy code can (almost) never find a bug, since it already passed
  with every bug in place, but it can still be a false alarm if it encodes a bug's behaviour.

So the agent works like this, for each unit of code and in several rounds with different focuses:
1. Plan (Python, planner.py, hints.py): split the package into units of at most ~220 source lines,
   order them by risk (how many comparisons, constants and operators a small bug could be in),
   find the documentation paragraphs about each unit, and compute test design hints from the code:
   boundary values from its constants and defaults, and input classes from its branches and
   parameters (boundary value analysis and equivalence partitioning).
2. Write (model): one short conversation that is shown the unit's code (odd rounds, white-box) or
   only its signatures and docstrings (even rounds, black-box: the writer cannot copy a bug it
   cannot see), the docs and the hints, and writes one test file in a single command.
3. Triage (Python, checks.py): add missing imports, run the file twice, and remove tests that
   crash with an error, change outcome between runs, or pass. Only tests that fail on an assertion
   remain: each one claims to have found a bug.
4. Blind oracle (model, oracle.py): each failing test is cut off before its failing assert, and
   model calls that see only the docs, signatures and the project's own tests predict the value.
   A claim is kept only if ORACLE_VOTES independent predictions all agree with the test, and only
   if, in the same calls, the oracle also answers known-answer control questions from the
   project's own tests correctly (oracle.controls: it has to show it understands this area).
   At the end, if the controls show the oracle is often wrong on this project (80% lower bound of
   its failure rate over 25%), the kept claims that never had a control are removed as well.
6. Oracle differences (oracle.differences): the blind oracle is also asked about the last value
   checked by tests that PASS. If two answers agree on a different value than the code returns,
   the test is rewritten to assert the documented value and goes through triage as a new claim
   (the writer sometimes copies a bug from the code into its test).
   (An earlier version then asked a KEEP/DROP reviewer, reviewer.py; on run oracle-r1 the reviewer
   dropped 39 real claims for 3 false alarms it caught, so it is off by default: TA_REVIEW=1.)

Every model call goes through `model`, the harness's budgeted model.
"""

import ast
import json
import math
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # the harness loads this file by path

import yaml
from jinja2 import StrictUndefined, Template
from minisweagent.exceptions import FormatError, LimitsExceeded

from checks import (add_missing_imports, clock_params, real_clock_tests, undocumented_claims, drop_unparsable_tests, fix_chained_comparisons, fix_imports, private_name_tests, rejection_reason, remove_tests, run_tests,
                    test_sources, uncollectable_files)
from hints import hints
from planner import by_risk, doc_snippets, existing_tests, plan, spec_view
from oracle import _function, controls, differences, oracle_tests, predict
from reviewer import review

MAX_QUERIES_PER_UNIT = 3      # model calls allowed to get the file written
MIN_TOKENS_FOR_UNIT = 15_000  # do not start a unit (write + review) with less budget than this
MAX_ROUNDS = 6
TEST_HEADROOM = 20            # stop a little before the evaluator's test limit
WORKERS = 4                   # units worked on at the same time (model calls are slow)
MAX_KEPT_PER_UNIT = 3         # bug-claiming tests kept per unit over all rounds
# Which claim filters run. The defaults are our final agent; the environment variables exist only so the
# ablation runs for the report can switch a technique off (docker run -e TA_ORACLE_VOTES=0 ...).
ORACLE_VOTES = int(os.environ.get("TA_ORACLE_VOTES", "2"))   # blind predictions that must all agree
USE_REVIEW = os.environ.get("TA_REVIEW", "0") == "1"         # the KEEP/DROP reviewer (reviewer.py)
CONTROLS = int(os.environ.get("TA_CONTROLS", "2"))           # known-answer questions per oracle call
ORACLE_DIFF = os.environ.get("TA_DIFF", "0") == "1"         # oracle on passing tests: off, measured net harmful (diff-r1/r2)
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
    salvaged = drop_unparsable_tests(test_file)
    reason = rejection_reason(test_file)
    if reason:
        test_file.unlink()
        return {"dropped_file": reason}, {}
    info = {"added_imports": add_missing_imports(test_file), "unparsable_removed": salvaged,
            "imports_moved": fix_imports(workspace, test_file, package),
            "chains_fixed": fix_chained_comparisons(test_file)}
    lines = {}
    first = run_tests(workspace, rel, lines=lines)
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
    real_clock = real_clock_tests(test_file, clock_params(workspace, package))
    info["real_clock"] = len(real_clock)
    undocumented = undocumented_claims(workspace, test_file, package)   # no spec, so no oracle for it
    info["undocumented"] = len(undocumented)
    private |= real_clock | undocumented
    failing = {k: v for k, v in first.items()
               if v.startswith("assertion") and second.get(k) == v and k not in private}
    info["fail_lines"] = {k: lines[k] for k in failing if k in lines}
    info["passing"] = sorted(k for k, v in first.items() if v == "passed" and second.get(k) == v and k not in private)
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


DISTRUST_RATE = 0.25   # distrust a project's unchecked claims if the oracle fails more than this...
CONFIDENCE_Z = 1.2816  # ...with 80% confidence (one-sided Wilson lower bound)


def _wilson_low(failures: int, n: int, z: float = CONFIDENCE_Z) -> float:
    """Lower confidence bound of a failure rate: small samples get the benefit of the doubt."""
    if n == 0:
        return 0.0
    p = failures / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / (1 + z * z / n)


def _safe(function, *args):
    """function(*args), or "" if it raises."""
    try:
        return function(*args)
    except Exception:
        return ""


def solve(task, model, env) -> None:
    student = Path(task.student_dir)
    prompts = yaml.safe_load((student / "prompts" / "agent.yaml").read_text())
    workspace = Path(task.workspace)
    traj_dir = Path(task.trajectory_path).parent / "units"
    traj_dir.mkdir(parents=True, exist_ok=True)

    # Unknown projects can have files our helpers do not expect: a failure here must cost one unit's
    # extras, never the whole project.
    try:
        units = by_risk(workspace, plan(workspace, task.package))   # most bug-prone first: budget may run out
    except Exception as error:
        print(f"[{task.project}] planning failed ({type(error).__name__}: {error}); no units", flush=True)
        units = []
    docs = {u.slug: _safe(doc_snippets, workspace, u) or "(no documentation found for this unit)" for u in units}
    specs = {u.slug: _safe(spec_view, workspace, u) or u.source for u in units}
    design = {u.slug: _safe(hints, workspace, u) or "(no computed hints for this unit)" for u in units}
    log = {"project": task.project, "units": len(units), "steps": []}
    state = {"tests": 0, "stop": False, "errors": 0}
    kept_per_unit = {u.slug: 0 for u in units}
    kept_sources = {u.slug: [] for u in units}   # tests already kept, shown to later rounds of that unit
    lock = threading.Lock()

    def differential(round_no, unit, rel, original, passing, base, counter, trajectory) -> dict | None:
        """Ask the blind oracle about this file's passing tests; keep rewritten tests that now fail.

        Has its own error handling: a failure here must not cost the unit's other claims."""
        new_rel = rel[:-3] + "_oracle.py"
        new_path = workspace / new_rel
        with lock:
            if kept_per_unit[unit.slug] >= MAX_KEPT_PER_UNIT or model.tokens_left < MIN_TOKENS_FOR_UNIT:
                return None
        step = {"round": round_no, "unit": unit.name, "kind": "oracle difference"}
        try:
            tree = ast.parse(original)
            funcs = [_function(tree, k) for k in passing]
            sources = [ast.get_source_segment(original, f) for f in funcs if f is not None]
            evidence = existing_tests(workspace, sources)
            context = base | {"existing": evidence or "(no existing tests use the same names)"}
            known = controls(workspace, sources, task.package, CONTROLS) if CONTROLS else []
            chosen, logs = differences(counter, prompts, context, original, passing, known, max(ORACLE_VOTES, 1))
            trajectory["differences"] = logs
            step |= {"asked": len(logs[0]["questions"]) if logs else 0,
                     "calibrated": bool(logs and logs[0].get("controls")),
                     "control_calls": sum(1 for g in logs if g.get("controls")),
                     "control_fails": sum(1 for g in logs if g.get("controls") and g.get("failed_controls"))}
            if not chosen:
                return step
            new_path.write_text(oracle_tests(original, chosen))
            info, failing = triage(workspace, new_path, new_rel, task.package)
            step |= {"file": new_rel, "proposed": {k: repr(q["proposed"]) for k, q in chosen.items()},
                     "claims": sorted(failing), "dropped_file": info.get("dropped_file")}
            keep = set(failing) & set(chosen)
            with lock:
                room = max(0, MAX_KEPT_PER_UNIT - kept_per_unit[unit.slug])
                keep = set(sorted(keep)[:room])
                kept_per_unit[unit.slug] += len(keep)
            if new_path.exists():
                step["kept"] = keep_only(workspace, new_path, new_rel, keep)
                step["kept_tests"] = sorted(keep)
            return step
        except Exception as error:
            new_path.unlink(missing_ok=True)
            return step | {"error": f"{type(error).__name__}: {str(error)[:120]}", "kept": 0}

    def do_unit(round_no: int, i: int, unit) -> None:
        with lock:
            if (state["stop"] or model.tokens_left < MIN_TOKENS_FOR_UNIT
                    or state["tests"] >= task.max_tests - TEST_HEADROOM):
                state["stop"] = True
                return
            if kept_per_unit[unit.slug] >= MAX_KEPT_PER_UNIT:
                return                 # this unit already has its share; spend the budget elsewhere
        focus = FOCUSES[(round_no - 1) % len(FOCUSES)]
        black_box = round_no % 2 == 0   # even rounds: the writer sees the promises, not the (buggy) bodies
        rel = f"{task.new_tests_dir}/test_r{round_no}_{i:02d}_{unit.slug}.py"
        path = workspace / rel
        counter = UsageCounter(model)
        base = dict(project=task.project, package=task.package, summary=task.summary, unit=unit.name,
                    module=unit.module, file=unit.file, first_line=unit.first_line, last_line=unit.last_line,
                    source=specs[unit.slug] if black_box else unit.source, docs=docs[unit.slug], spec=specs[unit.slug],
                    view=("signatures and docstrings only: the function bodies are hidden on purpose, so that you test "
                          "what is promised, not what the (possibly buggy) code happens to do") if black_box else
                         "the code (it may be buggy)",
                    hints=design[unit.slug])
        with lock:
            already = "\n\n".join(kept_sources[unit.slug]) or "(none yet)"
        entry = {"round": round_no, "unit": unit.name, "file": rel, "black_box": black_box}
        trajectory, extra = {}, None
        try:
            trajectory["write"] = write_unit(counter, env, prompts, base | {"focus": focus, "test_file": rel,
                                                                       "already": already}, path)
            if path.exists():
                info, failing = triage(workspace, path, rel, task.package)
                original = path.read_text() if path.exists() else ""     # before keep_only trims it
                passing = info.pop("passing", [])
                entry |= info
                entry["claims"] = sorted(failing)
                kept = set()
                if failing:
                    sources = test_sources(path)
                    evidence = existing_tests(workspace, [sources[k] for k in failing if k in sources])
                    context = base | {"existing": evidence or "(no existing tests use the same names)"}
                    fail_lines = entry.pop("fail_lines")
                    known = controls(workspace, [sources[k] for k in failing if k in sources], task.package,
                                     CONTROLS) if CONTROLS else []
                    agreed, unasked, trajectory["oracle"] = set(failing), set(), []
                    for _ in range(ORACLE_VOTES):     # independent blind predictions; all must agree
                        asked = {k: v for k, v in failing.items() if k in agreed}
                        if not asked:
                            break
                        agreed, still_unasked, oracle_log = predict(counter, prompts, context,
                                                                    path.read_text(), asked, fail_lines, known)
                        entry.setdefault("failed_controls", []).append(len(oracle_log.get("failed_controls", [])))
                        if oracle_log.get("controls"):     # one calibration sample for the whole project
                            entry["control_calls"] = entry.get("control_calls", 0) + 1
                            entry["control_fails"] = entry.get("control_fails", 0) + bool(oracle_log["failed_controls"])
                        unasked |= still_unasked
                        trajectory["oracle"].append(oracle_log)
                    entry["calibrated"] = bool(trajectory["oracle"] and trajectory["oracle"][0].get("controls"))
                    entry["oracle_agreed"], entry["oracle_unasked"] = sorted(agreed), sorted(unasked)
                    if USE_REVIEW:            # claims the oracle could not ask about go to the reviewer too
                        survivors = {k: v for k, v in failing.items() if k in agreed | unasked}
                        if survivors:
                            kept, trajectory["review"] = review(counter, prompts, context, survivors, sources)
                    else:
                        kept = agreed
                with lock:             # reserve this unit's remaining share
                    room = max(0, MAX_KEPT_PER_UNIT - kept_per_unit[unit.slug])
                    kept = set(sorted(kept)[:room])
                    kept_per_unit[unit.slug] += len(kept)
                if path.exists():
                    entry["kept"] = keep_only(workspace, path, rel, kept)
                    entry["kept_tests"] = sorted(kept)
                    if entry["kept"]:
                        sources = test_sources(path)
                        with lock:
                            kept_sources[unit.slug] += [sources[k] for k in sorted(kept) if k in sources]
                # Only after white-box rounds: a writer that never saw the code cannot have copied a bug.
                if ORACLE_DIFF and original and passing and not black_box:
                    extra = differential(round_no, unit, rel, original, passing, base, counter, trajectory)
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
            state["tests"] += entry.get("kept", 0) + (extra or {}).get("kept", 0)
            log["steps"].append(entry)
            if extra:
                log["steps"].append(extra)
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

    # Project-level calibration: the control questions measured how often the oracle is wrong on
    # this project. If we are fairly sure it is often wrong, the claims that were never checked
    # against a control are not trusted either.
    try:
        calls = sum(e.get("control_calls", 0) for e in log["steps"])
        fails = sum(e.get("control_fails", 0) for e in log["steps"])
        log["calibration"] = {"control_calls": calls, "failed": fails, "lower_bound": round(_wilson_low(fails, calls), 3)}
        if calls >= 4 and _wilson_low(fails, calls) > DISTRUST_RATE:
            removed = []
            for e in log["steps"]:
                path = workspace / (e.get("file") or "-")
                if e.get("file") and e.get("kept") and not e.get("calibrated") and path.exists():
                    remove_tests(path, set(test_sources(path)))
                    path.unlink()
                    removed += [f"{e['file']}::{k}" for k in e.get("kept_tests", [])]
            log["calibration"]["removed_uncalibrated"] = removed
            state["tests"] -= len(removed)
            print(f"[{task.project}] oracle failed {fails}/{calls} control calls: removed {len(removed)} "
                  "claims that had no control", flush=True)
    except Exception as error:
        log["calibration_error"] = f"{type(error).__name__}: {error}"

    # One file that pytest cannot collect stops the whole measured run, so check the folder as a whole.
    try:
        for bad in uncollectable_files(workspace, task.new_tests_dir):
            (workspace / bad).unlink(missing_ok=True)
            log.setdefault("removed_uncollectable", []).append(bad)
    except Exception as error:
        log["cleanup_error"] = f"{type(error).__name__}: {error}"
    log["tokens_used"] = model.total_tokens
    log["total_tests"] = state["tests"]
    Path(task.trajectory_path).write_text(json.dumps(log, indent=2))
