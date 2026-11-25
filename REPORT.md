# Report: Group-C

## 1. Team

| Name | Student number |
|---|---|
| Andrew Nong | 219661537 |
| Ethan Constantin | 220287447 |
| Matthew Magagna | 219804921 |
| Peyton Hislop | 217342957 |

## 2. Your agent

The starter agent gives the model a shell and lets it explore and write tests until the budget runs
out. Our agent instead runs a fixed pipeline in Python and uses the model only for narrow tasks:

1. **Plan** (`planner.plan`, `planner.by_risk`). The package is split into units of a few related
   classes or functions that fit in the context window. Units are ordered by a risk score computed from
   their AST (comparisons, arithmetic, boolean operators, negations, integer constants; private units
   count less), so the riskiest code is tested first while the budget lasts.
2. **Write** (`agent.solve`). For each unit the model writes one pytest file. The prompt contains
   the unit's source, the matching documentation sections, computed test-design hints (`hints.hints`)
   and the tests already kept for that unit. Units are processed in rounds (up to 6), 4 at a time.
3. **Triage in Python** (`agent.triage`, `checks.py`). The file is repaired (unparsable functions
   dropped, imports moved to the module that defines the name, missing imports added) and run twice.
   Only tests that fail with an assertion error both times are bug claims. Tests that use private
   names, the real clock or undocumented API are dropped, because their expected values cannot come
   from the specification.
4. **Blind oracle** (`oracle.predict`). Each claim is turned into a question: the test is cut off at the
   failing assert, comments, docstrings and the test name are removed, and the model is asked what the
   value should be, given the documentation. A claim is kept only if two independent answers agree with
   the test's expected value. Each oracle call also contains two control questions with known answers,
   taken from the project's own passing tests (`oracle.controls`).
5. **Keep and clean up.** At most 3 claims are kept per unit (`MAX_KEPT_PER_UNIT`). If the oracle
   fails its control questions on a project significantly more than 25% of the time (Wilson lower
   bound, `agent._wilson_low`), claims that were never checked by a passing control are removed.
   Finally any file that does not collect is deleted, because one bad file stops the evaluation.

## 3. Testing techniques

| Technique | Lecture topic | Where in the code | What it does |
|---|---|---|---|
| Boundary value analysis | Black-box test design | `hints.py`: `hints`, `_constants`, `_defaults` | Finds integer constants and parameter defaults the unit depends on (e.g. `maxsize=128`, a load factor) and tells the writer to test just below, at and above them. |
| Equivalence partitioning | Black-box / white-box test design | `hints.py`: `_branches`, `ROLES` | Lists the input classes from the code's own branches on a parameter (`x is None`, `callable(x)`, `x < 0`, ...) and from parameter roles (negative indexes, equal keys for `key=`, fake timer before and after expiry). |
| Test oracle / N-version voting | Test oracles | `oracle.py`: `_blind`, `_ask_blind`, `predict`, `agrees` | Decides whether a failing test's expected value is right, without showing the code that disagrees, using independent blind predictions. |
| Oracle calibration on known answers | Test oracles, test adequacy | `oracle.controls`, `agent._wilson_low` | Mixes questions with known answers into every oracle call, and distrusts a project's unchecked claims when the oracle's error rate is too high. |
| Risk-based prioritisation | Test planning | `planner.risk`, `planner.by_risk` | Tests units with the most decision logic first. |
| Specification mining | Specification-based testing | `planner.doc_files`, `checks.documented_names`, `hints._documented_signatures` | Reads README and docs, pairs them with units, and rejects tests of API the docs do not describe. |

## 4. Evidence that a technique finds new bugs

**Final agent against the starter agent** (run 1 and run 2 against the two starter runs):

```
$ python -m testagent.compare --run final-r1 --against starter-r2
cachetools: final-r1 3, starter-r2 1, both 0
  only final-r1: __init__-181-efd2101d
  only final-r1: __init__-581-4381440f
  only final-r1: __init__-686-c06d53e8
  only starter-r2: __init__-416-26ae3e21
sortedcontainers: final-r1 9, starter-r2 0, both 0
  only final-r1: sorteddict-447-a81e8156
  only final-r1: sorteddict-483-5f5d56d1
  only final-r1: sortedlist-1253-a2aae0c6
  only final-r1: sortedlist-1943-281a9af4
  only final-r1: sortedlist-1946-a9b457f5
  only final-r1: sortedlist-379-f462d251
  only final-r1: sortedlist-411-908d90ba
  only final-r1: sortedset-209-e1cd3889
  only final-r1: sortedset-400-0ae45178
Total: 12 bugs only final-r1 found, 1 only starter-r2 found.

$ python -m testagent.compare --run final-r2 --against starter-r1
Total: 11 bugs only final-r2 found, 0 only starter-r1 found.
```

The starter agent spends most of its budget exploring with shell commands and wrote no test file at
all for sortedcontainers. Planning by units means every part of the package gets a test file, and
the hints push the writer to the input classes where these bugs sit (for example `sortedlist-1943/1946`
are in `SortedKeyList.__contains__`, reached by the "different values with equal keys" class the
hints give for every `key=` parameter).

**With and without the test-design hints** (BVA and equivalence partitioning; the runs before
and after `hints.py` was added):

```
$ python -m testagent.compare --run hints-r1 --against oracle-r1
Total: 3 bugs only hints-r1 found, 1 only oracle-r1 found.
$ python -m testagent.compare --run hints-r2 --against oracle-r2
Total: 1 bugs only hints-r2 found, 1 only oracle-r2 found.
```

**With and without the oracle** (`TA_ORACLE_VOTES=0 TA_CONTROLS=0`):

```
$ python -m testagent.compare --run final-r1 --against nooracle-r1
Total: 3 bugs only final-r1 found, 5 only nooracle-r1 found.
```

The oracle does not find bugs. It is a filter: without it the agent finds 14-15 bugs but raises 36-46
false alarms and scores -9.0 and -3.0, against 10.5 and 10.0 with it (section 5). Two votes and two
controls use about 55% more tokens (about 810k against 520k), and that is what moves the score from
negative to positive.

## 5. Results on the development projects

| Run | Bugs found (of 40) | False alarms | Score | Tokens | Cost (USD) |
|---|---|---|---|---|---|
| Starter agent (run 1) | 1 | 1 | 0.5 | 821,366 | 0.0824 |
| Starter agent (run 2) | 1 | 2 | 0.0 | 828,441 | 0.0835 |
| Your final agent, run 1 | 12 | 3 | 10.5 | 814,479 | 0.0984 |
| Your final agent, run 2 | 12 | 4 | 10.0 | 808,121 | 0.0967 |
| Final agent without oracle, run 1 | 14 | 46 | -9.0 | 506,765 | 0.0605 |
| Final agent without oracle, run 2 | 15 | 36 | -3.0 | 533,598 | 0.0643 |

## 6. What you tried

Scores vary by about ±2 between runs of the same code, so we ran each change at least twice.

| Change | Bugs found | False alarms | Kept? |
|---|---|---|---|
| LLM reviewer reading the code and test (runs review-r3 to r6) | 9-11 | 4-6 | No; replaced by the blind oracle |
| Blind oracle, 2 votes (oracle-r1, r2) | 11, 11 | 3, 7 | Yes |
| Test-design hints, BVA + equivalence partitioning (hints-r1, r2) | 13, 11 | 8, 13 | Yes |
| Control questions in every oracle call (controls-r1 to r3) | 9-10 | 4-9 | Yes, after removing controls whose answer was "True", which the model always got right |
| Project calibration with a fixed 25% threshold | (offline replay) | | Replaced by a Wilson lower bound, which removed fewer real claims |
| Oracle on passing tests: rewrite a test when two blind answers agree on a different value (diff-r1, r2) | 8, 9 | 5, 2 | No: 3 claims, all false alarms. The oracle fails ~40% of controls on cachetools, so a disagreement with code that runs is far more often the oracle's mistake than a bug. Off by default (`TA_DIFF=0`). |
| Hint to test the exact expiry instant of timers | | | No; the docs do not define it, and it produced false alarms |
| Black-box rounds (writer sees docs and signatures only) in even rounds | 4, 5 (alone) | 2, 1 | Kept in the final runs, but splitting the final runs shows they found 0 bugs the white-box rounds missed (`compare final-r1-black --against final-r1-white`: 0 only black) |

## 7. Questions

**1. How does your agent decide what an expected value should be, given that the code it can run is
buggy?** It never trusts the code's output. The writer is told to assert what the documentation says,
and every test that fails is checked by the blind oracle: the model sees only the documentation and the
test up to the failing assert, without the code's actual value, and must predict the expected value
twice, independently. A claim survives only if both predictions match the test.

**2. How does your agent decide where to look in a project that does not fit in the model's context?**
`planner.plan` splits the package into units by class and function size, attaches the documentation
sections that mention each unit, and `planner.by_risk` orders them by the amount of decision logic in
their AST. Each model call sees one unit, not the project.

**3. How does your agent keep false alarms down?** Several layers in sequence: tests must fail with an
assertion error twice (no crashes or flaky tests); tests of private, undocumented or real-clock
behaviour are dropped; two blind oracle votes must agree; control questions measure how reliable the
oracle is on each project, and unchecked claims are removed when it is unreliable; at most 3 claims are
kept per unit. Without the oracle layers false alarms rise from 3-4 to 36-46.

**4. Why should your agent work on projects it has never seen? What did you do to avoid tuning it to the
development projects?** Nothing in `student/` names a development project, class or bug. Hints come from
generic patterns (parameter names, branches, constants), controls are taken from whatever passing tests
the project ships, and the calibration threshold is a statistical bound, not a tuned number. Every
precomputation is wrapped so that an unusual layout disables that step instead of crashing (no docs,
no tests, odd package layout). We removed prompt text that only helped on cachetools.

**5. Describe two bugs your agent still misses, and why, using its trajectories.**

- `func-75` (and five like it in `cachetools/func.py`): the bare-decorator form `@lru_cache` without
  arguments takes the `callable(maxsize)` branch, which uses a hard-coded size of 128 (changed to 129).
  The hints list that branch, but in the trajectory `r3_01_..._func_functions` every test calls
  `lru_cache(maxsize=10)` and checks `cache_parameters()["maxsize"] == 10`, so the branch is never run.
  The writer prefers the documented parameterised form, and nothing forces each listed branch to be
  covered.
- `__init__-555` and `__init__-574` (`TLRUCache`, `<` changed to `<=`): the bug only shows at the exact
  instant an item expires. We tell the writer to test well before and well after expiry, because the docs
  do not say whether an item is still valid at that instant, and an earlier hint that tested the exact
  instant produced false alarms. This trades these bugs for fewer false alarms.
