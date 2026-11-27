"""Shared engine for the sbst and mutants tools. Not a tool itself (the harness skips names starting with _).

It finds the package in the workspace, resolves a target (function, class or Class.method), builds
random call sequences for it, runs them and records what happened, and looks up what extract_spec
(step 1 of the prompt) says about the target, so both tools can print the spec next to their results.

A call sequence ("individual") is kept as Python source: a constructor call and a list of steps. That
makes every input the search finds printable exactly as it was run.
"""

import ast
import contextlib
import inspect
import io
import json
import random
import re
import signal
import subprocess
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
SKIP_DIRS = {"tests", "tests_new", "test", "docs", "doc", "build", "dist", "examples", "benchmarks"}
MAX_REPR = 160          # longest value printed
MAX_KEPT = 3000         # longest value kept for comparing two runs
ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")

# Method names whose calls are written as Python syntax when printed.
SYNTAX = {
    "__getitem__": "{o}[{0}]", "__setitem__": "{o}[{0}] = {1}", "__delitem__": "del {o}[{0}]",
    "__contains__": "{0} in {o}", "__len__": "len({o})", "__iter__": "list({o})",
    "__reversed__": "list(reversed({o}))", "__repr__": "repr({o})", "__str__": "str({o})",
    "__bool__": "bool({o})", "__eq__": "{o} == {0}", "__ne__": "{o} != {0}", "__lt__": "{o} < {0}",
    "__le__": "{o} <= {0}", "__gt__": "{o} > {0}", "__ge__": "{o} >= {0}", "__add__": "{o} + {0}",
    "__radd__": "{0} + {o}", "__sub__": "{o} - {0}", "__mul__": "{o} * {0}", "__rmul__": "{0} * {o}",
    "__and__": "{o} & {0}", "__or__": "{o} | {0}", "__xor__": "{o} ^ {0}", "__iadd__": "{o} += {0}",
    "__isub__": "{o} -= {0}", "__imul__": "{o} *= {0}", "__iand__": "{o} &= {0}", "__ior__": "{o} |= {0}",
    "__ixor__": "{o} ^= {0}", "__copy__": "{o}.__copy__()", "__call__": "{o}({args})",
}
STATEMENTS = {"__setitem__", "__delitem__", "__iadd__", "__isub__", "__imul__", "__iand__", "__ior__",
              "__ixor__"}
SKIP_METHODS = {"__init__", "__new__", "__init_subclass__", "__subclasshook__", "__class_getitem__",
                "__getattr__", "__getattribute__", "__setattr__", "__delattr__", "__dir__", "__hash__",
                "__format__", "__sizeof__", "__reduce__", "__reduce_ex__", "__getstate__", "__setstate__",
                "__missing__", "__del__", "__enter__", "__exit__", "__deepcopy__", "__getnewargs__",
                "__getnewargs_ex__", "__instancecheck__", "__subclasscheck__"}
MUTATORS = re.compile(r"^(__setitem__|__delitem__|__i\w+__|add|append|extend|update|insert|push|put|set\w*|"
                      r"pop\w*|remove|discard|clear|expire|__getitem__|get)$")

CALLABLE_NAMES = {"func", "function", "callback", "getsizeof", "user_function", "fn", "predicate", "ttu",
                  "cmp", "default_factory", "choice", "timer", "hashkey", "factory", "transform"}
SEQUENCE_NAMES = {"iterable", "values", "items", "seq", "sequence", "other", "iterables", "data",
                  "elements", "mapping", "args", "keys", "lst", "list", "collection", "others"}
REPEAT_NAMES = {"num", "times", "n", "repeat", "count"}   # list * num: keep these small
REPEAT_METHODS = {"__mul__", "__imul__", "__rmul__"}
INT_NAMES = {"maxsize", "size", "n", "num", "index", "idx", "count", "start", "stop", "step", "pos",
             "ttl", "load", "k", "i", "j", "level", "depth", "length", "limit", "times", "offset",
             "capacity", "number", "width", "precision", "base"}
CALLABLE_POOL = ["abs", "len", "str", "lambda x: -x", "lambda *a: 1"]


class Timeout(BaseException):
    """One run took too long. A BaseException so the code under test cannot swallow it."""


def _on_alarm(signum, frame):
    raise Timeout()


def limit_resources(memory_mb: int = 1500, wall_seconds: float | None = None) -> None:
    """Cap memory growth (a call such as obj *= 1000, repeated, would fill the machine) and total time."""
    try:
        import resource
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[0])
        current = pages * resource.getpagesize()
        limit = current + memory_mb * 1024 * 1024
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
    except (ImportError, OSError, ValueError):
        pass
    if wall_seconds:
        import threading, os as _os

        def stop():
            print(f"\n(stopped: the tool ran longer than {wall_seconds:.0f} seconds)", flush=True)
            _os._exit(3)
        timer = threading.Timer(wall_seconds, stop)
        timer.daemon = True
        timer.start()


# ----------------------------------------------------------------------------------------- package
def find_packages(root: Path, given: list[str] | None = None) -> list[str]:
    if given:
        return [Path(g).stem if g.endswith(".py") else Path(g).name for g in given]
    found = [p.name for p in sorted(root.iterdir())
             if p.is_dir() and (p / "__init__.py").exists() and p.name not in SKIP_DIRS
             and not p.name.startswith((".", "_"))]
    if not found:
        found = [p.stem for p in sorted(root.glob("*.py"))
                 if not p.name.startswith(("test", "setup", "conftest", "_"))]
    return found


def package_files(root: Path, packages: list[str]) -> dict[str, Path]:
    """Module name -> source file for every module of the packages."""
    files = {}
    for pkg in packages:
        base = root / pkg
        if base.is_dir():
            for path in sorted(base.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                parts = list(path.relative_to(root).with_suffix("").parts)
                if parts[-1] == "__init__":
                    parts.pop()
                files[".".join(parts)] = path.resolve()
        elif (root / f"{pkg}.py").exists():
            files[pkg] = (root / f"{pkg}.py").resolve()
    return files


def import_modules(names) -> dict[str, types.ModuleType]:
    modules = {}
    for name in sorted(names, key=lambda n: (n.count("."), n)):
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                modules[name] = __import__(name, fromlist=["_"])
        except BaseException:  # a broken optional module must not stop the tool
            pass
    return modules


def in_package(obj, packages) -> bool:
    module = getattr(obj, "__module__", None) or ""
    return any(module == p or module.startswith(p + ".") for p in packages)


# ------------------------------------------------------------------------------------------ target
@dataclass
class Target:
    kind: str            # function, class or method
    name: str            # qualified name inside the module, e.g. LRUCache.popitem
    fullname: str        # public module + name, e.g. cachetools.LRUCache.popitem
    import_line: str     # from cachetools import LRUCache
    bound: str           # the name the import binds: LRUCache
    obj: object          # the function or class
    cls: type | None = None
    method: str | None = None

    def scope_functions(self) -> list:
        """The functions whose code the target covers."""
        if self.kind == "function":
            return [unwrap(self.obj)]
        if self.kind == "method":
            return [f for f in [unwrap(raw_function(self.cls.__dict__.get(self.method)))] if f]
        funcs = []
        for value in vars(self.cls).values():
            f = unwrap(raw_function(value))
            if f is not None:
                funcs.append(f)
        return funcs


def raw_function(value):
    if isinstance(value, (staticmethod, classmethod)):
        return value.__func__
    if isinstance(value, property):
        return value.fget
    if isinstance(value, types.FunctionType):
        return value
    return None


def unwrap(func):
    while func is not None and hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func if isinstance(func, types.FunctionType) else None


def public_home(obj, modules) -> str:
    """The shortest module path that exposes obj, e.g. cachetools rather than cachetools.__init__."""
    homes = [name for name, mod in modules.items()
             if getattr(mod, getattr(obj, "__name__", ""), None) is obj]
    return min(homes, key=lambda n: (n.count("."), len(n))) if homes else obj.__module__


def candidates(modules, packages):
    """(fullname, Target) for every class, method and function defined in the packages."""
    seen = set()
    for mod in modules.values():
        for attr, obj in sorted(vars(mod).items()):
            if attr.startswith("_") or id(obj) in seen or not in_package(obj, packages):
                continue
            if isinstance(obj, type) or isinstance(obj, types.FunctionType):
                seen.add(id(obj))
                home = public_home(obj, modules)
                imp, bound = f"from {home} import {obj.__name__}", obj.__name__
                if isinstance(obj, type):
                    yield Target("class", obj.__name__, f"{home}.{obj.__name__}", imp, bound, obj, cls=obj)
                    for mname, value in vars(obj).items():
                        if raw_function(value) is not None and (not mname.startswith("_") or
                                                                 mname.startswith("__") and mname.endswith("__")):
                            yield Target("method", f"{obj.__name__}.{mname}", f"{home}.{obj.__name__}.{mname}",
                                         imp, bound, value, cls=obj, method=mname)
                else:
                    yield Target("function", obj.__name__, f"{home}.{obj.__name__}", imp, bound, obj)


def resolve(text: str, modules, packages) -> list[Target]:
    found = list(candidates(modules, packages))
    exact = [t for t in found if t.fullname == text or t.name == text]
    return exact or [t for t in found if t.fullname.endswith("." + text)]


# --------------------------------------------------------------------------------------- generation
@dataclass
class Param:
    name: str
    kind: str              # pos, kw, var
    required: bool
    default: object = None


@dataclass
class Callee:
    name: str              # method name ('' for the target function itself)
    params: list
    weight: float = 1.0
    is_property: bool = False


def params_of(func, drop_first: bool) -> list[Param] | None:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    params = list(sig.parameters.values())
    if drop_first and params:
        params = params[1:]
    result = []
    for p in params:
        if p.kind is p.VAR_KEYWORD:
            continue
        if p.kind is p.VAR_POSITIONAL:
            result.append(Param(p.name, "var", False))
            continue
        kind = "pos" if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) else "kw"
        if p.default is p.empty:
            result.append(Param(p.name, kind, True))
        elif callable(p.default):
            continue   # internal hooks such as cache_setitem=Cache.__setitem__ or timer=time.monotonic
        else:
            result.append(Param(p.name, "kw" if kind == "pos" and p.kind is not p.POSITIONAL_ONLY else kind,
                                False, p.default))
    return result


def callees_of(cls, packages, focus: str | None = None) -> list[Callee]:
    """Methods and properties a call sequence on an instance of cls may use."""
    found = {}
    for klass in reversed(cls.__mro__):
        if klass is object or klass.__module__ == "builtins":
            continue
        for name, value in vars(klass).items():
            if name in SKIP_METHODS or (name.startswith("_") and not (name.startswith("__") and name.endswith("__"))):
                continue
            if name.startswith("__") and name not in SYNTAX:
                continue
            if isinstance(value, property):
                found[name] = Callee(name, [], 0.5, is_property=True)
                continue
            func = raw_function(value) if not isinstance(value, (staticmethod, classmethod)) else None
            if func is None and callable(value) and not isinstance(value, type):
                func = value
            if func is None:
                continue
            params = params_of(func, drop_first=True)
            if params is None:
                continue
            weight = 3.0 if MUTATORS.match(name) else 1.0
            if not in_package(klass, packages):
                weight *= 0.5
            found[name] = Callee(name, params, weight)
    if focus and focus in found:
        found[focus].weight = max(found[focus].weight, 4.0)
    return list(found.values())


class Pool:
    """Values for arguments, as source text. Seeded with the constants of the code under test."""

    def __init__(self, rng: random.Random, constants=()):
        self.rng = rng
        ints = {-2, -1, 0, 1, 2, 3, 4, 5, 10}
        self.big = set()
        for c in constants:
            if isinstance(c, bool) or not isinstance(c, int) or abs(c) > 100_000:
                continue
            ints |= {c - 1, c, c + 1}
            if c > 20:
                self.big |= {c - 1, c, c + 1, 2 * c + 1}
        self.ints = sorted(ints)
        self.theme = "int"   # one value type per sequence: mixed types mostly just raise TypeError
        self.strs = ["'a'", "'b'", "''", "'abc'"] + sorted(
            {repr(c) for c in constants if isinstance(c, str) and 0 < len(c) <= 12 and c.isprintable()})[:8]

    def small_int(self) -> str:
        return str(self.rng.choice([0, 1, 2, 3, 4, 5]) if self.rng.random() < 0.6 else self.rng.choice(self.ints))

    def int(self) -> str:
        return str(self.rng.choice(self.ints))

    def seq(self) -> str:
        r = self.rng.random()
        if r < 0.15:
            return "[]"
        if r < 0.25 and self.big:
            return f"list(range({self.rng.choice(sorted(self.big))}))"
        if r < 0.3 and self.theme != "int":
            return "'abc'"
        length = self.rng.choice([1, 2, 3, 3, 4, 5, 8])
        item = self.scalar if self.theme != "int" else self.small_int
        return "[" + ", ".join(item() for _ in range(length)) + "]"

    def scalar(self) -> str:
        if self.theme == "int":
            return self.small_int()
        if self.theme == "str":
            return self.rng.choice(self.strs[:4] + ["'c'", "'d'"])
        return self.rng.choice([self.small_int(), self.rng.choice(self.strs), "None", "0.5"])

    def like(self, default) -> str | None:
        if isinstance(default, bool):
            return self.rng.choice(["True", "False"])
        if isinstance(default, int):
            return self.int()
        if isinstance(default, float):
            return self.rng.choice(["0.0", "0.5", "1.0", "-1.0", "2.5"])
        if isinstance(default, str):
            return self.rng.choice(self.strs)
        if isinstance(default, tuple) and default and all(isinstance(x, bool) for x in default):
            return "(" + ", ".join(self.rng.choice(["True", "False"]) for _ in default) + ("," if len(default) == 1 else "") + ")"
        return None

    def value(self, name: str, default=inspect.Parameter.empty) -> str:
        low = name.lower()
        rng = self.rng
        if default is not inspect.Parameter.empty and default is not None:
            text = self.like(default)
            if text is not None:
                return text
        if low in CALLABLE_NAMES or low.endswith(("func", "_fn")) or (low == "key" and default is None):
            return "None" if rng.random() < 0.2 else rng.choice(CALLABLE_POOL)
        if low in SEQUENCE_NAMES or low.endswith("s") and low not in INT_NAMES and len(low) > 3:
            return self.seq()
        if low in REPEAT_NAMES:
            return str(rng.choice([-1, 0, 1, 2, 3]))
        if low in INT_NAMES or low.startswith(("max", "min", "num", "n_")) or low.endswith(("size", "count", "index")):
            if low in {"index", "idx"} and rng.random() < 0.15:
                return rng.choice(["slice(1, 3)", "slice(None, None, -1)", "slice(-2, None)", "slice(0, 5, 2)"])
            return self.int()
        r = rng.random()
        if r < 0.82:
            return self.scalar()
        if r < 0.87:
            return "None"
        return self.seq()


@dataclass
class Step:
    name: str                       # method name; '' calls the target function
    args: list = field(default_factory=list)
    kwargs: list = field(default_factory=list)   # (name, source)
    is_property: bool = False

    def copy(self):
        return Step(self.name, list(self.args), list(self.kwargs), self.is_property)


@dataclass
class Individual:
    ctor: Step | None
    steps: list

    def copy(self):
        return Individual(self.ctor.copy() if self.ctor else None, [s.copy() for s in self.steps])

    def key(self):
        return repr((self.ctor, self.steps))


class Generator:
    """Random call sequences for a target, and mutations of them."""

    def __init__(self, target: Target, packages, rng: random.Random, max_steps: int = 10):
        self.target, self.rng, self.max_steps = target, rng, max_steps
        self.pool = Pool(rng, constants_of(target))
        self.ctor_params = []
        self.callees = []
        if target.cls is not None:
            self.ctor_params = params_of(target.cls, drop_first=False) or []
            self.callees = callees_of(target.cls, packages, target.method)
            if target.kind == "method" and not any(c.name == target.method for c in self.callees):
                func = raw_function(target.cls.__dict__.get(target.method))
                params = params_of(func, drop_first=not isinstance(target.cls.__dict__.get(target.method), staticmethod)) if func else None
                if params is not None:
                    self.callees.append(Callee(target.method, params, 4.0,
                                               isinstance(target.cls.__dict__.get(target.method), property)))
        else:
            params = params_of(target.obj, drop_first=False) or []
            self.callees = [Callee("", params, 1.0)]
        self.weights = [c.weight for c in self.callees]

    def args_for(self, params) -> tuple[list, list]:
        args, kwargs = [], []
        positional_ok = True
        for p in params:
            if p.kind == "var":
                if positional_ok:
                    args += [self.pool.value(p.name) for _ in range(self.rng.choice([0, 0, 1, 2]))]
                continue
            if p.required:
                text = self.pool.value(p.name)
                if p.kind == "pos" and positional_ok:
                    args.append(text)
                else:
                    kwargs.append((p.name, text))
            elif self.rng.random() < 0.35:
                text = self.pool.value(p.name, p.default)
                if text == repr(p.default):     # passing the default only adds noise to the printed case
                    if p.kind == "pos":
                        positional_ok = False
                    continue
                if p.kind == "pos" and positional_ok:
                    args.append(text)
                else:
                    kwargs.append((p.name, text))
            elif p.kind == "pos":
                positional_ok = False
        return args, kwargs

    def step(self, callee: Callee | None = None) -> Step:
        callee = callee or self.rng.choices(self.callees, self.weights)[0]
        if callee.is_property:
            return Step(callee.name, is_property=True)
        args, kwargs = self.args_for(callee.params)
        if callee.name in REPEAT_METHODS:
            args = [str(self.rng.choice([-1, 0, 1, 2, 3])) for _ in args]
        return Step(callee.name, args, kwargs)

    def ctor(self) -> Step | None:
        if self.target.cls is None:
            return None
        args, kwargs = self.args_for(self.ctor_params)
        return Step("<init>", args, kwargs)

    def random(self, min_steps: int = 0) -> Individual:
        self.pool.theme = self.rng.choices(["int", "str", "mixed"], [0.75, 0.12, 0.13])[0]
        if self.target.cls is None:
            return Individual(None, [self.step() for _ in range(self.rng.choice([1, 1, 1, 2]))])
        steps = [self.step() for _ in range(self.rng.randint(min_steps, max(min_steps, self.max_steps // 2)))]
        if self.target.kind == "method":
            focus = [c for c in self.callees if c.name == self.target.method]
            if focus:
                steps.append(self.step(focus[0]))
        return Individual(self.ctor(), steps)

    def tweak(self, text: str) -> str:
        try:
            value = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(value, bool):
            return str(not value)
        if isinstance(value, int):
            return str(value + self.rng.choice([-1, 1, -2, 2, 1, -1]))
        return text

    def mutate(self, parent: Individual, other: Individual | None = None) -> Individual:
        ind = parent.copy()
        for _ in range(self.rng.choice([1, 1, 1, 2, 3])):
            op = self.rng.random()
            steps = ind.steps
            if op < 0.3 and (steps or ind.ctor):
                holder = self.rng.choice(steps + ([ind.ctor] if ind.ctor else []))
                slots = [("a", i) for i in range(len(holder.args))] + [("k", i) for i in range(len(holder.kwargs))]
                if not slots:
                    continue
                where, i = self.rng.choice(slots)
                if holder.name in REPEAT_METHODS:
                    continue
                old = holder.args[i] if where == "a" else holder.kwargs[i][1]
                name = holder.kwargs[i][0] if where == "k" else "x"
                new = self.tweak(old) if self.rng.random() < 0.6 else self.pool.value(name)
                if where == "a":
                    holder.args[i] = new
                else:
                    holder.kwargs[i] = (holder.kwargs[i][0], new)
            elif op < 0.5 and len(steps) < self.max_steps:
                steps.insert(self.rng.randint(0, len(steps)), self.step())
            elif op < 0.65 and len(steps) > 1:
                del steps[self.rng.randrange(len(steps))]
            elif op < 0.85 and steps and len(steps) < self.max_steps:
                i = self.rng.randrange(len(steps))
                dup = steps[i].copy()
                if dup.name not in REPEAT_METHODS:
                    dup.args = [self.tweak(a) for a in dup.args]
                steps.insert(i + 1, dup)
            elif op < 0.92 and ind.ctor is not None:
                ind.ctor = self.ctor()
            elif other is not None and other.steps and self.target.cls is not None:
                cut = self.rng.randint(0, len(steps))
                ind.steps = (steps[:cut] + other.steps[self.rng.randint(0, len(other.steps) - 1):])[:self.max_steps]
            else:
                steps.append(self.step())
        if self.target.cls is None and not ind.steps:
            ind.steps.append(self.step())
        return ind


def constants_of(target: Target) -> list:
    """Literal constants in the target's code, used to seed boundary values."""
    found = []
    funcs = target.scope_functions()
    if target.cls is not None:   # a method's boundaries often come from the rest of its class
        funcs = [f for f in (unwrap(raw_function(v)) for v in vars(target.cls).values()) if f]
    for func in funcs:
        try:
            tree = ast.parse(_dedent(inspect.getsource(func)))
        except (OSError, TypeError, SyntaxError):
            continue
        found += [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)]
    if target.cls is not None:
        for klass in target.cls.__mro__:
            for value in vars(klass).values():
                if isinstance(value, int) and not isinstance(value, bool):
                    found.append(value)
    return found


def _dedent(source: str) -> str:
    lines = source.splitlines()
    indent = min((len(l) - len(l.lstrip()) for l in lines if l.strip()), default=0)
    return "\n".join(l[indent:] for l in lines)


# ----------------------------------------------------------------------------------------- running
def render_step(target: Target, step: Step, listify: bool = False) -> str:
    args = step.args + [f"{k}={v}" for k, v in step.kwargs]
    if step.name == "<init>":
        return f"obj = {target.bound}({', '.join(args)})"
    if step.name == "":
        text = f"{target.bound}({', '.join(args)})"
    elif step.is_property:
        text = f"obj.{step.name}"
    elif step.name in SYNTAX and not step.kwargs and len(step.args) == len(set(re.findall(r"\{\d\}", SYNTAX[step.name]))):
        text = SYNTAX[step.name].format(*step.args, o="obj", args=", ".join(args))
    else:
        text = f"obj.{step.name}({', '.join(args)})"
    return f"list({text})" if listify else text


def is_statement(step: Step) -> bool:
    return step.name in STATEMENTS and not step.kwargs


def describe(value, limit: int = MAX_KEPT) -> str:
    try:
        text = repr(value)
    except BaseException as error:  # a broken __repr__ is itself worth seeing
        text = f"<repr failed: {type(error).__name__}>"
    text = ADDRESS.sub("", text)
    return clip(text, limit)


def clip(text: str, limit: int = MAX_REPR) -> str:
    return text if len(text) <= limit else text[:limit - 3] + "..."


def excerpt(a: str, b: str, width: int = 60) -> tuple[str, str]:
    """The parts of two long values around their first difference."""
    if len(a) <= MAX_REPR and len(b) <= MAX_REPR:
        return a, b
    p = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    start = max(0, p - width // 2)
    cut = lambda s: ("..." if start else "") + s[start:start + width] + ("..." if start + width < len(s) else "")
    return cut(a), cut(b)


def is_lazy(value) -> bool:
    return isinstance(value, (types.GeneratorType, map, filter, zip, reversed)) or (
        hasattr(value, "__next__") and hasattr(value, "__iter__") and not isinstance(value, (str, bytes)))


@dataclass
class Outcome:
    observations: list     # one string per step; the ctor's comes first for classes
    listify: list          # whether each step's result was an iterator turned into a list
    state: str = ""        # repr of the object after the last step


def execute(target: Target, ind: Individual, time_limit: float = 1.0, tracer=None) -> Outcome:
    """Run an individual in a fresh namespace. Never raises for errors in the code under test."""
    namespace = {}
    observations, listify = [], []
    previous = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, time_limit)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            exec(target.import_line, namespace)
            steps = ([ind.ctor] if ind.ctor else []) + ind.steps
            for step in steps:
                if step is not ind.ctor and target.cls is not None and "obj" not in namespace:
                    observations.append("skipped: no object")
                    listify.append(False)
                    continue
                code = render_step(target, step)
                lazy = False
                if tracer:
                    sys.settrace(tracer)
                try:
                    if step.name == "<init>" or is_statement(step):
                        exec(code, namespace)
                        result = None
                    else:
                        result = eval(code, namespace)
                    if is_lazy(result):
                        lazy, result = True, list(_take(result, 200))
                    text = "None" if step.name == "<init>" or is_statement(step) else describe(result)
                except Timeout:
                    raise
                except BaseException as error:
                    text = f"raises {type(error).__name__}" + (f": {describe(str(error))}" if str(error) else "")
                finally:
                    if tracer:
                        sys.settrace(None)
                observations.append(text)
                listify.append(lazy)
            state = describe(namespace["obj"]) if "obj" in namespace else ""
    except Timeout:
        sys.settrace(None)
        observations.append("timeout")
        listify.append(False)
        state = "timeout"
    except BaseException as error:
        sys.settrace(None)
        observations.append(f"raises {type(error).__name__}")
        listify.append(False)
        state = ""
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    return Outcome(observations, listify, state)


def _take(iterator, n):
    for i, item in enumerate(iterator):
        if i >= n:
            break
        yield item


def render(target: Target, ind: Individual, outcome: Outcome | None = None, indent: str = "    ",
           marks: dict | None = None) -> str:
    """The individual as code, with what each step returned on the current code as a comment."""
    lines = [indent + target.import_line]
    steps = ([ind.ctor] if ind.ctor else []) + ind.steps
    for i, step in enumerate(steps):
        lazy = bool(outcome and i < len(outcome.listify) and outcome.listify[i])
        code = render_step(target, step, lazy)
        note = ""
        if outcome and i < len(outcome.observations):
            obs = outcome.observations[i]
            if marks and i in marks:
                note = marks[i]
            elif obs.startswith("raises") or not (step.name == "<init>" or is_statement(step)):
                note = f"current: {clip(obs)}"
        lines.append(f"{indent}{code}" + (f"    # {note}" if note else ""))
    if outcome and outcome.state and target.cls is not None:
        lines.append(f"{indent}# obj afterwards (current code): {clip(outcome.state)}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------- spec
def spec_claims(names: list[str], packages: list[str], per_entity: int = 4, timeout: int = 60) -> dict:
    """What extract_spec (step 1) says about these entities: {fullname: [claim text]}."""
    names = sorted(set(n for n in names if n))
    if not names or not (TOOLS_DIR / "extract_spec").exists():
        return {}
    pattern = "(^|\\.)(" + "|".join(re.escape(n.split(".", 1)[1] if n.count(".") >= 2 else n) for n in names) + ")$"
    command = [sys.executable, str(TOOLS_DIR / "extract_spec"), "--name", pattern, "--compact"]
    for p in packages:
        command += ["--package", p]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        data = json.loads(result.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return {}
    found = {}
    for entity in data.get("entities", []):
        claims = []
        for claim in entity.get("claims", []):
            text = " ".join(str(claim.get("text", "")).split())
            if claim.get("kind") == "example" and claim.get("code"):
                text = f"{one_line(claim['code'])}  ->  {one_line(str(claim.get('expected', '')))}"
            if text and text not in claims:
                claims.append(f"{claim.get('kind', 'claim')}: {text[:220]}")
        found[entity["name"]] = claims[:per_entity]
    return found


def one_line(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def format_claims(claims: dict, order: list[str]) -> str:
    lines, done, empty = [], set(), []
    for name in order:
        matches = [k for k in claims if k == name or k.endswith("." + name.split(".", 1)[-1])]
        for key in matches[:1]:
            if key in done:
                continue
            done.add(key)
            if claims[key]:
                lines.append(f"  {key}:")
                lines += [f"    - {c}" for c in claims[key]]
            else:
                empty.append(key)
    if empty:
        lines.append("  no claims of their own (use the class's claims, the README and docs/): "
                     + ", ".join(empty[:8]))
    return "\n".join(lines) if lines else "  (extract_spec found nothing for this target)"
