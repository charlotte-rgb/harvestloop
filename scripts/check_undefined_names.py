"""
Scope-aware undefined-name check for the harvestloop package.

Catches renamed-variable leftovers that only blow up at runtime, the
class of bug py_compile and import smoke tests cannot see, because
the code path only runs inside Isaac.

Usage:
    python3 scripts/check_undefined_names.py
"""

import ast
import builtins
import sys
from pathlib import Path

PACKAGE_DIRS = [
    Path("/home/charlotte/harvestloop_sim/scripts/harvestloop"),
    Path("/home/charlotte/harvestloop_sim/scripts/scenegen"),
]

SCRIPTS_DIR = Path("/home/charlotte/harvestloop_sim/scripts")


def module_toplevel_names(path):
    """Names a module defines at top level."""
    names = set()
    tree = ast.parse(path.read_text(), filename=str(path))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                for sub in ast.walk(target):
                    if isinstance(sub, ast.Name):
                        names.add(sub.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "*":
                    continue
                names.add(alias.asname or alias.name.split(".")[0])

    return names


class ScopeChecker(ast.NodeVisitor):
    """Walks functions, tracking which names are bound in each scope."""

    def __init__(self, path, global_names):
        self.path = path
        self.problems = []
        self.scopes = [set(global_names)]

    # -- scope helpers -------------------------------------------------

    def bound(self, name):
        return any(name in scope for scope in self.scopes)

    def bind(self, name):
        self.scopes[-1].add(name)

    def bind_target(self, node):
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                self.scopes[-1].add(sub.id)

    def bind_arguments(self, args):
        for arg in (
            list(args.posonlyargs)
            + list(args.args)
            + list(args.kwonlyargs)
            + ([args.vararg] if args.vararg else [])
            + ([args.kwarg] if args.kwarg else [])
        ):
            self.bind(arg.arg)

    # -- scopes --------------------------------------------------------

    def visit_FunctionDef(self, node):
        self.bind(node.name)
        self._function_scope(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _function_scope(self, node):
        for decorator in node.decorator_list:
            self.visit(decorator)

        for default in node.args.defaults + [
            d for d in node.args.kw_defaults if d is not None
        ]:
            self.visit(default)

        self.scopes.append(set())
        self.bind_arguments(node.args)

        # Pre-bind everything assigned anywhere in the body: Python
        # binds on execution, but flagging a name used before its
        # assignment line is a different (rarer) bug than a rename.
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = (
                    sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                )
                for target in targets:
                    self.bind_target(target)
            elif isinstance(sub, (ast.For, ast.AsyncFor)):
                self.bind_target(sub.target)
            elif isinstance(sub, (ast.With, ast.AsyncWith)):
                for item in sub.items:
                    if item.optional_vars is not None:
                        self.bind_target(item.optional_vars)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                self.bind(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for alias in sub.names:
                    if alias.name != "*":
                        self.bind(alias.asname or alias.name.split(".")[0])
            elif isinstance(sub, ast.comprehension):
                self.bind_target(sub.target)
            elif isinstance(sub, ast.Lambda):
                self.bind_arguments(sub.args)
            elif isinstance(
                sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                self.bind(sub.name)
            elif isinstance(sub, ast.Global):
                for name in sub.names:
                    self.bind(name)
            elif isinstance(sub, ast.NamedExpr):
                self.bind_target(sub.target)

        for statement in node.body:
            self.visit(statement)

        self.scopes.pop()

    def visit_ClassDef(self, node):
        self.bind(node.name)
        self.scopes.append(set())
        for statement in node.body:
            self.visit(statement)
        self.scopes.pop()

    def visit_Lambda(self, node):
        self.scopes.append(set())
        self.bind_arguments(node.args)
        self.visit(node.body)
        self.scopes.pop()

    def visit_Name(self, node):
        if not isinstance(node.ctx, ast.Load):
            self.bind(node.id)
            return

        if self.bound(node.id) or hasattr(builtins, node.id):
            return

        self.problems.append((node.lineno, node.id))


def star_import_names(path, node):
    """Top-level names pulled in by one `from X import *`."""
    if node.level:
        # Relative: resolve against this module's package.
        source = path.parent / f"{node.module}.py"
    else:
        source = SCRIPTS_DIR / f"{node.module.replace('.', '/')}.py"

    if not source.exists():
        print(f"[WARN] cannot resolve star import in {path}: {node.module}")
        return set()

    return module_toplevel_names(source)


def main():
    problems = []

    for package in PACKAGE_DIRS:
        for path in sorted(package.glob("*.py")):
            tree = ast.parse(path.read_text(), filename=str(path))

            globals_here = module_toplevel_names(path)

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and any(
                    alias.name == "*" for alias in node.names
                ):
                    globals_here |= star_import_names(path, node)

            checker = ScopeChecker(path, globals_here)

            for statement in tree.body:
                checker.visit(statement)

            for lineno, name in checker.problems:
                problems.append((path, lineno, name))

    for path, lineno, name in problems:
        print(f"{path}:{lineno}: undefined name '{name}'")

    print()
    if problems:
        print(f"[FAIL] {len(problems)} undefined name(s)")
        return 1

    print("[OK] no undefined names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
