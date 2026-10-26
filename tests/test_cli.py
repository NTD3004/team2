"""CLI integration tests runnable without the scientific Python dependencies."""
import ast
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch

SRC = Path(__file__).resolve().parents[1] / "src"
spec = importlib.util.spec_from_file_location("cli", SRC / "main.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class CommandTests(unittest.TestCase):
    def dispatch(self, arguments):
        calls = {name: Mock() for name in
                 ("run_kmeans", "run_gmm", "run_gmm_bootstrap", "run_kde")}
        modules = {name: types.ModuleType(name) for name in ("kmeans", "gmm", "kde")}
        for name, function in calls.items():
            module = "gmm" if name.startswith("run_gmm") else name[4:]
            setattr(modules[module], name, function)
        with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["main.py", *arguments]):
            cli.main()
        return calls

    def test_kmeans_controls(self):
        self.dispatch(["kmeans", "--k", "4", "--n-init", "7", "--no-use-death-weights"])[
            "run_kmeans"].assert_called_once_with(k=4, n_init=7, use_death_weights=False)

    def test_gmm_controls_and_bootstrap(self):
        for bootstrap in (False, True):
            arguments = ["gmm", "--k", "3", "--n-init", "9", "--reg-covar", "0.01",
                         "--no-use-death-weights"]
            if bootstrap:
                arguments += ["--bootstrap", "--samples", "2", "--seed", "5"]
            calls = self.dispatch(arguments)
            expected = dict(k=3, n_init=9, reg_covar=0.01, use_death_weights=False,
                            max_iter=300, tol=1e-6)
            if bootstrap:
                expected.update(number_of_samples=2, random_state=5)
            calls["run_gmm_bootstrap" if bootstrap else "run_gmm"].assert_called_once_with(**expected)
            calls["run_gmm" if bootstrap else "run_gmm_bootstrap"].assert_not_called()

    def test_kde_controls(self):
        self.dispatch(["kde", "--kernel", "box", "--bandwidth", "25"])[
            "run_kde"].assert_called_once_with(kernel_type="box", bandwidth=25., use_death_weights=True)

    def test_invalid_controls(self):
        for arguments in (["kmeans", "--n-init", "0"], ["gmm", "--reg-covar", "nan"],
                          ["gmm", "--k", "-1"], ["kde", "--bandwidth", "inf"],
                          ["kde", "--k", "8"]):
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    self.dispatch(arguments)
                self.assertEqual(error.exception.code, 2)

    def test_internal_keyword_signatures(self):
        trees = [ast.parse(path.read_text()) for path in SRC.glob("*.py")]
        signatures = {node.name: {arg.arg for arg in node.args.args + node.args.kwonlyargs}
                      for tree in trees for node in tree.body if isinstance(node, ast.FunctionDef)}
        for tree in trees:
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in signatures:
                    for kw in node.keywords:
                        if kw.arg is not None:
                            self.assertIn(kw.arg, signatures[node.func.id], node.func.id)


if __name__ == "__main__":
    unittest.main()
