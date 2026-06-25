import pathlib
import subprocess
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_seg.py"


class CheckSegCliTest(unittest.TestCase):

    def test_help_exposes_required_cli_options_without_running_model(self):
        result = subprocess.run(
            ["python", str(SCRIPT), "--help"],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--config", result.stdout)
        self.assertIn("--checkpoint", result.stdout)
        self.assertIn("--out-dir", result.stdout)
        self.assertIn("--split", result.stdout)
        self.assertIn("--max-frames", result.stdout)

    def test_module_adds_repo_root_to_python_path(self):
        code = (
            "import importlib.util, sys; "
            f"spec = importlib.util.spec_from_file_location('check_seg_cli', {str(SCRIPT)!r}); "
            "module = importlib.util.module_from_spec(spec); "
            "spec.loader.exec_module(module); "
            "print(module.REPO_ROOT); "
            "print(sys.path[0])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd="/tmp",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        lines = result.stdout.strip().splitlines()
        self.assertEqual(lines[0], str(ROOT))
        self.assertEqual(lines[1], str(ROOT))


if __name__ == "__main__":
    unittest.main()
