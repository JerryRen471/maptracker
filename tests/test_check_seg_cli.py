import pathlib
import subprocess
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


if __name__ == "__main__":
    unittest.main()
