import ast
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "visualization" / "vis_global.py"


def load_helper(name):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if not helpers:
        raise AssertionError(f"{name} helper is missing")
    module = ast.Module(body=helpers, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(SCRIPT), "exec"), namespace)
    return namespace[name]


class VisGlobalTest(unittest.TestCase):

    def test_vis_pred_skips_scene_when_prediction_vectors_are_empty(self):
        has_vectors = load_helper("scene_has_prediction_vectors")
        pred_results = [
            {
                "scene_name": "scene_empty",
                "vectors": [],
                "labels": [],
                "global_ids": [],
                "local_idx": 0,
            }
        ]

        self.assertFalse(has_vectors("scene_empty", pred_results))
        self.assertFalse(has_vectors("missing_scene", pred_results))
        self.assertTrue(
            has_vectors(
                "scene_full",
                [{"scene_name": "scene_full", "vectors": [[[0.0, 0.0], [1.0, 1.0]]]}],
            )
        )


if __name__ == "__main__":
    unittest.main()
