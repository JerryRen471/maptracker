import ast
import pathlib
import struct
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "check_seg.py"


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


class CheckSegTest(unittest.TestCase):

    def test_bev_border_box_spans_full_canvas(self):
        border_box = load_helper("bev_border_box")

        self.assertEqual(border_box(width=200, height=100), (0, 0, 199, 99))
        self.assertEqual(border_box(width=1, height=1), (0, 0, 0, 0))

    def test_save_visibility_mask_writes_png(self):
        save_visibility_mask = load_helper("save_visibility_mask")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = pathlib.Path(tmp_dir) / "visibility_mask.png"
            save_visibility_mask(
                [[1.0, 0.0], [0.0, 1.0]],
                out_path,
                draw_bev_range=False,
            )

            self.assertTrue(out_path.is_file())

            data = out_path.read_bytes()
            self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(data[12:16], b"IHDR")
            width, height, bit_depth, color_type = struct.unpack(
                ">IIBB", data[16:26])
            self.assertEqual((width, height), (2, 2))
            self.assertEqual(bit_depth, 8)
            self.assertEqual(color_type, 0)


if __name__ == "__main__":
    unittest.main()
