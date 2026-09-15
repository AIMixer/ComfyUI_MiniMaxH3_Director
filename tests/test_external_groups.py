"""CPU-only checks for ignored AddGuide external payloads."""

import ast
import copy
import unittest
from pathlib import Path

from test_addguide import load_module


ROOT = Path(__file__).resolve().parents[1]
tasks = load_module("external_group_task_prompts", "lib/task_prompts.py")
tree = ast.parse((ROOT / "director/external_groups.py").read_text(encoding="utf-8"))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"connected_external_group_inputs", "normalize_groups_list", "validate_external_group_inputs"}]
namespace = {"Any": object, "resolve_task_key": tasks.resolve_task_key,
             "I2V_FAMILY": {"t2v", "i2v", "fl2v"},
             "ADDGUIDE_EXTERNAL_GROUP_ERROR": "disconnect groups"}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "director/external_groups.py"), "exec"), namespace)
validate = namespace["validate_external_group_inputs"]
director_source = (ROOT / "nodes/director.py").read_text(encoding="utf-8")
director_tree = ast.parse(director_source)
director_class = next(node for node in director_tree.body
                      if isinstance(node, ast.ClassDef) and node.name == "MiniMaxH3Director")
lazy_method = next(node for node in director_class.body
                   if isinstance(node, ast.FunctionDef) and node.name == "check_lazy_status")
lazy_namespace = {
    "resolve_task_key": tasks.resolve_task_key,
    "connected_external_group_inputs": namespace["connected_external_group_inputs"],
}
lazy_module = ast.fix_missing_locations(ast.Module(
    body=[ast.ClassDef(name="LazyProbe", bases=[], keywords=[],
                       body=[lazy_method], decorator_list=[])],
    type_ignores=[],
))
exec(compile(lazy_module,
             str(ROOT / "nodes/director.py"), "exec"), lazy_namespace)


class ExternalGroupTests(unittest.TestCase):
    def test_addguide_requires_external_groups_to_be_disconnected(self):
        self.assertEqual(validate(task_type="addguide", i2v_groups=None, r2v_groups=None),
                         ("addguide", None, None))
        for value in ([], {}, {"kind": "i2v"}, [{"prompt": "draft"}], "invalid"):
            with self.subTest(value=value):
                before = copy.deepcopy(value)
                with self.assertRaisesRegex(ValueError, "disconnect groups"):
                    validate(task_type="addguide", i2v_groups=value, r2v_groups=None)
                self.assertEqual(value, before)

    def test_connected_inputs_are_detected_from_prompt_without_reading_values(self):
        detect = namespace["connected_external_group_inputs"]
        prompt = {"12": {"inputs": {"i2v_groups": ["4", 0], "task_type": "addguide"}}}
        self.assertEqual(detect(prompt, "12"), ["i2v_groups"])
        self.assertEqual(detect(prompt, "missing"), [])

    def test_addguide_does_not_request_lazy_group_values(self):
        probe = lazy_namespace["LazyProbe"]()
        prompt = {"12": {"inputs": {"i2v_groups": ["4", 0]}}}
        self.assertGreaterEqual(director_source.count('"lazy": True'), 2)
        self.assertEqual(probe.check_lazy_status("addguide", prompt=prompt, unique_id="12"), [])
        self.assertEqual(
            probe.check_lazy_status("i2v", prompt=prompt, unique_id="12", i2v_groups=None),
            ["i2v_groups"],
        )

    def test_supported_group_modes_still_consume_and_validate_payloads(self):
        group = {"kind": "t2v", "prompt": "draft"}
        self.assertEqual(validate(task_type="t2v", i2v_groups=[group], r2v_groups=None),
                         ("t2v", [group], "i2v"))
        with self.assertRaises(ValueError):
            validate(task_type="i2v", i2v_groups=[], r2v_groups=None)
        with self.assertRaisesRegex(ValueError, "mixed mode"):
            validate(task_type="mixed", i2v_groups=[group], r2v_groups=None)
