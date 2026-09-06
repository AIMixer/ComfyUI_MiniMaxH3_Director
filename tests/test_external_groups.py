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
             and node.name in {"normalize_groups_list", "validate_external_group_inputs"}]
namespace = {"Any": object, "resolve_task_key": tasks.resolve_task_key,
             "I2V_FAMILY": {"t2v", "i2v", "fl2v"}}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(ROOT / "director/external_groups.py"), "exec"), namespace)
validate = namespace["validate_external_group_inputs"]


class ExternalGroupTests(unittest.TestCase):
    def test_addguide_ignores_empty_incomplete_and_both_inputs_without_mutation(self):
        for value in (None, [], {}, {"kind": "i2v"}, [{"prompt": "draft"}], "invalid"):
            for ports in ((value, None), (None, value), (value, value)):
                with self.subTest(ports=ports):
                    before = copy.deepcopy(ports)
                    self.assertEqual(validate(task_type="addguide", i2v_groups=ports[0],
                                              r2v_groups=ports[1]), ("addguide", None, None))
                    self.assertEqual(ports, before)

    def test_supported_group_modes_still_consume_and_validate_payloads(self):
        group = {"kind": "t2v", "prompt": "draft"}
        self.assertEqual(validate(task_type="t2v", i2v_groups=[group], r2v_groups=None),
                         ("t2v", [group], "i2v"))
        with self.assertRaises(ValueError):
            validate(task_type="i2v", i2v_groups=[], r2v_groups=None)
        with self.assertRaisesRegex(ValueError, "mixed mode"):
            validate(task_type="mixed", i2v_groups=[group], r2v_groups=None)
