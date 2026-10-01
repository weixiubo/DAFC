#!/usr/bin/env python3
"""Offline tests for the post-result integral-coordinate loader repair."""

from __future__ import annotations

import copy
import math
import unittest

import analyze_android_recovered_source_followup_v2_integral_coordinate_repaired as repair


def payload(coordinates):
    return {
        "logical": {
            "kind": "qwen25vl",
            "steps": [
                {
                    "action_executed": True,
                    "action_xy": coordinates,
                    "decision": {
                        "executed_action_xy": copy.deepcopy(coordinates)
                    },
                }
            ],
        }
    }


class IntegralCoordinateRepairTest(unittest.TestCase):
    def test_float_pair_is_normalized_only_in_copy(self) -> None:
        original = payload([17.0, 29.0])
        normalized, audit = repair.normalize_integral_action_coordinates(
            original
        )
        self.assertEqual(original["logical"]["steps"][0]["action_xy"], [17.0, 29.0])
        self.assertEqual(normalized["logical"]["steps"][0]["action_xy"], [17, 29])
        self.assertTrue(
            all(
                isinstance(value, int)
                for value in normalized["logical"]["steps"][0]["action_xy"]
            )
        )
        self.assertEqual(audit["qwen25vl"]["executed_steps"], 1)
        self.assertEqual(audit["qwen25vl"]["original_float_pairs"], 1)
        self.assertEqual(audit["qwen25vl"]["normalized_pairs"], 1)

    def test_integer_pair_remains_valid(self) -> None:
        normalized, audit = repair.normalize_integral_action_coordinates(
            payload([17, 29])
        )
        self.assertEqual(normalized["logical"]["steps"][0]["action_xy"], [17, 29])
        self.assertEqual(audit["qwen25vl"]["original_int_pairs"], 1)

    def test_fractional_bool_nonfinite_and_mismatch_are_rejected(self) -> None:
        cases = [
            payload([17.5, 29.0]),
            payload([True, 29.0]),
            payload([math.nan, 29.0]),
            payload([math.inf, 29.0]),
        ]
        mismatch = payload([17.0, 29.0])
        mismatch["logical"]["steps"][0]["decision"]["executed_action_xy"] = [
            18.0,
            29.0,
        ]
        cases.append(mismatch)
        for row in cases:
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                repair.normalize_integral_action_coordinates(row)

    def test_no_action_step_is_not_changed(self) -> None:
        original = {
            "logical": {
                "kind": "uitars",
                "steps": [
                    {
                        "action_executed": False,
                        "action_xy": None,
                        "decision": {},
                    }
                ],
            }
        }
        normalized, audit = repair.normalize_integral_action_coordinates(
            original
        )
        self.assertEqual(normalized, original)
        self.assertEqual(audit["uitars"]["executed_steps"], 0)


if __name__ == "__main__":
    unittest.main()
