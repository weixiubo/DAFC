from __future__ import annotations

import unittest

import dafc.evaluation.audit_transitionguard_postcheck_baselines as MODULE
from dafc.evaluation import postcheck_report


def unit(classification: str, change: bool, absolute: bool, relative: bool):
    return {
        "actor": "qwen25vl",
        "contract": "native_zoom_80",
        "decisions": {
            "change_only": change,
            "absolute_template": absolute,
            "directional_effect": relative,
            "transition_contract": absolute or relative,
        },
        "logical_id": classification,
        "semantic_class": classification,
        "source_cluster_id": f"cluster-{classification}",
        "step": 0,
        "task": "click-option",
    }


class PostcheckBaselineAuditTest(unittest.TestCase):
    def test_detector_decisions_validate_logged_or(self):
        post = {
            "absolute_postcondition": {"accepted": False},
            "relative_effect": {"accepted": True, "checks": {"target_change": True}},
            "post_verified": True,
        }
        self.assertEqual(
            MODULE.detector_decisions(post),
            {
                "change_only": True,
                "absolute_template": False,
                "directional_effect": True,
                "transition_contract": True,
            },
        )

    def test_detector_decisions_reject_inconsistent_log(self):
        post = {
            "absolute_postcondition": {"accepted": False},
            "relative_effect": {"accepted": False, "checks": {"target_change": True}},
            "post_verified": True,
        }
        with self.assertRaisesRegex(ValueError, "disagrees"):
            MODULE.detector_decisions(post)

    def test_natural_summary_counts_relative_only_semantics(self):
        units = [
            unit("advanced", True, True, True),
            unit("advanced", True, False, True),
            unit("nonadvancing", True, False, False),
            unit("harmful", False, False, False),
        ]
        summary = MODULE.summarize_natural(units)
        self.assertEqual(summary["relative_only_acceptances"]["count"], 1)
        self.assertEqual(summary["relative_only_acceptances"]["semantics"]["advanced"], 1)
        self.assertEqual(summary["detectors"]["transition_contract"]["accepted_advanced"], 2)
        self.assertEqual(summary["detectors"]["change_only"]["accepted_adverse"], 1)

    def test_fault_summary_preserves_paired_disagreement(self):
        units = [
            unit("advanced", True, False, False),
            unit("advanced", True, True, False),
            unit("advanced", False, False, False),
        ]
        summary = MODULE.summarize_fault(units)
        self.assertEqual(summary["detectors"]["transition_contract"]["rejected"], 2)
        self.assertEqual(
            summary["paired_disagreement"]["change_only"],
            {"both_accept": 1, "contract_only": 0, "baseline_only": 1, "both_reject": 1},
        )

    def test_percentile_interpolates(self):
        self.assertEqual(MODULE._percentile([0.0, 1.0], 0.5), 0.5)

    def test_latex_table_keeps_secondary_claim_boundary(self):
        natural = MODULE.summarize_natural(
            [unit("advanced", True, True, True), unit("nonadvancing", True, False, False)]
        )
        fault = MODULE.summarize_fault([unit("advanced", False, False, False)])
        table = postcheck_report.latex_table(
            {
                "natural": {"pooled": natural},
                "faults": {"adjacent": {"pooled": fault}, "drop": {"pooled": fault}},
            }
        )
        self.assertIn("Complete contract", table)
        self.assertIn("read-only secondary analysis", table)
        self.assertIn("no counterfactual episode", table)
        number_report = {
            "natural": {
                "pooled": natural,
                "advanced_contract_minus_absolute_bootstrap": {
                    "pooled": {"estimate": 0.5, "confidence_interval_95": [0.1, 0.9]}
                },
            },
            "faults": {"adjacent": {"pooled": fault}, "drop": {"pooled": fault}},
        }
        numbers = postcheck_report.latex_numbers(number_report)
        self.assertIn("\\TGPostNaturalActions", numbers)


if __name__ == "__main__":
    unittest.main()
