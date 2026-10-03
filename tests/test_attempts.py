import unittest

from dafc.evaluation import attempts as audit


class AttemptLifecycleTests(unittest.TestCase):
    def test_single_outcome_attempt_is_terminal(self):
        rows = [{"attempt": 0, "status": "complete", "outcome_bearing": True, "retry_permitted": False}]
        self.assertEqual(audit.sequence_errors(rows, require_complete=True), [])

    def test_one_infrastructure_retry_then_outcome_is_valid(self):
        rows = [
            {
                "attempt": 0,
                "status": "infrastructure_failure",
                "outcome_bearing": False,
                "retry_permitted": True,
            },
            {"attempt": 1, "status": "outcome_failure", "outcome_bearing": True, "retry_permitted": False},
        ]
        self.assertEqual(audit.sequence_errors(rows, require_complete=True), [])

    def test_outcome_retry_is_rejected(self):
        rows = [
            {"attempt": 0, "status": "complete", "outcome_bearing": True, "retry_permitted": False},
            {"attempt": 1, "status": "complete", "outcome_bearing": True, "retry_permitted": False},
        ]
        messages = audit.sequence_errors(rows, require_complete=True)
        self.assertTrue(any(("more than one" in message for message in messages)))
        self.assertTrue(any(("retried" in message for message in messages)))

    def test_partial_retry_boundary_is_not_misreported_complete(self):
        rows = [
            {
                "attempt": 0,
                "status": "infrastructure_failure",
                "outcome_bearing": False,
                "retry_permitted": True,
            }
        ]
        self.assertEqual(audit.sequence_errors(rows, require_complete=False), [])
        self.assertTrue(audit.sequence_errors(rows, require_complete=True))
