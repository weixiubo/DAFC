import unittest, tempfile
from pathlib import Path
import cv2
import numpy as np
from dafc.experiments import android_atomic as atomic
from dafc.evaluation import attempts as audit
class AuthorizationTests(unittest.TestCase):
    def test_failed_receipt_forces_fallback_for_every_hybrid(self):
        source = {"source_success": False, "atomic_memory": None}
        target = np.zeros((120, 160, 3), dtype=np.uint8)
        for policy in ("s_only", "standard_ransac", "dafc"):
            receipt = atomic.authorize(policy, source, target)
            self.assertFalse(receipt["accepted"])
            self.assertEqual(receipt["reason"], "source_receipt_failed")

    def test_standard_identity_maps_the_action(self):
        rng = np.random.default_rng(585001)
        image = rng.integers(0, 256, size=(360, 480, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        action = [211.0, 143.0]
        receipt = atomic.standard_registration(image, image.copy(), action)
        self.assertTrue(receipt["accepted"], receipt)
        self.assertGreaterEqual(receipt["inliers"], atomic.STANDARD_MIN_INLIERS)
        self.assertAlmostEqual(receipt["mapped_action_xy"][0], action[0], places=3)
        self.assertAlmostEqual(receipt["mapped_action_xy"][1], action[1], places=3)

    def test_model_coordinate_normalization_is_bounded(self):
        image = np.zeros((240, 108, 3), dtype=np.uint8)
        self.assertEqual(
            atomic.normalized_to_pixel(
                {"valid": True, "normalized_xy": [1.2, -0.1]}, image
            ),
            [107.0, 0.0],
        )
        self.assertIsNone(
            atomic.normalized_to_pixel(
                {"valid": False, "normalized_xy": [0.5, 0.5]}, image
            )
        )

    def test_target_prompt_exposes_no_setup_label(self):
        prompt = atomic.target_instruction("Turn wifi on.")
        self.assertIn("current screenshot", prompt)
        for label in ("Network & internet", "Internet", "Wi-Fi"):
            self.assertNotIn(label, prompt)

class RetentionTests(unittest.TestCase):
    def test_exactly_one_predecision_retry(self):
        job = atomic.jobs_for_actor("qwen25vl")[0]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(atomic.next_attempt(root, "qwen25vl", job), 0)
            path = atomic.result_path(root, "qwen25vl", job, 0)
            atomic.write_new_json(
                path,
                {
                    "outcome_bearing": False,
                    "retry_permitted": True,
                },
            )
            self.assertEqual(atomic.next_attempt(root, "qwen25vl", job), 1)
            second = atomic.result_path(root, "qwen25vl", job, 1)
            atomic.write_new_json(
                second,
                {
                    "outcome_bearing": True,
                    "retry_permitted": False,
                },
            )
            self.assertIsNone(atomic.next_attempt(root, "qwen25vl", job))

