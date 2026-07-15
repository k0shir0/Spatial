from __future__ import annotations

import json
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from local3d.shape_router import (
    DEFAULT_MODEL_PATH,
    MaskSample,
    ShapeRouterError,
    _draw_rounded_rectangle,
    _synthetic_clip,
    aggregate_clip_features,
    aggregate_frame_features,
    classify_shape,
    extract_frame_features,
    load_mask_sequence,
    load_model,
)


ROOT = Path(__file__).resolve().parents[1]


class ShapeRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model(DEFAULT_MODEL_PATH)

    def test_basic_silhouette_features_separate_circle_and_rectangle(self) -> None:
        circle = np.zeros((256, 256), np.uint8)
        cv2.circle(circle, (128, 128), 80, 255, cv2.FILLED)
        rectangle = np.zeros_like(circle)
        cv2.rectangle(rectangle, (38, 64), (218, 192), 255, cv2.FILLED)

        circle_features = extract_frame_features(circle).values
        rectangle_features = extract_frame_features(rectangle).values

        self.assertLess(circle_features["rectangularity"], 0.82)
        self.assertGreater(rectangle_features["rectangularity"], 0.98)
        self.assertGreater(circle_features["circularity"], rectangle_features["circularity"])
        self.assertEqual(circle_features["rounded_radius_ratio"], 0.5)
        self.assertEqual(rectangle_features["rounded_radius_ratio"], 0.0)

    def test_features_are_stable_under_rotation_and_scale(self) -> None:
        first = _draw_rounded_rectangle(116, 84, 0.14, 0, size=192)
        second = _draw_rounded_rectangle(92, 67, 0.14, 37, size=192)
        a = extract_frame_features(first).values
        b = extract_frame_features(second).values
        for key in ("rectangularity", "solidity", "rounded_rect_iou", "rounded_radius_ratio"):
            self.assertAlmostEqual(a[key], b[key], delta=0.08, msg=key)

    def test_repeated_face_views_abstain(self) -> None:
        mask = _draw_rounded_rectangle(118, 112, 0.12, 5, size=192)
        frames = [extract_frame_features(mask) for _ in range(12)]
        clip = aggregate_frame_features(frames)
        decision = classify_shape(clip, self.model, mask_provenance="generic_reviewed")
        self.assertEqual(decision["family"], "unknown")
        self.assertEqual(decision["decision"], "review")
        self.assertTrue(
            any(
                "view cluster" in reason or "orbit coverage" in reason
                for reason in decision["reject_reasons"]
            ),
            decision["reject_reasons"],
        )

    def test_synthetic_families_smoke_route(self) -> None:
        rng = np.random.default_rng(20260714)
        for expected in (
            "planar",
            "rounded_slab",
            "rectangular_prism",
            "cylinder",
            "bottle",
            "revolved",
            "free_form",
        ):
            masks = _synthetic_clip(expected, rng, corruption=0.12)
            clip = aggregate_frame_features([extract_frame_features(mask) for mask in masks])
            decision = classify_shape(clip, self.model, mask_provenance="generic_reviewed")
            self.assertEqual(decision["predicted_family"], expected)

    def test_portrait_phone_routes_to_rounded_slab_for_review(self) -> None:
        masks = []
        for yaw in np.linspace(0.0, math.pi, 18, endpoint=False):
            projected_width = abs(65.0 * math.cos(yaw)) + abs(8.0 * math.sin(yaw))
            masks.append(_draw_rounded_rectangle(projected_width, 140, 0.12, 7, size=192))
        clip = aggregate_frame_features([extract_frame_features(mask) for mask in masks])
        decision = classify_shape(clip, self.model, mask_provenance="generic_reviewed")
        self.assertEqual(decision["family"], "rounded_slab")
        self.assertTrue(decision["candidate_valid"])
        self.assertEqual(decision["decision"], "review")
        self.assertFalse(decision["auto_route_eligible"])

    def test_thin_sharp_book_routes_to_rectangular_prism_for_review(self) -> None:
        masks = []
        for yaw in np.linspace(0.0, math.pi, 20, endpoint=False):
            projected_width = abs(90.0 * math.cos(yaw)) + abs(8.0 * math.sin(yaw))
            masks.append(_draw_rounded_rectangle(projected_width, 140, 0.005, -3, size=192))
        clip = aggregate_frame_features([extract_frame_features(mask) for mask in masks])
        decision = classify_shape(clip, self.model, mask_provenance="generic_reviewed")
        self.assertEqual(decision["family"], "rectangular_prism")
        self.assertTrue(decision["candidate_valid"])
        self.assertEqual(decision["decision"], "review")
        self.assertLess(decision["evidence"]["estimated_thickness_ratio"], 0.15)

    def test_saved_model_rejects_invalid_normalization_scale(self) -> None:
        model = json.loads(DEFAULT_MODEL_PATH.read_text(encoding="utf-8"))
        model["normalization"]["scale"][0] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-model.json"
            path.write_text(json.dumps(model), encoding="utf-8")
            with self.assertRaisesRegex(ShapeRouterError, "scales must be positive"):
                load_model(path)

    def test_placeholder_segmentation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "masks.json"
            path.write_text(
                json.dumps({"schema_version": "1.0", "placeholder": True, "frames": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ShapeRouterError, "placeholder"):
                load_mask_sequence(segmentation_manifest_path=path)

    def test_canonical_manifest_verifies_a_real_binary_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mask_path = root / "mask.png"
            mask = _draw_rounded_rectangle(90, 70, 0.12, 0, size=128)
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            checksum = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            manifest_path = root / "masks.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "placeholder": False,
                        "frames": [
                            {
                                "candidate_index": 4,
                                "timestamp_s": 1.25,
                                "width": 128,
                                "height": 128,
                                "object_mask_path": "mask.png",
                                "object_mask_sha256": checksum,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            samples = load_mask_sequence(segmentation_manifest_path=manifest_path)
            clip = aggregate_clip_features(samples)
            self.assertEqual(len(samples), 1)
            self.assertEqual(clip.evidence["valid_masks"], 1)
            self.assertEqual(samples[0].candidate_index, 4)

    def test_low_segmentation_confidence_does_not_count_as_a_good_mask(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            mask_path = Path(directory) / "mask.png"
            mask = _draw_rounded_rectangle(90, 70, 0.12, 0, size=128)
            self.assertTrue(cv2.imwrite(str(mask_path), mask))
            samples = [MaskSample(mask_path, confidence=0.01) for _ in range(8)]
            with self.assertRaisesRegex(ShapeRouterError, "no valid masks"):
                aggregate_clip_features(samples)

    def test_significant_silhouette_hole_forces_review_unknown(self) -> None:
        ring = np.zeros((192, 192), np.uint8)
        cv2.circle(ring, (96, 96), 68, 255, cv2.FILLED)
        cv2.circle(ring, (96, 96), 30, 0, cv2.FILLED)
        frames = [extract_frame_features(ring) for _ in range(10)]
        clip = aggregate_frame_features(frames)
        decision = classify_shape(
            clip,
            self.model,
            mask_provenance="generic_reviewed",
            rotation_coverage_confirmed=True,
        )
        self.assertEqual(decision["family"], "unknown")
        self.assertFalse(decision["candidate_valid"])
        self.assertTrue(
            any("silhouette holes" in reason for reason in decision["reject_reasons"])
        )

    def test_mint_regression_routes_to_rounded_slab_without_dispatching_builder(self) -> None:
        masks = ROOT / "runs" / "mint-tin" / "masks-dense-cpu"
        analysis = ROOT / "runs" / "mint-tin" / "ingest" / "analysis.json"
        if not masks.is_dir() or not analysis.is_file():
            self.skipTest("mint-tin regression fixture is not present")
        samples = load_mask_sequence(analysis, masks_dir=masks)
        decision = classify_shape(
            aggregate_clip_features(samples),
            self.model,
            mask_provenance="object_specific",
        )
        self.assertEqual(decision["family"], "rounded_slab")
        self.assertEqual(decision["decision"], "review")
        self.assertTrue(decision["candidate_valid"])
        self.assertFalse(decision["auto_route_eligible"])
        self.assertFalse(decision["builder_supported"])
        self.assertGreaterEqual(decision["evidence"]["face_views"], 1)
        self.assertGreaterEqual(decision["evidence"]["edge_views"], 1)
        self.assertGreaterEqual(decision["evidence"]["bootstrap_agreement"], 0.75)


if __name__ == "__main__":
    unittest.main()
