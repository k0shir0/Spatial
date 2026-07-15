from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh

from local3d.auto_soft import (
    AutoSoftError,
    SoftViewCandidate,
    _assess_pair,
    _build_volume_evidence,
    _mesh_from_volume,
    _normalize_view,
    _projected_parts,
    _select_views,
    _validate_exported_soft_glb,
)
from local3d.parametric_assets import Material, export_glb


class AutoSoftHelperTests(unittest.TestCase):
    """Synthetic source-gate tests; no exporter or model builder is invoked."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="auto-soft-test-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_pair(self, *, touches_border: bool = False) -> tuple[Path, Path]:
        height, width = 240, 320
        frame = np.full((height, width, 3), (42, 58, 76), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (0, 120) if touches_border else (160, 120)
        cv2.ellipse(mask, center, (62, 82), 0, 0, 360, 255, -1)
        frame[mask > 0] = (70, 105, 190)
        for y in range(55, 190, 12):
            cv2.line(frame, (105, y), (215, y), (235, 225, 40), 3, cv2.LINE_AA)
        frame_path = self.root / "frame_000001.jpg"
        mask_path = self.root / "frame_000001.png"
        self.assertTrue(cv2.imwrite(str(frame_path), frame))
        self.assertTrue(cv2.imwrite(str(mask_path), mask))
        return frame_path, mask_path

    def test_source_assessment_accepts_isolated_detailed_silhouette(self) -> None:
        frame, mask = self._write_pair()
        assessment = _assess_pair(frame, mask, 0)
        self.assertEqual(assessment.reasons, ())
        self.assertIsNotNone(assessment.candidate)
        self.assertGreater(assessment.measurements["margin_fraction"], 0.01)
        self.assertGreater(assessment.measurements["internal_sharpness"], 8.0)

    def test_source_assessment_rejects_frame_border_contact(self) -> None:
        frame, mask = self._write_pair(touches_border=True)
        assessment = _assess_pair(frame, mask, 0)
        self.assertIsNone(assessment.candidate)
        self.assertIn("mask_touches_frame_border", assessment.reasons)
        self.assertIn("silhouette_too_close_to_frame_border", assessment.reasons)

    def test_source_assessment_does_not_count_outer_edge_as_detail(self) -> None:
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        mask = np.zeros((240, 320), dtype=np.uint8)
        cv2.ellipse(mask, (160, 120), (62, 82), 0, 0, 360, 255, -1)
        frame[mask > 0] = (230, 230, 230)
        frame_path = self.root / "frame_000002.png"
        mask_path = self.root / "frame_000002_mask.png"
        self.assertTrue(cv2.imwrite(str(frame_path), frame))
        self.assertTrue(cv2.imwrite(str(mask_path), mask))
        assessment = _assess_pair(frame_path, mask_path, 0)
        self.assertIsNone(assessment.candidate)
        self.assertIn("insufficient_detail", assessment.reasons)
        self.assertLess(assessment.measurements["internal_sharpness"], 1.0)

    def test_source_assessment_accepts_smooth_but_tonally_detailed_plush(self) -> None:
        height, width = 240, 320
        frame = np.full((height, width, 3), (225, 225, 225), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (160, 120), (72, 88), 0, 0, 360, 255, -1)
        vertical = np.linspace(-24.0, 24.0, height, dtype=np.float32)[:, None, None]
        plush = np.clip(
            np.asarray((155, 175, 220), dtype=np.float32)[None, None, :] + vertical,
            0,
            255,
        ).astype(np.uint8)
        plush = np.broadcast_to(plush, frame.shape)
        frame[mask > 0] = plush[mask > 0]
        cv2.circle(frame, (137, 103), 10, (105, 120, 175), -1, cv2.LINE_AA)
        cv2.circle(frame, (183, 103), 10, (105, 120, 175), -1, cv2.LINE_AA)
        cv2.ellipse(frame, (160, 140), (18, 12), 0, 0, 360, (120, 135, 195), -1)
        frame = cv2.GaussianBlur(frame, (0, 0), 2.0)
        frame_path = self.root / "frame_smooth.png"
        mask_path = self.root / "frame_smooth_mask.png"
        self.assertTrue(cv2.imwrite(str(frame_path), frame))
        self.assertTrue(cv2.imwrite(str(mask_path), mask))

        assessment = _assess_pair(frame_path, mask_path, 0)

        self.assertEqual(assessment.reasons, ())
        self.assertIsNotNone(assessment.candidate)
        self.assertTrue(
            assessment.measurements["internal_sharpness"] > 1.0
            or assessment.measurements["interior_entropy_bits"] > 1.7
        )
        self.assertGreaterEqual(assessment.measurements["multiscale_detail"], 2.5)
        self.assertGreaterEqual(assessment.measurements["marking_score"], 0.8)

    def test_smooth_exposure_gradients_cannot_become_primary_evidence(self) -> None:
        height, width = 240, 320
        yy, xx = np.mgrid[:height, :width].astype(np.float32)
        fields = {
            "linear": (xx - width * 0.5) / width,
            "radial": np.sqrt(
                ((xx - width * 0.5) / width) ** 2
                + ((yy - height * 0.5) / height) ** 2
            ),
            "exposure": ((xx - width * 0.5) / width) + 0.4 * (
                (yy - height * 0.5) / height
            ),
        }
        for index, (name, field) in enumerate(fields.items()):
            with self.subTest(name=name):
                frame = np.full((height, width, 3), 225, dtype=np.uint8)
                mask = np.zeros((height, width), dtype=np.uint8)
                cv2.ellipse(mask, (160, 120), (72, 88), 0, 0, 360, 255, -1)
                values = np.clip(165.0 + field * 28.0, 0, 255).astype(np.uint8)
                for channel, offset in enumerate((0, 8, 25)):
                    plane = np.clip(values.astype(np.int16) + offset, 0, 255).astype(np.uint8)
                    frame[..., channel][mask > 0] = plane[mask > 0]
                frame_path = self.root / f"frame_gradient_{index}.png"
                mask_path = self.root / f"frame_gradient_{index}_mask.png"
                self.assertTrue(cv2.imwrite(str(frame_path), frame))
                self.assertTrue(cv2.imwrite(str(mask_path), mask))

                assessment = _assess_pair(frame_path, mask_path, index)

                self.assertIsNotNone(assessment.support_candidate)
                self.assertLess(assessment.measurements["multiscale_detail"], 2.5)
                self.assertLess(assessment.measurements["marking_score"], 0.55)

    def test_normalization_replaces_background_without_removing_object(self) -> None:
        frame = np.full((120, 160, 3), (5, 10, 15), dtype=np.uint8)
        mask = np.zeros((120, 160), dtype=np.uint8)
        cv2.rectangle(mask, (45, 20), (115, 100), 1, -1)
        frame[mask > 0] = (70, 90, 130)
        cleaned, normalized_mask, aspect = _normalize_view(
            frame, mask, (45, 20, 71, 81), size=128
        )
        self.assertEqual(cleaned.shape, (128, 128, 3))
        self.assertEqual(normalized_mask.shape, (128, 128))
        self.assertGreater(aspect, 0.5)
        self.assertTrue(np.all(cleaned[normalized_mask == 0] == (70, 90, 130)))

    def test_normalization_flat_fills_occluder_hole_instead_of_copying_it(self) -> None:
        frame = np.full((160, 200, 3), (15, 20, 25), dtype=np.uint8)
        mask = np.zeros((160, 200), dtype=np.uint8)
        cv2.ellipse(mask, (100, 80), (65, 55), 0, 0, 360, 1, -1)
        frame[mask > 0] = (90, 130, 190)
        cv2.circle(mask, (100, 80), 16, 0, -1)
        cv2.circle(frame, (100, 80), 16, (20, 230, 20), -1)

        cleaned, normalized_mask, _aspect = _normalize_view(
            frame, mask, (35, 25, 131, 111), size=160
        )

        hole = normalized_mask == 0
        self.assertFalse(np.any(np.all(cleaned[hole] == (20, 230, 20), axis=1)))
        self.assertTrue(np.any(np.all(cleaned[hole] == (90, 130, 190), axis=1)))

    @staticmethod
    def _candidate(
        index: int,
        score: float,
        feature: np.ndarray,
        *,
        wide: bool = False,
        marking_score: float = 1.0,
        timestamp_ms: int | None = None,
        solidity: float = 0.9,
        extent: float = 0.7,
        multiscale_detail: float | None = None,
    ) -> SoftViewCandidate:
        mask = np.zeros((64, 64), dtype=np.uint8)
        axes = (25, 15) if wide else (18, 25)
        cv2.ellipse(mask, (32, 32), axes, 0, 0, 360, 1, -1)
        return SoftViewCandidate(
            frame=Path(f"frame_{index:06d}.jpg"),
            mask_path=Path(f"frame_{index:06d}.png"),
            sequence_index=index,
            frame_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
            mask=mask,
            bbox_xywh=(7, 7, 50, 50),
            coverage=0.25,
            component_fraction=1.0,
            solidity=solidity,
            extent=extent,
            margin_fraction=0.1,
            border_fraction=0.0,
            sharpness=100.0,
            detail_density=0.2,
            interior_entropy_bits=4.0,
            possible_skin_fraction=0.0,
            score=score,
            feature=feature.astype(np.float32),
            normalized_silhouette=mask,
            capture_time_ms=index * 500 if timestamp_ms is None else timestamp_ms,
            crop_aspect=1.6 if wide else 0.75,
            multiscale_detail=(
                multiscale_detail
                if multiscale_detail is not None
                else (4.0 if marking_score >= 0.8 else 1.0)
            ),
            marking_fraction=0.12 if marking_score >= 0.8 else 0.01,
            marking_score=marking_score,
            rotation_features=(feature.astype(np.float32),) * 4,
        )

    def test_selection_uses_supported_families_not_dissimilar_outlier(self) -> None:
        primary_feature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        secondary_feature = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        candidates = [
            self._candidate(0, 1.0, primary_feature),
            self._candidate(1, 0.7, primary_feature),
            self._candidate(2, 0.7, primary_feature),
            self._candidate(
                3, 0.7, secondary_feature, marking_score=0.1, timestamp_ms=5_000
            ),
            self._candidate(
                4, 0.75, secondary_feature, marking_score=0.1, timestamp_ms=5_500
            ),
            self._candidate(
                5, 0.8, secondary_feature, marking_score=0.1, timestamp_ms=6_000
            ),
            # A very dissimilar but structurally damaged view is not part of a
            # supported temporal family and cannot win secondary selection.
            self._candidate(
                6,
                0.99,
                np.asarray([0.0, 0.0, 1.0], np.float32),
                marking_score=0.1,
                timestamp_ms=12_000,
                solidity=0.51,
                extent=0.39,
            ),
        ]
        selection = _select_views(candidates)
        self.assertEqual(selection.primary.sequence_index, 0)
        self.assertEqual(selection.secondary.sequence_index, 5)
        self.assertGreater(selection.combined_dissimilarity, 0.5)
        self.assertEqual(selection.primary_family_support, 3)
        self.assertEqual(selection.secondary_family_support, 3)

    def test_selection_rejects_unsupported_single_appearance_family(self) -> None:
        feature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        candidates = [self._candidate(index, 0.8, feature) for index in range(6)]

        with self.assertRaisesRegex(AutoSoftError, "two supported"):
            _select_views(candidates)

    def test_distance_prior_is_bounded_and_fails_on_thin_evidence(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(mask, (64, 64), (35, 48), 0, 0, 360, 1, -1)
        evidence = _build_volume_evidence(mask, 0.8, resolution=80)
        self.assertGreaterEqual(evidence.inferred_depth_to_height, 0.24)
        self.assertLessEqual(evidence.inferred_depth_to_height, 0.62)
        self.assertGreater(int(evidence.occupancy.sum()), 4_000)

        thin = np.zeros((128, 128), dtype=np.uint8)
        cv2.line(thin, (10, 64), (118, 64), 1, 1)
        with self.assertRaises(AutoSoftError):
            _build_volume_evidence(thin, 1.0, resolution=80)

    def test_forced_soft_mesh_decimation_preserves_closed_topology(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(mask, (64, 64), (40, 50), 0, 0, 360, 1, -1)
        evidence = _build_volume_evidence(mask, 0.85, resolution=80)

        vertices, faces = _mesh_from_volume(evidence, max_triangles=5_000)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        self.assertLessEqual(len(faces), 5_000)
        self.assertTrue(mesh.is_watertight)
        self.assertTrue(mesh.is_winding_consistent)
        self.assertEqual(mesh.body_count, 1)
        self.assertEqual(mesh.euler_number, 2)

    def test_exported_glb_reloads_as_closed_position_welded_textured_body(self) -> None:
        mask = np.zeros((128, 128), dtype=np.uint8)
        cv2.ellipse(mask, (64, 64), (38, 49), 0, 0, 360, 1, -1)
        evidence = _build_volume_evidence(mask, 0.82, resolution=72)
        vertices, faces = _mesh_from_volume(evidence, max_triangles=4_000)
        atlas = np.zeros((128, 256, 3), dtype=np.uint8)
        atlas[:, :128] = (35, 80, 210)
        atlas[:, 128:] = (190, 120, 45)
        atlas_path = self.root / "atlas.png"
        self.assertTrue(cv2.imwrite(str(atlas_path), atlas))
        material = Material(
            "soft_surface",
            color_rgb=(180, 120, 90),
            metallic=0.0,
            roughness=0.88,
        )
        parts = _projected_parts(vertices, faces, evidence, atlas, material)
        glb_path = self.root / "soft.glb"
        export_glb(parts, glb_path, {"soft_atlas": atlas_path}, "soft_model")
        expected = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        validation = _validate_exported_soft_glb(
            glb_path,
            expected_triangles=len(faces),
            expected_vertices_after_position_merge=len(expected.vertices),
            expected_material="soft_surface",
        )

        self.assertTrue(validation["quality_gate_passed"])
        welded = validation["position_welded"]
        self.assertTrue(welded["watertight"])
        self.assertTrue(welded["winding_consistent"])
        self.assertEqual(welded["body_count"], 1)
        self.assertEqual(welded["euler_number"], 2)
        self.assertEqual(welded["triangles"], len(faces))
        primitives = {
            record["geometry"]: record
            for record in validation["projection_primitives"]
        }
        self.assertEqual(
            set(primitives), {"SoftPrimaryProjection", "SoftSecondaryProjection"}
        )
        self.assertTrue(all(record["embedded_texture"] for record in primitives.values()))
        self.assertTrue(all(record["material"] == "soft_surface" for record in primitives.values()))
        self.assertLess(primitives["SoftPrimaryProjection"]["uv_bounds"][1][0], 0.5)
        self.assertGreater(primitives["SoftSecondaryProjection"]["uv_bounds"][0][0], 0.5)
        self.assertFalse(
            validation["attribute_indexed_before_position_merge"]["watertight"]
        )


if __name__ == "__main__":
    unittest.main()
