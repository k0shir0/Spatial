from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_usda_mesh", ROOT / "scripts" / "render_usda_mesh.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ReconstructionGateTests(unittest.TestCase):
    def base_analysis(self):
        return {
            "connected_components": 1,
            "boundary_edges": 0,
            "nonmanifold_edges": 0,
            "triangles": 500,
            "dimensions_m": {"x": 0.1, "y": 0.08, "z": 0.03},
        }

    def test_closed_manifold_can_advance_but_scale_stays_unverified(self):
        result = MODULE.reconstruction_gate(self.base_analysis())
        self.assertTrue(result["may_advance_to_higher_detail"])
        self.assertEqual(result["status"], "needs_visual_and_reprojection_review")
        self.assertIn("metric_scale_unverified_without_depth_marker_or_measurement", result["warnings"])

    def test_open_folded_preview_fails_closed(self):
        analysis = self.base_analysis()
        analysis["boundary_edges"] = 30
        result = MODULE.reconstruction_gate(analysis)
        self.assertFalse(result["may_advance_to_higher_detail"])
        self.assertEqual(result["status"], "needs_recapture")
        self.assertIn("mesh_has_open_boundary_edges", result["failures"])


if __name__ == "__main__":
    unittest.main()
