"""Tests for optional mesh post-processing (requires the ``mesh`` extra)."""

from __future__ import annotations

import unittest

import numpy as np
import trimesh

try:
    import fast_simplification  # noqa: F401
    import manifold3d  # noqa: F401

    HAVE_MESH_EXTRA = True
except ImportError:  # pragma: no cover
    HAVE_MESH_EXTRA = False

if HAVE_MESH_EXTRA:
    from local3d.mesh_post import decimate, mesh_report, postprocess, weld_to_manifold


@unittest.skipUnless(HAVE_MESH_EXTRA, "mesh extra (fast-simplification, manifold3d) not installed")
class MeshPostTests(unittest.TestCase):
    def dense_sphere(self) -> trimesh.Trimesh:
        return trimesh.creation.icosphere(subdivisions=4)

    def test_decimate_reaches_target_and_stays_watertight(self) -> None:
        sphere = self.dense_sphere()
        vertices, faces, report = postprocess(
            np.asarray(sphere.vertices), np.asarray(sphere.faces), target_triangles=500
        )
        self.assertLessEqual(report["after"]["triangles"], 520)
        self.assertTrue(report["after"]["watertight"])
        self.assertLess(report["after"]["triangles"], report["before"]["triangles"])
        self.assertEqual(vertices.dtype, np.float32)
        self.assertEqual(faces.dtype, np.int32)

    def test_decimate_is_deterministic(self) -> None:
        sphere = self.dense_sphere()
        first = decimate(np.asarray(sphere.vertices), np.asarray(sphere.faces), target_triangles=400)
        second = decimate(np.asarray(sphere.vertices), np.asarray(sphere.faces), target_triangles=400)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])

    def test_decimate_noop_below_target(self) -> None:
        box = trimesh.creation.box()
        vertices, faces = decimate(np.asarray(box.vertices), np.asarray(box.faces), target_triangles=100)
        self.assertEqual(len(faces), len(box.faces))

    def test_weld_recloses_unwelded_mesh(self) -> None:
        sphere = trimesh.creation.icosphere(subdivisions=2)
        # Split every face into isolated triangles: topologically open.
        vertices = np.asarray(sphere.vertices)[np.asarray(sphere.faces)].reshape(-1, 3)
        faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
        self.assertFalse(mesh_report(vertices, faces)["watertight"])
        welded_vertices, welded_faces = weld_to_manifold(vertices, faces)
        self.assertTrue(mesh_report(welded_vertices, welded_faces)["watertight"])

    def test_postprocess_fails_closed_on_unrepairable_mesh(self) -> None:
        # A single triangle has open edges that welding cannot close.
        vertices = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        faces = np.asarray([[0, 1, 2]])
        with self.assertRaises(ValueError):
            postprocess(vertices, faces)


if __name__ == "__main__":
    unittest.main()
