from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from local3d.backends import write_glb_material_parts, write_glb_mesh
from local3d.soft_parts import combine_parts, ellipsoid_mesh, superellipsoid_mesh, tube_mesh


def parse_glb(path: Path) -> tuple[dict, bytes]:
    payload = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or total != len(payload):
        raise AssertionError("invalid GLB header")
    json_length, kind = struct.unpack_from("<I4s", payload, 12)
    if kind != b"JSON":
        raise AssertionError("missing JSON chunk")
    return json.loads(payload[20:20 + json_length]), payload


class SoftPartsTests(unittest.TestCase):
    def test_colored_normalled_assembly_exports_valid_accessors(self):
        ellipsoid = ellipsoid_mesh([0, 0, 0], [1, 0.8, 0.6], rings=8, segments=12)
        tube = tube_mesh([[0, 0, 0], [0, 0.5, 0.4], [0.2, 0.8, 0.6]], 0.08, segments=8)
        vertices, faces, normals, colors = combine_parts([
            (*ellipsoid, [230, 170, 180, 255]),
            (*tube, [80, 40, 50, 255]),
        ])
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.isfinite(normals).all())
        self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-5))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "soft.glb"
            write_glb_mesh(
                path, vertices.tolist(), faces.tolist(), vertex_colors=colors.tolist(),
                normals=normals.tolist(), extras={"geometry": "inferred"},
            )
            document, payload = parse_glb(path)
            attributes = document["meshes"][0]["primitives"][0]["attributes"]
            self.assertEqual(set(attributes), {"POSITION", "COLOR_0", "NORMAL"})
            self.assertTrue(document["accessors"][attributes["COLOR_0"]]["normalized"])
            self.assertEqual(document["accessors"][attributes["NORMAL"]]["count"], len(vertices))
            binary_declared = document["buffers"][0]["byteLength"]
            json_length = struct.unpack_from("<I", payload, 12)[0]
            self.assertEqual(binary_declared, len(payload) - (12 + 8 + json_length + 8))

    def test_invalid_part_parameters_fail(self):
        with self.assertRaises(ValueError):
            ellipsoid_mesh([0, 0, 0], [1, 0, 0.5])
        with self.assertRaises(ValueError):
            tube_mesh([[0, 0, 0]], 0.1)
        with self.assertRaises(ValueError):
            superellipsoid_mesh([0, 0, 0], [1, 1, 1], vertical_exponent=0.05)

    def test_superellipsoid_is_a_finite_rounded_loaf(self):
        vertices, faces, normals = superellipsoid_mesh(
            [0, 0, 0], [0.6, 0.5, 1.0], vertical_exponent=0.62,
            horizontal_exponent=0.72, rings=12, segments=20,
        )
        self.assertTrue(np.isfinite(vertices).all())
        self.assertTrue(np.isfinite(normals).all())
        self.assertTrue(np.allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5))
        self.assertEqual(len(faces), 2 * 20 * (12 - 1))
        equator = vertices[np.argmin(np.abs(vertices[:, 1]))]
        self.assertGreater(abs(float(equator[0])) + abs(float(equator[2])), 0.55)

    def test_explicit_part_materials_for_preview_compatibility(self):
        first = ellipsoid_mesh([0, 0, 0], [1, 1, 1], rings=6, segments=8)
        second = ellipsoid_mesh([1, 0, 0], [0.2, 0.3, 0.4], rings=6, segments=8)
        parts = [
            (first[0].tolist(), first[1].tolist(), first[2].tolist(), [240, 180, 190, 255]),
            (second[0].tolist(), second[1].tolist(), second[2].tolist(), [80, 40, 50, 255]),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "materials.glb"
            write_glb_material_parts(path, parts)
            document, _ = parse_glb(path)
            self.assertEqual(len(document["materials"]), 2)
            self.assertEqual(len(document["meshes"][0]["primitives"]), 2)
            self.assertEqual(document["meshes"][0]["primitives"][1]["material"], 1)
            self.assertNotIn("COLOR_0", document["meshes"][0]["primitives"][0]["attributes"])


if __name__ == "__main__":
    unittest.main()
