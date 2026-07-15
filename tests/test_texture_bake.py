from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.local3d.gltf_texture import TexturedMeshPart, write_textured_glb
from src.local3d.soft_parts import ellipsoid_mesh, tube_mesh
from src.local3d.texture_bake import (
    BaseMeshPart,
    _encode_image,
    _gltf_uvs_to_usd,
    _map_face_uvs_to_atlas,
    build_material_maps,
    build_textured_parts,
    harmonize_projected_atlas,
    prepare_safe_mask,
)


def parse_glb(path: Path) -> tuple[dict, bytes, int]:
    payload = path.read_bytes()
    magic, version, total = struct.unpack_from("<4sII", payload, 0)
    if magic != b"glTF" or version != 2 or total != len(payload):
        raise AssertionError("invalid GLB header")
    json_length, kind = struct.unpack_from("<I4s", payload, 12)
    if kind != b"JSON":
        raise AssertionError("missing JSON chunk")
    document = json.loads(payload[20 : 20 + json_length])
    binary_start = 20 + json_length + 8
    return document, payload, binary_start


class TextureBakeTests(unittest.TestCase):
    def test_atlas_uvs_use_gltf_upper_left_image_origin(self) -> None:
        local = np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float64)
        mapped = _map_face_uvs_to_atlas(local, (32, 64, 128, 96), 256, 4)
        self.assertAlmostEqual(float(mapped[0, 0, 0]), 36.0 / 255.0)
        self.assertAlmostEqual(float(mapped[0, 0, 1]), 68.0 / 255.0)
        self.assertGreater(float(mapped[0, 2, 1]), float(mapped[0, 0, 1]))

    def test_usd_export_flips_gltf_texture_v(self) -> None:
        gltf_uvs = np.array([[0.2, 0.1], [0.4, 0.75]], dtype=np.float32)
        usd_uvs = _gltf_uvs_to_usd(gltf_uvs)
        self.assertTrue(np.allclose(usd_uvs, [[0.2, 0.9], [0.4, 0.25]]))
        self.assertTrue(np.allclose(gltf_uvs, [[0.2, 0.1], [0.4, 0.75]]))


    def test_safe_mask_fills_small_feature_hole_but_applies_exclusion(self) -> None:
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[8:56, 8:56] = 255
        mask[24:28, 24:28] = 0
        view = {
            "name": "synthetic",
            "max_hole_area_fraction": 0.01,
            "component_seed_normalized": [0.3, 0.3],
            "valid_polygons_normalized": [],
            "trusted_polygons_normalized": [],
            "exclude_polygons_normalized": [
                [(0.75, 0.0), (1.0, 0.0), (1.0, 1.0), (0.75, 1.0)]
            ],
            "erode_pixels": 0,
        }
        safe = prepare_safe_mask(mask, view)
        self.assertEqual(int(safe[25, 25]), 255)
        self.assertEqual(int(safe[30, 52]), 0)
        self.assertEqual(int(safe[2, 2]), 0)

    def test_soft_parts_expand_to_finite_uvs_and_tangent_frames(self) -> None:
        ellipsoid = ellipsoid_mesh([0, 0, 0], [1.0, 0.8, 0.6], rings=8, segments=12)
        tube = tube_mesh([[0, 0, 0], [0.2, 0.4, 0.1], [0.4, 0.7, 0.2]], 0.08, segments=8)
        parts = [
            BaseMeshPart(
                0,
                "body",
                {
                    "type": "ellipsoid",
                    "center": [0, 0, 0],
                    "radii": [1.0, 0.8, 0.6],
                    "rings": 8,
                    "segments": 12,
                },
                *ellipsoid,
                "fabric",
                None,
                0,
            ),
            BaseMeshPart(
                1,
                "stitch",
                {
                    "type": "tube",
                    "points": [[0, 0, 0], [0.2, 0.4, 0.1], [0.4, 0.7, 0.2]],
                    "radius": 0.08,
                    "segments": 8,
                },
                *tube,
                "thread",
                (60, 40, 45),
                len(ellipsoid[1]),
            ),
        ]
        textured = build_textured_parts(
            parts,
            [(0, 0, 128, 256), (128, 0, 128, 256)],
            256,
            4,
        )
        self.assertEqual(sum(len(part.faces) for part in textured), len(ellipsoid[1]) + len(tube[1]))
        for part in textured:
            self.assertTrue(np.isfinite(part.uvs).all())
            self.assertGreaterEqual(float(part.uvs.min()), 0.0)
            self.assertLessEqual(float(part.uvs.max()), 1.0)
            self.assertTrue(np.allclose(np.linalg.norm(part.tangents[:, :3], axis=1), 1.0, atol=2e-4))
            self.assertLess(
                float(np.max(np.abs(np.einsum("ij,ij->i", part.normals, part.tangents[:, :3])))),
                2e-4,
            )
            self.assertTrue(np.isin(part.tangents[:, 3], [-1.0, 1.0]).all())

    def test_glb_embeds_three_maps_and_references_them_from_every_material(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        part = TexturedMeshPart(
            name="triangle",
            vertices=vertices,
            faces=np.array([[0, 1, 2]], dtype=np.int32),
            normals=np.tile([0, 0, 1], (3, 1)).astype(np.float32),
            uvs=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
            tangents=np.tile([1, 0, 0, 1], (3, 1)).astype(np.float32),
            material_class="fabric",
        )
        base = np.full((8, 8, 3), (120, 150, 190), dtype=np.uint8)
        normal = np.full((8, 8, 3), (255, 128, 128), dtype=np.uint8)
        mr = np.empty((8, 8, 3), dtype=np.uint8)
        mr[:, :, 0] = 0
        mr[:, :, 1] = 230
        mr[:, :, 2] = 255
        base_payload = _encode_image(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, 90])
        normal_payload = _encode_image(".png", normal, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        mr_payload = _encode_image(".png", mr, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "textured.glb"
            write_textured_glb(
                output,
                [part],
                base_color=base_payload,
                base_color_mime="image/jpeg",
                normal_map=normal_payload,
                metallic_roughness=mr_payload,
            )
            document, payload, binary_start = parse_glb(output)
            attributes = document["meshes"][0]["primitives"][0]["attributes"]
            self.assertEqual(set(attributes), {"POSITION", "NORMAL", "TEXCOORD_0", "TANGENT"})
            self.assertEqual([item["mimeType"] for item in document["images"]], ["image/jpeg", "image/png", "image/png"])
            material = document["materials"][0]
            self.assertEqual(material["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0)
            self.assertEqual(material["normalTexture"]["index"], 1)
            self.assertEqual(material["pbrMetallicRoughness"]["metallicRoughnessTexture"]["index"], 2)
            self.assertEqual(material["pbrMetallicRoughness"]["metallicFactor"], 0.0)
            self.assertEqual(document["samplers"][0]["minFilter"], 9729)
            for image in document["images"]:
                view = document["bufferViews"][image["bufferView"]]
                self.assertNotIn("target", view)
                self.assertGreater(view["byteLength"], 20)
                start = binary_start + view["byteOffset"]
                self.assertEqual(len(payload[start : start + view["byteLength"]]), view["byteLength"])

    def test_harmonization_rejects_background_sentinel_and_feathers_edges(self) -> None:
        shape = (32, 32)
        fallback = np.full((*shape, 3), (120, 125, 168), dtype=np.float32)
        projected = np.full((*shape, 3), (116, 132, 176), dtype=np.float32)
        projected[15, 15] = (255, 255, 255)
        observed = np.full(shape, 255, dtype=np.uint8)
        selected = np.zeros(shape, dtype=np.int16)
        observed[[0, -1], :] = 0
        observed[:, [0, -1]] = 0
        selected[[0, -1], :] = -1
        selected[:, [0, -1]] = -1
        surface = np.full(shape, 255, dtype=np.uint8)
        part_map = np.zeros(shape, dtype=np.int16)
        part = BaseMeshPart(
            0,
            "body",
            {"type": "ellipsoid"},
            np.zeros((3, 3), dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int32),
            np.tile([0, 0, 1], (3, 1)).astype(np.float32),
            "fabric",
            None,
        )
        result, clean_observed, clean_selected, metrics = harmonize_projected_atlas(
            projected,
            fallback,
            observed,
            selected,
            surface,
            part_map,
            [part],
            {"fabric": {"source_color_max_distance": 50, "source_color_strength": 0.8}},
            feather_pixels=4,
        )
        self.assertEqual(int(clean_observed[15, 15]), 0)
        self.assertEqual(int(clean_selected[15, 15]), -1)
        self.assertTrue(np.array_equal(result[15, 15], fallback[15, 15]))
        self.assertEqual(metrics["source_pixels_rejected_as_color_outliers"], 1)
        self.assertLess(float(np.linalg.norm(result[0, 0] - fallback[0, 0])), 1e-5)
        self.assertGreater(float(np.linalg.norm(result[16, 16] - fallback[16, 16])), 0.0)

    def test_material_maps_are_repeatable_and_metallic_is_zero(self) -> None:
        shape = (64, 64)
        color = np.full((*shape, 3), (120, 125, 168), dtype=np.float32)
        observed = np.full(shape, 255, dtype=np.uint8)
        selected = np.zeros(shape, dtype=np.int16)
        surface = np.full(shape, 255, dtype=np.uint8)
        part_map = np.zeros(shape, dtype=np.int16)
        part = BaseMeshPart(
            0,
            "body",
            {"type": "ellipsoid"},
            np.zeros((3, 3), dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.int32),
            np.tile([0, 0, 1], (3, 1)).astype(np.float32),
            "fabric",
            None,
        )
        detail = np.arange(32 * 32, dtype=np.float32).reshape(32, 32) % 17
        config = {
            "materials": {
                "fabric": {
                    "kind": "fabric",
                    "fabric_detail": True,
                    "roughness": 0.94,
                    "roughness_min": 0.85,
                    "roughness_max": 0.98,
                }
            },
            "fabric": {
                "fiber_periods_px": [[7.0, 19.0], [13.0, -9.0]],
                "detail_tile_size": 32,
            },
        }
        first = build_material_maps(
            color,
            observed,
            selected,
            surface,
            part_map,
            [part],
            [],
            [detail],
            config,
        )
        second = build_material_maps(
            color,
            observed,
            selected,
            surface,
            part_map,
            [part],
            [],
            [detail],
            config,
        )
        for left, right in zip(first[:3], second[:3]):
            self.assertTrue(np.array_equal(left, right))
        self.assertEqual(int(first[2][:, :, 0].max()), 0)
        self.assertGreaterEqual(first[3]["roughness_min"], 0.85)
        self.assertLessEqual(first[3]["roughness_max"], 0.98)


if __name__ == "__main__":
    unittest.main()
