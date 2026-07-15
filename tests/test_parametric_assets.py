from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import trimesh


from local3d.parametric_assets import (
    MAX_TEXTURE_SIZE,
    build_asset,
    load_config,
    rectify_face,
)


class ParametricAssetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="parametric-assets-test-")
        self.root = Path(self.temporary.name)
        self.front = self.root / "front.jpg"
        self.back = self.root / "back.jpg"
        height, width = 260, 180
        grid_x = np.tile(np.arange(width, dtype=np.uint8), (height, 1))
        grid_y = np.tile(np.arange(height, dtype=np.uint16)[:, None], (1, width)).astype(np.uint8)
        front = np.dstack((grid_x, grid_y, np.full((height, width), 180, dtype=np.uint8)))
        back = np.dstack((np.full((height, width), 70, dtype=np.uint8), grid_x, grid_y))
        cv2.putText(front, "FRONT", (38, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (5, 5, 5), 2)
        cv2.putText(back, "BACK", (47, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (5, 5, 5), 2)
        self.assertTrue(cv2.imwrite(str(self.front), front))
        self.assertTrue(cv2.imwrite(str(self.back), back))
        self.quad = [[22, 16], [157, 20], [151, 242], [27, 239]]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, payload: dict[str, object], name: str = "config.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def phone_config(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "asset_name": "Test Phone",
            "kind": "phone",
            "output_name": "test_phone",
            "front": {"image": self.front.name, "quad_px": self.quad},
            "back": {"image": self.back.name, "quad_px": self.quad},
            "dimensions_mm": {
                "width": 70.0,
                "height": 145.0,
                "depth": 8.0,
                "corner_radius": 8.0,
                "bevel": 0.8,
            },
            "texture_size": 256,
            "output_rotation_deg": [0, 0, 0],
            "materials": {
                "body": {"color_rgb": [44, 46, 50], "metallic": 0.4, "roughness": 0.3},
                "camera_bump": [35, 37, 41],
                "lens": [12, 17, 22],
            },
            "phone": {
                "camera_bump": {
                    "side": "back",
                    "center_mm": [-19, 45],
                    "size_mm": [26, 28],
                    "protrusion_mm": 2.0,
                    "corner_radius_mm": 4.5,
                    "lenses": [
                        {
                            "center_mm": [-5, 5],
                            "relative_to": "bump",
                            "radius_mm": 4.5,
                            "protrusion_mm": 1.2,
                            "segments": 16,
                        }
                    ],
                }
            },
        }

    def book_config(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "asset_name": "Test Book",
            "kind": "book",
            "output_name": "test_book",
            "front": {"image": str(self.front), "quad_px": self.quad},
            "back": {"image": str(self.back), "quad_px": self.quad},
            "dimensions_mm": {"width": 150, "height": 220, "depth": 22},
            "texture_size": 256,
            "output_rotation_deg": [0, 0, 90],
            "materials": {
                "cover": {"color_rgb": [85, 30, 24], "roughness": 0.6},
                "pages": [235, 225, 198],
                "spine": [75, 24, 20],
            },
            "book": {"spine_side": "left"},
        }

    def test_rectification_preserves_configured_face_aspect(self) -> None:
        texture, metrics = rectify_face(
            self.front,
            self.quad,
            (124, 256),
            (30, 30, 30),
        )
        self.assertEqual(texture.shape, (256, 124, 3))
        self.assertEqual(metrics["texture_dimensions_px"], [124, 256])
        self.assertFalse(metrics["automatic_detection"])
        self.assertGreater(metrics["quad_area_px"], 20_000)

    def test_rectification_supports_explicit_clockwise_quarter_turn(self) -> None:
        texture, metrics = rectify_face(
            self.front,
            self.quad,
            (124, 256),
            (30, 30, 30),
            rotate_quarter_turns=1,
        )
        self.assertEqual(texture.shape, (256, 124, 3))
        self.assertEqual(metrics["rotate_quarter_turns_clockwise"], 1)

    def test_phone_end_to_end_is_bounded_and_watertight(self) -> None:
        config = self.write_config(self.phone_config())
        output = self.root / "phone-output"
        manifest = build_asset(config, output, allow_usdz=False)
        topology = manifest["geometry"]["topology"]
        self.assertTrue(topology["watertight"])
        self.assertTrue(topology["winding_consistent"])
        self.assertLess(topology["triangles"], 1000)
        self.assertGreaterEqual(topology["body_count"], 3)
        self.assertFalse(manifest["execution"]["learned_inference"])
        self.assertFalse(manifest["execution"]["network_access"])
        self.assertFalse(manifest["execution"]["torch"])
        self.assertFalse(manifest["execution"]["gpu"])
        for name in (
            "test_phone.glb",
            "test_phone.usda",
            "front_rectified.png",
            "back_rectified.png",
            "qa_model_contact.png",
            "qa_texture_contact.png",
            "manifest.json",
        ):
            self.assertTrue((output / name).is_file(), name)
        scene = trimesh.load(output / "test_phone.glb", force="scene")
        self.assertGreater(len(scene.geometry), 0)
        front = cv2.imread(str(output / "front_rectified.png"))
        self.assertLessEqual(max(front.shape[:2]), MAX_TEXTURE_SIZE)

    def test_view_texture_mode_accepts_source_and_material(self) -> None:
        payload = self.phone_config()
        payload["front"]["texture_mode"] = "source"  # type: ignore[index]
        payload["back"]["texture_mode"] = "material"  # type: ignore[index]
        normalized = load_config(self.write_config(payload))
        self.assertEqual(normalized["front"]["texture_mode"], "source")
        self.assertEqual(normalized["back"]["texture_mode"], "material")

        payload["back"]["texture_mode"] = "invented"  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "texture_mode"):
            load_config(self.write_config(payload, "invalid-texture-mode.json"))

    def test_material_mode_back_exports_without_source_texture(self) -> None:
        payload = self.phone_config()
        payload["back"]["texture_mode"] = "material"  # type: ignore[index]
        payload["materials"]["back"] = {  # type: ignore[index]
            "color_rgb": [61, 89, 104],
            "metallic": 0.0,
            "roughness": 0.58,
        }
        output = self.root / "material-back-output"
        build_asset(self.write_config(payload), output, allow_usdz=False)

        scene = trimesh.load(output / "test_phone.glb", force="scene")
        front = scene.geometry["ObservedFront"]
        back = scene.geometry["ObservedBack"]
        self.assertIsNotNone(getattr(front.visual.material, "baseColorTexture", None))
        self.assertIsNone(getattr(back.visual.material, "baseColorTexture", None))
        self.assertEqual(
            list(np.asarray(back.visual.material.baseColorFactor, dtype=np.uint8)),
            [61, 89, 104, 255],
        )
        usda = (output / "test_phone.usda").read_text(encoding="utf-8")
        self.assertIn('def Material "back"', usda)
        self.assertNotIn("@back_rectified.png@", usda)

    def test_phone_decorations_are_watertight_closed_primitives(self) -> None:
        payload = self.phone_config()
        payload["phone"] = {
            "camera_bump": None,
            "decorations": [
                {
                    "name": "TestRing",
                    "type": "annulus",
                    "side": "back",
                    "center_mm": [0.0, -8.0],
                    "radius_mm": 9.0,
                    "inner_radius_mm": 7.0,
                    "offset_mm": 0.1,
                    "protrusion_mm": 0.45,
                    "segments": 20,
                    "material": "magsafe",
                },
                {
                    "name": "TestFlash",
                    "type": "cylinder",
                    "side": "back",
                    "center_mm": [20.0, 42.0],
                    "radius_mm": 2.2,
                    "offset_mm": 0.1,
                    "protrusion_mm": 0.35,
                    "segments": 16,
                    "material": "flash",
                },
                {
                    "name": "TestBezel",
                    "type": "rounded_prism",
                    "side": "front",
                    "center_mm": [0.0, 27.0],
                    "size_mm": [15.0, 6.0],
                    "corner_radius_mm": 2.5,
                    "offset_mm": 0.1,
                    "protrusion_mm": 0.4,
                    "material": "button",
                },
                {
                    "name": "TestSideControl",
                    "type": "box",
                    "center_mm": [-35.2, -18.0, 0.0],
                    "size_mm": [1.2, 13.0, 3.2],
                    "material": "button",
                },
            ],
        }
        output = self.root / "decorated-phone-output"
        manifest = build_asset(self.write_config(payload), output, allow_usdz=False)
        topology = manifest["geometry"]["topology"]
        self.assertTrue(topology["watertight"])
        self.assertTrue(topology["winding_consistent"])
        self.assertLess(topology["triangles"], 5000)

        scene = trimesh.load(output / "test_phone.glb", force="scene")
        for prefix in ("TestRing", "TestFlash", "TestBezel", "TestSideControl"):
            pieces = [
                geometry
                for name, geometry in scene.geometry.items()
                if name.startswith(prefix)
            ]
            self.assertGreater(len(pieces), 0, prefix)
            joined = trimesh.util.concatenate(pieces)
            joined.merge_vertices(digits_vertex=8)
            joined.remove_unreferenced_vertices()
            self.assertTrue(joined.is_watertight, prefix)
            self.assertTrue(joined.is_winding_consistent, prefix)

    def test_phone_rejects_annulus_with_invalid_inner_radius(self) -> None:
        payload = self.phone_config()
        payload["phone"] = {
            "camera_bump": None,
            "decorations": [
                {
                    "name": "InvalidRing",
                    "type": "annulus",
                    "side": "back",
                    "center_mm": [0.0, 0.0],
                    "radius_mm": 8.0,
                    "inner_radius_mm": 8.0,
                    "offset_mm": 0.1,
                    "protrusion_mm": 0.4,
                    "segments": 20,
                    "material": "magsafe",
                }
            ],
        }
        with self.assertRaises(ValueError):
            load_config(self.write_config(payload, "invalid-annulus.json"))

    def test_book_has_cover_page_and_spine_materials(self) -> None:
        config = self.write_config(self.book_config(), "book.json")
        output = self.root / "book-output"
        manifest = build_asset(config, output, allow_usdz=False)
        topology = manifest["geometry"]["topology"]
        self.assertTrue(topology["watertight"])
        self.assertTrue(topology["winding_consistent"])
        self.assertEqual(topology["body_count"], 1)
        # A 90-degree Z rotation swaps configured X and Y extents.
        self.assertAlmostEqual(topology["extents_m"][0], 0.22, places=6)
        self.assertAlmostEqual(topology["extents_m"][1], 0.15, places=6)
        usda = (output / "test_book.usda").read_text(encoding="utf-8")
        self.assertIn('def Material "front_cover"', usda)
        self.assertIn('def Material "back_cover"', usda)
        self.assertIn('def Material "pages"', usda)
        self.assertIn('def Material "spine"', usda)
        self.assertIn('def Mesh "Spine"', usda)

    def test_rejects_oversized_texture(self) -> None:
        payload = self.phone_config()
        payload["texture_size"] = MAX_TEXTURE_SIZE + 1
        config = self.write_config(payload)
        with self.assertRaisesRegex(ValueError, "texture_size"):
            load_config(config)

    def test_rejects_quad_outside_source(self) -> None:
        bad_quad = [[-20, 10], [150, 10], [150, 240], [10, 240]]
        with self.assertRaisesRegex(ValueError, "outside source image bounds"):
            rectify_face(self.front, bad_quad, (128, 256), (0, 0, 0))

    def test_manifest_checksums_every_artifact(self) -> None:
        config = self.write_config(self.book_config())
        output = self.root / "checksums"
        manifest = build_asset(config, output, allow_usdz=False)
        for name, record in manifest["artifacts"].items():
            self.assertEqual(len(record["sha256"]), 64, name)
            self.assertEqual(record["bytes"], (output / name).stat().st_size)

    def test_manifest_is_repeatable_without_wall_clock_metadata(self) -> None:
        config = self.write_config(self.phone_config())
        first = build_asset(config, self.root / "repeat-a", allow_usdz=False)
        second = build_asset(config, self.root / "repeat-b", allow_usdz=False)
        self.assertIsNone(first["created_utc"])
        self.assertEqual(first, second)
        self.assertEqual(
            (self.root / "repeat-a" / "test_phone.glb").read_bytes(),
            (self.root / "repeat-b" / "test_phone.glb").read_bytes(),
        )

    @unittest.skipUnless(
        Path("/usr/bin/usdcat").is_file() and Path("/usr/bin/usdzip").is_file(),
        "Apple USD command-line tools are unavailable",
    )
    def test_apple_usdz_package_contains_both_textures(self) -> None:
        config = self.write_config(self.book_config())
        output = self.root / "usdz"
        manifest = build_asset(config, output, allow_usdz=True)
        self.assertTrue(manifest["usd_package"]["created"])
        self.assertTrue((output / "test_book.usdz").is_file())
        if Path("/usr/bin/usdchecker").is_file():
            self.assertTrue(manifest["usd_package"]["validation"]["passed"])
        listing = "\n".join(manifest["usd_package"]["archive_entries"])
        self.assertIn("front_rectified.png", listing)
        self.assertIn("back_rectified.png", listing)

    @unittest.skipUnless(
        Path("/usr/bin/usdcat").is_file() and Path("/usr/bin/usdzip").is_file(),
        "Apple USD command-line tools are unavailable",
    )
    def test_apple_usdz_package_is_byte_repeatable(self) -> None:
        config = self.write_config(self.phone_config())
        first = self.root / "usdz-repeat-a"
        second = self.root / "usdz-repeat-b"
        first_manifest = build_asset(config, first, allow_usdz=True)
        second_manifest = build_asset(config, second, allow_usdz=True)
        self.assertEqual(
            (first / "test_phone.usdz").read_bytes(),
            (second / "test_phone.usdz").read_bytes(),
        )
        self.assertEqual(first_manifest, second_manifest)


if __name__ == "__main__":
    unittest.main()
