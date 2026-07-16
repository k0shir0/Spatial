"""Tests for the pure-Python deterministic USDZ packer."""

from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from local3d.usdz_pack import (
    pack_usdz,
    package_usdz,
    referenced_assets,
    verify_usdz_layout,
)

MINIMAL_USDA = """#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Root" (
    kind = "component"
)
{
    def Scope "Looks"
    {
        def Material "skin"
        {
            token outputs:surface.connect = </Root/Looks/skin/PreviewSurface.outputs:surface>
            def Shader "PreviewSurface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Root/Looks/skin/Texture.outputs:rgb>
                token outputs:surface
            }
            def Shader "Texture"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @texture.png@
                float3 outputs:rgb
            }
        }
    }
}
"""

# Placeholder texture payload: the packer treats assets as opaque bytes, so
# these tests do not need a decodable image.
TINY_PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(48))


class UsdzPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.usda = self.root / "asset.usda"
        self.usda.write_text(MINIMAL_USDA, encoding="utf-8")
        self.texture = self.root / "texture.png"
        self.texture.write_bytes(TINY_PNG)

    def test_referenced_assets_finds_texture(self) -> None:
        self.assertEqual(referenced_assets(self.usda), [self.texture])

    def test_referenced_assets_rejects_missing_file(self) -> None:
        self.texture.unlink()
        with self.assertRaises(FileNotFoundError):
            referenced_assets(self.usda)

    def test_referenced_assets_rejects_escaping_paths(self) -> None:
        bad = self.root / "bad.usda"
        bad.write_text('#usda 1.0\ndef "X" { asset inputs:file = @../secret.png@ }\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            referenced_assets(bad)
        bad.write_text('#usda 1.0\ndef "X" { asset inputs:file = @/etc/hosts@ }\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            referenced_assets(bad)

    def test_package_layout_and_readability(self) -> None:
        usdz = self.root / "asset.usdz"
        report = package_usdz(self.usda, usdz)
        self.assertTrue(report["created"])
        self.assertEqual(report["archive_entries"], ["asset.usda", "texture.png"])
        layout = report["layout"]
        self.assertTrue(layout["passed"], layout["problems"])
        for entry in layout["entries"]:
            self.assertEqual(entry["data_offset"] % 64, 0, entry)
            self.assertTrue(entry["stored"], entry)
            self.assertTrue(entry["epoch_timestamp"], entry)
        # The archive must also be readable by an ordinary ZIP implementation.
        with zipfile.ZipFile(usdz) as archive:
            self.assertEqual(archive.namelist(), ["asset.usda", "texture.png"])
            self.assertIsNone(archive.testzip())
            self.assertEqual(archive.read("texture.png"), TINY_PNG)
            self.assertEqual(
                archive.read("asset.usda").decode("utf-8"),
                MINIMAL_USDA,
            )

    def test_pack_is_byte_deterministic(self) -> None:
        first = self.root / "first.usdz"
        second = self.root / "second.usdz"
        package_usdz(self.usda, first)
        package_usdz(self.usda, second)
        self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_first_entry_must_be_usd_layer(self) -> None:
        usdz = self.root / "asset.usdz"
        with self.assertRaises(ValueError):
            pack_usdz(usdz, [self.texture, self.usda])

    def test_duplicate_entry_names_rejected(self) -> None:
        nested = self.root / "nested"
        nested.mkdir()
        clone = nested / "texture.png"
        clone.write_bytes(TINY_PNG)
        with self.assertRaises(ValueError):
            pack_usdz(self.root / "asset.usdz", [self.usda, self.texture, clone])

    def test_alignment_survives_varied_name_lengths(self) -> None:
        # Sweep name lengths so padding hits every remainder class, including
        # the pad < 4 branch that must add a full alignment block.
        for length in range(1, 70):
            usda = self.root / f"{'a' * length}.usda"
            usda.write_text(MINIMAL_USDA, encoding="utf-8")
            usdz = self.root / f"out{length}.usdz"
            pack_usdz(usdz, [usda, self.texture])
            layout = verify_usdz_layout(usdz)
            self.assertTrue(layout["passed"], (length, layout["problems"]))

    def test_verify_flags_apple_style_wall_clock_timestamps(self) -> None:
        usdz = self.root / "bad.usdz"
        with zipfile.ZipFile(usdz, "w", zipfile.ZIP_STORED) as archive:
            info = zipfile.ZipInfo("asset.usda", date_time=(2026, 7, 14, 12, 0, 0))
            archive.writestr(info, MINIMAL_USDA)
        layout = verify_usdz_layout(usdz)
        self.assertFalse(layout["passed"])
        self.assertTrue(any("timestamp" in problem or "aligned" in problem for problem in layout["problems"]))


if __name__ == "__main__":
    unittest.main()
