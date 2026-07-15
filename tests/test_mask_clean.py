"""Synthetic-data tests for deterministic object-mask clean-up."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from local3d.mask_clean import clean_mask, clean_mask_sequence, erode_for_sfm

HEIGHT, WIDTH = 480, 640

# Ellipse "object": centred, large; rows 70-290, cols 170-470.
ELLIPSE_CENTER = (320, 180)  # (col, row)
ELLIPSE_AXES = (150, 110)    # (a, b)

# Thin "arm" entering from the bottom border into the object.
ARM_COLS = (314, 326)   # width 12 px (< opening kernel -> severable)
ARM_TOP_ROW = 260

# Skin-coloured finger patch straddling the right edge of the ellipse.
FINGER = (458, 168, 482, 192)  # (x0, y0, x1, y1)
SKIN_BGR = (120, 140, 200)     # verified inside the YCrCb gate below.

# Wide "arm" for the depth tests: 140 px across, far too fat for the opening
# kernel to sever, so ONLY depth pruning can remove it.  It enters from the
# bottom border and overlaps the ellipse's lower edge (rows 260-289) so the two
# are one connected blob.
WIDE_ARM_COLS = (250, 390)
WIDE_ARM_TOP_ROW = 260


def _blank_mask() -> np.ndarray:
    return np.zeros((HEIGHT, WIDTH), dtype=np.uint8)


def object_mask() -> np.ndarray:
    """Ellipse object + thin bottom-border arm, as a 0/255 uint8 mask."""

    mask = _blank_mask()
    cv2.ellipse(mask, ELLIPSE_CENTER, ELLIPSE_AXES, 0, 0, 360, 255, -1)
    mask[ARM_TOP_ROW:HEIGHT, ARM_COLS[0]:ARM_COLS[1]] = 255
    return mask


def object_mask_with_finger() -> np.ndarray:
    mask = object_mask()
    x0, y0, x1, y1 = FINGER
    mask[y0:y1, x0:x1] = 255
    return mask


def base_image(object_color: tuple[int, int, int]) -> np.ndarray:
    """BGR frame: solid background, object region painted ``object_color``."""

    image = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)
    region = _blank_mask()
    cv2.ellipse(region, ELLIPSE_CENTER, ELLIPSE_AXES, 0, 0, 360, 255, -1)
    region[ARM_TOP_ROW:HEIGHT, ARM_COLS[0]:ARM_COLS[1]] = 255
    image[region > 0] = object_color
    return image


def image_with_skin_finger() -> np.ndarray:
    """Non-skin object with a skin-coloured finger patch at the edge."""

    image = base_image((200, 200, 200))  # light grey object (not skin)
    x0, y0, x1, y1 = FINGER
    image[y0:y1, x0:x1] = SKIN_BGR
    return image


def _ellipse_only() -> np.ndarray:
    mask = _blank_mask()
    cv2.ellipse(mask, ELLIPSE_CENTER, ELLIPSE_AXES, 0, 0, 360, 255, -1)
    return mask > 0


def wide_arm_mask() -> np.ndarray:
    """Ellipse object + a fat bottom-border arm (unseverable by morphology)."""

    mask = _blank_mask()
    cv2.ellipse(mask, ELLIPSE_CENTER, ELLIPSE_AXES, 0, 0, 360, 255, -1)
    mask[WIDE_ARM_TOP_ROW:HEIGHT, WIDE_ARM_COLS[0]:WIDE_ARM_COLS[1]] = 255
    return mask


def arm_disparity(
    *, ellipse_val: float = 0.8, arm_top: float = 0.95, arm_bot: float = 0.30,
    background: float = 0.0,
) -> np.ndarray:
    """Disparity for :func:`wide_arm_mask`.

    The ellipse sits at a compact ``ellipse_val`` (nearest); the arm ramps from
    ``arm_top`` where it meets the ellipse down to ``arm_bot`` at the border
    (sloping away, exactly the held-object-plus-forearm depth signature); the
    background is far.  Larger = nearer, float32, per the module contract.
    """

    disp = np.full((HEIGHT, WIDTH), background, dtype=np.float32)
    arm = np.zeros((HEIGHT, WIDTH), dtype=bool)
    arm[WIDE_ARM_TOP_ROW:HEIGHT, WIDE_ARM_COLS[0]:WIDE_ARM_COLS[1]] = True
    rows = np.arange(HEIGHT, dtype=np.float32)
    frac = np.clip((rows - WIDE_ARM_TOP_ROW) / (HEIGHT - 1 - WIDE_ARM_TOP_ROW), 0.0, 1.0)
    ramp = arm_top + (arm_bot - arm_top) * frac
    disp[arm] = np.repeat(ramp[:, None], WIDTH, axis=1)[arm]
    ellipse = _blank_mask()
    cv2.ellipse(ellipse, ELLIPSE_CENTER, ELLIPSE_AXES, 0, 0, 360, 255, -1)
    disp[ellipse > 0] = ellipse_val
    return disp


def uniform_disparity(value: float = 0.5) -> np.ndarray:
    return np.full((HEIGHT, WIDTH), value, dtype=np.float32)


def _ellipse_interior() -> np.ndarray:
    ellipse = _ellipse_only()
    return cv2.erode(ellipse.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0


class SkinGateSanityTests(unittest.TestCase):
    def test_chosen_bgr_is_inside_the_ycrcb_gate(self) -> None:
        pixel = np.array([[list(SKIN_BGR)]], dtype=np.uint8)
        ycrcb = cv2.cvtColor(pixel, cv2.COLOR_BGR2YCrCb)[0, 0]
        _, cr, cb = int(ycrcb[0]), int(ycrcb[1]), int(ycrcb[2])
        self.assertTrue(133 <= cr <= 173, f"Cr={cr} outside gate")
        self.assertTrue(77 <= cb <= 127, f"Cb={cb} outside gate")


class CleanMaskTests(unittest.TestCase):
    def test_arm_removed_from_bottom_rows(self) -> None:
        tight, report = clean_mask(object_mask())
        bottom = tight[int(HEIGHT * 0.85):, :]
        self.assertEqual(int(bottom.sum()), 0, "arm survived into bottom 15% rows")
        self.assertIn("arm_pruned", report["flags"])

    def test_object_interior_preserved(self) -> None:
        tight, _ = clean_mask(object_mask())
        ellipse = _ellipse_only()
        # Interior = pixels well inside the ellipse (erode to drop the boundary band).
        interior = cv2.erode(ellipse.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        kept = float((tight & interior).sum()) / float(interior.sum())
        self.assertGreater(kept, 0.90, f"only {kept:.2%} of interior kept")

    def test_skin_finger_removed_when_object_not_skin(self) -> None:
        x0, y0, x1, y1 = FINGER
        without_image, _ = clean_mask(object_mask_with_finger())
        with_image, report = clean_mask(
            object_mask_with_finger(), image_with_skin_finger()
        )
        # Without colour info the finger patch is retained.
        self.assertGreater(int(without_image[y0:y1, x0:x1].sum()), 0)
        # With colour info the skin finger is suppressed.
        self.assertEqual(int(with_image[y0:y1, x0:x1].sum()), 0)
        self.assertNotIn("skin_suppression_skipped", report["flags"])

    def test_skin_object_skips_suppression(self) -> None:
        # Whole object painted skin colour -> suppression must self-disable.
        skin_image = base_image(SKIN_BGR)
        tight, report = clean_mask(object_mask(), skin_image)
        self.assertIn("skin_suppression_skipped", report["flags"])
        ellipse = _ellipse_only()
        interior = cv2.erode(ellipse.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        kept = float((tight & interior).sum()) / float(interior.sum())
        self.assertGreater(kept, 0.90, "skin object was eaten by suppression")

    def test_empty_mask_is_handled(self) -> None:
        tight, report = clean_mask(_blank_mask())
        self.assertEqual(int(tight.sum()), 0)
        self.assertEqual(report["coverage"], 0.0)
        self.assertEqual(report["components"], 0)

    def test_report_fields_present(self) -> None:
        _, report = clean_mask(object_mask())
        for key in ("coverage", "border_touch", "removed_fraction", "components", "flags"):
            self.assertIn(key, report)
        self.assertGreater(report["removed_fraction"], 0.0)

    def test_determinism_identical_bytes(self) -> None:
        first, _ = clean_mask(object_mask_with_finger(), image_with_skin_finger())
        second, _ = clean_mask(object_mask_with_finger(), image_with_skin_finger())
        self.assertEqual(first.tobytes(), second.tobytes())


class ErodeForSfmTests(unittest.TestCase):
    def test_erosion_strictly_shrinks_and_is_subset(self) -> None:
        tight, _ = clean_mask(object_mask())
        eroded = erode_for_sfm(tight)
        self.assertLess(int(eroded.sum()), int(tight.sum()))
        self.assertEqual(int((eroded & ~tight).sum()), 0, "erosion added pixels")
        self.assertEqual(eroded.dtype, bool)

    def test_erosion_of_empty_is_empty(self) -> None:
        self.assertEqual(int(erode_for_sfm(_blank_mask() > 0).sum()), 0)

    def test_erosion_is_deterministic(self) -> None:
        tight, _ = clean_mask(object_mask())
        self.assertEqual(erode_for_sfm(tight).tobytes(), erode_for_sfm(tight).tobytes())


class SequenceTests(unittest.TestCase):
    def test_sequence_flags_area_outlier_without_dropping(self) -> None:
        masks = [object_mask() for _ in range(9)]
        # Frame 4 loses most of its area (a re-grip): shrink the ellipse.
        small = _blank_mask()
        cv2.ellipse(small, ELLIPSE_CENTER, (60, 45), 0, 0, 360, 255, -1)
        masks[4] = small
        tight_masks, eroded_masks, report = clean_mask_sequence(masks)
        self.assertEqual(len(tight_masks), 9)
        self.assertEqual(len(eroded_masks), 9)
        self.assertIn(4, report["area_outlier_frames"])
        self.assertTrue(report["regrip_outlier"])
        self.assertIn("regrip_outlier", report["flags"])
        self.assertIn("area_outlier", report["frames"][4]["flags"])
        # The outlier frame is flagged, not dropped.
        self.assertGreater(int(tight_masks[4].sum()), 0)

    def test_uniform_sequence_has_no_outliers(self) -> None:
        masks = [object_mask() for _ in range(5)]
        _, _, report = clean_mask_sequence(masks)
        self.assertEqual(report["area_outlier_frames"], [])
        self.assertFalse(report["regrip_outlier"])

    def test_sequence_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            clean_mask_sequence([object_mask()], [base_image((200, 200, 200))] * 2)

    def test_sequence_with_images_runs_suppression(self) -> None:
        masks = [object_mask_with_finger() for _ in range(3)]
        images = [image_with_skin_finger() for _ in range(3)]
        tight_masks, _, report = clean_mask_sequence(masks, images)
        x0, y0, x1, y1 = FINGER
        self.assertEqual(int(tight_masks[0][y0:y1, x0:x1].sum()), 0)
        self.assertEqual(report["frame_count"], 3)


class DepthPruneTests(unittest.TestCase):
    def test_wide_arm_survives_without_disparity(self) -> None:
        # Morphology alone cannot sever a 140 px arm: it is still there, and no
        # depth flags appear (the depth pass never runs).
        tight, report = clean_mask(wide_arm_mask())
        self.assertGreater(
            int(tight[int(HEIGHT * 0.9):, :].sum()), 0, "wide arm should survive"
        )
        self.assertNotIn("depth_pruned", report["flags"])
        self.assertNotIn("depth_prune_skipped", report["flags"])
        self.assertEqual(report["depth_pruned_fraction"], 0.0)

    def test_depth_prune_removes_arm_keeps_ellipse(self) -> None:
        tight, report = clean_mask(wide_arm_mask(), disparity=arm_disparity())
        interior = _ellipse_interior()
        kept = float((tight & interior).sum()) / float(interior.sum())
        self.assertGreater(kept, 0.98, f"ellipse interior not preserved ({kept:.2%})")
        # The far (border-touching) portion of the arm is severed.
        self.assertEqual(
            int(tight[HEIGHT - 15:, :].sum()), 0, "far arm survived depth pruning"
        )
        self.assertIn("depth_pruned", report["flags"])
        self.assertNotIn("depth_prune_skipped", report["flags"])
        self.assertGreater(report["depth_pruned_fraction"], 0.03)

    def test_uniform_depth_triggers_failsafe_and_no_change(self) -> None:
        baseline, _ = clean_mask(wide_arm_mask())
        tight, report = clean_mask(wide_arm_mask(), disparity=uniform_disparity(0.5))
        self.assertIn("depth_prune_skipped", report["flags"])
        self.assertNotIn("depth_pruned", report["flags"])
        self.assertEqual(report["depth_pruned_fraction"], 0.0)
        # Fail-safe returns the un-depth-pruned mask untouched, byte for byte.
        self.assertEqual(tight.tobytes(), baseline.tobytes())

    def test_compact_object_preserved_by_depth(self) -> None:
        # A compact object with a mild internal depth gradient and no arm (the
        # tin case) must be kept essentially whole — depth pruning does no harm.
        ellipse = _blank_mask()
        cv2.ellipse(ellipse, ELLIPSE_CENTER, ELLIPSE_AXES, 0, 0, 360, 255, -1)
        disp = np.full((HEIGHT, WIDTH), 0.0, dtype=np.float32)
        rows = np.arange(HEIGHT, dtype=np.float32)
        grad = np.clip(0.75 + 0.10 * (rows - 70) / 220.0, 0.75, 0.85).astype(np.float32)
        mask_bool = ellipse > 0
        disp[mask_bool] = np.repeat(grad[:, None], WIDTH, axis=1)[mask_bool]
        tight, report = clean_mask(ellipse, disparity=disp)
        kept = float(int(tight.sum())) / float(int(mask_bool.sum()))
        self.assertGreater(kept, 0.90, f"compact object eaten by depth ({kept:.2%})")

    def test_depth_prune_is_deterministic(self) -> None:
        disp = arm_disparity()
        first, _ = clean_mask(wide_arm_mask(), disparity=disp)
        second, _ = clean_mask(wide_arm_mask(), disparity=disp)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_depth_pruned_fraction_always_reported(self) -> None:
        _, no_disp = clean_mask(wide_arm_mask())
        _, with_disp = clean_mask(wide_arm_mask(), disparity=arm_disparity())
        _, empty = clean_mask(_blank_mask(), disparity=arm_disparity())
        for report in (no_disp, with_disp, empty):
            self.assertIn("depth_pruned_fraction", report)

    def test_image_and_disparity_together(self) -> None:
        # Positional image + keyword disparity still both take effect.
        image = base_image((200, 200, 200))
        tight, report = clean_mask(
            wide_arm_mask(), image, disparity=arm_disparity()
        )
        self.assertEqual(int(tight[HEIGHT - 15:, :].sum()), 0)
        self.assertIn("depth_pruned", report["flags"])


class SequenceDepthTests(unittest.TestCase):
    def test_sequence_applies_disparities(self) -> None:
        masks = [wide_arm_mask() for _ in range(3)]
        disparities = [arm_disparity() for _ in range(3)]
        tight_masks, eroded_masks, report = clean_mask_sequence(
            masks, disparities=disparities
        )
        self.assertEqual(len(tight_masks), 3)
        self.assertEqual(len(eroded_masks), 3)
        for index in range(3):
            self.assertEqual(int(tight_masks[index][HEIGHT - 15:, :].sum()), 0)
            self.assertIn("depth_pruned", report["frames"][index]["flags"])
            self.assertGreater(report["frames"][index]["depth_pruned_fraction"], 0.03)

    def test_sequence_without_disparities_unchanged(self) -> None:
        masks = [wide_arm_mask() for _ in range(3)]
        tight_masks, _, report = clean_mask_sequence(masks)
        for index in range(3):
            self.assertGreater(int(tight_masks[index][int(HEIGHT * 0.9):, :].sum()), 0)
            self.assertNotIn("depth_pruned", report["frames"][index]["flags"])

    def test_sequence_disparity_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            clean_mask_sequence(
                [wide_arm_mask()], disparities=[arm_disparity(), arm_disparity()]
            )


if __name__ == "__main__":
    unittest.main()
