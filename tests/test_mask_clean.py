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


if __name__ == "__main__":
    unittest.main()
