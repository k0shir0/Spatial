from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local3d.ingest import (  # noqa: E402
    IngestConfig,
    IngestError,
    analyze_video,
    compute_frame_metrics,
    probe_video,
    write_analysis_json,
)


def _write_synthetic_video(path: Path, *, frame_count: int = 24, fps: float = 8.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (320, 240),
    )
    if not writer.isOpened():
        raise unittest.SkipTest("OpenCV build cannot create an MJPG test video")
    try:
        for index in range(frame_count):
            frame = np.full((240, 320, 3), 205, dtype=np.uint8)
            # Static background structure should not be mistaken for view change.
            cv2.line(frame, (0, 40), (319, 40), (170, 170, 170), 2)
            cv2.line(frame, (0, 200), (319, 200), (170, 170, 170), 2)
            center_x = 70 + round(index * 170 / max(1, frame_count - 1))
            color = (40 + index * 4, 150, 225 - index * 3)
            cv2.rectangle(frame, (center_x - 34, 82), (center_x + 34, 158), color, -1)
            cv2.circle(frame, (center_x, 120), 18, (20, 20, 20), 2)
            cv2.putText(frame, str(index), (center_x - 12, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (5, 5, 5), 1)
            if index == 9:  # One deliberately poor candidate.
                frame = cv2.GaussianBlur(frame, (31, 31), 8)
            writer.write(frame)
    finally:
        writer.release()


@unittest.skipUnless(shutil.which("ffprobe"), "ffprobe is required")
class VideoIngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.video = self.root / "clip with unicode   space.avi"
        _write_synthetic_video(self.video)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_probe_video_preserves_unicode_path_and_metadata(self) -> None:
        metadata = probe_video(self.video)
        self.assertEqual(metadata.source_path, str(self.video.resolve()))
        self.assertEqual((metadata.display_width, metadata.display_height), (320, 240))
        self.assertAlmostEqual(metadata.fps, 8.0, places=2)
        self.assertGreater(metadata.duration_s, 2.5)
        self.assertEqual(metadata.codec, "mjpeg")
        self.assertIn("aspect_ratio", metadata.to_dict())

    def test_analysis_extracts_frames_selects_diverse_keyframes_and_serializes(self) -> None:
        config = IngestConfig(
            sample_fps=4.0,
            max_candidates=20,
            keyframe_count=5,
            min_keyframe_gap_s=0.25,
            analysis_long_side=256,
        )
        analysis = analyze_video(self.video, self.root / "frames", config=config)

        self.assertGreaterEqual(len(analysis.frames), 10)
        self.assertEqual(len(analysis.keyframes), 5)
        timestamps = [frame.timestamp_s for frame in analysis.keyframes]
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertGreater(max(timestamps) - min(timestamps), 1.5)
        self.assertTrue(all(Path(frame.path).is_file() for frame in analysis.frames))
        self.assertTrue(all(frame.metrics.coverage_source == "temporal_change" for frame in analysis.frames))
        self.assertTrue(all(0.0 <= frame.metrics.quality_score <= 1.0 for frame in analysis.frames))

        artifact = write_analysis_json(analysis, self.root / "analysis.json")
        decoded = json.loads(artifact.read_text(encoding="utf-8"))
        self.assertEqual(decoded["schema_version"], "1.0")
        self.assertEqual(len(decoded["frames"]), len(analysis.frames))
        self.assertEqual(len(decoded["keyframes"]), 5)
        self.assertTrue(all(item["is_keyframe"] for item in decoded["keyframes"]))
        self.assertTrue(all(Path(item["path"]).is_absolute() for item in decoded["keyframes"]))

    def test_true_mask_is_used_for_subject_coverage(self) -> None:
        image = np.full((100, 200, 3), 120, dtype=np.uint8)
        # Probability masks from segmenters are often smaller float arrays.
        mask = np.zeros((50, 100), dtype=np.float32)
        mask[12:38, 25:75] = 0.75
        metrics = compute_frame_metrics(image, subject_mask=mask)
        self.assertEqual(metrics.coverage_source, "object_mask")
        self.assertAlmostEqual(metrics.subject_coverage, 0.26, places=2)
        self.assertGreater(metrics.central_subject_coverage, metrics.subject_coverage)
        self.assertEqual(metrics.subject_border_touch, 0.0)

    def test_invalid_input_and_config_fail_with_clear_errors(self) -> None:
        with self.assertRaises(IngestError):
            probe_video(self.root / "missing.mov")
        with self.assertRaises(ValueError):
            IngestConfig(sample_fps=0).validate()
        with self.assertRaises(ValueError):
            IngestConfig(image_format="webp").validate()


if __name__ == "__main__":
    unittest.main()
