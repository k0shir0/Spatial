from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from local3d.backends import (
    FullFrameSegmentationBackend,
    PlaceholderReconstructionBackend,
    ensure_glb,
    write_obj_mesh,
)
from local3d.core import (
    ArtifactSpec,
    ArtifactValidationError,
    JobLockedError,
    JobStatus,
    LocalJobStore,
    LocalPipelineRunner,
    PipelineStage,
    StageResult,
    StageStatus,
)


def _source_video(tmp_path: Path) -> Path:
    path = tmp_path / "input video.mov"
    path.write_bytes(b"not decoded by core")
    return path


def _parse_glb(path: Path) -> dict:
    payload = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<4sII", payload, 0)
    assert magic == b"glTF"
    assert version == 2
    assert total_length == len(payload)
    json_length, chunk_type = struct.unpack_from("<I4s", payload, 12)
    assert chunk_type == b"JSON"
    return json.loads(payload[20 : 20 + json_length].decode("utf-8"))


def test_manifest_round_trip_and_safe_job_ids(tmp_path: Path) -> None:
    store = LocalJobStore(tmp_path / "jobs")
    manifest = store.create_job(
        _source_video(tmp_path),
        job_id="capture-001",
        settings={"mode": "accurate", "checkpoint": Path("models/local")},
    )

    loaded = store.load("capture-001")
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.settings["checkpoint"] == "models/local"
    assert loaded.status is JobStatus.CREATED
    assert set(loaded.stages) == set(PipelineStage)

    with pytest.raises(ValueError):
        store.job_dir("../escape")


def test_fallback_pipeline_is_resumable_and_emits_valid_labelled_glb(tmp_path: Path) -> None:
    store = LocalJobStore(tmp_path / "jobs")
    manifest = store.create_job(_source_video(tmp_path), job_id="fallback")
    job_dir = store.job_dir(manifest.job_id)
    frame = job_dir / "frame 0001.jpg"
    frame.write_bytes(b"frame bytes are not decoded by fallback")
    analysis = job_dir / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "metadata": {"display_width": 8, "display_height": 6},
                "keyframes": [
                    {
                        "candidate_index": 1,
                        "source_frame_index": 10,
                        "timestamp_s": 0.5,
                        "path": str(frame),
                        "width": 8,
                        "height": 6,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    store.complete_external_stage(
        manifest.job_id,
        PipelineStage.INGEST,
        [ArtifactSpec("ingest.analysis", analysis, "application/json")],
        backend="test-ingest",
    )

    runner = LocalPipelineRunner(
        store,
        segmentation_backend=FullFrameSegmentationBackend(),
        reconstruction_backend=PlaceholderReconstructionBackend(),
    )
    completed = runner.run(manifest.job_id)
    assert completed.status is JobStatus.COMPLETED
    assert completed.stages[PipelineStage.SEGMENTATION].status is StageStatus.COMPLETED
    assert completed.stages[PipelineStage.RECONSTRUCTION].status is StageStatus.COMPLETED
    assert completed.stages[PipelineStage.SEGMENTATION].attempts == 1
    assert completed.stages[PipelineStage.RECONSTRUCTION].attempts == 1
    assert all(record.validate(job_dir) for record in completed.artifacts.values())

    masks = json.loads(
        (job_dir / completed.artifacts["segmentation.manifest"].path).read_text(encoding="utf-8")
    )
    assert masks["placeholder"] is True
    assert len(masks["frames"]) == 1
    assert Path(masks["frames"][0]["object_mask_path"]).read_bytes().startswith(b"P4\n8 6\n")

    glb_path = job_dir / completed.artifacts["reconstruction.model.glb"].path
    gltf = _parse_glb(glb_path)
    assert gltf["asset"]["extras"]["placeholder"] is True
    assert gltf["meshes"][0]["primitives"][0]["mode"] == 4

    resumed = runner.run(manifest.job_id)
    assert resumed.stages[PipelineStage.SEGMENTATION].attempts == 1
    assert resumed.stages[PipelineStage.RECONSTRUCTION].attempts == 1

    forced = runner.run(manifest.job_id, force=True)
    assert forced.stages[PipelineStage.SEGMENTATION].attempts == 2
    assert forced.stages[PipelineStage.RECONSTRUCTION].attempts == 2


def test_runner_rejects_artifacts_outside_job_directory(tmp_path: Path) -> None:
    class UnsafeSegmenter:
        name = "unsafe-test"

        def is_available(self) -> bool:
            return True

        def segment(self, context):
            outside = context.job_dir.parent / "outside.json"
            outside.write_text("{}", encoding="utf-8")
            return StageResult.of([ArtifactSpec("bad", outside, "application/json")])

    store = LocalJobStore(tmp_path / "jobs")
    manifest = store.create_job(_source_video(tmp_path), job_id="unsafe")
    runner = LocalPipelineRunner(
        store,
        segmentation_backend=UnsafeSegmenter(),
        reconstruction_backend=PlaceholderReconstructionBackend(),
    )

    with pytest.raises(ArtifactValidationError):
        runner.run(manifest.job_id)
    failed = store.load(manifest.job_id)
    assert failed.status is JobStatus.FAILED
    assert failed.stages[PipelineStage.SEGMENTATION].status is StageStatus.FAILED
    assert "outside the job directory" in (failed.error or "")


def test_job_lock_prevents_concurrent_runner(tmp_path: Path) -> None:
    store = LocalJobStore(tmp_path / "jobs")
    manifest = store.create_job(_source_video(tmp_path), job_id="locked")
    with store.lock(manifest.job_id):
        with pytest.raises(JobLockedError):
            with store.lock(manifest.job_id):
                pass


def test_dependency_free_obj_to_glb_conversion(tmp_path: Path) -> None:
    obj = write_obj_mesh(
        tmp_path / "triangle.obj",
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        [(0, 1, 2)],
    )
    glb = ensure_glb(obj, tmp_path / "triangle.glb")
    document = _parse_glb(glb)
    assert document["accessors"][0]["count"] == 3
    assert document["accessors"][1]["count"] == 3
