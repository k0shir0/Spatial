"""Focused tests for the hybrid reconstruction script's delivery helpers.

These tests deliberately exercise the orchestration boundary rather than the
individual reconstruction backends.  In particular, contradictory candidate
state must fail closed and promotion must never leave a partial or overwritten
stable delivery directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spatial_reconstruct_object", ROOT / "scripts" / "reconstruct_object.py"
)
assert SPEC is not None and SPEC.loader is not None
RECONSTRUCT_OBJECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECONSTRUCT_OBJECT)

import local3d.auto_parametric as auto_parametric  # noqa: E402
import local3d.auto_soft as auto_soft  # noqa: E402


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _prior(name: str, *, ok: bool = True, artifact: Path | None = None) -> dict:
    candidate = {
        "ok": ok,
        "classification": f"explicit {name} fallback classification",
    }
    if artifact is not None:
        candidate["artifacts"] = {"glb": artifact}
    return candidate


def test_selection_prefers_a_built_general_candidate_that_passed_its_gate() -> None:
    general = {
        "ok": True,
        "evidence_gate": {"pass": True, "reasons": []},
        "artifacts": {"glb": Path("general.glb")},
    }
    priors = {
        "rounded_slab": _prior("rounded slab"),
        "soft_volume": _prior("soft volume"),
    }

    selected = RECONSTRUCT_OBJECT.select_candidate(general, priors)

    assert selected["name"] == "general_reconstruction"
    assert selected["candidate"] is general
    assert "source-supported" in selected["classification"]


def test_selection_falls_back_in_narrowest_prior_order() -> None:
    rejected_general = {
        "ok": True,
        "evidence_gate": {"pass": False, "reasons": ["weak_source_support"]},
    }
    slab = _prior("rounded slab")
    soft = _prior("soft volume")

    selected = RECONSTRUCT_OBJECT.select_candidate(
        rejected_general, {"rounded_slab": slab, "soft_volume": soft}
    )
    assert selected["name"] == "rounded_slab"
    assert selected["candidate"] is slab
    assert selected["classification"] == slab["classification"]

    selected_without_slab = RECONSTRUCT_OBJECT.select_candidate(
        rejected_general,
        {"rounded_slab": _prior("rounded slab", ok=False), "soft_volume": soft},
    )
    assert selected_without_slab["name"] == "soft_volume"
    assert selected_without_slab["candidate"] is soft
    assert selected_without_slab["classification"] == soft["classification"]


def test_selection_rejects_contradictory_general_state_fail_closed() -> None:
    """A passing nested gate cannot revive a candidate whose build failed."""

    contradictory_general = {
        "ok": False,
        "error": "RuntimeError: export failed after evidence assessment",
        "evidence_gate": {"pass": True, "reasons": []},
    }
    slab = _prior("rounded slab")

    selected = RECONSTRUCT_OBJECT.select_candidate(
        contradictory_general, {"rounded_slab": slab}
    )

    assert selected["name"] == "rounded_slab"
    assert selected["candidate"] is slab


def test_selection_requires_recapture_when_every_candidate_failed() -> None:
    general = {
        "ok": False,
        "error": "RuntimeError: no coherent SfM model",
        "evidence_gate": {
            "pass": False,
            "reasons": ["degenerate_camera_sweep", "weak_source_support"],
        },
    }
    priors = {
        "rounded_slab": {
            "ok": False,
            "error": "AutoFitError: not slab-like",
            "classification": "rounded slab",
        },
        "soft_volume": {
            "ok": False,
            "error": "AutoSoftError: profile evidence missing",
            "classification": "soft volume",
        },
    }

    with pytest.raises(SystemExit, match="needs_recapture") as caught:
        RECONSTRUCT_OBJECT.select_candidate(general, priors)

    message = str(caught.value)
    assert "degenerate_camera_sweep" in message
    assert "RuntimeError: no coherent SfM model" in message
    assert "rounded_slab: AutoFitError: not slab-like" in message
    assert "soft_volume: AutoSoftError: profile evidence missing" in message


def test_promote_candidate_uses_stable_names_and_content_hashes(tmp_path: Path) -> None:
    source = tmp_path / "candidate"
    source.mkdir()
    payloads = {
        "glb": b"synthetic glb payload",
        "texture": b"synthetic texture payload",
        "qa_model": b"synthetic qa payload",
    }
    artifacts = {}
    for key, payload in payloads.items():
        path = source / f"source-{key}.bin"
        path.write_bytes(payload)
        artifacts[key] = path
    selection = {"candidate": {"artifacts": artifacts}}
    delivery = tmp_path / "delivery"

    promoted = RECONSTRUCT_OBJECT.promote_candidate(selection, delivery)

    expected_names = {
        "glb": "model.glb",
        "texture": "texture.png",
        "qa_model": "qa_model.png",
    }
    for key, destination_name in expected_names.items():
        destination = delivery / destination_name
        assert destination.read_bytes() == payloads[key]
        assert promoted[key] == str(destination)
        assert promoted[f"{key}_bytes"] == len(payloads[key])
        assert promoted[f"{key}_sha256"] == _sha256(payloads[key])

    # Promotion is a copy: later candidate mutation cannot alter the delivery.
    artifacts["glb"].write_bytes(b"changed after promotion")
    assert (delivery / "model.glb").read_bytes() == payloads["glb"]


def test_promote_candidate_refuses_to_overwrite_an_existing_delivery(tmp_path: Path) -> None:
    source_glb = tmp_path / "source.glb"
    source_glb.write_bytes(b"new candidate")
    delivery = tmp_path / "delivery"
    delivery.mkdir()
    existing = delivery / "model.glb"
    existing.write_bytes(b"keep this delivery")

    with pytest.raises(FileExistsError):
        RECONSTRUCT_OBJECT.promote_candidate(
            {"candidate": {"artifacts": {"glb": source_glb}}}, delivery
        )

    assert existing.read_bytes() == b"keep this delivery"
    assert sorted(path.name for path in delivery.iterdir()) == ["model.glb"]


def test_promote_candidate_missing_glb_leaves_no_partial_delivery(tmp_path: Path) -> None:
    """The helper promises atomic publication, including failure cleanup."""

    texture = tmp_path / "candidate-texture.png"
    texture.write_bytes(b"texture without a mesh")
    delivery = tmp_path / "delivery"

    with pytest.raises(SystemExit, match="did not produce a GLB"):
        RECONSTRUCT_OBJECT.promote_candidate(
            {"candidate": {"artifacts": {"texture": texture}}}, delivery
        )

    assert not delivery.exists()


def _write_textured_icosphere(path: Path) -> int:
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    radii = np.linalg.norm(vertices, axis=1)
    u = 0.5 + np.arctan2(vertices[:, 2], vertices[:, 0]) / (2.0 * np.pi)
    v = 0.5 - np.arcsin(vertices[:, 1] / radii) / np.pi
    atlas = Image.fromarray(
        np.array(
            [
                [[240, 45, 35], [30, 210, 70]],
                [[25, 75, 230], [245, 220, 40]],
            ],
            dtype=np.uint8,
        ),
        mode="RGB",
    )
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.column_stack((u, v)),
        material=trimesh.visual.material.SimpleMaterial(image=atlas),
    )
    mesh.export(path)
    return int(len(mesh.faces))


def test_validate_exported_glb_reloads_textured_delivery_and_checks_triangle_count(
    tmp_path: Path,
) -> None:
    glb = tmp_path / "textured.glb"
    triangles = _write_textured_icosphere(glb)

    valid = RECONSTRUCT_OBJECT.validate_exported_glb(
        glb, expected_triangles=triangles
    )

    assert valid["pass"], valid["reasons"]
    assert valid["reloaded"]
    assert valid["triangles"] == triangles
    assert valid["mesh_primitives"] == 1
    assert valid["textured_mesh_primitives"] == 1
    assert valid["topology"]["pass"]

    mismatch = RECONSTRUCT_OBJECT.validate_exported_glb(
        glb, expected_triangles=triangles + 1
    )
    assert not mismatch["pass"]
    assert any("triangle count changed" in reason for reason in mismatch["reasons"])


def test_prior_candidates_stage_only_exact_frame_mask_pairs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()
    frames = []
    paired_indices = {1, 4}
    paired_stems: set[str] = set()
    for index in range(6):
        frame = frames_dir / f"frame_{index:04d}_{index * 333:09d}ms.jpg"
        frame.write_bytes(f"frame-{index}".encode())
        frames.append(frame)
        if index in paired_indices:
            paired_stems.add(frame.stem)
            (masks_dir / f"{frame.stem}.png").write_bytes(b"mask")

    observed: list[tuple[set[str], set[str]]] = []

    def fake_fit(frames_input: Path, masks_input: Path, output: Path) -> dict:
        frame_stems = {path.stem for path in frames_input.glob("*.jpg")}
        mask_stems = {path.stem for path in masks_input.glob("*.png")}
        observed.append((frame_stems, mask_stems))
        output.mkdir()
        return {"quality_gate_passed": True}

    monkeypatch.setattr(auto_parametric, "fit_rounded_slab", fake_fit)
    monkeypatch.setattr(auto_soft, "fit_soft_volume", fake_fit)

    candidates = RECONSTRUCT_OBJECT.stage_prior_candidates(
        frames, {"prior_dir": masks_dir}, tmp_path / "candidates"
    )

    assert candidates["rounded_slab"]["ok"]
    assert candidates["soft_volume"]["ok"]
    assert observed == [(paired_stems, paired_stems), (paired_stems, paired_stems)]
    assert candidates["soft_volume"]["input_summary"]["object_supported_frames"] == 2


def test_prior_input_stream_uses_fixed_sparse_cadence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_frames = [tmp_path / "sparse" / f"frame_{index:04d}.jpg" for index in range(8)]
    calls: dict[str, object] = {}

    def fake_ingest(
        video: Path, out_dir: Path, *, sample_fps: float, max_frames: int
    ) -> tuple[list[Path], object]:
        calls["ingest"] = (video, out_dir, sample_fps, max_frames)
        return fake_frames, object()

    def fake_masks(
        frames_directory: Path,
        output_directory: Path,
        review_directory: Path,
        *,
        model_name: str,
    ) -> dict:
        calls["masks"] = (
            frames_directory,
            output_directory,
            review_directory,
            model_name,
        )
        return {"frames": [{"mask": "one.png"}, {"mask": "two.png"}]}

    import local3d.masking as masking

    monkeypatch.setattr(RECONSTRUCT_OBJECT, "stage_ingest", fake_ingest)
    monkeypatch.setattr(masking, "generate_color_anchored_masks", fake_masks)

    bundle = RECONSTRUCT_OBJECT.stage_prior_inputs(
        tmp_path / "capture.mov", tmp_path / "prior"
    )

    assert calls["ingest"][-2:] == (3.0, 60)
    assert calls["masks"][0] == fake_frames[0].parent
    assert bundle["frames"] == fake_frames
    assert len(bundle["report"]["frames"]) == 2


@pytest.mark.parametrize(
    ("report", "frame_count", "expected"),
    [
        ({"acceptedFrames": 18, "persistentSoftPink": True}, 60, True),
        ({"acceptedFrames": 18}, 60, False),
        ({"acceptedFrames": 55}, 100, True),
        ({"acceptedFrames": 5, "persistentGreen": True}, 60, False),
    ],
)
def test_general_mask_router_never_discards_a_supported_persistent_object_anchor(
    report: dict, frame_count: int, expected: bool
) -> None:
    assert (
        RECONSTRUCT_OBJECT._use_object_anchor_for_general(report, frame_count)
        is expected
    )
