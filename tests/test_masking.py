from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

import local3d.masking as masking

from local3d.masking import (
    MaskingError,
    _appearance_anchor,
    _complete_centered_rigid_hull,
    _green_anchor,
    _has_soft_pink_evidence,
    _new_cached_u2netp_session,
    _soft_color_anchor,
    _u2netp_model_path,
    generate_color_anchored_masks,
)


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = np.logical_and(left > 0, right > 0).sum()
    union = np.logical_or(left > 0, right > 0).sum()
    return float(intersection / max(union, 1))


def test_rigid_appearance_anchor_separates_blue_phone_from_skin_holder() -> None:
    frame = np.full((320, 480, 3), 236, dtype=np.uint8)
    expected = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(expected, (92, 92), (388, 226), 1, -1)
    cv2.rectangle(frame, (92, 92), (388, 226), (132, 92, 54), -1)
    coarse = expected.copy()
    for center in ((75, 150), (405, 150)):
        cv2.circle(frame, center, 55, (72, 100, 154), -1)
        cv2.circle(coarse, center, 55, 1, -1)

    recovered = _appearance_anchor(frame, coarse)

    assert recovered is not None
    assert _iou(recovered, expected) > 0.88
    assert not recovered[0].any() and not recovered[-1].any()


def test_centered_rigid_completion_recovers_skin_colored_half_face() -> None:
    coarse = np.zeros((240, 360), dtype=np.uint8)
    cv2.rectangle(coarse, (60, 58), (300, 168), 1, -1)
    half = np.zeros_like(coarse)
    cv2.rectangle(half, (60, 58), (180, 168), 1, -1)

    completed, changed = _complete_centered_rigid_hull(half, coarse)

    assert changed
    assert _iou(completed, coarse) > 0.96


def test_soft_pink_anchor_prunes_border_limbs_without_convex_hull() -> None:
    frame = np.full((320, 480, 3), 230, dtype=np.uint8)
    target = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.ellipse(target, (240, 155), (105, 82), 0, 0, 360, 1, -1)
    cv2.ellipse(target, (155, 105), (60, 18), -18, 0, 360, 1, -1)
    cv2.ellipse(target, (325, 105), (60, 18), 18, 0, 360, 1, -1)
    cv2.rectangle(target, (202, 202), (226, 254), 1, -1)
    cv2.rectangle(target, (254, 202), (278, 254), 1, -1)
    frame[target > 0] = (100, 112, 158)
    # Similar warm holder limbs touch the target through narrow necks and run
    # to the border.  The soft path must prune them rather than convex-hull them.
    cv2.line(frame, (140, 170), (0, 300), (74, 92, 137), 18)
    cv2.line(frame, (340, 170), (479, 300), (74, 92, 137), 18)

    recovered = _soft_color_anchor(frame, np.ones(frame.shape[:2], dtype=np.uint8))

    assert _has_soft_pink_evidence(frame)
    assert recovered is not None
    assert _iou(recovered, target) > 0.72
    assert not recovered[0].any() and not recovered[-1].any()
    assert not recovered[:, 0].any() and not recovered[:, -1].any()


def test_soft_pink_anchor_recovers_safe_disconnected_ears_only() -> None:
    frame = np.full((320, 480, 3), 230, dtype=np.uint8)
    target = np.zeros(frame.shape[:2], dtype=np.uint8)
    body_color = (100, 112, 158)
    neck_color = (80, 90, 135)  # broad Lab evidence, outside the tight delta-E component
    cv2.ellipse(target, (240, 165), (95, 75), 0, 0, 360, 1, -1)
    cv2.ellipse(target, (120, 105), (48, 15), -12, 0, 360, 1, -1)
    cv2.ellipse(target, (360, 105), (48, 15), 12, 0, 360, 1, -1)
    frame[target > 0] = body_color
    # Narrow darker necks keep the object physically connected in broad color
    # evidence while making both ears separate strict-color components.
    for start, end in (((162, 112), (181, 128)), ((318, 128), (338, 112))):
        cv2.line(target, start, end, 1, 11, cv2.LINE_AA)
        cv2.line(frame, start, end, neck_color, 11, cv2.LINE_AA)

    # Warm limbs reach the frame and a strict-color elongated decoy sits near
    # the left border.  Neither is allowed to join the torso.
    cv2.line(frame, (150, 175), (0, 300), (74, 92, 137), 20)
    cv2.line(frame, (330, 175), (479, 300), (74, 92, 137), 20)
    cv2.ellipse(frame, (18, 80), (14, 48), 0, 0, 360, body_color, -1)
    diagnostics: dict[str, object] = {}

    recovered = _soft_color_anchor(
        frame,
        np.ones(frame.shape[:2], dtype=np.uint8),
        diagnostics=diagnostics,
    )

    assert recovered is not None
    assert _iou(recovered, target) > 0.80
    assert recovered[105, 120] and recovered[105, 360]
    assert not recovered[:, 0].any() and not recovered[:, -1].any()
    assert diagnostics["safeAppendageCount"] >= 1
    assert diagnostics["appendageGrowthRatio"] <= 1.55


def test_green_anchor_retains_existing_full_object_hull() -> None:
    frame = np.full((240, 360, 3), 225, dtype=np.uint8)
    expected = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.rectangle(expected, (85, 60), (275, 180), 1, -1)
    frame[expected > 0] = (45, 175, 74)

    recovered = _green_anchor(frame, expected)

    assert recovered is not None
    assert _iou(recovered, expected) > 0.98


def test_u2netp_cache_path_matches_rembg_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit_cache = tmp_path / "explicit-cache"
    monkeypatch.setenv("U2NET_HOME", str(explicit_cache))

    assert _u2netp_model_path("u2netp") == explicit_cache / "u2netp.onnx"

    monkeypatch.delenv("U2NET_HOME")
    xdg_data = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))

    assert _u2netp_model_path("u2netp") == xdg_data / ".u2net" / "u2netp.onnx"


def test_missing_cached_model_fails_before_loading_rembg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    assert cv2.imwrite(
        str(frames / "frame_0001.jpg"), np.zeros((32, 48, 3), np.uint8)
    )
    monkeypatch.setenv("U2NET_HOME", str(tmp_path / "empty-cache"))
    backend_loaded = False

    def load_backend() -> object:
        nonlocal backend_loaded
        backend_loaded = True
        raise AssertionError("rembg must not load before cache preflight")

    monkeypatch.setattr(masking, "_load_rembg_backend", load_backend)

    with pytest.raises(MaskingError, match="automatic model downloads are disabled"):
        generate_color_anchored_masks(frames, tmp_path / "masks", tmp_path / "review")

    assert backend_loaded is False


def test_cached_session_redirects_and_restores_rembg_model_resolver(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "u2netp.onnx"
    model_path.write_bytes(b"cached-model")
    original_calls: list[str] = []

    class FakeU2netpSession:
        @classmethod
        def download_models(cls) -> str:
            original_calls.append("would-download")
            return "network-model"

    def new_session(model_name: str) -> str:
        assert model_name == "u2netp"
        return FakeU2netpSession.download_models()

    session = _new_cached_u2netp_session(new_session, FakeU2netpSession, model_path)

    assert session == str(model_path)
    assert original_calls == []
    assert FakeU2netpSession.download_models() == "network-model"
    assert original_calls == ["would-download"]


def test_session_failure_is_wrapped_as_masking_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    assert cv2.imwrite(
        str(frames / "frame_0001.jpg"), np.zeros((32, 48, 3), np.uint8)
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "u2netp.onnx").write_bytes(b"cached-model")
    monkeypatch.setenv("U2NET_HOME", str(cache))

    class FakeU2netpSession:
        @classmethod
        def download_models(cls) -> str:
            raise AssertionError("network resolver must remain unreachable")

    def fail_session(_model_name: str) -> object:
        raise RuntimeError("onnx initialization failed")

    monkeypatch.setattr(
        masking,
        "_load_rembg_backend",
        lambda: (
            fail_session,
            lambda *_args, **_kwargs: None,
            FakeU2netpSession,
            "2.test",
        ),
    )

    with pytest.raises(MaskingError, match="could not initialize cached local U2Net session"):
        generate_color_anchored_masks(frames, tmp_path / "masks", tmp_path / "review")


def test_inference_failure_is_wrapped_as_masking_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    assert cv2.imwrite(
        str(frames / "frame_0001.jpg"), np.zeros((32, 48, 3), np.uint8)
    )
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "u2netp.onnx").write_bytes(b"cached-model")
    monkeypatch.setenv("U2NET_HOME", str(cache))

    class FakeU2netpSession:
        @classmethod
        def download_models(cls) -> str:
            raise AssertionError("network resolver must remain unreachable")

    def fail_inference(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("onnx inference failed")

    monkeypatch.setattr(
        masking,
        "_load_rembg_backend",
        lambda: (lambda _name: object(), fail_inference, FakeU2netpSession, "2.test"),
    )

    with pytest.raises(MaskingError, match="local U2Net inference failed"):
        generate_color_anchored_masks(frames, tmp_path / "masks", tmp_path / "review")


def test_mask_report_records_cached_model_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frames = tmp_path / "frames"
    frames.mkdir()
    frame = np.full((64, 96, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(frames / "frame_0001.jpg"), frame)
    monkeypatch.delenv("U2NET_HOME", raising=False)
    xdg_data = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg_data))
    cache = xdg_data / ".u2net"
    cache.mkdir(parents=True)
    model_bytes = b"preexisting-u2netp-model"
    (cache / "u2netp.onnx").write_bytes(model_bytes)

    class FakeU2netpSession:
        @classmethod
        def download_models(cls) -> str:
            raise AssertionError("network resolver must remain unreachable")

    def remove(source: Image.Image, **_kwargs: object) -> Image.Image:
        return Image.new("L", source.size, 255)

    monkeypatch.setattr(
        masking,
        "_load_rembg_backend",
        lambda: (lambda _name: object(), remove, FakeU2netpSession, "2.0.test"),
    )
    monkeypatch.setattr(masking, "_has_green_evidence", lambda _frame: False)
    monkeypatch.setattr(masking, "_has_soft_pink_evidence", lambda _frame: False)
    monkeypatch.setattr(
        masking,
        "_appearance_anchor",
        lambda source, _coarse: np.ones(source.shape[:2], dtype=np.uint8),
    )

    report = generate_color_anchored_masks(frames, tmp_path / "masks", tmp_path / "review")

    assert report["model"] == {
        "name": "u2netp",
        "path": str(cache / "u2netp.onnx"),
        "bytes": len(model_bytes),
        "sha256": hashlib.sha256(model_bytes).hexdigest(),
        "provenance": {
            "kind": "preexisting_local_cache",
            "cachePathSource": "XDG_DATA_HOME",
            "networkAttempted": False,
        },
    }
    assert report["runtime"] == {"package": "rembg", "version": "2.0.test"}
