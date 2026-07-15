from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from local3d.recon_common import make_view
from local3d.sfm_video import _track_evidence


class _Track:
    def __init__(self, elements):
        self.elements = elements

    def length(self):
        return len(self.elements)


class _Rec:
    def __init__(self):
        elements = [
            SimpleNamespace(image_id=1, point2D_idx=0),
            SimpleNamespace(image_id=2, point2D_idx=0),
            SimpleNamespace(image_id=3, point2D_idx=0),
        ]
        self.points3D = {
            10: SimpleNamespace(
                xyz=np.array([0.0, 0.0, 2.0]),
                track=_Track(elements),
                has_error=True,
                error=0.25,
            ),
            # Too-short track: excluded globally and from every view.
            20: SimpleNamespace(
                xyz=np.array([1.0, 0.0, 2.0]),
                track=_Track(elements[:2]),
                has_error=True,
                error=0.5,
            ),
        }
        self._images = {
            1: SimpleNamespace(
                name="a.jpg",
                points2D=[SimpleNamespace(xy=np.array([10.5, 20.5]))],
            ),
            2: SimpleNamespace(
                name="b.jpg",
                points2D=[SimpleNamespace(xy=np.array([11.5, 21.5]))],
            ),
            # The track exists, but this image is not a retained coherent view.
            3: SimpleNamespace(
                name="discarded.jpg",
                points2D=[SimpleNamespace(xy=np.array([12.5, 22.5]))],
            ),
        }

    def image(self, image_id):
        return self._images[image_id]


def _view(name: str):
    return make_view(
        name=name,
        image_path=name,
        rotation=np.eye(3),
        translation=np.zeros(3),
    )


def test_track_evidence_keeps_only_actual_retained_observations():
    evidence = _track_evidence(_Rec(), [_view("b.jpg"), _view("a.jpg")])

    assert evidence["point3d_ids"].tolist() == [10]
    assert sorted(evidence["views"]) == ["a.jpg", "b.jpg"]
    assert evidence["views"]["a.jpg"]["point3d_ids"].tolist() == [10]
    assert evidence["views"]["b.jpg"]["point3d_ids"].tolist() == [10]
    np.testing.assert_allclose(evidence["views"]["a.jpg"]["xy"], [[10.5, 20.5]])
    np.testing.assert_allclose(evidence["views"]["a.jpg"]["z_camera"], [2.0])


def test_track_evidence_uses_each_view_pose_for_camera_depth():
    shifted = _view("a.jpg")
    shifted["translation"] = np.array([0.0, 0.0, 3.0])
    evidence = _track_evidence(_Rec(), [shifted])

    np.testing.assert_allclose(evidence["views"]["a.jpg"]["z_camera"], [5.0])
