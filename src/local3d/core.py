"""Local, resumable orchestration primitives for the video-to-3D pipeline.

This module intentionally has no model, image, cloud, or database dependency.
Model integrations implement the small contracts in :mod:`local3d.backends` and
write their outputs into a job directory.  The runner records every output in an
atomic, checksummed manifest, making long GPU jobs safe to inspect and resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TYPE_CHECKING

if TYPE_CHECKING:  # Avoid a core <-> backends import cycle at runtime.
    from .backends import ReconstructionBackend, SegmentationBackend


SCHEMA_VERSION = "1.0"
_JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class PipelineError(RuntimeError):
    """Base error for an invalid or failed local pipeline job."""


class JobNotFoundError(PipelineError):
    """Raised when a requested local job does not exist."""


class JobLockedError(PipelineError):
    """Raised when another process is already executing the job."""


class ArtifactValidationError(PipelineError):
    """Raised when a backend emits a missing or unsafe artifact."""


class PipelineStage(str, Enum):
    """Stable stage names persisted in job manifests.

    ``INGEST`` is intentionally external to :class:`LocalPipelineRunner`; the
    ingest module can complete it before this runner starts.  Segmentation and
    reconstruction are the model-pluggable stages implemented here.
    """

    INGEST = "ingest"
    SEGMENTATION = "segmentation"
    RECONSTRUCTION = "reconstruction"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    """Convert common local configuration values into JSON-compatible values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n").encode("utf-8")
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


@dataclass(frozen=True)
class ArtifactSpec:
    """A backend-produced file awaiting validation and manifest registration."""

    key: str
    path: str | Path
    media_type: str = "application/octet-stream"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactRecord:
    """A checksummed file path relative to its job directory."""

    key: str
    path: str
    media_type: str
    size_bytes: int
    sha256: str
    produced_by: PipelineStage
    created_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "path": self.path,
            "media_type": self.media_type,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "produced_by": self.produced_by.value,
            "created_at": self.created_at,
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ArtifactRecord":
        return cls(
            key=str(data["key"]),
            path=str(data["path"]),
            media_type=str(data.get("media_type", "application/octet-stream")),
            size_bytes=int(data["size_bytes"]),
            sha256=str(data["sha256"]),
            produced_by=PipelineStage(str(data["produced_by"])),
            created_at=str(data["created_at"]),
            metadata=dict(data.get("metadata", {})),
        )

    def validate(self, job_dir: Path, *, checksum: bool = True) -> bool:
        candidate = (job_dir / self.path).resolve()
        try:
            candidate.relative_to(job_dir.resolve())
        except ValueError:
            return False
        if not candidate.is_file() or candidate.stat().st_size != self.size_bytes:
            return False
        return not checksum or _sha256(candidate) == self.sha256


@dataclass
class StageRecord:
    status: StageStatus = StageStatus.PENDING
    backend: str | None = None
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    output_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "backend": self.backend,
            "attempts": self.attempts,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "output_keys": list(self.output_keys),
            "metadata": _json_safe(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StageRecord":
        return cls(
            status=StageStatus(str(data.get("status", StageStatus.PENDING.value))),
            backend=data.get("backend"),
            attempts=int(data.get("attempts", 0)),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            error=data.get("error"),
            output_keys=[str(item) for item in data.get("output_keys", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class JobManifest:
    schema_version: str
    job_id: str
    source_video: str
    created_at: str
    updated_at: str
    status: JobStatus = JobStatus.CREATED
    active_stage: PipelineStage | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    stages: dict[PipelineStage, StageRecord] = field(default_factory=dict)
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def create(
        cls,
        job_id: str,
        source_video: Path,
        settings: Mapping[str, Any] | None = None,
    ) -> "JobManifest":
        now = _utc_now()
        return cls(
            schema_version=SCHEMA_VERSION,
            job_id=job_id,
            source_video=str(source_video),
            created_at=now,
            updated_at=now,
            settings=dict(_json_safe(settings or {})),
            stages={stage: StageRecord() for stage in PipelineStage},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "job_id": self.job_id,
            "source_video": self.source_video,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status.value,
            "active_stage": self.active_stage.value if self.active_stage else None,
            "settings": _json_safe(self.settings),
            "stages": {stage.value: record.to_dict() for stage, record in self.stages.items()},
            "artifacts": {key: record.to_dict() for key, record in self.artifacts.items()},
            "warnings": list(self.warnings),
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "JobManifest":
        schema_version = str(data.get("schema_version", ""))
        if schema_version != SCHEMA_VERSION:
            raise PipelineError(
                f"unsupported job manifest schema {schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        stages_data = data.get("stages", {})
        stages = {
            stage: StageRecord.from_dict(stages_data.get(stage.value, {}))
            for stage in PipelineStage
        }
        artifacts = {
            str(key): ArtifactRecord.from_dict(value)
            for key, value in dict(data.get("artifacts", {})).items()
        }
        active_stage = data.get("active_stage")
        return cls(
            schema_version=schema_version,
            job_id=str(data["job_id"]),
            source_video=str(data["source_video"]),
            created_at=str(data["created_at"]),
            updated_at=str(data["updated_at"]),
            status=JobStatus(str(data.get("status", JobStatus.CREATED.value))),
            active_stage=PipelineStage(str(active_stage)) if active_stage else None,
            settings=dict(data.get("settings", {})),
            stages=stages,
            artifacts=artifacts,
            warnings=[str(item) for item in data.get("warnings", [])],
            error=data.get("error"),
        )


@dataclass(frozen=True)
class StageResult:
    """The common result returned by a local model backend."""

    artifacts: tuple[ArtifactSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        artifacts: Sequence[ArtifactSpec],
        *,
        metadata: Mapping[str, Any] | None = None,
        warnings: Sequence[str] = (),
    ) -> "StageResult":
        return cls(tuple(artifacts), metadata or {}, tuple(warnings))


@dataclass
class PipelineContext:
    """Read-only job inputs plus safe output helpers passed to backends."""

    job_dir: Path
    manifest: JobManifest
    stage: PipelineStage

    @property
    def source_video(self) -> Path:
        return Path(self.manifest.source_video)

    @property
    def settings(self) -> Mapping[str, Any]:
        return self.manifest.settings

    def output_dir(self) -> Path:
        output = self.job_dir / "artifacts" / self.stage.value
        output.mkdir(parents=True, exist_ok=True)
        return output

    def output_path(self, *parts: str) -> Path:
        candidate = self.output_dir().joinpath(*parts).resolve()
        try:
            candidate.relative_to(self.job_dir.resolve())
        except ValueError as exc:
            raise ArtifactValidationError(f"unsafe output path: {candidate}") from exc
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate

    def artifact_path(self, key: str) -> Path | None:
        record = self.manifest.artifacts.get(key)
        return (self.job_dir / record.path).resolve() if record else None

    def analysis_path(self) -> Path | None:
        registered = self.artifact_path("ingest.analysis")
        if registered and registered.is_file():
            return registered
        conventional = self.job_dir / "analysis.json"
        return conventional if conventional.is_file() else None

    def read_analysis(self) -> dict[str, Any] | None:
        path = self.analysis_path()
        if path is None:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(f"could not read ingest analysis {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PipelineError(f"ingest analysis must be a JSON object: {path}")
        return payload


class LocalJobStore:
    """Filesystem-backed manifest store with per-job execution locks."""

    manifest_filename = "manifest.json"
    lock_filename = ".runner.lock"

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def validate_job_id(job_id: str) -> str:
        if not _JOB_ID_RE.fullmatch(job_id):
            raise ValueError(
                "job id must be 1-128 characters using letters, numbers, '.', '_' or '-'"
            )
        return job_id

    def job_dir(self, job_id: str) -> Path:
        return self.root / self.validate_job_id(job_id)

    def manifest_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / self.manifest_filename

    def create_job(
        self,
        source_video: str | Path,
        *,
        job_id: str | None = None,
        settings: Mapping[str, Any] | None = None,
    ) -> JobManifest:
        source = Path(source_video).expanduser().resolve()
        if not source.is_file():
            raise PipelineError(f"source video does not exist or is not a file: {source}")
        identifier = self.validate_job_id(job_id or uuid.uuid4().hex[:12])
        destination = self.job_dir(identifier)
        try:
            destination.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise PipelineError(f"job already exists: {identifier}") from exc
        (destination / "artifacts").mkdir()
        manifest = JobManifest.create(identifier, source, settings)
        self.save(manifest)
        return manifest

    def load(self, job_id: str) -> JobManifest:
        path = self.manifest_path(job_id)
        if not path.is_file():
            raise JobNotFoundError(f"job does not exist: {job_id}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PipelineError(f"could not read job manifest {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PipelineError(f"job manifest must be a JSON object: {path}")
        manifest = JobManifest.from_dict(payload)
        if manifest.job_id != job_id:
            raise PipelineError(
                f"job manifest id mismatch: requested {job_id!r}, found {manifest.job_id!r}"
            )
        return manifest

    def save(self, manifest: JobManifest) -> None:
        self.validate_job_id(manifest.job_id)
        job_dir = self.job_dir(manifest.job_id)
        if not job_dir.is_dir():
            raise JobNotFoundError(f"job directory does not exist: {manifest.job_id}")
        manifest.updated_at = _utc_now()
        _atomic_json_write(job_dir / self.manifest_filename, manifest.to_dict())

    def register_artifact(
        self,
        manifest: JobManifest,
        spec: ArtifactSpec,
        stage: PipelineStage,
    ) -> ArtifactRecord:
        key = spec.key.strip()
        if not key:
            raise ArtifactValidationError("artifact key cannot be empty")
        job_dir = self.job_dir(manifest.job_id).resolve()
        candidate = Path(spec.path)
        if not candidate.is_absolute():
            candidate = job_dir / candidate
        candidate = candidate.resolve()
        try:
            relative = candidate.relative_to(job_dir)
        except ValueError as exc:
            raise ArtifactValidationError(
                f"artifact {key!r} is outside the job directory: {candidate}"
            ) from exc
        if not candidate.is_file():
            raise ArtifactValidationError(f"artifact {key!r} is not a file: {candidate}")
        record = ArtifactRecord(
            key=key,
            path=relative.as_posix(),
            media_type=spec.media_type,
            size_bytes=candidate.stat().st_size,
            sha256=_sha256(candidate),
            produced_by=stage,
            created_at=_utc_now(),
            metadata=dict(_json_safe(spec.metadata)),
        )
        manifest.artifacts[key] = record
        return record

    def complete_external_stage(
        self,
        job_id: str,
        stage: PipelineStage,
        artifacts: Sequence[ArtifactSpec],
        *,
        backend: str = "local",
        metadata: Mapping[str, Any] | None = None,
        warnings: Sequence[str] = (),
    ) -> JobManifest:
        """Register outputs from ingest or another caller-managed local stage."""

        with self.lock(job_id):
            manifest = self.load(job_id)
            record = manifest.stages[stage]
            record.status = StageStatus.RUNNING
            record.backend = backend
            record.attempts += 1
            record.started_at = _utc_now()
            record.error = None
            self.save(manifest)
            output_keys: list[str] = []
            try:
                for spec in artifacts:
                    output_keys.append(self.register_artifact(manifest, spec, stage).key)
            except Exception as exc:
                record.status = StageStatus.FAILED
                record.error = f"{type(exc).__name__}: {exc}"
                manifest.status = JobStatus.FAILED
                manifest.error = record.error
                self.save(manifest)
                raise
            record.status = StageStatus.COMPLETED
            record.output_keys = output_keys
            record.completed_at = _utc_now()
            record.metadata = dict(_json_safe(metadata or {}))
            self._append_warnings(manifest, warnings)
            self.save(manifest)
            return manifest

    @staticmethod
    def _append_warnings(manifest: JobManifest, warnings: Sequence[str]) -> None:
        for warning in warnings:
            text = str(warning)
            if text and text not in manifest.warnings:
                manifest.warnings.append(text)

    def _lock_is_stale(self, path: Path) -> bool:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload["pid"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        except (PermissionError, OSError):
            return False
        return False

    @contextmanager
    def lock(self, job_id: str) -> Iterator[None]:
        job_dir = self.job_dir(job_id)
        if not job_dir.is_dir():
            raise JobNotFoundError(f"job does not exist: {job_id}")
        lock_path = job_dir / self.lock_filename
        lock_payload = json.dumps({"pid": os.getpid(), "created_at": _utc_now()}) + "\n"
        for attempt in range(2):
            try:
                descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    stream.write(lock_payload)
                break
            except FileExistsError as exc:
                if attempt == 0 and self._lock_is_stale(lock_path):
                    try:
                        lock_path.unlink()
                    except FileNotFoundError:
                        pass
                    continue
                raise JobLockedError(f"job is already running: {job_id}") from exc
        try:
            yield
        finally:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass


class LocalPipelineRunner:
    """Run segmentation and reconstruction entirely inside a local job folder."""

    stages = (PipelineStage.SEGMENTATION, PipelineStage.RECONSTRUCTION)

    def __init__(
        self,
        store: LocalJobStore,
        *,
        segmentation_backend: "SegmentationBackend | None" = None,
        reconstruction_backend: "ReconstructionBackend | None" = None,
    ):
        if segmentation_backend is None or reconstruction_backend is None:
            from .backends import FullFrameSegmentationBackend, PlaceholderReconstructionBackend

            segmentation_backend = segmentation_backend or FullFrameSegmentationBackend()
            reconstruction_backend = reconstruction_backend or PlaceholderReconstructionBackend()
        self.store = store
        self.segmentation_backend = segmentation_backend
        self.reconstruction_backend = reconstruction_backend

    @staticmethod
    def _backend_name(backend: Any) -> str:
        name = getattr(backend, "name", type(backend).__name__)
        return str(name)

    @staticmethod
    def _check_available(backend: Any) -> None:
        predicate = getattr(backend, "is_available", None)
        if callable(predicate) and not predicate():
            raise PipelineError(f"local backend is unavailable: {LocalPipelineRunner._backend_name(backend)}")

    def _stage_valid(self, manifest: JobManifest, stage: PipelineStage) -> bool:
        record = manifest.stages[stage]
        if record.status is not StageStatus.COMPLETED or not record.output_keys:
            return False
        job_dir = self.store.job_dir(manifest.job_id)
        return all(
            key in manifest.artifacts and manifest.artifacts[key].validate(job_dir)
            for key in record.output_keys
        )

    def _reset_stage(self, manifest: JobManifest, stage: PipelineStage) -> None:
        old = manifest.stages[stage]
        for key in old.output_keys:
            artifact = manifest.artifacts.get(key)
            if artifact and artifact.produced_by is stage:
                manifest.artifacts.pop(key, None)
        manifest.stages[stage] = StageRecord(attempts=old.attempts)

    def _invoke(self, stage: PipelineStage, context: PipelineContext) -> StageResult:
        if stage is PipelineStage.SEGMENTATION:
            backend = self.segmentation_backend
            self._check_available(backend)
            return backend.segment(context)
        backend = self.reconstruction_backend
        self._check_available(backend)
        return backend.reconstruct(context)

    def run(self, job_id: str, *, force: bool = False) -> JobManifest:
        """Execute or resume a job.

        Completed stages are skipped only when every checksummed output still
        validates.  ``force=True`` re-runs model stages but deliberately leaves
        old files on disk; only their manifest records are replaced.
        """

        current_stage: PipelineStage | None = None
        with self.store.lock(job_id):
            manifest = self.store.load(job_id)
            try:
                if force:
                    for stage in self.stages:
                        self._reset_stage(manifest, stage)
                if not force and all(self._stage_valid(manifest, stage) for stage in self.stages):
                    manifest.status = JobStatus.COMPLETED
                    manifest.active_stage = None
                    manifest.error = None
                    self.store.save(manifest)
                    return manifest

                manifest.status = JobStatus.RUNNING
                manifest.error = None
                self.store.save(manifest)

                for stage in self.stages:
                    current_stage = stage
                    if not force and self._stage_valid(manifest, stage):
                        continue
                    self._reset_stage(manifest, stage)
                    stage_record = manifest.stages[stage]
                    backend = (
                        self.segmentation_backend
                        if stage is PipelineStage.SEGMENTATION
                        else self.reconstruction_backend
                    )
                    stage_record.status = StageStatus.RUNNING
                    stage_record.backend = self._backend_name(backend)
                    stage_record.attempts += 1
                    stage_record.started_at = _utc_now()
                    stage_record.error = None
                    manifest.active_stage = stage
                    self.store.save(manifest)

                    context = PipelineContext(self.store.job_dir(job_id), manifest, stage)
                    result = self._invoke(stage, context)
                    if not isinstance(result, StageResult):
                        raise PipelineError(
                            f"backend {stage_record.backend!r} returned {type(result).__name__}, "
                            "expected StageResult"
                        )
                    if not result.artifacts:
                        raise ArtifactValidationError(
                            f"backend {stage_record.backend!r} produced no artifacts"
                        )
                    output_keys = [
                        self.store.register_artifact(manifest, spec, stage).key
                        for spec in result.artifacts
                    ]
                    if len(output_keys) != len(set(output_keys)):
                        raise ArtifactValidationError(
                            f"backend {stage_record.backend!r} emitted duplicate artifact keys"
                        )
                    stage_record.status = StageStatus.COMPLETED
                    stage_record.completed_at = _utc_now()
                    stage_record.output_keys = output_keys
                    stage_record.metadata = dict(_json_safe(result.metadata))
                    self.store._append_warnings(manifest, result.warnings)
                    manifest.active_stage = None
                    self.store.save(manifest)

                manifest.status = JobStatus.COMPLETED
                manifest.active_stage = None
                manifest.error = None
                self.store.save(manifest)
                return manifest
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                if current_stage is not None:
                    record = manifest.stages[current_stage]
                    record.status = StageStatus.FAILED
                    record.error = message
                    record.completed_at = _utc_now()
                manifest.status = JobStatus.FAILED
                manifest.active_stage = None
                manifest.error = message
                self.store.save(manifest)
                raise


def summarize_manifest(manifest: JobManifest) -> dict[str, Any]:
    """Return a concise JSON-friendly status view for CLIs and local UIs."""

    return {
        "job_id": manifest.job_id,
        "status": manifest.status.value,
        "source_video": manifest.source_video,
        "active_stage": manifest.active_stage.value if manifest.active_stage else None,
        "stages": {
            stage.value: {
                "status": record.status.value,
                "backend": record.backend,
                "attempts": record.attempts,
                "error": record.error,
            }
            for stage, record in manifest.stages.items()
        },
        "artifacts": {
            key: {"path": record.path, "media_type": record.media_type}
            for key, record in manifest.artifacts.items()
        },
        "warnings": list(manifest.warnings),
        "error": manifest.error,
    }
