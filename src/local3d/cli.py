"""Command-line access to the dependency-light local pipeline core.

This CLI never downloads a checkpoint or contacts a service.  Until a real
backend is wired in by the application, ``run`` uses the explicitly labelled
full-frame segmentation and unit-cube reconstruction fallbacks.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

from .core import (
    ArtifactSpec,
    LocalJobStore,
    LocalPipelineRunner,
    PipelineError,
    PipelineStage,
    summarize_manifest,
)


def _default_jobs_dir() -> Path:
    configured = os.environ.get("LOCAL3D_JOBS_DIR")
    return Path(configured).expanduser() if configured else Path.cwd() / ".local3d" / "jobs"


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _load_settings(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineError(f"could not read settings JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PipelineError("settings JSON must contain an object at the top level")
    return value


def _store(arguments: argparse.Namespace) -> LocalJobStore:
    return LocalJobStore(arguments.jobs_dir)


def _command_create(arguments: argparse.Namespace) -> int:
    store = _store(arguments)
    manifest = store.create_job(
        arguments.video,
        job_id=arguments.job_id,
        settings=_load_settings(arguments.settings_json),
    )
    result = summarize_manifest(manifest)
    result["job_dir"] = str(store.job_dir(manifest.job_id))
    _print_json(result)
    return 0


def _command_attach_analysis(arguments: argparse.Namespace) -> int:
    store = _store(arguments)
    manifest = store.load(arguments.job_id)
    source = arguments.analysis.expanduser().resolve()
    if not source.is_file():
        raise PipelineError(f"analysis does not exist or is not a file: {source}")
    destination = store.job_dir(manifest.job_id) / "analysis.json"
    if source != destination.resolve():
        if destination.exists() and not arguments.replace:
            raise PipelineError(
                f"analysis already exists at {destination}; pass --replace to overwrite it"
            )
        shutil.copy2(source, destination)
    manifest = store.complete_external_stage(
        manifest.job_id,
        PipelineStage.INGEST,
        [ArtifactSpec("ingest.analysis", destination, "application/json")],
        backend="local-video-ingest",
    )
    _print_json(summarize_manifest(manifest))
    return 0


def _command_run(arguments: argparse.Namespace) -> int:
    store = _store(arguments)
    runner = LocalPipelineRunner(store)
    manifest = runner.run(arguments.job_id, force=arguments.force)
    result = summarize_manifest(manifest)
    result["notice"] = (
        "Default CLI backends are placeholders. The emitted cube is not a reconstruction; "
        "wire local model backends through LocalPipelineRunner for real output."
    )
    _print_json(result)
    return 0


def _command_status(arguments: argparse.Namespace) -> int:
    store = _store(arguments)
    manifest = store.load(arguments.job_id)
    result = summarize_manifest(manifest)
    result["job_dir"] = str(store.job_dir(manifest.job_id))
    if arguments.verify:
        result["artifact_validation"] = {
            key: record.validate(store.job_dir(manifest.job_id))
            for key, record in manifest.artifacts.items()
        }
    _print_json(result)
    return 0


def _command_list(arguments: argparse.Namespace) -> int:
    store = _store(arguments)
    jobs: list[dict[str, Any]] = []
    for manifest_path in sorted(store.root.glob(f"*/{store.manifest_filename}")):
        try:
            manifest = store.load(manifest_path.parent.name)
        except PipelineError as exc:
            jobs.append(
                {
                    "job_id": manifest_path.parent.name,
                    "status": "unreadable",
                    "error": str(exc),
                }
            )
            continue
        jobs.append(summarize_manifest(manifest))
    _print_json({"jobs_dir": str(store.root), "jobs": jobs})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="local3d",
        description="Manage resumable, entirely local video-to-3D jobs.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=_default_jobs_dir(),
        help="local job root (default: LOCAL3D_JOBS_DIR or ./.local3d/jobs)",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="create a local job for a source video")
    create.add_argument("video", type=Path)
    create.add_argument("--job-id")
    create.add_argument("--settings-json", type=Path)
    create.set_defaults(handler=_command_create)

    attach = subparsers.add_parser(
        "attach-analysis", help="register a local ingest analysis.json with a job"
    )
    attach.add_argument("job_id")
    attach.add_argument("analysis", type=Path)
    attach.add_argument("--replace", action="store_true")
    attach.set_defaults(handler=_command_attach_analysis)

    run = subparsers.add_parser(
        "run", help="run/resume model stages (dependency-free build uses labelled placeholders)"
    )
    run.add_argument("job_id")
    run.add_argument("--force", action="store_true", help="rerun completed model stages")
    run.set_defaults(handler=_command_run)

    status = subparsers.add_parser("status", help="show a job manifest summary")
    status.add_argument("job_id")
    status.add_argument("--verify", action="store_true", help="rehash and validate all artifacts")
    status.set_defaults(handler=_command_status)

    listing = subparsers.add_parser("list", help="list jobs under the local job root")
    listing.set_defaults(handler=_command_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        return int(arguments.handler(arguments))
    except (PipelineError, ValueError) as exc:
        print(f"local3d: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("local3d: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
