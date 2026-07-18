"""Validate canonical HCM AI JSONL exports before adapting an official schema.

The official HCM2026 submission format is not published yet.  This command
therefore validates the stable internal contract: IDs, timestamps, temporal
ordering, grounding, and (optionally) renderable image artifact paths.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from _cli_common import emit_json, iter_jsonl

from hcm_ai.exporting import to_primitive
from hcm_ai.validation import ValidationError, validate_answer, validate_moment, validate_sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Canonical JSONL export")
    parser.add_argument("--task", default="auto", choices=["auto", "KIS", "TRAKE", "QA"])
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Resolve relative image_path values under this root when checking files",
    )
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Fail when a referenced image artifact cannot be found on this machine/Drive mount",
    )
    parser.add_argument("--max-errors", type=int, default=20)
    return parser


def _infer_task(row: Mapping[str, Any]) -> str:
    if "events" in row:
        return "TRAKE"
    if "answer" in row or "evidence" in row:
        return "QA"
    return "KIS"


def _validate(row: Mapping[str, Any], task: str) -> None:
    if task == "KIS":
        validate_moment(row)
    elif task == "TRAKE":
        validate_sequence(row)
    elif task == "QA":
        validate_answer(row)
    else:  # Defensive, parser choices prevent this branch.
        raise ValueError(f"unsupported task {task!r}")


def _image_paths(row: Mapping[str, Any], task: str) -> Iterable[str]:
    if task == "KIS":
        image_path = row.get("image_path")
        if isinstance(image_path, str):
            yield image_path
    elif task == "TRAKE":
        events = row.get("events", [])
        if isinstance(events, Sequence) and not isinstance(events, (str, bytes)):
            for event in events:
                if isinstance(event, Mapping) and isinstance(event.get("image_path"), str):
                    yield event["image_path"]
    elif task == "QA":
        evidence = row.get("evidence", [])
        if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes)):
            for moment in evidence:
                if isinstance(moment, Mapping) and isinstance(moment.get("image_path"), str):
                    yield moment["image_path"]


def _resolve_image(path: str, artifact_root: Path | None) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute() and artifact_root is not None:
        candidate = artifact_root / candidate
    return candidate


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_errors <= 0:
        raise ValueError("--max-errors must be positive")
    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    errors: list[str] = []
    counts: dict[str, int] = {"KIS": 0, "TRAKE": 0, "QA": 0}
    total = 0
    for line_number, row in enumerate(iter_jsonl(args.input), start=1):
        total += 1
        task = args.task if args.task != "auto" else _infer_task(row)
        try:
            _validate(row, task)
            if args.require_artifacts:
                for image_path in _image_paths(row, task):
                    resolved = _resolve_image(image_path, args.artifact_root)
                    if not resolved.is_file():
                        raise ValidationError(f"missing image artifact: {resolved}")
            counts[task] += 1
        except (ValidationError, ValueError, TypeError) as error:
            errors.append(f"line {line_number} ({task}): {error}")
            if len(errors) >= args.max_errors:
                break

    emit_json(
        {
            "input": args.input,
            "valid": not errors,
            "records_checked": total,
            "counts": counts,
            "errors": errors,
        }
    )
    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover - exercised by notebook/CLI use
    raise SystemExit(main())

