#!/usr/bin/env python3
"""Validate the public JSONL schema without importing the training stack."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FIELDS = {
    "key",
    "source_text",
    "target_text",
    "emotion_text_prompt",
    "answer_cosyvoice_speech_token",
}


def validate_row(row: object, line_number: int) -> list[str]:
    if not isinstance(row, dict):
        return [f"line {line_number}: expected a JSON object"]

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS.difference(row))
    if missing:
        errors.append(f"line {line_number}: missing fields: {', '.join(missing)}")

    key = row.get("key")
    if not isinstance(key, str) or not key.strip():
        errors.append(f"line {line_number}: key must be a non-empty string")

    tokens = row.get("answer_cosyvoice_speech_token")
    if not isinstance(tokens, list) or not tokens:
        errors.append(
            f"line {line_number}: answer_cosyvoice_speech_token must be a non-empty list"
        )
    elif not all(isinstance(stream, list) for stream in tokens):
        errors.append(
            f"line {line_number}: audio tokens must be represented as a list of codebook lists"
        )
    elif not all(isinstance(token, int) for stream in tokens for token in stream):
        errors.append(f"line {line_number}: every audio token must be an integer")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--check-audio",
        action="store_true",
        help="also require target_wav and neutral_speaker_wav to exist",
    )
    args = parser.parse_args()

    errors: list[str] = []
    rows = 0
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            rows += 1
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: invalid JSON: {exc.msg}")
                continue
            errors.extend(validate_row(row, line_number))

            if args.check_audio and isinstance(row, dict):
                for field in ("target_wav", "neutral_speaker_wav"):
                    value = row.get(field)
                    if not isinstance(value, str) or not value:
                        errors.append(f"line {line_number}: missing {field}")
                        continue
                    candidate = Path(value)
                    if not candidate.is_absolute():
                        candidate = args.manifest.parent / candidate
                    if not candidate.exists():
                        errors.append(
                            f"line {line_number}: {field} does not exist: {value}"
                        )

    if rows == 0:
        errors.append("manifest contains no records")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print(f"OK: validated {rows} record(s) in {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

