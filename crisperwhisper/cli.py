"""Command-line interface for local and containerized transcription."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from crisperwhisper import CrisperWhisperModel, __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crisperwhisper",
        description="Transcribe audio with CrisperWhisper.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    transcribe = commands.add_parser(
        "transcribe", help="transcribe one audio file"
    )
    transcribe.add_argument("audio", type=Path, help="path to the audio file")
    transcribe.add_argument(
        "--model",
        default=os.getenv("CW_MODEL", "small"),
        help="model shorthand or Hugging Face ID (default: small)",
    )
    transcribe.add_argument(
        "--language", default="en", help="ISO 639-1 language code (default: en)"
    )
    transcribe.add_argument(
        "--mode",
        choices=("verbatim", "intended"),
        default="verbatim",
        help="transcription style (default: verbatim)",
    )
    transcribe.add_argument(
        "--word-timestamps",
        action="store_true",
        help="include word-level timestamps",
    )
    transcribe.add_argument(
        "--hotword",
        action="append",
        default=[],
        help="bias toward a word or phrase; may be repeated (Pro models only)",
    )
    transcribe.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="maximum generated tokens per chunk (default: 256)",
    )
    transcribe.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="output format (default: text)",
    )
    transcribe.add_argument(
        "--output", "-o", type=Path, help="write output to this file"
    )
    transcribe.add_argument(
        "--backend",
        choices=("auto", "ct2", "transformers"),
        default=os.getenv("CW_BACKEND", "transformers"),
        help="inference backend (default: transformers)",
    )
    transcribe.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default=os.getenv("CW_DEVICE", "cpu"),
        help="inference device (default: cpu)",
    )
    transcribe.add_argument(
        "--compute-type",
        default=os.getenv("CW_COMPUTE_TYPE", "float32"),
        help="model numeric type (default: float32)",
    )
    transcribe.set_defaults(handler=_transcribe)
    return parser


def _transcribe(args: argparse.Namespace) -> int:
    if not args.audio.is_file():
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    model = CrisperWhisperModel(
        args.model,
        backend=args.backend,
        device=args.device,
        compute_type=args.compute_type,
    )
    result = model.transcribe(
        args.audio,
        language=args.language,
        mode=args.mode,
        hotwords=args.hotword or None,
        max_new_tokens=args.max_new_tokens,
        word_timestamps=args.word_timestamps,
    )

    if args.format == "json":
        rendered = json.dumps(asdict(result), ensure_ascii=False, indent=2)
    else:
        rendered = result.text

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
