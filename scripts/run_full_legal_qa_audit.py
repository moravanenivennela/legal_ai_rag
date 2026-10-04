"""Run the resumable local-PDF audit of every discovered legal QA candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.legal_qa_audit_core import ROOT as PROJECT_ROOT  # noqa: E402
from scripts.legal_qa_audit_core import run_full_audit  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit discovered legal QA candidate datasets against the supplied "
            "Constitution and Consumer Protection Act PDFs. This is not legal "
            "verification and does not change source data."
        )
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Candidate identities per resumable checkpoint batch (default: 100).",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root; defaults to this script's repository.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="PDF source directory; defaults to <root>/data.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Report output directory; defaults to <root>/reports.",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume an interrupted audit from its timestamped report directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = args.root.resolve()
    data_dir = args.data_dir.resolve() if args.data_dir else root / "data"
    output_root = (
        args.output_root.resolve() if args.output_root else root / "reports"
    )
    resume_from = args.resume_from.resolve() if args.resume_from else None
    try:
        output_dir = run_full_audit(
            root=root,
            data_dir=data_dir,
            output_root=output_root,
            batch_size=args.batch_size,
            resume_from=resume_from,
        )
    except (FileNotFoundError, FileExistsError, OSError, ValueError) as error:
        parser.exit(2, f"Audit could not complete safely: {error}\n")
    print(f"Output directory: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
