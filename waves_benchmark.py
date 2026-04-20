from __future__ import annotations

import argparse
import sys
import uuid
from datetime import datetime
from pathlib import Path

from models.stego_engine import SteganographyEngine
from WAVES import WavesBenchmarkSuite, attack_names, select_attacks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Phase 4 WAVES adversarial benchmark suite.",
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--covers-dir",
        type=Path,
        help="Directory of clean cover images. The suite will generate stego images first.",
    )
    source_group.add_argument(
        "--manifest-csv",
        type=Path,
        help="CSV containing sample_id, cover_path, stego_path, and target_uuid/uuid columns.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_runs") / f"waves_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Directory where CSV outputs and optional artifacts will be written.",
    )
    parser.add_argument(
        "--encoder-weights",
        type=Path,
        default=Path("saved_models/production/best_encoder_full.pth"),
        help="Path to the trained encoder weights.",
    )
    parser.add_argument(
        "--decoder-weights",
        type=Path,
        default=Path("saved_models/production/best_decoder_full.pth"),
        help="Path to the trained decoder weights.",
    )
    parser.add_argument(
        "--target-uuid",
        type=str,
        help="Fixed UUID to embed for generated stego sets. Defaults to one UUID per run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit the number of images loaded from covers-dir or manifest-csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed used for deterministic random crops.",
    )
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=0.001,
        help="Desired false-positive rate for the BER detection threshold. Default: 0.001 (0.1%%).",
    )
    parser.add_argument(
        "--save-attacked-images",
        action="store_true",
        help="Save attacked image artifacts under the output directory.",
    )
    parser.add_argument(
        "--skip-roc-plot",
        action="store_true",
        help="Skip ROC plot generation even if matplotlib is installed.",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        help=f"Optional subset of attack names. Available: {', '.join(attack_names())}",
    )
    parser.add_argument(
        "--list-attacks",
        action="store_true",
        help="Print the available attack names and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_attacks:
        for name in attack_names():
            print(name)
        return 0

    if args.covers_dir is None and args.manifest_csv is None:
        parser.error("one of the arguments --covers-dir --manifest-csv is required")

    engine = SteganographyEngine(
        encoder_weights=str(args.encoder_weights),
        decoder_weights=str(args.decoder_weights),
    )
    suite = WavesBenchmarkSuite(
        engine,
        args.output_dir,
        seed=args.seed,
        save_attacked_images=args.save_attacked_images,
    )

    if args.manifest_csv is not None:
        samples = suite.load_samples_from_manifest(args.manifest_csv, limit=args.limit)
        target_uuid = None
    else:
        target_uuid = uuid.UUID(args.target_uuid) if args.target_uuid else uuid.uuid4()
        samples = suite.create_samples_from_covers(
            args.covers_dir,
            target_uuid=target_uuid,
            limit=args.limit,
        )

    artifacts = suite.run(
        samples,
        attacks=select_attacks(args.attacks),
        target_fpr=args.target_fpr,
        write_roc_plot=not args.skip_roc_plot,
    )

    print(f"Output directory: {args.output_dir.resolve()}")
    print(f"Samples benchmarked: {len(samples)}")
    if target_uuid is not None:
        print(f"Target UUID: {target_uuid}")
    print(f"Positive results: {artifacts['positive_results']}")
    print(f"Negative controls: {artifacts['negative_results']}")
    print(f"Attack summary: {artifacts['summary']}")
    print(f"ROC points: {artifacts['roc_points']}")
    print(f"ROC summary: {artifacts['roc_summary']}")
    if artifacts["roc_plot"] is not None:
        print(f"ROC plot: {artifacts['roc_plot']}")
    else:
        print("ROC plot: skipped or matplotlib not installed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
