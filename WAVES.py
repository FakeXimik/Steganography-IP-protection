from __future__ import annotations

import csv
import io
import json
import math
import random
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageFilter

from models.stego_engine import SteganographyEngine


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class AttackStep:
    kind: str
    value: float | int


@dataclass(frozen=True)
class AttackSpec:
    name: str
    steps: tuple[AttackStep, ...]


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    cover_path: Path
    stego_path: Path
    target_uuid: uuid.UUID


def default_attack_suite() -> list[AttackSpec]:
    return [
        AttackSpec("identity", tuple()),
        AttackSpec("jpeg_q90", (AttackStep("jpeg", 90),)),
        AttackSpec("jpeg_q70", (AttackStep("jpeg", 70),)),
        AttackSpec("jpeg_q50", (AttackStep("jpeg", 50),)),
        AttackSpec("jpeg_q30", (AttackStep("jpeg", 30),)),
        AttackSpec("jpeg_q10", (AttackStep("jpeg", 10),)),
        AttackSpec("crop_10pct", (AttackStep("crop", 0.10),)),
        AttackSpec("crop_30pct", (AttackStep("crop", 0.30),)),
        AttackSpec("crop_50pct", (AttackStep("crop", 0.50),)),
        AttackSpec("scale_down_50pct_restore", (AttackStep("scale", 0.50),)),
        AttackSpec("rotate_5deg", (AttackStep("rotate", 5),)),
        AttackSpec("rotate_10deg", (AttackStep("rotate", 10),)),
        AttackSpec("rotate_15deg", (AttackStep("rotate", 15),)),
        AttackSpec("gaussian_blur_r1_0", (AttackStep("gaussian", 1.0),)),
        AttackSpec("gaussian_blur_r2_0", (AttackStep("gaussian", 2.0),)),
        AttackSpec("median_filter_3", (AttackStep("median", 3),)),
        AttackSpec("median_filter_5", (AttackStep("median", 5),)),
        AttackSpec(
            "jpeg_q30_crop_20pct",
            (AttackStep("jpeg", 30), AttackStep("crop", 0.20)),
        ),
        AttackSpec(
            "jpeg_q30_rotate_10deg",
            (AttackStep("jpeg", 30), AttackStep("rotate", 10)),
        ),
        AttackSpec(
            "jpeg_q50_gaussian_blur_r1_5",
            (AttackStep("jpeg", 50), AttackStep("gaussian", 1.5)),
        ),
    ]


def attack_names(attacks: Sequence[AttackSpec] | None = None) -> list[str]:
    return [attack.name for attack in (attacks or default_attack_suite())]


def discover_images(directory: Path, limit: int | None = None) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Image directory does not exist: {directory}")

    files = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if limit is not None:
        files = files[:limit]
    if not files:
        raise ValueError(f"No supported images found in {directory}")
    return files


def resolve_manifest_path(manifest_path: Path, candidate: str) -> Path:
    path = Path(candidate)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def load_samples_from_manifest(manifest_path: Path, limit: int | None = None) -> list[BenchmarkSample]:
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if limit is not None:
        rows = rows[:limit]

    samples = []
    for index, row in enumerate(rows, start=1):
        sample_id = row.get("sample_id") or f"sample_{index:04d}"
        cover_value = row.get("cover_path")
        stego_value = row.get("stego_path")
        uuid_value = row.get("target_uuid") or row.get("uuid")

        if not cover_value or not stego_value or not uuid_value:
            raise ValueError(
                "Manifest rows must include cover_path, stego_path, and target_uuid/uuid columns."
            )

        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                cover_path=resolve_manifest_path(manifest_path, cover_value),
                stego_path=resolve_manifest_path(manifest_path, stego_value),
                target_uuid=uuid.UUID(uuid_value),
            )
        )

    if not samples:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return samples


def _safe_sample_id(index: int, path: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)
    return f"sample_{index:04d}_{stem}"


def create_stego_samples(
    engine: SteganographyEngine,
    covers_dir: Path,
    stego_output_dir: Path,
    *,
    target_uuid: uuid.UUID,
    limit: int | None = None,
) -> list[BenchmarkSample]:
    cover_paths = discover_images(covers_dir, limit=limit)
    stego_output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for index, cover_path in enumerate(cover_paths, start=1):
        sample_id = _safe_sample_id(index, cover_path)
        stego_path = stego_output_dir / f"{sample_id}.png"
        engine.embed_uuid(cover_path, stego_path, target_uuid=target_uuid)
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                cover_path=cover_path.resolve(),
                stego_path=stego_path.resolve(),
                target_uuid=target_uuid,
            )
        )
    return samples


def write_csv(path: Path, rows: Sequence[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def save_manifest(path: Path, samples: Sequence[BenchmarkSample]) -> Path:
    rows = [
        {
            "sample_id": sample.sample_id,
            "cover_path": str(sample.cover_path),
            "stego_path": str(sample.stego_path),
            "target_uuid": str(sample.target_uuid),
        }
        for sample in samples
    ]
    return write_csv(path, rows)


def open_rgb_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def pil_to_float_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def calculate_psnr(reference: Image.Image, candidate: Image.Image) -> float:
    ref = pil_to_float_array(reference)
    cmp = pil_to_float_array(candidate)
    if ref.shape != cmp.shape:
        raise ValueError("PSNR requires images with identical dimensions.")

    mse = float(np.mean((ref - cmp) ** 2))
    if mse == 0.0:
        return math.inf
    return 20.0 * math.log10(255.0) - 10.0 * math.log10(mse)


def _ssim_single_channel(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = reference.astype(np.float64)
    candidate = candidate.astype(np.float64)

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    mu_ref = cv2.GaussianBlur(reference, (11, 11), 1.5)
    mu_cmp = cv2.GaussianBlur(candidate, (11, 11), 1.5)

    mu_ref_sq = mu_ref * mu_ref
    mu_cmp_sq = mu_cmp * mu_cmp
    mu_ref_cmp = mu_ref * mu_cmp

    sigma_ref_sq = cv2.GaussianBlur(reference * reference, (11, 11), 1.5) - mu_ref_sq
    sigma_cmp_sq = cv2.GaussianBlur(candidate * candidate, (11, 11), 1.5) - mu_cmp_sq
    sigma_ref_cmp = cv2.GaussianBlur(reference * candidate, (11, 11), 1.5) - mu_ref_cmp

    numerator = (2 * mu_ref_cmp + c1) * (2 * sigma_ref_cmp + c2)
    denominator = (mu_ref_sq + mu_cmp_sq + c1) * (sigma_ref_sq + sigma_cmp_sq + c2)
    return float((numerator / (denominator + 1e-12)).mean())


def calculate_ssim(reference: Image.Image, candidate: Image.Image) -> float:
    ref = pil_to_float_array(reference)
    cmp = pil_to_float_array(candidate)
    if ref.shape != cmp.shape:
        raise ValueError("SSIM requires images with identical dimensions.")

    return float(np.mean([_ssim_single_channel(ref[:, :, idx], cmp[:, :, idx]) for idx in range(3)]))


def bit_error_rate(observed_bits: Sequence[int], expected_bits: Sequence[int]) -> float:
    compare_length = min(len(observed_bits), len(expected_bits))
    if compare_length == 0:
        raise ValueError("Cannot calculate BER for empty payloads.")

    errors = sum(
        int(observed_bits[index]) != int(expected_bits[index])
        for index in range(compare_length)
    )
    return errors / compare_length


def ber_threshold_at_fpr(negative_bers: Sequence[float], target_fpr: float) -> tuple[float, float]:
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError(f"target_fpr must be between 0.0 and 1.0, got {target_fpr}")

    negatives = np.sort(np.asarray(negative_bers, dtype=np.float64))
    if negatives.size == 0:
        raise ValueError("Need at least one negative BER value to calculate a threshold.")

    unique = np.unique(negatives)
    candidates = [float(unique[0] - 1e-12)]
    if unique.size > 1:
        candidates.extend(((unique[:-1] + unique[1:]) / 2.0).tolist())
    candidates.extend(unique.tolist())

    best_threshold = candidates[0]
    achieved_fpr = float(np.mean(negatives <= best_threshold))
    for threshold in candidates:
        fpr = float(np.mean(negatives <= threshold))
        if fpr <= target_fpr + 1e-12 and threshold >= best_threshold:
            best_threshold = float(threshold)
            achieved_fpr = fpr

    return best_threshold, achieved_fpr


def compute_roc_points(negative_bers: Sequence[float], positive_bers: Sequence[float]) -> list[dict]:
    negatives = np.asarray(negative_bers, dtype=np.float64)
    positives = np.asarray(positive_bers, dtype=np.float64)
    if negatives.size == 0 or positives.size == 0:
        raise ValueError("ROC calculation requires both negative and positive BER values.")

    all_values = np.unique(np.concatenate([negatives, positives]))
    thresholds = [float(all_values[0] - 1e-12), *all_values.tolist(), float(all_values[-1] + 1e-12)]

    points = []
    for threshold in thresholds:
        fpr = float(np.mean(negatives <= threshold))
        tpr = float(np.mean(positives <= threshold))
        points.append(
            {
                "threshold_ber": threshold,
                "threshold_bit_accuracy": 1.0 - threshold,
                "fpr": fpr,
                "tpr": tpr,
            }
        )
    return points


def maybe_plot_roc_curve(roc_points: Sequence[dict], output_path: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(
        [point["fpr"] for point in roc_points],
        [point["tpr"] for point in roc_points],
        linewidth=2,
    )
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("WAVES ROC Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _mean_fill_color(image: Image.Image) -> tuple[int, int, int]:
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    fill = pixels.mean(axis=(0, 1)).round().astype(np.uint8)
    return int(fill[0]), int(fill[1]), int(fill[2])


def _jpeg_recompress(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _random_crop_resize(image: Image.Image, remove_fraction: float, rng: random.Random) -> Image.Image:
    if not 0.0 <= remove_fraction < 1.0:
        raise ValueError(f"remove_fraction must be in [0, 1), got {remove_fraction}")

    width, height = image.size
    keep_ratio = math.sqrt(1.0 - float(remove_fraction))
    crop_width = max(1, int(round(width * keep_ratio)))
    crop_height = max(1, int(round(height * keep_ratio)))
    max_left = max(0, width - crop_width)
    max_top = max(0, height - crop_height)
    left = rng.randint(0, max_left) if max_left else 0
    top = rng.randint(0, max_top) if max_top else 0
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def _scale_restore(image: Image.Image, scale_factor: float) -> Image.Image:
    if not 0.0 < scale_factor <= 1.0:
        raise ValueError(f"scale_factor must be in (0, 1], got {scale_factor}")

    width, height = image.size
    down_width = max(1, int(round(width * scale_factor)))
    down_height = max(1, int(round(height * scale_factor)))
    downscaled = image.resize((down_width, down_height), Image.Resampling.LANCZOS)
    return downscaled.resize((width, height), Image.Resampling.BICUBIC)


def _rotate(image: Image.Image, degrees: float) -> Image.Image:
    return image.rotate(
        float(degrees),
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=_mean_fill_color(image),
    )


def apply_attack_step(image: Image.Image, step: AttackStep, rng: random.Random) -> Image.Image:
    if step.kind == "jpeg":
        return _jpeg_recompress(image, int(step.value))
    if step.kind == "crop":
        return _random_crop_resize(image, float(step.value), rng)
    if step.kind == "scale":
        return _scale_restore(image, float(step.value))
    if step.kind == "rotate":
        return _rotate(image, float(step.value))
    if step.kind == "gaussian":
        return image.filter(ImageFilter.GaussianBlur(radius=float(step.value)))
    if step.kind == "median":
        return image.filter(ImageFilter.MedianFilter(size=int(step.value)))
    raise ValueError(f"Unknown attack step kind: {step.kind}")


def apply_attack(image: Image.Image, attack: AttackSpec, rng: random.Random) -> Image.Image:
    attacked = image.convert("RGB")
    for step in attack.steps:
        attacked = apply_attack_step(attacked, step, rng)
    return attacked


def select_attacks(requested_names: Sequence[str] | None) -> list[AttackSpec]:
    available = {attack.name: attack for attack in default_attack_suite()}
    if not requested_names:
        return list(available.values())

    missing = [name for name in requested_names if name not in available]
    if missing:
        raise ValueError(f"Unknown attack names: {', '.join(sorted(missing))}")
    return [available[name] for name in requested_names]


def _attack_rng(seed: int, sample_id: str, attack_name: str) -> random.Random:
    return random.Random(f"{seed}:{sample_id}:{attack_name}")


def _serialize_attack_steps(attack: AttackSpec) -> str:
    if not attack.steps:
        return "identity"
    return " -> ".join(f"{step.kind}:{step.value}" for step in attack.steps)


def summarize_attack_rows(rows: Sequence[dict], detection_threshold: float) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["attack"], []).append(row)

    summary_rows = []
    for attack_name in sorted(grouped):
        attack_rows = grouped[attack_name]
        summary_rows.append(
            {
                "attack": attack_name,
                "samples": len(attack_rows),
                "mean_psnr": float(np.mean([row["psnr"] for row in attack_rows])),
                "mean_ssim": float(np.mean([row["ssim"] for row in attack_rows])),
                "mean_raw_ber": float(np.mean([row["raw_ber"] for row in attack_rows])),
                "mean_bit_accuracy": float(np.mean([row["bit_accuracy"] for row in attack_rows])),
                "recovery_rate": float(np.mean([row["fec_recovered"] for row in attack_rows])),
                "detection_rate_at_threshold": float(
                    np.mean([row["raw_ber"] <= detection_threshold for row in attack_rows])
                ),
                "min_raw_ber": float(np.min([row["raw_ber"] for row in attack_rows])),
                "max_raw_ber": float(np.max([row["raw_ber"] for row in attack_rows])),
            }
        )
    return summary_rows


class WavesBenchmarkSuite:
    def __init__(
        self,
        engine: SteganographyEngine,
        output_dir: Path,
        *,
        seed: int = 1337,
        save_attacked_images: bool = False,
    ) -> None:
        self.engine = engine
        self.output_dir = output_dir.resolve()
        self.seed = seed
        self.save_attacked_images = save_attacked_images

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_stego_dir = self.output_dir / "generated_stego"
        self.attacked_dir = self.output_dir / "attacked"

    def create_samples_from_covers(
        self,
        covers_dir: Path,
        *,
        target_uuid: uuid.UUID,
        limit: int | None = None,
    ) -> list[BenchmarkSample]:
        samples = create_stego_samples(
            self.engine,
            covers_dir,
            self.generated_stego_dir,
            target_uuid=target_uuid,
            limit=limit,
        )
        save_manifest(self.output_dir / "stego_manifest.csv", samples)
        return samples

    def load_samples_from_manifest(
        self,
        manifest_path: Path,
        *,
        limit: int | None = None,
    ) -> list[BenchmarkSample]:
        samples = load_samples_from_manifest(manifest_path, limit=limit)
        save_manifest(self.output_dir / "stego_manifest.csv", samples)
        return samples

    def run(
        self,
        samples: Sequence[BenchmarkSample],
        *,
        attacks: Sequence[AttackSpec] | None = None,
        target_fpr: float = 0.001,
        write_roc_plot: bool = True,
    ) -> dict[str, Path | None]:
        selected_attacks = list(attacks or default_attack_suite())
        positive_rows: list[dict] = []
        negative_rows: list[dict] = []

        for sample in samples:
            cover_image = open_rgb_image(sample.cover_path)
            stego_image = open_rgb_image(sample.stego_path)
            expected_bits = self.engine.uuid_to_payload_bits(sample.target_uuid)

            clean_details = self.engine.extract_payload_details(cover_image)
            clean_ber = bit_error_rate(clean_details["bits"], expected_bits)
            negative_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "cover_path": str(sample.cover_path),
                    "target_uuid": str(sample.target_uuid),
                    "raw_ber": clean_ber,
                    "bit_accuracy": 1.0 - clean_ber,
                    "mask_active_pct": float(clean_details["mask_active_pct"]),
                    "fec_recovered": int(clean_details["decoded_uuid"] == sample.target_uuid),
                    "decoded_uuid": str(clean_details["decoded_uuid"] or ""),
                    "decode_error": clean_details["decode_error"] or "",
                    "payload_hex": clean_details["payload_bytes"].hex(),
                }
            )

            for attack in selected_attacks:
                rng = _attack_rng(self.seed, sample.sample_id, attack.name)
                attacked_image = apply_attack(stego_image, attack, rng)
                attacked_path = ""
                if self.save_attacked_images:
                    attack_dir = self.attacked_dir / attack.name
                    attack_dir.mkdir(parents=True, exist_ok=True)
                    attacked_path = str((attack_dir / f"{sample.sample_id}.png").resolve())
                    attacked_image.save(attacked_path, format="PNG")

                attack_details = self.engine.extract_payload_details(attacked_image)
                raw_ber = bit_error_rate(attack_details["bits"], expected_bits)
                decoded_uuid = attack_details["decoded_uuid"]

                positive_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "attack": attack.name,
                        "attack_steps": _serialize_attack_steps(attack),
                        "cover_path": str(sample.cover_path),
                        "stego_path": str(sample.stego_path),
                        "attacked_path": attacked_path,
                        "target_uuid": str(sample.target_uuid),
                        "decoded_uuid": str(decoded_uuid or ""),
                        "decode_error": attack_details["decode_error"] or "",
                        "psnr": calculate_psnr(cover_image, attacked_image),
                        "ssim": calculate_ssim(cover_image, attacked_image),
                        "raw_ber": raw_ber,
                        "bit_accuracy": 1.0 - raw_ber,
                        "mask_active_pct": float(attack_details["mask_active_pct"]),
                        "fec_recovered": int(decoded_uuid == sample.target_uuid),
                        "payload_hex": attack_details["payload_bytes"].hex(),
                    }
                )

        negative_bers = [row["raw_ber"] for row in negative_rows]
        positive_bers = [row["raw_ber"] for row in positive_rows]
        threshold_ber, achieved_fpr = ber_threshold_at_fpr(negative_bers, target_fpr)
        roc_points = compute_roc_points(negative_bers, positive_bers)
        attack_summary = summarize_attack_rows(positive_rows, threshold_ber)

        positive_results_path = write_csv(self.output_dir / "attack_results.csv", positive_rows)
        negative_results_path = write_csv(self.output_dir / "negative_controls.csv", negative_rows)
        summary_path = write_csv(self.output_dir / "attack_summary.csv", attack_summary)
        roc_points_path = write_csv(self.output_dir / "roc_points.csv", roc_points)

        roc_summary = {
            "requested_fpr": target_fpr,
            "achieved_fpr": achieved_fpr,
            "threshold_ber": threshold_ber,
            "threshold_bit_accuracy": 1.0 - threshold_ber,
            "overall_tpr": float(np.mean([ber <= threshold_ber for ber in positive_bers])),
            "samples": len(samples),
            "positive_examples": len(positive_rows),
            "negative_examples": len(negative_rows),
        }
        roc_summary_path = write_json(self.output_dir / "roc_summary.json", roc_summary)
        roc_plot_path = maybe_plot_roc_curve(roc_points, self.output_dir / "roc_curve.png") if write_roc_plot else None

        return {
            "positive_results": positive_results_path,
            "negative_results": negative_results_path,
            "summary": summary_path,
            "roc_points": roc_points_path,
            "roc_summary": roc_summary_path,
            "roc_plot": roc_plot_path,
        }


__all__ = [
    "AttackSpec",
    "AttackStep",
    "BenchmarkSample",
    "WavesBenchmarkSuite",
    "apply_attack",
    "attack_names",
    "ber_threshold_at_fpr",
    "bit_error_rate",
    "calculate_psnr",
    "calculate_ssim",
    "compute_roc_points",
    "default_attack_suite",
    "load_samples_from_manifest",
    "select_attacks",
]
