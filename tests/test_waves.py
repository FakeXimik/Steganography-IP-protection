import random

import numpy as np
import pytest
from PIL import Image

from WAVES import (
    AttackSpec,
    AttackStep,
    apply_attack,
    ber_threshold_at_fpr,
    calculate_psnr,
    calculate_ssim,
    compute_roc_points,
    default_attack_suite,
    resize_to_max_dimension,
)


def make_test_image(size: int = 64) -> Image.Image:
    x = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    y = x.T
    z = ((x.astype(np.uint16) + y.astype(np.uint16)) // 2).astype(np.uint8)
    rgb = np.stack([x, y, z], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def test_default_attack_suite_includes_phase4_core_attacks():
    names = {attack.name for attack in default_attack_suite()}

    assert "jpeg_q10" in names
    assert "crop_50pct" in names
    assert "scale_down_50pct_restore" in names
    assert "rotate_15deg" in names
    assert "gaussian_blur_r2_0" in names
    assert "median_filter_5" in names
    assert "jpeg_q30_crop_20pct" in names


def test_crop_attack_preserves_original_dimensions():
    image = make_test_image()
    attack = AttackSpec("crop_30pct", (AttackStep("crop", 0.30),))

    attacked = apply_attack(image, attack, random.Random(1337))

    assert attacked.size == image.size


def test_resize_to_max_dimension_caps_longest_side_and_preserves_aspect_ratio():
    image = make_test_image(size=160)
    image = image.resize((1600, 800))

    resized = resize_to_max_dimension(image, 512)

    assert resized.size == (512, 256)


def test_identical_images_have_perfect_similarity_metrics():
    image = make_test_image()

    psnr = calculate_psnr(image, image)
    ssim = calculate_ssim(image, image)

    assert psnr == pytest.approx(float("inf"))
    assert ssim == pytest.approx(1.0, abs=1e-6)


def test_threshold_at_fpr_does_not_exceed_requested_rate():
    negative_bers = [0.41, 0.43, 0.44, 0.49]

    threshold, achieved_fpr = ber_threshold_at_fpr(negative_bers, 0.10)

    assert threshold < min(negative_bers)
    assert achieved_fpr <= 0.10


def test_roc_points_are_monotonic():
    points = compute_roc_points(
        negative_bers=[0.41, 0.43, 0.50],
        positive_bers=[0.10, 0.12, 0.20],
    )

    fprs = [point["fpr"] for point in points]
    tprs = [point["tpr"] for point in points]

    assert fprs == sorted(fprs)
    assert tprs == sorted(tprs)
