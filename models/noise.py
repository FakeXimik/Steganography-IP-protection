from __future__ import annotations

from typing import Tuple

import kornia.augmentation as K
import torch
from torch import Tensor, nn


class DifferentiableNoiseLayer(nn.Module):
    """Differentiable corruption layer for training robust watermark models.

    JPEG simulation is always present during training because it is the critical
    distortion the decoder must learn to survive. Lighter geometric and blur
    corruptions can be mixed in to mimic common post-processing pipelines.

    The legacy ``cover_img`` argument is accepted for compatibility with the
    current training loop, but it is not used by the differentiable pipeline.
    """

    def __init__(
        self,
        jpeg_quality: Tuple[float, float] = (35.0, 90.0),
        *,
        same_on_batch: bool = False,
        apply_in_eval: bool = False,
        affine_probability: float = 0.30,
        affine_translate: Tuple[float, float] = (0.03, 0.03),
        affine_scale: Tuple[float, float] = (0.97, 1.03),
        perspective_probability: float = 0.10,
        perspective_distortion: float = 0.15,
        blur_probability: float = 0.10,
        blur_kernel_size: Tuple[int, int] = (3, 3),
        blur_sigma: Tuple[float, float] = (0.1, 1.2),
    ) -> None:
        super().__init__()
        self.apply_in_eval = apply_in_eval
        self.jpeg_quality = jpeg_quality
        self.num_augmentations = 0
        self._validate_range("jpeg_quality", jpeg_quality, min_value=0.0, max_value=100.0)
        self._validate_probability("affine_probability", affine_probability)
        self._validate_probability("perspective_probability", perspective_probability)
        self._validate_probability("blur_probability", blur_probability)
        self._validate_probability("perspective_distortion", perspective_distortion)

        augmentations: list[nn.Module] = [
            K.RandomJPEG(
                jpeg_quality=jpeg_quality,
                same_on_batch=same_on_batch,
                p=1.0,
                keepdim=True,
            )
        ]

        if affine_probability > 0.0:
            augmentations.append(
                K.RandomAffine(
                    degrees=0.0,
                    translate=affine_translate,
                    scale=affine_scale,
                    padding_mode="border",
                    same_on_batch=same_on_batch,
                    p=affine_probability,
                    keepdim=True,
                )
            )

        if perspective_probability > 0.0:
            augmentations.append(
                K.RandomPerspective(
                    distortion_scale=perspective_distortion,
                    same_on_batch=same_on_batch,
                    p=perspective_probability,
                    keepdim=True,
                )
            )

        if blur_probability > 0.0:
            augmentations.append(
                K.RandomGaussianBlur(
                    kernel_size=blur_kernel_size,
                    sigma=blur_sigma,
                    same_on_batch=same_on_batch,
                    p=blur_probability,
                    keepdim=True,
                )
            )

        self.pipeline = K.AugmentationSequential(
            *augmentations,
            data_keys=["input"],
            same_on_batch=same_on_batch,
            keepdim=True,
        )
        self.num_augmentations = len(augmentations)

    def forward(self, stego_img: Tensor, cover_img: Tensor | None = None) -> Tensor:
        del cover_img

        image, squeeze_batch = self._prepare_input(stego_img)
        image = image.clamp(0.0, 1.0)

        if not self.training and not self.apply_in_eval:
            output = image
        else:
            output = self.pipeline(image)

        output = output.clamp(0.0, 1.0)
        return output.squeeze(0) if squeeze_batch else output

    def extra_repr(self) -> str:
        return (
            f"jpeg_quality={self.jpeg_quality}, "
            f"apply_in_eval={self.apply_in_eval}, "
            f"augmentations={self.num_augmentations}"
        )

    @staticmethod
    def _prepare_input(stego_img: Tensor) -> tuple[Tensor, bool]:
        if not torch.is_floating_point(stego_img):
            raise TypeError("DifferentiableNoiseLayer expects a floating point tensor.")

        if stego_img.ndim == 3:
            if stego_img.shape[0] != 3:
                raise ValueError("Expected input shape (3, H, W) for a single RGB image.")
            return stego_img.unsqueeze(0), True

        if stego_img.ndim == 4:
            if stego_img.shape[1] != 3:
                raise ValueError("Expected input shape (B, 3, H, W) for batched RGB images.")
            return stego_img, False

        raise ValueError("DifferentiableNoiseLayer expects a tensor shaped (3, H, W) or (B, 3, H, W).")

    @staticmethod
    def _validate_probability(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}.")

    @staticmethod
    def _validate_range(
        name: str,
        value: Tuple[float, float],
        *,
        min_value: float,
        max_value: float,
    ) -> None:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two values.")

        start, end = value
        if start > end:
            raise ValueError(f"{name} must be an increasing range, got {value}.")
        if start < min_value or end > max_value:
            raise ValueError(
                f"{name} must stay within [{min_value}, {max_value}], got {value}."
            )


class StandardNoiseLayer(DifferentiableNoiseLayer):
    """Backward-compatible alias for the training loop."""


__all__ = ["DifferentiableNoiseLayer", "StandardNoiseLayer"]
