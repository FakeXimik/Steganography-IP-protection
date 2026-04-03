from __future__ import annotations

from typing import Tuple

import kornia.augmentation as K
import torch
from torch import Tensor, nn
from torchvision.io import ImageReadMode, decode_jpeg, encode_jpeg


class AdvancedNoiseLayer(nn.Module):
    """Phase 3 noise layer with MBRS JPEG STE and burst-error augmentations.

    The forward pass uses a real JPEG codec while the backward pass routes
    gradients through Kornia's differentiable JPEG approximation. This follows
    the Mini-Batch of Real and Simulated (MBRS) strategy from the Phase 3 spec.

    The legacy ``cover_img`` argument is accepted for compatibility with the
    current training loop, but it is not used by the augmentation pipeline.
    """

    def __init__(
        self,
        jpeg_quality: Tuple[float, float] = (35.0, 90.0),
        *,
        same_on_batch: bool = False,
        apply_in_eval: bool = False,
        affine_probability: float = 0.30,
        affine_translate: Tuple[float, float] = (0.05, 0.05),
        affine_scale: Tuple[float, float] = (0.85, 1.15),
        affine_degrees: Tuple[float, float] = (-5.0, 5.0),
        perspective_probability: float = 0.10,
        perspective_distortion: float = 0.15,
        erasing_probability: float = 0.50,
        erasing_scale: Tuple[float, float] = (0.10, 0.30),
        erasing_value: float = 0.0,
        blur_probability: float = 0.10,
        blur_kernel_size: Tuple[int, int] = (3, 3),
        blur_sigma: Tuple[float, float] = (0.1, 1.2),
    ) -> None:
        super().__init__()

        self.apply_in_eval = apply_in_eval
        self.same_on_batch = same_on_batch
        self.jpeg_quality = jpeg_quality
        self.affine_translate = affine_translate
        self.affine_scale = affine_scale
        self.affine_degrees = affine_degrees
        self.perspective_distortion = perspective_distortion
        self.erasing_scale = erasing_scale
        self.num_augmentations = 0

        self._validate_range("jpeg_quality", jpeg_quality, min_value=0.0, max_value=100.0)
        self._validate_probability("affine_probability", affine_probability)
        self._validate_probability("perspective_probability", perspective_probability)
        self._validate_probability("erasing_probability", erasing_probability)
        self._validate_probability("blur_probability", blur_probability)
        self._validate_probability("perspective_distortion", perspective_distortion)
        self._validate_range("affine_translate", affine_translate, min_value=0.0, max_value=1.0)
        self._validate_range("affine_scale", affine_scale, min_value=0.0, max_value=float("inf"))
        self._validate_range("affine_degrees", affine_degrees, min_value=-360.0, max_value=360.0)
        self._validate_range("erasing_scale", erasing_scale, min_value=0.0, max_value=1.0)

        pre_jpeg_augmentations: list[nn.Module] = []
        if erasing_probability > 0.0:
            pre_jpeg_augmentations.append(
                K.RandomErasing(
                    scale=erasing_scale,
                    value=erasing_value,
                    same_on_batch=same_on_batch,
                    p=erasing_probability,
                    keepdim=True,
                )
            )

        if affine_probability > 0.0:
            pre_jpeg_augmentations.append(
                K.RandomAffine(
                    degrees=affine_degrees,
                    translate=affine_translate,
                    scale=affine_scale,
                    padding_mode="border",
                    same_on_batch=same_on_batch,
                    p=affine_probability,
                    keepdim=True,
                )
            )

        if perspective_probability > 0.0:
            pre_jpeg_augmentations.append(
                K.RandomPerspective(
                    distortion_scale=perspective_distortion,
                    same_on_batch=same_on_batch,
                    p=perspective_probability,
                    keepdim=True,
                )
            )

        post_jpeg_augmentations: list[nn.Module] = []
        if blur_probability > 0.0:
            post_jpeg_augmentations.append(
                K.RandomGaussianBlur(
                    kernel_size=blur_kernel_size,
                    sigma=blur_sigma,
                    same_on_batch=same_on_batch,
                    p=blur_probability,
                    keepdim=True,
                )
            )

        self.pre_jpeg_pipeline = (
            K.AugmentationSequential(
                *pre_jpeg_augmentations,
                data_keys=["input"],
                same_on_batch=same_on_batch,
                keepdim=True,
            )
            if pre_jpeg_augmentations
            else nn.Identity()
        )
        self.simulated_jpeg = K.RandomJPEG(
            jpeg_quality=jpeg_quality,
            same_on_batch=same_on_batch,
            p=1.0,
            keepdim=True,
        )
        self.post_jpeg_pipeline = (
            K.AugmentationSequential(
                *post_jpeg_augmentations,
                data_keys=["input"],
                same_on_batch=same_on_batch,
                keepdim=True,
            )
            if post_jpeg_augmentations
            else nn.Identity()
        )
        self.num_augmentations = len(pre_jpeg_augmentations) + 1 + len(post_jpeg_augmentations)

    def forward(self, stego_img: Tensor, cover_img: Tensor | None = None) -> Tensor:
        del cover_img

        image, squeeze_batch = self._prepare_input(stego_img)
        image = image.clamp(0.0, 1.0)

        if not self.training and not self.apply_in_eval:
            output = image
        else:
            output = self.pre_jpeg_pipeline(image)
            output = self._apply_jpeg_ste(output)
            output = self.post_jpeg_pipeline(output)

        output = output.clamp(0.0, 1.0)
        return output.squeeze(0) if squeeze_batch else output

    def extra_repr(self) -> str:
        return (
            f"jpeg_quality={self.jpeg_quality}, "
            f"affine_translate={self.affine_translate}, "
            f"affine_scale={self.affine_scale}, "
            f"affine_degrees={self.affine_degrees}, "
            f"erasing_scale={self.erasing_scale}, "
            f"apply_in_eval={self.apply_in_eval}, "
            f"augmentations={self.num_augmentations}"
        )

    def _apply_jpeg_ste(self, image: Tensor) -> Tensor:
        jpeg_params = self.simulated_jpeg.forward_parameters(image.shape)
        sim_jpeg = self.simulated_jpeg(image, params=jpeg_params)
        real_jpeg = self._apply_real_jpeg(image.detach(), jpeg_params["jpeg_quality"])
        return real_jpeg + sim_jpeg - sim_jpeg.detach()

    @staticmethod
    def _apply_real_jpeg(image: Tensor, jpeg_quality: Tensor) -> Tensor:
        cpu_image = image.clamp(0.0, 1.0).mul(255.0).round().to(dtype=torch.uint8, device="cpu")
        quality_values = jpeg_quality.detach().flatten().to(dtype=torch.float32, device="cpu")

        decoded_images = []
        for sample, quality in zip(cpu_image, quality_values, strict=True):
            encoded = encode_jpeg(sample.contiguous(), quality=int(quality.round().clamp(1, 100).item()))
            decoded = decode_jpeg(encoded, mode=ImageReadMode.RGB)
            decoded_images.append(decoded.to(dtype=torch.float32).div_(255.0))

        return torch.stack(decoded_images, dim=0).to(device=image.device, dtype=image.dtype)

    @staticmethod
    def _prepare_input(stego_img: Tensor) -> tuple[Tensor, bool]:
        if not torch.is_floating_point(stego_img):
            raise TypeError("AdvancedNoiseLayer expects a floating point tensor.")

        if stego_img.ndim == 3:
            if stego_img.shape[0] != 3:
                raise ValueError("Expected input shape (3, H, W) for a single RGB image.")
            return stego_img.unsqueeze(0), True

        if stego_img.ndim == 4:
            if stego_img.shape[1] != 3:
                raise ValueError("Expected input shape (B, 3, H, W) for batched RGB images.")
            return stego_img, False

        raise ValueError("AdvancedNoiseLayer expects a tensor shaped (3, H, W) or (B, 3, H, W).")

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
            raise ValueError(f"{name} must stay within [{min_value}, {max_value}], got {value}.")


class DifferentiableNoiseLayer(AdvancedNoiseLayer):
    """Backward-compatible alias for the Phase 3 implementation."""


class StandardNoiseLayer(AdvancedNoiseLayer):
    """Backward-compatible alias for the training loop."""


__all__ = ["AdvancedNoiseLayer", "DifferentiableNoiseLayer", "StandardNoiseLayer"]
