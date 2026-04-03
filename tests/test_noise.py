import torch

from models.noise import AdvancedNoiseLayer, DifferentiableNoiseLayer, StandardNoiseLayer


def test_advanced_noise_layer_defaults_match_phase_3_spec():
    layer = AdvancedNoiseLayer()

    assert layer.affine_translate == (0.05, 0.05)
    assert layer.affine_scale == (0.85, 1.15)
    assert layer.affine_degrees == (-5.0, 5.0)
    assert layer.erasing_scale == (0.10, 0.30)


def test_noise_layer_preserves_shape_and_output_range():
    layer = DifferentiableNoiseLayer()
    layer.train()

    images = torch.rand(4, 3, 32, 32)
    noised = layer(images)

    assert noised.shape == images.shape
    assert float(noised.min()) >= 0.0
    assert float(noised.max()) <= 1.0


def test_noise_layer_backpropagates_gradients_through_jpeg_ste():
    layer = AdvancedNoiseLayer(
        jpeg_quality=(70.0, 70.0),
        affine_probability=0.0,
        perspective_probability=0.0,
        erasing_probability=0.0,
        blur_probability=0.0,
    )
    layer.train()

    images = torch.rand(2, 3, 32, 32, requires_grad=True)
    noised = layer(images)
    noised.mean().backward()

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert torch.count_nonzero(images.grad).item() > 0


def test_noise_layer_supports_single_image_inputs():
    layer = AdvancedNoiseLayer()
    layer.train()

    image = torch.rand(3, 32, 32)
    noised = layer(image)

    assert noised.shape == image.shape


def test_noise_layer_is_passthrough_in_eval_mode_by_default():
    layer = DifferentiableNoiseLayer()
    layer.eval()

    images = torch.rand(2, 3, 32, 32)
    noised = layer(images)

    assert torch.allclose(noised, images)


def test_standard_noise_layer_alias_uses_new_implementation():
    layer = StandardNoiseLayer()
    layer.train()

    images = torch.rand(2, 3, 32, 32)
    cover = torch.rand(2, 3, 32, 32)
    noised = layer(images, cover)

    assert isinstance(layer, AdvancedNoiseLayer)
    assert noised.shape == images.shape
