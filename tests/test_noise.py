import torch

from models.noise import DifferentiableNoiseLayer, StandardNoiseLayer


def test_noise_layer_preserves_shape_and_output_range():
    layer = DifferentiableNoiseLayer()
    layer.train()

    images = torch.rand(4, 3, 32, 32)
    noised = layer(images)

    assert noised.shape == images.shape
    assert float(noised.min()) >= 0.0
    assert float(noised.max()) <= 1.0


def test_noise_layer_backpropagates_gradients():
    layer = DifferentiableNoiseLayer()
    layer.train()

    images = torch.rand(2, 3, 32, 32, requires_grad=True)
    noised = layer(images)
    noised.mean().backward()

    assert images.grad is not None
    assert torch.isfinite(images.grad).all()
    assert torch.count_nonzero(images.grad).item() > 0


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

    assert noised.shape == images.shape
