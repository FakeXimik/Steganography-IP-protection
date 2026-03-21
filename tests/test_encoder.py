import torch
import pytest
from models.encoder_CIFAR import EncoderCIFAR

def test_encoder_dimensions():
    # 1. Setup variables
    batch_size = 4
    message_length = 64
    
    # 2. Initialize the model
    encoder = EncoderCIFAR(message_length=message_length)
    
    # 3. Generate dummy data
    dummy_images = torch.randn(batch_size, 3, 32, 32)
    dummy_payloads = torch.randint(0, 2, (batch_size, message_length)).float()
    
    # 4. Forward Pass
    stego_images = encoder(dummy_images, dummy_payloads)
    
    # 5. Verify the output shape matches a standard image exactly
    assert stego_images.shape == (batch_size, 3, 32, 32), "Encoder warped the image dimensions!"