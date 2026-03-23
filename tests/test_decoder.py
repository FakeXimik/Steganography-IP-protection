import torch
import pytest
from models.decoder_CIFAR import DecoderCIFAR

def test_decoder_dimensions():
    # 1. Setup variables
    batch_size = 4
    message_length = 64
    
    # 2. Initialize the model
    decoder = DecoderCIFAR(message_length=message_length)
    
    # 3. Generate dummy stego-image (simulating the Encoder's output)
    dummy_stego_images = torch.randn(batch_size, 3, 32, 32)
    
    # 4. Extraction Pass
    extracted_payloads = decoder(dummy_stego_images)
    
    # 5. Verify the output shape matches the original payload perfectly
    assert extracted_payloads.shape == (batch_size, message_length), "Decoder failed to extract the correct payload length!"