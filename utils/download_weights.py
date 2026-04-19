import torch
import torch.optim as optim
from steganogan import SteganoGAN
import sys
import types
import os

print("Applying PyTorch 2.6 Security & Compatibility bypasses...")

# --- PATCH 1: Bypass the PyTorch 2.6 strict loading security ---
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

# --- PATCH 2: Dummy Adam Optimizer ---
class DummyAdam:
    def __init__(self, *args, **kwargs):
        self.defaults = {}
    def __setstate__(self, state):
        self.__dict__.update(state)
        # Ensure 'defaults' exists to satisfy PyTorch 2.6's internal checks
        self.defaults = getattr(self, 'defaults', {})

_original_adam = optim.Adam
optim.Adam = DummyAdam

# --- PATCH 3: The System Injection ---
fake_module = types.ModuleType('torch.optim.adam')
fake_module.Adam = DummyAdam
sys.modules['torch.optim.adam'] = fake_module

print("Downloading and unpacking SteganoGAN Dense weights from AWS...")
steg = SteganoGAN.load(architecture='dense')

# --- NEW: Create the weights directory ---
print("\nPreparing the 'weights/' directory...")
os.makedirs('weights', exist_ok=True)

print("Extracting pure Encoder and Decoder state dictionaries...")
encoder_path = os.path.join('weights', 'encoder_dense_pretrained.pth')
decoder_path = os.path.join('weights', 'decoder_dense_pretrained.pth')

# Save directly into the new folder
torch.save(steg.encoder.state_dict(), encoder_path)
torch.save(steg.decoder.state_dict(), decoder_path)

print("\nSuccess! Weights saved as standalone .pth files:")
print(f" -> {encoder_path}")
print(f" -> {decoder_path}")

# Clean up our patches
torch.load = _original_load
optim.Adam = _original_adam
del sys.modules['torch.optim.adam']
