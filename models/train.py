import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import uuid

# ==========================================
# 1. TEAM IMPORTS (Modular Architecture)
# ==========================================
# Importing from the newly named CIFAR files
from models.encoder_CIFAR import EncoderCIFAR
from models.decoder_CIFAR import DecoderCIFAR
from models.noise import StandardNoiseLayer

from utils.fec import RSCodecPipeline

# ==========================================
# 2. DISCRIMINATOR (Kept local since no file exists yet)
# ==========================================
class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class HiddenDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(*[ConvBNRelu(3 if i==0 else 64, 64) for i in range(3)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 1) 

    def forward(self, img):
        x = self.convs(img)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.linear(x)

# ==========================================
# 3. LOSS FUNCTIONS & METRICS
# ==========================================
class HiDDeNLoss(nn.Module):
    def __init__(self, lambda_i=0.7, lambda_g=0.001):
        super().__init__()
        self.mse_loss = nn.MSELoss() 
        self.bce_with_logits = nn.BCEWithLogitsLoss() 
        self.lambda_i = lambda_i
        self.lambda_g = lambda_g

    def forward(self, cover, stego, msg_in, msg_out, discriminator_logits):
        l_i = self.mse_loss(stego, cover)
        l_m = self.bce_with_logits(msg_out, msg_in)
        real_labels = torch.ones_like(discriminator_logits)
        l_g = self.bce_with_logits(discriminator_logits, real_labels)
        total_loss = l_m + (self.lambda_i * l_i) + (self.lambda_g * l_g)
        return total_loss, l_i, l_m, l_g

def calculate_ber(decoded_logits, original_message):
    probabilities = torch.sigmoid(decoded_logits)
    predicted_bits = (probabilities > 0.5).float()
    errors = (predicted_bits != original_message).sum().item()
    return errors / original_message.numel()

# ==========================================
# 4. REAL UUID PAYLOAD GENERATOR
# ==========================================
def generate_real_payloads(batch_size, device):
    """Generates real UUIDs, applies Reed-Solomon FEC, and converts to bit tensors."""
    fec = RSCodecPipeline(parity_symbols=10)
    batch_payloads = []
    
    for _ in range(batch_size):
        dummy_uuid = str(uuid.uuid4())
        encoded_bytes = fec.encode_uuid(dummy_uuid) # 16 UUID + 10 Parity = 26 bytes
        
        # Convert bytes to bits (0s and 1s)
        bits = []
        for byte in encoded_bytes:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        batch_payloads.append(bits)
        
    return torch.tensor(batch_payloads, dtype=torch.float32).to(device)

# ==========================================
# 5. THE MASTER TRAINING LOOP
# ==========================================
def run_training_loop():
    print("\n--- INITIALIZING HiDDeN PHASE 3 (PRODUCTION PIPELINE) ---")
    batch_size = 128 
    payload_length = 208 # 26 bytes * 8 bits
    epochs = 5
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    print("Ingesting CIFAR-100 Dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    
    # SWAPPED TO CIFAR-100
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    
    # Initialize imported networks
    # Note: Assuming your teammates kept the class names 'HiddenEncoder', 'HiddenDecoder', etc.
    encoder = EncoderCIFAR(payload_length).to(device)
    decoder = DecoderCIFAR(payload_length).to(device)
    discriminator = HiddenDiscriminator().to(device) 
    noise_layer = StandardNoiseLayer().to(device)
    
    criterion = HiDDeNLoss(lambda_i=0.7, lambda_g=0.001).to(device)
    opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            cover_images = images.to(device)
            current_batch_size = cover_images.size(0)
            
            # USE REAL REED-SOLOMON UUIDS INSTEAD OF RANDOM BITS
            payloads = generate_real_payloads(current_batch_size, device)
            
            # STEP A: Train Discriminator
            opt_disc.zero_grad()
            stego_images = encoder(cover_images, payloads).detach() 
            d_real_logits = discriminator(cover_images)
            d_fake_logits = discriminator(stego_images)
            d_loss_real = criterion.bce_with_logits(d_real_logits, torch.ones_like(d_real_logits))
            d_loss_fake = criterion.bce_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_disc.step()
            
            # STEP B: Train Encoder/Decoder
            opt_enc_dec.zero_grad()
            stego_images = encoder(cover_images, payloads)
            noisy_images = noise_layer(stego_images, cover_images) 
            extracted_payloads = decoder(noisy_images)
            d_fake_logits_for_gen = discriminator(stego_images)
            
            loss, l2_loss, msg_loss, g_loss = criterion(cover_images, stego_images, payloads, extracted_payloads, d_fake_logits_for_gen)
            loss.backward()
            opt_enc_dec.step()
            
            ber = calculate_ber(extracted_payloads, payloads)
            
            if (i+1) % 100 == 0: 
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}] "
                      f"| Total Loss: {loss.item():.4f} | D_Loss: {d_loss.item():.4f} | BER: {ber:.2%}")

        print(f"--> Epoch {epoch+1} Completed!")

    print("\n--- SAVING PRODUCTION MODELS ---")
    os.makedirs("saved_models", exist_ok=True)
    torch.save(encoder.state_dict(), "saved_models/hidden_encoder.pth")
    torch.save(decoder.state_dict(), "saved_models/hidden_decoder.pth")
    print("Model weights saved to /saved_models/")

if __name__ == "__main__":
    run_training_loop()
