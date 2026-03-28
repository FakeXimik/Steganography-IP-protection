import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import uuid

# ==========================================
# 1. TEAM IMPORTS 
# ==========================================
from models.encoder_CIFAR import EncoderCIFAR
from models.decoder_CIFAR import DecoderCIFAR
from models.noise import StandardNoiseLayer
from data.cifar_loader import CIFAR100
from utils.fec import RSCodecPipeline

# ==========================================
# 2. DISCRIMINATOR 
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
    # ADDED lambda_m=10.0 to the initialization
    def __init__(self, lambda_m=10.0, lambda_i=0.1, lambda_g=0.001):
        super().__init__()
        self.mse_loss = nn.MSELoss() 
        self.bce_with_logits = nn.BCEWithLogitsLoss() 
        self.lambda_m = lambda_m
        self.lambda_i = lambda_i
        self.lambda_g = lambda_g

    def forward(self, cover, stego, msg_in, msg_out, discriminator_logits):
        l_i = self.mse_loss(stego, cover)
        l_m = self.bce_with_logits(msg_out, msg_in)
        real_labels = torch.ones_like(discriminator_logits)
        l_g = self.bce_with_logits(discriminator_logits, real_labels)
        
        # Aself.lambda_m to l_m
        total_loss = (self.lambda_m * l_m) + (self.lambda_i * l_i) + (self.lambda_g * l_g)
        return total_loss, l_i, l_m, l_g

def calculate_ber(decoded_logits, original_message):
    probabilities = torch.sigmoid(decoded_logits)
    predicted_bits = (probabilities > 0.5).float()
    errors = (predicted_bits != original_message).sum().item()
    return errors / original_message.numel()

# ==========================================
# GLOBAL FEC INITIALIZATION
# ==========================================
FEC_PIPELINE = RSCodecPipeline(parity_symbols=10)

def generate_real_payloads(batch_size, device):
    all_bits = []
    for _ in range(batch_size):
        raw_uuid_bytes = uuid.uuid4().bytes
        encoded_bytes = FEC_PIPELINE.rs.encode(raw_uuid_bytes)
        
        bits = []
        for byte in encoded_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        all_bits.append(bits)
        
    return torch.tensor(all_bits, dtype=torch.float32).to(device)

# ==========================================
# THE MASTER TRAINING LOOP
# ==========================================
def run_training_loop():
    print("\n--- INITIALIZING HiDDeN PRODUCTION PIPELINE (RTX 4080 OPTIMIZED) ---")
    
    # --- Configuration ---
    batch_size = 64 
    payload_length = 208 
    epochs = 80         
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device} | Batch Size: {batch_size} | Payload: {payload_length} bits")
    
    # --- Data & Models ---
    cifar_dataset = CIFAR100(batch_size=batch_size)
    train_loader = cifar_dataset.train_loader
    
    encoder = EncoderCIFAR(payload_length).to(device)
    decoder = DecoderCIFAR(payload_length).to(device)
    discriminator = HiddenDiscriminator().to(device) 
    noise_layer = StandardNoiseLayer().to(device)
    
    # lambda_m=10.0
    criterion = HiDDeNLoss(lambda_m=10.0, lambda_i=0.1, lambda_g=0.001).to(device)
    
    opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)
    
    print("Starting full dataset training...")

    for epoch in range(epochs):
        # --- CURRICULUM LEARNING ---
        if epoch == 25:
            print("\n[SYSTEM] Increasing Image Fidelity Constraint (lambda_i -> 0.5)")
            criterion.lambda_i = 0.7
            
        encoder.train()
        decoder.train()

        for i, (images, _) in enumerate(train_loader):
            cover_images = images.to(device)
            current_batch_size = cover_images.size(0)
            
            payloads = generate_real_payloads(current_batch_size, device)
            
            # ----------------------------
            # STEP A: Train Discriminator
            # ----------------------------
            opt_disc.zero_grad()
            
            stego_images = encoder(cover_images, payloads).detach() 
            d_real_logits = discriminator(cover_images)
            d_fake_logits = discriminator(stego_images)
            
            d_loss_real = criterion.bce_with_logits(d_real_logits, torch.ones_like(d_real_logits))
            d_loss_fake = criterion.bce_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            d_loss.backward()
            opt_disc.step()
            
            # ----------------------------
            # STEP B: Train Encoder/Decode
            # ----------------------------
            opt_enc_dec.zero_grad()
            
            stego_images = encoder(cover_images, payloads)
            
            # NOISE RAMPING
            if epoch > 15:
                noisy_images = noise_layer(stego_images, cover_images) 
            else:
                noisy_images = stego_images
                
            extracted_payloads = decoder(noisy_images)
            d_fake_logits_for_gen = discriminator(stego_images)
            
            loss, l2_loss, msg_loss, g_loss = criterion(
                cover_images, stego_images, payloads, extracted_payloads, d_fake_logits_for_gen
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            
            opt_enc_dec.step()
            
            # ----------------------------
            # STEP C: Logging
            # ----------------------------
            if (i+1) % 100 == 0: 
                ber = calculate_ber(extracted_payloads, payloads)
                noise_status = "ON" if epoch > 15 else "OFF"
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(train_loader)}] "
                      f"| BER: {ber:.2%} | Noise: {noise_status} | L_Img: {l2_loss.item():.4f}")

        # Save checkpoint after every 10 epochs
        if (epoch + 1) % 10 == 0:
            os.makedirs("saved_models", exist_ok=True)
            torch.save(encoder.state_dict(), f"saved_models/encoder_epoch_{epoch+1}.pth")
            # ADDED DECODER SAVE
            torch.save(decoder.state_dict(), f"saved_models/decoder_epoch_{epoch+1}.pth")
            print(f"--> Checkpoint saved for Epoch {epoch+1}")

    print("\n--- FINALIZING PRODUCTION MODELS ---")
    torch.save(encoder.state_dict(), "saved_models/hidden_encoder_final.pth")
    torch.save(decoder.state_dict(), "saved_models/hidden_decoder_final.pth")
    print("Training Complete. Final weights saved to /saved_models/")
    
if __name__ == "__main__":
    run_training_loop()