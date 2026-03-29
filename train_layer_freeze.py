import torch
import torch.nn as nn
import torch.optim as optim
import os
import uuid

# ==========================================
# 1. PHASE 3 IMPORTS 
# ==========================================
from models.encoder_dense import EncoderDense
from models.decoder_dense import DecoderDense
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
    def __init__(self, lambda_m=10.0, lambda_i=0.3, lambda_g=0.001):
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
        
        total_loss = (self.lambda_m * l_m) + (self.lambda_i * l_i) + (self.lambda_g * l_g)
        return total_loss, l_i, l_m, l_g

def calculate_ber(decoded_logits, original_message, payload_length=208):
    """
    PHASE 3 UPDATE: Flattens the 32x32 spatial map and isolates 
    only the first 208 bits to calculate the true UUID error rate.
    """
    flat_logits = decoded_logits.view(decoded_logits.size(0), -1)[:, :payload_length]
    flat_original = original_message.view(original_message.size(0), -1)[:, :payload_length]
    
    probabilities = torch.sigmoid(flat_logits)
    predicted_bits = (probabilities > 0.5).float()
    errors = (predicted_bits != flat_original).sum().item()
    return errors / flat_original.numel()

# ==========================================
# GLOBAL FEC INITIALIZATION & PAYLOAD GENERATOR
# ==========================================
FEC_PIPELINE = RSCodecPipeline(parity_symbols=10)

def generate_spatial_payloads(batch_size, device, spatial_size=32, data_depth=8):
    """
    PHASE 3 UPDATE: Converts the 208-bit UUID into a padded 
    8-channel spatial map to perfectly align with SteganoGAN's weights.
    """
    all_bits = []
    for _ in range(batch_size):
        raw_uuid_bytes = uuid.uuid4().bytes
        encoded_bytes = FEC_PIPELINE.rs.encode(raw_uuid_bytes)
        
        bits = []
        for byte in encoded_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        total_capacity = data_depth * spatial_size * spatial_size
        padding_length = total_capacity - len(bits)
        bits.extend([0] * padding_length)
        
        bits_tensor = torch.tensor(bits, dtype=torch.float32).view(data_depth, spatial_size, spatial_size)
        all_bits.append(bits_tensor)
        
    return torch.stack(all_bits).to(device)


# ==========================================
# PHASE 3 INITIALIZATION & LAYER FREEZE
# ==========================================
def build_and_freeze_phase3_models(device):
    print("\n--- INITIATING PHASE 3: HYBRID VRAM DEFENSE ---")
    data_depth = 8 
    
    encoder = EncoderDense(data_depth=data_depth).to(device)
    decoder = DecoderDense(data_depth=data_depth).to(device)
    
    print("Loading pre-trained Dense weights from /weights/...")
    encoder_path = os.path.join('weights', 'encoder_dense_pretrained.pth')
    decoder_path = os.path.join('weights', 'decoder_dense_pretrained.pth')
    
    encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
    decoder.load_state_dict(torch.load(decoder_path, weights_only=False))
    
    # ==========================================
    # HYBRID LAYER FREEZING 
    # ==========================================
    print("Executing Hybrid Layer Freeze to protect VRAM...")
    
    # ENCODER: FROZEN
    for param in encoder.conv1.parameters(): param.requires_grad = False
    for param in encoder.conv2.parameters(): param.requires_grad = False
    for param in encoder.conv3.parameters(): param.requires_grad = False
    
    # DECODER: UNFROZEN.

    enc_frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    enc_active = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_frozen = sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    dec_active = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    print(f"[Encoder] Frozen: {enc_frozen:,} | Active: {enc_active:,} (VRAM Protected)")
    print(f"[Decoder] Frozen: {dec_frozen:,} | Active: {dec_active:,} (Fully Active)")
    
    return encoder, decoder

# ==========================================
# THE MASTER TRAINING LOOP
# ==========================================
def run_training_loop():
    print("\n--- INITIALIZING HiDDeN PRODUCTION PIPELINE (PHASE 3) ---")
    
    # --- Configuration ---
    # We drop the physical batch size to 32, but accumulate 4 times to simulate 64.
    # This prepares the loop architecture for 512x512 images later.
    physical_batch_size = 32
    accumulation_steps = 4 
    
    epochs = 80         
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Device: {device} | Phys Batch: {physical_batch_size} | Accumulation: {accumulation_steps}")
    
    # --- Data & Models ---
    cifar_dataset = CIFAR100(batch_size=physical_batch_size)
    train_loader = cifar_dataset.train_loader
    
    encoder, decoder = build_and_freeze_phase3_models(device)
    discriminator = HiddenDiscriminator().to(device) 
    noise_layer = StandardNoiseLayer().to(device)
    
    criterion = HiDDeNLoss(lambda_m=10.0, lambda_i=0.3, lambda_g=0.001).to(device)
    
    # CRITICAL: We MUST filter out frozen parameters from the optimizer!
    active_params = [p for p in encoder.parameters() if p.requires_grad] + \
                    [p for p in decoder.parameters() if p.requires_grad]
                    
    opt_enc_dec = optim.Adam(active_params, lr=lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)
    
    print("Starting Phase 3 training...")

    opt_disc.zero_grad()
    opt_enc_dec.zero_grad()

    for epoch in range(epochs):
        # --- CURRICULUM LEARNING ---
        if epoch == 10:
            print("\n[SYSTEM] Increasing Image Fidelity Constraint (lambda_i -> 0.7)")
            criterion.lambda_i = 0.7
            
        encoder.train()
        decoder.train()

        for i, (images, _) in enumerate(train_loader):
            cover_images = images.to(device)
            current_batch_size = cover_images.size(0)
            
            payloads = generate_spatial_payloads(current_batch_size, device)
            
            # ----------------------------
            # STEP A: Train Discriminator
            # ----------------------------
            stego_images = encoder(cover_images, payloads).detach() 
            d_real_logits = discriminator(cover_images)
            d_fake_logits = discriminator(stego_images)
            
            d_loss_real = criterion.bce_with_logits(d_real_logits, torch.ones_like(d_real_logits))
            d_loss_fake = criterion.bce_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            
            # Divide loss by accumulation steps
            d_loss = ((d_loss_real + d_loss_fake) / 2) / accumulation_steps
            d_loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                opt_disc.step()
                opt_disc.zero_grad()
            
            # ----------------------------
            # STEP B: Train Encoder/Decode
            # ----------------------------
            stego_images = encoder(cover_images, payloads)
            
            # NOISE RAMPING
            if epoch > 1:
                noisy_images = noise_layer(stego_images, cover_images) 
            else:
                noisy_images = stego_images
                
            extracted_payloads = decoder(noisy_images)
            d_fake_logits_for_gen = discriminator(stego_images)
            
            loss, l2_loss, msg_loss, g_loss = criterion(
                cover_images, stego_images, payloads, extracted_payloads, d_fake_logits_for_gen
            )
            
            # Divide loss by accumulation steps
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(active_params, 1.0)
                opt_enc_dec.step()
                opt_enc_dec.zero_grad()
            
            # ----------------------------
            # STEP C: Logging
            # ----------------------------
            if (i+1) % 100 == 0: 
                ber = calculate_ber(extracted_payloads, payloads)
                noise_status = "ON" if epoch > 1 else "OFF"
                print(f"Epoch [{epoch+1}/{epochs}] | Step [{i+1}/{len(train_loader)}] "
                      f"| BER: {ber:.2%} | Noise: {noise_status} | L_Img: {l2_loss.item():.4f}")

        # Save checkpoints
        if (epoch + 1) % 10 == 0:
            os.makedirs("saved_models", exist_ok=True)
            torch.save(encoder.state_dict(), f"saved_models/encoder_epoch_{epoch+1}.pth")
            torch.save(decoder.state_dict(), f"saved_models/decoder_epoch_{epoch+1}.pth")
            print(f"--> Checkpoint saved for Epoch {epoch+1}")

    print("\n--- FINALIZING PRODUCTION MODELS ---")
    torch.save(encoder.state_dict(), "saved_models/hidden_encoder_final.pth")
    torch.save(decoder.state_dict(), "saved_models/hidden_decoder_final.pth")
    print("Training Complete. Final weights saved to /saved_models/")
    
if __name__ == "__main__":
    run_training_loop()