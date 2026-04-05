import os
import logging
from datetime import datetime
import uuid
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import lpips  # Perceptual Loss

# ==========================================
# IMPORTS
# ==========================================
from models.encoder_dense import EncoderDense
from models.decoder_dense import DecoderDense
from models.noise import AdvancedNoiseLayer
from utils.fec import RSCodecPipeline

# ==========================================
# LOGGING
# ==========================================

os.makedirs("logs", exist_ok=True)

log_filename = datetime.now().strftime("logs/training_%Y-%m-%d_%H-%M-%S.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Writes to the file
        logging.StreamHandler()             # Prints to the console
    ]
)

logger = logging.getLogger(__name__)

logger.info("--- TRAINING SCRIPT STARTED ---")

# ==========================================
# HIGH-RESOLUTION DATA PIPELINE
# ==========================================
class HighResImageFolder(Dataset):
    """
    Custom loader for massive images (MS-COCO/Div2K) to prevent GPU bottlenecks.
    Pulls pristine 512x512 patches instead of warping the image.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Extracts pristine patches without distorting the frequency domain
        self.transform = transforms.Compose([
            transforms.RandomCrop(512, pad_if_needed=True), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except Exception:
            # Fallback for corrupted images in massive datasets
            return self.__getitem__((idx + 1) % len(self))

# ==========================================
# DISCRIMINATOR
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
#  PERCEPTUAL LOSS & METRICS
# ==========================================
class HighResHiDDeNLoss(nn.Module):
    def __init__(self, device, lambda_m=10.0, lambda_i=0.3, lambda_g=0.001, lambda_p=1.5):
        super().__init__()
        self.mse_loss = nn.MSELoss() 
        self.bce_with_logits = nn.BCEWithLogitsLoss() 
        # Integration of Learned Perceptual Image Patch Similarity
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        
        self.lambda_m = lambda_m
        self.lambda_i = lambda_i
        self.lambda_g = lambda_g
        self.lambda_p = lambda_p

    def forward(self, cover, stego, msg_in, msg_out, discriminator_logits):
        l_i = self.mse_loss(stego, cover)
        l_m = self.bce_with_logits(msg_out, msg_in)
        real_labels = torch.ones_like(discriminator_logits)
        l_g = self.bce_with_logits(discriminator_logits, real_labels)
        
        # Calculate Perceptual Loss across the batch
        l_p = self.lpips_loss(stego, cover).mean()
        
        total_loss = (self.lambda_m * l_m) + (self.lambda_i * l_i) + (self.lambda_g * l_g) + (self.lambda_p * l_p)
        return total_loss, l_i, l_m, l_g, l_p

def calculate_ber(decoded_logits, original_payloads, payload_length=208):
    """
    The Voting Box
    Averages predictions from all 512x512 tiled copies to defeat localized noise.
    """
    batch_size = decoded_logits.size(0)
    flat_logits = decoded_logits.view(batch_size, -1)
    
    complete_copies_length = (flat_logits.size(1) // payload_length) * payload_length
    truncated_logits = flat_logits[:, :complete_copies_length]
    copies = truncated_logits.view(batch_size, -1, payload_length)
    
    # Vote
    averaged_logits = copies.mean(dim=1)
    probabilities = torch.sigmoid(averaged_logits)
    predicted_bits = (probabilities > 0.5).float()
    
    flat_original = original_payloads.view(batch_size, -1)[:, :payload_length]
    errors = (predicted_bits != flat_original).sum().item()
    return errors / flat_original.numel()

# ==========================================
# GLOBAL FEC & PAYLOAD GENERATOR
# ==========================================
FEC_PIPELINE = RSCodecPipeline(parity_symbols=10)

def generate_spatial_payloads(batch_size, device, spatial_size=512, data_depth=8):
    """
    Uses Spatial Broadcasting (Repeating) to stamp the UUID everywhere.
    """
    all_bits = []
    total_capacity = data_depth * spatial_size * spatial_size
    
    for _ in range(batch_size):
        raw_uuid_bytes = uuid.uuid4().bytes
        encoded_bytes = FEC_PIPELINE.rs.encode(raw_uuid_bytes)
        
        bits = []
        for byte in encoded_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        bits_tensor = torch.tensor(bits, dtype=torch.float32)
        repeats = (total_capacity // len(bits_tensor)) + 1
        bits_tensor = bits_tensor.repeat(repeats)[:total_capacity]
        
        bits_tensor = bits_tensor.view(data_depth, spatial_size, spatial_size)
        all_bits.append(bits_tensor)
        
    return torch.stack(all_bits).to(device)

# ==========================================
# INITIALIZATION & LAYER FREEZE
# ==========================================
def build_and_freeze_phase3_models(device):
    logger.info("\n--- INITIATING PHASE 3: HYBRID VRAM DEFENSE ---")
    data_depth = 8 
    
    encoder = EncoderDense(data_depth=data_depth).to(device)
    decoder = DecoderDense(data_depth=data_depth).to(device)
    
    logger.info("Loading pre-trained Dense weights from /weights/...")
    encoder_path = os.path.join('weights', 'encoder_dense_pretrained.pth')
    decoder_path = os.path.join('weights', 'decoder_dense_pretrained.pth')
    
    encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
    decoder.load_state_dict(torch.load(decoder_path, weights_only=False), strict=False)
       
    logger.info("Executing Hybrid Layer Freeze to protect VRAM on 512x512 images...")
    
    # ENCODER: Freeze early massive layers.
    for param in encoder.conv1.parameters(): param.requires_grad = False
    for param in encoder.conv2.parameters(): param.requires_grad = False
    for param in encoder.conv3.parameters(): param.requires_grad = False
    
    # DECODER: Fully Active to learn the noise.

    enc_frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    enc_active = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_frozen = sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    dec_active = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    logger.info(f"[Encoder] Frozen: {enc_frozen:,} | Active: {enc_active:,} (VRAM Protected)")
    logger.info (f"[Decoder] Frozen: {dec_frozen:,} | Active: {dec_active:,} (Fully Active)")
    
    return encoder, decoder

# ==========================================
# MASTER TRAINING LOOP
# ==========================================
def run_training_loop():
    logger.info("\n--- INITIALIZING HiDDeN HIGH-RES PRODUCTION PIPELINE ---")
    
    physical_batch_size = 2   
    accumulation_steps = 16   
    epochs = 80         
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_path = "./data/fast_patches" 
    os.makedirs(dataset_path, exist_ok=True)
    
    logger.info("Mounting High-Resolution Dataset...")
    dataset = HighResImageFolder(dataset_path)
    if len(dataset) == 0:
        logger.info(f"ERROR: Put some images in '{dataset_path}' to begin training.")
        return
        
    train_loader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True, num_workers=0, pin_memory=True)
    
    encoder, decoder = build_and_freeze_phase3_models(device)
    discriminator = HiddenDiscriminator().to(device) 
    
    noise_layer = AdvancedNoiseLayer().to(device)
    # Starting with a low image penalty (0.05) so the network focuses on surviving the noise
    criterion = HighResHiDDeNLoss(device=device, lambda_m=500.0, lambda_i=0.05, lambda_p=0.5).to(device)
    
    active_params = [p for p in encoder.parameters() if p.requires_grad] + \
                    [p for p in decoder.parameters() if p.requires_grad]
                    
    opt_enc_dec = optim.Adam(active_params, lr=lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)
    
    logger.info("Starting MBRS Phase 3 training...")
    opt_disc.zero_grad()
    opt_enc_dec.zero_grad()

    best_real_ber = float('inf')

    for epoch in range(epochs):
        
        # ==========================================
        #  CURRICULUM LEARNING
        # ==========================================
        if epoch == 2:
            logger.info("\n[SYSTEM] Increasing Image Fidelity Constraint (lambda_i -> 0.25)")
            criterion.lambda_i = 0.25
            
        if epoch == 5:
            logger.info("\n[SYSTEM] Executing Mid-Flight Layer Freeze on Encoder...")
            for param in encoder.parameters(): 
                param.requires_grad = False
                
            # Update optimizer to only train the Decoder from this point forward
            active_params = [p for p in decoder.parameters() if p.requires_grad]
            opt_enc_dec = optim.Adam(active_params, lr=lr)    
        # ==========================================

        encoder.train()
        decoder.train()

        for i, cover_images in enumerate(train_loader):
            cover_images = cover_images.to(device)
            current_batch_size = cover_images.size(0)
            
            payloads = generate_spatial_payloads(current_batch_size, device, spatial_size=512)
            
            # --- Train Discriminator ---
            stego_images = encoder(cover_images, payloads).detach() 
            d_real_logits = discriminator(cover_images)
            d_fake_logits = discriminator(stego_images)
            
            d_loss_real = criterion.bce_with_logits(d_real_logits, torch.ones_like(d_real_logits))
            d_loss_fake = criterion.bce_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            
            d_loss = ((d_loss_real + d_loss_fake) / 2) / accumulation_steps
            d_loss.backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                opt_disc.step()
                opt_disc.zero_grad()
            
            # --- Train Encoder/Decoder ---
            stego_images = encoder(cover_images, payloads)
            
            #  NOISE WARM-UP    
            if epoch > 1:
                noisy_sim_images = noise_layer(stego_images, cover_images) 
            else:
                noisy_sim_images = stego_images
                
            decoded_sim_payloads = decoder(noisy_sim_images)
            d_fake_logits_for_gen = discriminator(stego_images)
            
            loss, l2_loss, msg_loss, g_loss, p_loss = criterion(
                cover_images, stego_images, payloads, decoded_sim_payloads, d_fake_logits_for_gen
            )
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (i + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(active_params, 1.0)
                opt_enc_dec.step()
                opt_enc_dec.zero_grad()
            
            # --- FORENSIC VALIDATION ---
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                encoder.eval()
                decoder.eval()
                
                with torch.no_grad(): 
                    # Clean BER
                    decoded_clean = decoder(stego_images)
                    clean_ber = calculate_ber(decoded_clean, payloads)
                    
                    # Sim BER
                    sim_ber = calculate_ber(decoded_sim_payloads, payloads)
                    
                    # Real BER (Only apply Real JPEG if we are past the noise warm-up!)
                    if epoch > 1:
                        real_quality = torch.full((current_batch_size,), 50.0).to(device)
                        real_jpeg_images = AdvancedNoiseLayer._apply_real_jpeg(stego_images, real_quality)
                        decoded_real = decoder(real_jpeg_images)
                        real_ber = calculate_ber(decoded_real, payloads)
                    else:
                        real_ber = clean_ber # Default to clean BER during warmup

            # Save checkpoints
            if (epoch + 1) % 10 == 0:
                os.makedirs("saved_models", exist_ok=True)
                torch.save(encoder.state_dict(), f"saved_models/encoder_epoch_{epoch+1}.pth")
                torch.save(decoder.state_dict(), f"saved_models/decoder_epoch_{epoch+1}.pth")
                logger.info(f"--> Checkpoint saved for Epoch {epoch+1}")

                noise_status = "ON" if epoch > 1 else "OFF (Warm-up)"
                logger.info(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}] | Loss: {loss.item() * accumulation_steps:.4f} | LPIPS: {p_loss.item():.4f} | Noise: {noise_status}")
                logger.info(f"  -> BER | Clean: {clean_ber:.2%} | Simulated: {sim_ber:.2%} | REAL: {real_ber:.2%}")
                
                if epoch > 1 and real_ber < 0.10 and real_ber < best_real_ber:
                    best_real_ber = real_ber
                    logger.info(f"  🌟 [SUCCESS] New Best Real BER ({real_ber:.2%}). Saving Commercial Model...")
                    os.makedirs("saved_models", exist_ok=True)
                    torch.save(encoder.state_dict(), "saved_models/best_commercial_encoder.pth")
                    torch.save(decoder.state_dict(), "saved_models/best_commercial_decoder.pth")
                
                encoder.train()
                decoder.train()

if __name__ == "__main__":
    try:
        run_training_loop()
        logger.info("\n--- TRAINING COMPLETE ---")
        
    except Exception as e:
        logger.error("A FATAL ERROR OCCURRED DURING TRAINING!", exc_info=True)