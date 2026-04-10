import os
import logging
import time
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

logger = logging.getLogger(__name__)

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
def build_and_freeze_phase3_models(device, resume_epoch=0):
    logger.info("\n--- INITIATING PHASE 3: HYBRID VRAM DEFENSE ---")
    data_depth = 8 
    
    encoder = EncoderDense(data_depth=data_depth).to(device)
    decoder = DecoderDense(data_depth=data_depth).to(device)
    
    if resume_epoch > 0:
        logger.info(f"Attempting to resume from Epoch {resume_epoch} checkpoints...")
        encoder_path = os.path.join('saved_models', f'encoder_epoch_{resume_epoch}.pth')
        decoder_path = os.path.join('saved_models', f'decoder_epoch_{resume_epoch}.pth')
        
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
            decoder.load_state_dict(torch.load(decoder_path, weights_only=False))
            logger.info(" Resumed successfully from saved checkpoints.")
        else:
            logger.error(f"Checkpoints for Epoch {resume_epoch} not found! Falling back to fresh start.")
            resume_epoch = 0 # Force a fresh start

    if resume_epoch == 0:
        logger.info("Loading base pre-trained Dense weights from /weights/...")
        encoder_path = os.path.join('weights', 'encoder_dense_pretrained.pth')
        decoder_path = os.path.join('weights', 'decoder_dense_pretrained.pth')
        
        encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
        decoder.load_state_dict(torch.load(decoder_path, weights_only=False), strict=False)
       
    logger.info("Executing Hybrid Layer Freeze to protect VRAM on 512x512 images...")
    
    # ENCODER: Freeze early massive layers.
    for param in encoder.conv1.parameters(): param.requires_grad = False
    for param in encoder.conv2.parameters(): param.requires_grad = False
    for param in encoder.conv3.parameters(): param.requires_grad = False
    
    enc_frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    enc_active = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_frozen = sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    dec_active = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    logger.info(f"[Encoder] Frozen: {enc_frozen:,} | Active: {enc_active:,} (VRAM Protected)")
    logger.info (f"[Decoder] Frozen: {dec_frozen:,} | Active: {dec_active:,} (Fully Active)")
    
    return encoder, decoder, resume_epoch

# ==========================================
# MASTER TRAINING LOOP
# ==========================================
def run_training_loop(resume_epoch=0):
    logger.info("\n--- INITIALIZING HiDDeN HIGH-RES PRODUCTION PIPELINE ---")
    
    physical_batch_size = 4   
    accumulation_steps = 8  
    epochs = 10         
    lr = 1e-3
    if resume_epoch >= 8:
        logger.info("\n[SYSTEM] Late-stage resume detected. Dropping Learning Rate to 1e-4.")
        lr = 1e-4  # Slow down for final fine-tuning
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_path = "./data/fast_patches" 
    os.makedirs(dataset_path, exist_ok=True)
    
    logger.info("Mounting High-Resolution Dataset...")
    dataset = HighResImageFolder(dataset_path)
    if len(dataset) == 0:
        logger.info(f"ERROR: Put some images in '{dataset_path}' to begin training.")
        return
        
    train_loader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    encoder, decoder, resume_epoch = build_and_freeze_phase3_models(device, resume_epoch)
    discriminator = HiddenDiscriminator().to(device) 
    
    noise_layer = AdvancedNoiseLayer().to(device)
    criterion = HighResHiDDeNLoss(device=device, lambda_m=500.0, lambda_i=0.3, lambda_p=0.5).to(device)

    active_params = [p for p in encoder.parameters() if p.requires_grad] + \
                    [p for p in decoder.parameters() if p.requires_grad]
                    
    opt_enc_dec = optim.Adam(active_params, lr=lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)

    scaler = torch.amp.GradScaler('cuda')
    
    logger.info(f"Starting MBRS Phase 3 training from Epoch {resume_epoch}...")
    opt_disc.zero_grad()
    opt_enc_dec.zero_grad()

    best_real_ber = float('inf')

    for epoch in range(resume_epoch, epochs):
        
        # ==========================================
        #  CURRICULUM LEARNING
        # ==========================================
        # Epoch 0-1: Noise is OFF.
        # Epoch 1-3: Noise turns ON 
        
        new_lambda_i = min(10.0 + (epoch * 15.0), 50.0)
        new_lambda_p = min(5.0 + (epoch * 10.0), 30.0)
        
        if criterion.lambda_i != new_lambda_i:
            logger.info(f"\n[SYSTEM] Curriculum Learning: Image Fidelity (lambda_i -> {new_lambda_i:.1f}) | Perceptual (lambda_p -> {new_lambda_p:.1f})")
            criterion.lambda_i = new_lambda_i
            criterion.lambda_p = new_lambda_p
                
        if epoch == 8 and resume_epoch < 8: 
            logger.info("\n[SYSTEM] Dropping Learning Rate to 1e-4 for Joint Fine-Tuning...")
            for param_group in opt_enc_dec.param_groups:
                param_group['lr'] = 1e-4
            
        # ==========================================

        encoder.train()
        decoder.train()

        for i, cover_images in enumerate(train_loader):
            cover_images = cover_images.to(device)
            current_batch_size = cover_images.size(0)
            
            payloads = generate_spatial_payloads(current_batch_size, device, spatial_size=512)
            
            # ------------------------------------------
            # Train Discriminator 
            # ------------------------------------------
            with torch.autocast(device_type='cuda'):
                stego_images = encoder(cover_images, payloads).detach() 
                d_real_logits = discriminator(cover_images)
                d_fake_logits = discriminator(stego_images)
                
                d_loss_real = criterion.bce_with_logits(d_real_logits, torch.ones_like(d_real_logits))
                d_loss_fake = criterion.bce_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
                
                d_loss = ((d_loss_real + d_loss_fake) / 2) / accumulation_steps
                
            scaler.scale(d_loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt_disc)
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0) # The Savior
                scaler.step(opt_disc)
                opt_disc.zero_grad()
                # Do NOT call scaler.update() yet!
            
            # ------------------------------------------
            # Train Encoder/Decoder 
            # ------------------------------------------
            with torch.autocast(device_type='cuda'):
                stego_images = encoder(cover_images, payloads)
                
                if epoch > 0:
                    # 20% of the time, feed the Decoder a clean image so it doesn't forget!
                    if torch.rand(1).item() < 0.2:
                        noisy_sim_images = stego_images
                    else:
                        noisy_sim_images = noise_layer(stego_images, cover_images) 
                else:
                    noisy_sim_images = stego_images
                    
                decoded_sim_payloads = decoder(noisy_sim_images)
                d_fake_logits_for_gen = discriminator(stego_images)
                
                loss, l2_loss, msg_loss, g_loss, p_loss = criterion(
                    cover_images, stego_images, payloads, decoded_sim_payloads, d_fake_logits_for_gen
                )
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt_enc_dec)
                torch.nn.utils.clip_grad_norm_(active_params, 1.0)
                scaler.step(opt_enc_dec)
                
                scaler.update()
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
                    if epoch > 0:
                        real_quality = torch.full((current_batch_size,), 50.0).to(device)
                        real_jpeg_images = AdvancedNoiseLayer._apply_real_jpeg(stego_images, real_quality)
                        decoded_real = decoder(real_jpeg_images)
                        real_ber = calculate_ber(decoded_real, payloads)
                    else:
                        real_ber = clean_ber # Default to clean BER during warmup

                noise_status = "ON" if epoch > 0 else "OFF "
                logger.info(f"Epoch [{epoch+1}/{epochs}] Step [{i+1}] | Loss: {loss.item() * accumulation_steps:.4f} | LPIPS: {p_loss.item():.4f} | Noise: {noise_status}")
                logger.info(f"  -> BER | Clean: {clean_ber:.2%} | Simulated: {sim_ber:.2%} | REAL: {real_ber:.2%}")
                
                if epoch > 1 and real_ber < 0.10 and real_ber < best_real_ber:
                    best_real_ber = real_ber
                    logger.info(f"   [SUCCESS] New Best Real BER ({real_ber:.2%}). Saving Commercial Model...")
                    os.makedirs("saved_models", exist_ok=True)
                    torch.save(encoder.state_dict(), "saved_models/best_commercial_encoder.pth")
                    torch.save(decoder.state_dict(), "saved_models/best_commercial_decoder.pth")
                
                encoder.train()
                decoder.train()

        if (epoch + 1) % 1 == 0:
            os.makedirs("saved_models", exist_ok=True)
            torch.save(encoder.state_dict(), f"saved_models/encoder_epoch_{epoch+1}.pth")
            torch.save(decoder.state_dict(), f"saved_models/decoder_epoch_{epoch+1}.pth")
            logger.info(f"--> Checkpoint saved for Epoch {epoch+1}")

if __name__ == "__main__":

    os.makedirs("logs", exist_ok=True)

    log_filename = datetime.now().strftime("logs/training_%Y-%m-%d_%H-%M-%S.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_filename),  
            logging.StreamHandler()             
        ]
    )

    logger = logging.getLogger(__name__)

    logger.info("--- TRAINING SCRIPT STARTED ---")

    print("\n" + "="*50)
    print("  HiDDeN HIGH-RES TRAINING PROTOCOL")
    print("="*50)
    print("1. Start a fresh training process (Loads Base Weights)")
    print("2. Resume from a saved checkpoint")
    print("="*50)
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    resume_epoch = 0
    
    if choice == '2':
        epoch_input = input("Enter the exact epoch number you want to resume from (e.g., 2, 4, 8): ").strip()
        try:
            resume_epoch = int(epoch_input)
        except ValueError:
            print("Invalid input! Falling back to a fresh start.")
            resume_epoch = 0

    try:
        run_training_loop(resume_epoch=resume_epoch)
        logger.info("\n--- TRAINING COMPLETE ---")
        
    except Exception as e:
        logger.error("A FATAL ERROR OCCURRED DURING TRAINING!", exc_info=True)