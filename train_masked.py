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
# MASK CLASS
# ==========================================

import torch.nn.functional as F

class DirectionalTextureMask(nn.Module):
    """
    Spatial mask for high-frequency texture areas (like buildings)
    """
    def __init__(self):
        super().__init__()
        # 4 filters to detect edges and textures
        k1 = torch.tensor([[[[ 0.,  0.,  0.], [-1.,  2., -1.], [ 0.,  0.,  0.]]]]) 
        k2 = torch.tensor([[[[ 0., -1.,  0.], [ 0.,  2.,  0.], [ 0., -1.,  0.]]]]) 
        k3 = torch.tensor([[[[-1.,  0.,  0.], [ 0.,  2.,  0.], [ 0.,  0., -1.]]]]) 
        k4 = torch.tensor([[[[ 0., -1., -1.], [ 0.,  2.,  0.], [-1.,  0.,  0.]]]]) 
        # Stack filters into a single convolutional weight tensor
        self.register_buffer('filters', torch.cat([k1, k2, k3, k4], dim=0) / 4.0)

    def forward(self, img):
        img_01 = img * 0.5 + 0.5
        gray = 0.299 * img_01[:, 0:1] + 0.587 * img_01[:, 1:2] + 0.114 * img_01[:, 2:3]
        
        residuals = F.conv2d(gray, self.filters, padding=1)
        energy = torch.max(torch.abs(residuals), dim=1, keepdim=True)[0]
        energy = F.max_pool2d(energy, kernel_size=3, stride=1, padding=1)
        
        mask = torch.clamp((energy - 0.02) * 100.0, 0.0, 1.0)
        mask = F.avg_pool2d(mask, 3, stride=1, padding=1)
        return mask

# ==========================================
# LOGGING
# ==========================================

logger = logging.getLogger(__name__)

# ==========================================
# HIGH-RESOLUTION DATA PIPELINE
# ==========================================

class HighResImageFolder(Dataset):
    """
    loader for MS-COCO dataset to prevent GPU bottlenecks.
    Pulls pristine 512x512 patches instead of warping the image.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Standard transformations for training: random crop, horizontal flip, and normalization
        self.transform = transforms.Compose([
            transforms.RandomCrop(512, pad_if_needed=True), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            return self.transform(image)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

# ==========================================
# DISCRIMINATOR
# ==========================================
class ConvBNRelu(nn.Module):
    """A standard Convolution -> BatchNorm -> ReLU block used in the discriminator."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class HiddenDiscriminator(nn.Module):
    """
     discriminator that tries to classify images as 'real' or 'fake'
    """
    def __init__(self):
        super().__init__()
        # Build 3 convolutional layers
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
    """
    Calculates the combined loss for the Encoder/Decoder network, balancing
    message recovery (BCE), image fidelity (MSE + LPIPS), and stealth (GAN loss).
    """
    def __init__(self, device, payload_length=208, lambda_m=100.0, lambda_i=10.0, lambda_p=5.0, lambda_g=0.001):
        super().__init__()
        self.mse_loss = nn.MSELoss() 
        self.bce_with_logits = nn.BCEWithLogitsLoss() 
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device) # Perceptual loss using VGG features
        
        self.payload_length = payload_length
        self.lambda_m = lambda_m #
        self.lambda_i = lambda_i 
        self.lambda_p = lambda_p 
        self.lambda_g = lambda_g 

    def forward(self, cover, stego, original_payloads, msg_out, discriminator_logits, mask, stamp_size=32):
        l_i = self.mse_loss(stego, cover)
        l_p = self.lpips_loss(stego, cover).mean()

        # Prepare to fold the image into 32x32 stamps for message loss calculation
        batch_size, data_depth, H, W = msg_out.shape
        NH, NW = H // stamp_size, W // stamp_size
        
        # Reshape decoded logits into a sequence of 32x32 stamps
        s_logits = msg_out.view(batch_size, data_depth, NH, stamp_size, NW, stamp_size)
        s_logits = s_logits.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, NH * NW, -1)
        
        # Reshape the texture mask to match the stamp geometry
        s_mask = mask.view(batch_size, 1, NH, stamp_size, NW, stamp_size)
        s_mask = s_mask.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, NH * NW, -1)
        s_mask = s_mask.repeat(1, 1, data_depth) 
        
        weighted_sum = (s_logits * s_mask).sum(dim=1)
        total_weight = s_mask.sum(dim=1) + 1e-8
        voted_stamp = weighted_sum / total_weight
        
        voted_logits = voted_stamp[:, :self.payload_length]
        
        # Extract the original target bits from the first 32x32 stamp
        stamp_original = original_payloads[:, :, :stamp_size, :stamp_size]
        flat_original = stamp_original.reshape(batch_size, -1)[:, :self.payload_length]
        
        l_m = self.bce_with_logits(voted_logits, flat_original)
        l_g = -torch.mean(discriminator_logits)

        total_loss = (self.lambda_m * l_m) + (self.lambda_i * l_i) + (self.lambda_p * l_p) + (self.lambda_g * l_g)
        return total_loss, l_i, l_m, l_g, l_p

def calculate_weighted_ber(decoded_logits, original_payloads, mask, payload_length=208, stamp_size=32):
    """
    Calculates the Bit Error Rate (BER) by folding the image into stamps and 
    performing a weighted average based on the texture mask.
    """
    batch_size, data_depth, H, W = decoded_logits.shape
    NH, NW = H // stamp_size, W // stamp_size
    
    # Reshape logits into stamps
    s_logits = decoded_logits.view(batch_size, data_depth, NH, stamp_size, NW, stamp_size)
    s_logits = s_logits.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, NH * NW, -1)
    
    # Reshape mask into stamps
    s_mask = mask.view(batch_size, 1, NH, stamp_size, NW, stamp_size)
    s_mask = s_mask.permute(0, 2, 4, 1, 3, 5).reshape(batch_size, NH * NW, -1)
    s_mask = s_mask.repeat(1, 1, data_depth) 

    weighted_sum = (s_logits * s_mask).sum(dim=1)
    total_weight = s_mask.sum(dim=1) + 1e-8
    voted_stamp = weighted_sum / total_weight 

    probabilities = torch.sigmoid(voted_stamp[:, :payload_length])
    predicted_bits = (probabilities > 0.5).float()
    
    stamp_original = original_payloads[:, :, :stamp_size, :stamp_size]
    flat_original = stamp_original.reshape(batch_size, -1)[:, :payload_length]
    
    errors = (predicted_bits != flat_original).sum().item()
    return errors / flat_original.numel()

# ==========================================
# GLOBAL FEC & PAYLOAD GENERATOR
# ==========================================
# Initialize Reed-Solomon Error Correction pipeline
FEC_PIPELINE = RSCodecPipeline(parity_symbols=10)

def generate_spatial_payloads(batch_size, device, spatial_size=512, data_depth=8):
    """
    Generates a random UUID, encodes it with FEC, formats it into a 32x32 stamp,
    and then tiles that stamp across the entire 512x512 image tensor.
    """
    stamp_size = 32
    all_bits = []
    for _ in range(batch_size):
        raw_uuid_bytes = uuid.uuid4().bytes
        encoded_bytes = FEC_PIPELINE.rs.encode(raw_uuid_bytes)
        
        # Convert bytes to a flat list of floats (0.0 or 1.0)
        bits_tensor = torch.tensor([float((b >> i) & 1) for b in encoded_bytes for i in range(7, -1, -1)]).to(device)

        # Create a single 32x32 stamp and fill it with the bits
        stamp = torch.zeros((data_depth, stamp_size, stamp_size)).to(device)
        stamp.view(-1)[:len(bits_tensor)] = bits_tensor
        
        # Tile the stamp to fill the specified spatial_size (e.g., 512x512)
        repeats = spatial_size // stamp_size
        all_bits.append(stamp.repeat(1, repeats, repeats))
        
    return torch.stack(all_bits).to(device)

# ==========================================
# INITIALIZATION & LAYER FREEZE
# ==========================================
def build_and_freeze_phase3_models(device, resume_epoch=0):
    """
    Initializes the Encoder and Decoder networks, loads pretrained weights,
    and sets up layer freezing and L2 regularization anchors.
    """
    logger.info("\n--- INITIATING PHASE 3: CONV3 UNFREEZE + SOFT REGULARISATION ---")
    data_depth = 8 
    
    # Initialize architectures
    encoder = EncoderDense(data_depth=data_depth).to(device)
    decoder = DecoderDense(data_depth=data_depth).to(device)
    
    # Disable the Spatial Transformer Network to maintain strict 32x32 grid alignment
    decoder.stn = nn.Identity()

    # Attempt to load checkpoint if resuming
    if resume_epoch > 0:
        logger.info(f"Attempting to resume from Epoch {resume_epoch} checkpoints...")
        encoder_path = os.path.join('saved_models/checkpoints', f'encoder_epoch_{resume_epoch}.pth')
        decoder_path = os.path.join('saved_models/checkpoints', f'decoder_epoch_{resume_epoch}.pth')
        
        if os.path.exists(encoder_path) and os.path.exists(decoder_path):
            encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
            decoder.load_state_dict(torch.load(decoder_path, weights_only=False))
            logger.info(" Resumed successfully from saved checkpoints.")
        else:
            logger.error(f"Checkpoints for Epoch {resume_epoch} not found! Falling back to fresh start.")
            resume_epoch = 0

    # Load base pretrained weights for a fresh start
    if resume_epoch == 0:
        logger.info("Loading base pre-trained Dense weights from /weights/...")
        encoder_path = os.path.join('weights', 'encoder_dense_pretrained.pth')
        decoder_path = os.path.join('weights', 'decoder_dense_pretrained.pth')
        
        encoder.load_state_dict(torch.load(encoder_path, weights_only=False))
        decoder.load_state_dict(torch.load(decoder_path, weights_only=False), strict=False)

    # Capture the initial state of conv2 and conv3 to use as anchors for L2 regularization
    pretrained_conv2_state = {
        name: param.data.clone().to(device)
        for name, param in encoder.conv2.named_parameters()
    }
    pretrained_conv3_state = {
        name: param.data.clone().to(device)
        for name, param in encoder.conv3.named_parameters()
    }
    logger.info("Captured pretrained conv2 + conv3 anchors for soft L2 regularisation.")

    # Freeze the first convolutional block to preserve core feature extraction
    logger.info("Applying partial freeze: conv1 frozen | conv2 + conv3 + conv4 active...")
    for param in encoder.conv1.parameters(): param.requires_grad = False

    # Log parameter counts
    enc_frozen = sum(p.numel() for p in encoder.parameters() if not p.requires_grad)
    enc_active = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    dec_frozen = sum(p.numel() for p in decoder.parameters() if not p.requires_grad)
    dec_active = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    
    logger.info(f"[Encoder] Frozen: {enc_frozen:,} | Active: {enc_active:,}")
    logger.info(f"[Decoder] Frozen: {dec_frozen:,} | Active: {dec_active:,} (Fully Active)")
    
    return encoder, decoder, resume_epoch, pretrained_conv2_state, pretrained_conv3_state

# ==========================================
# MASTER TRAINING LOOP
# ==========================================
def run_training_loop(resume_epoch=0, manual_tier=None):
    logger.info("\n--- INITIALIZING HiDDeN HIGH-RES PRODUCTION PIPELINE ---")

    LAMBDA_ENC_REG = 1e-3
    CLEAN_WARMUP_STEPS = 800

    JPEG_QUALITY_TIERS = [
        (80.0, 95.0,  0.40),  # tier 0: warmup seed
        (70.0, 90.0,  0.30),  # tier 1
        (60.0, 85.0,  0.25),  # tier 2
        (55.0, 80.0,  0.22),  # tier 3
        (50.0, 80.0,  0.20),  # tier 4
        (45.0, 75.0,  0.18),  # tier 5
        (40.0, 70.0,  0.15),  # tier 6
        (35.0, 90.0,  0.00),  # tier 7: full range
    ]

    CLEAN_BER_WATCHDOG = 0.10

    physical_batch_size = 4   
    accumulation_steps = 8  
    steps_per_epoch = 5000
    epochs = 25         
    lr = 1e-3
    if resume_epoch >= 8:
        logger.info("\n[SYSTEM] Late-stage resume detected. Dropping Learning Rate to 1e-4.")
        lr = 1e-4 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_path = "./data/fast_patches" 
    os.makedirs(dataset_path, exist_ok=True)
    
    # Initialize Dataset and DataLoader
    logger.info("Mounting High-Resolution Dataset...")
    dataset = HighResImageFolder(dataset_path)
    if len(dataset) == 0:
        logger.info(f"ERROR: Put some images in '{dataset_path}' to begin training.")
        return
        
    train_loader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    # Load Models
    encoder, decoder, resume_epoch, pretrained_conv2_state, pretrained_conv3_state = \
        build_and_freeze_phase3_models(device, resume_epoch)
    discriminator = HiddenDiscriminator().to(device) 

    # Initialize Mask and Loss Criterion
    attention_mask = DirectionalTextureMask().to(device)
    criterion = HighResHiDDeNLoss(device=device, lambda_m=100.0, lambda_i=0.3, lambda_p=0.5).to(device)

    # Filter out frozen parameters for the optimizer
    encoder_active_params = [p for p in encoder.parameters() if p.requires_grad]
    decoder_active_params = [p for p in decoder.parameters() if p.requires_grad]

    # Initialize Optimizers (Encoder learns slower than Decoder for stability)
    opt_enc_dec = optim.Adam([
        {'params': encoder_active_params, 'lr': lr * 0.3},
        {'params': decoder_active_params, 'lr': lr},
    ])
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)

    # Initialize Mixed Precision Scaler
    scaler = torch.amp.GradScaler('cuda')
    
    logger.info(f"Starting MBRS Phase 3 training from Epoch {resume_epoch}...")

    # Initialize Exponential Moving Averages for metrics
    real_ber_ema = 0.03
    clean_ber_ema = 0.03

    quality_tier_idx = manual_tier if manual_tier is not None else 1

    opt_disc.zero_grad()
    opt_enc_dec.zero_grad()

    best_scores_per_tier = {i: float('inf') for i in range(len(JPEG_QUALITY_TIERS))}

    for epoch in range(resume_epoch, epochs):
        
        # ==========================================
        #  CURRICULUM LEARNING 
        # ==========================================
        if real_ber_ema < 0.35:
            new_lambda_i = min(10.0 + (epoch * 5), 50.0)
            new_lambda_p = min(5.0 + (epoch * 3.5), 35.0)
            new_lambda_m = min(50.0 + (epoch * 25.0), 400.0)
        else:
            new_lambda_i = criterion.lambda_i
            new_lambda_p = criterion.lambda_p
            new_lambda_m = criterion.lambda_m

        if (criterion.lambda_i != new_lambda_i or
                criterion.lambda_m != new_lambda_m or
                criterion.lambda_p != new_lambda_p):
            logger.info(
                f"\n[SYSTEM] Curriculum Learning: "
                f"Image ({new_lambda_i}) | Perceptual ({new_lambda_p}) | Message ({new_lambda_m})"
            )
            criterion.lambda_i = new_lambda_i
            criterion.lambda_p = new_lambda_p
            criterion.lambda_m = new_lambda_m

        # Gradually reduce the chance of skipping noise augmentation
        bypass_rate = max(0.10, 0.20 - (max(0, epoch - 3) * 0.01))

        # Bypass warmup tier if manual tier is set
        if epoch == 0 and manual_tier is None:
            current_quality = JPEG_QUALITY_TIERS[0][:2]
        else:
            current_tier = JPEG_QUALITY_TIERS[quality_tier_idx]
            unlock_threshold = current_tier[2]
            max_tier_idx = len(JPEG_QUALITY_TIERS) - 1
            
            # Check if performance warrants advancing to a harder tier
            if real_ber_ema < unlock_threshold and quality_tier_idx < max_tier_idx:
                quality_tier_idx += 1
                new_tier = JPEG_QUALITY_TIERS[quality_tier_idx]
                logger.info(
                    f"\n[TIER UP] BER EMA {real_ber_ema:.2%} < {unlock_threshold:.0%} threshold — "
                    f"advancing to tier {quality_tier_idx} "
                    f"(Q{int(new_tier[0])}-{int(new_tier[1])})"
                )
            else:
                if quality_tier_idx < max_tier_idx:
                    logger.info(
                        f"\n[TIER HOLD] BER EMA {real_ber_ema:.2%} >= "
                        f"{unlock_threshold:.0%} threshold — "
                        f"staying at tier {quality_tier_idx} "
                        f"(Q{int(current_tier[0])}-{int(current_tier[1])})"
                    )
            current_quality = JPEG_QUALITY_TIERS[quality_tier_idx][:2]

        # Dynamic L2 Regularization to prevent catastrophic forgetting
        if clean_ber_ema > CLEAN_BER_WATCHDOG:
            active_lambda_enc_reg = LAMBDA_ENC_REG * 2.0
        elif clean_ber_ema < CLEAN_BER_WATCHDOG * 0.5:
            active_lambda_enc_reg = LAMBDA_ENC_REG
        else:
            active_lambda_enc_reg = LAMBDA_ENC_REG

        noise_layer = AdvancedNoiseLayer(jpeg_quality=current_quality).to(device)
        logger.info(
            f"[SYSTEM] Epoch {epoch + 1} | "
            f"JPEG quality: Q{int(current_quality[0])}-{int(current_quality[1])} "
            f"(tier {quality_tier_idx if epoch > 0 else 0}) | "
            f"Bypass: {bypass_rate:.0%} | "
            f"BER EMA: {real_ber_ema:.2%} | "
            f"Clean EMA: {clean_ber_ema:.2%}"
        )

        # Late-stage fine-tuning schedule
        if epoch == 15 and resume_epoch < 15: 
            logger.info("\n[SYSTEM] Dropping Learning Rate to 1e-4 for Joint Fine-Tuning...")
            opt_enc_dec.param_groups[0]['lr'] = 1e-4 * 0.3 
            opt_enc_dec.param_groups[1]['lr'] = 1e-4          
            
        encoder.train()
        decoder.train()

        for i, cover_images in enumerate(train_loader):
            cover_images = cover_images.to(device)
            current_batch_size = cover_images.size(0)
            
            # Generate the spatial tiled payload for this batch
            payloads = generate_spatial_payloads(current_batch_size, device, spatial_size=512)

            # Generate the texture mask for this batch
            current_mask = attention_mask(cover_images)

            # ------------------------------------------
            # Train Discriminator 
            # ------------------------------------------
            with torch.autocast(device_type='cuda'):
                # Embed data and apply mask
                raw_stego = encoder(cover_images, payloads)
                residual = raw_stego - cover_images
                stego_images = torch.clamp(cover_images + (residual * current_mask), -1.0, 1.0).detach()
                
                # Get discriminator predictions
                d_real_logits = discriminator(cover_images)
                d_fake_logits = discriminator(stego_images)
                
                # Calculate Wasserstein GAN loss
                d_loss = (torch.mean(d_fake_logits) - torch.mean(d_real_logits)) / accumulation_steps
                
            scaler.scale(d_loss).backward()
            
            # Step Discriminator optimizer at accumulation intervals
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt_disc)
                scaler.step(opt_disc)
                opt_disc.zero_grad()
                # Weight clipping for Wasserstein GAN
                for p in discriminator.parameters():
                    p.data.clamp_(-0.1, 0.1)
            
            # ------------------------------------------
            # Train Encoder/Decoder 
            # ------------------------------------------
            with torch.autocast(device_type='cuda'):
                # Generate new stego images
                raw_stego = encoder(cover_images, payloads)
                residual = raw_stego - cover_images
                stego_images = torch.clamp(cover_images + (residual * current_mask), -1.0, 1.0)
                
                # Shift to [0, 1] range for noise layers
                stego_01 = stego_images * 0.5 + 0.5
                stego_01 = stego_01.clamp(0.0, 1.0)

                # Determine if we apply noise or bypass it
                bypass_chance = 0.30 if (epoch == 0 and manual_tier is None) else bypass_rate
                
                if torch.rand(1).item() < bypass_chance:
                    noisy_01 = stego_01
                else:
                    noisy_01 = noise_layer(stego_01)

                # Shift back to [-1, 1] for decoder
                noisy_sim_images = noisy_01 * 2.0 - 1.0
                    
                # Decode payloads from noisy images
                decoded_sim_payloads = decoder(noisy_sim_images)
                # Assess generator against discriminator
                d_fake_logits_for_gen = discriminator(stego_images)

                # Calculate primary multi-objective loss
                loss, l2_loss, msg_loss, g_loss, p_loss = criterion(
                    cover_images, stego_images, payloads, decoded_sim_payloads, d_fake_logits_for_gen, current_mask
                )
                loss = loss / accumulation_steps

                # Add Soft L2 Regularization to prevent feature forgetting
                enc_reg = sum(
                    torch.norm(p.float() - pretrained_conv3_state[n].to(device).float()) ** 2
                    for n, p in encoder.conv3.named_parameters()
                    if p.requires_grad
                )
                enc_reg += sum(
                    torch.norm(p.float() - pretrained_conv2_state[n].to(device).float()) ** 2
                    for n, p in encoder.conv2.named_parameters()
                    if p.requires_grad
                ) * 2.0
                loss = loss + (enc_reg * active_lambda_enc_reg) / accumulation_steps

            scaler.scale(loss).backward()
            
            # Step Encoder/Decoder optimizer at accumulation intervals
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.unscale_(opt_enc_dec)
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(encoder_active_params + decoder_active_params, 1.0)
                scaler.step(opt_enc_dec)
                
                scaler.update()
                opt_enc_dec.zero_grad()
            
            # ------------------------------------------
            # Validation & Logging
            # ------------------------------------------
            if (i + 1) % 100 == 0 or (i + 1) == len(train_loader):
                encoder.eval()
                decoder.eval()

                with torch.no_grad(): 
                    # Calculate masked BER on clean images
                    extraction_mask = attention_mask(stego_images)
                    decoded_clean = decoder(stego_images)
                    clean_ber = calculate_weighted_ber(decoded_clean, payloads, extraction_mask)
                    
                    # Calculate masked BER on differentiable simulated noise
                    sim_ber = calculate_weighted_ber(decoded_sim_payloads, payloads, extraction_mask)
                    
                    # Calculate masked BER on non-differentiable REAL JPEG compression
                    val_quality = current_quality[0]
                    real_quality_t = torch.full((current_batch_size,), val_quality).to(device)
                    stego_01_val = (stego_images * 0.5 + 0.5).clamp(0.0, 1.0)
                    real_jpeg_01 = AdvancedNoiseLayer._apply_real_jpeg(stego_01_val, real_quality_t)
                    real_jpeg_images = real_jpeg_01 * 2.0 - 1.0
                    decoded_real = decoder(real_jpeg_images)
                    real_extraction_mask = attention_mask(real_jpeg_images)
                    real_ber = calculate_weighted_ber(decoded_real, payloads, real_extraction_mask)

                    real_ber_ema = 0.9 * real_ber_ema + 0.1 * real_ber
                    clean_ber_ema = 0.9 * clean_ber_ema + 0.1 * clean_ber

                time.sleep(0.1)

                vram_peak_gb = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                torch.cuda.reset_peak_memory_stats(device)

                noise_status = "MILD" if (epoch == 0 and manual_tier is None) else "ON"
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] Step [{i+1}] | "
                    f"Loss: {loss.item() * accumulation_steps:.4f} | "
                    f"LPIPS: {p_loss.item():.4f} | "
                    f"Noise: {noise_status} (Q{int(current_quality[0])}-{int(current_quality[1])}) | "
                    f"Bypass: {bypass_rate:.0%} | "
                    f"VRAM: {vram_peak_gb:.2f} GB"
                )
                logger.info(
                    f"  -> BER | Clean: {clean_ber:.2%} | "
                    f"Simulated: {sim_ber:.2%} | "
                    f"REAL@Q{int(val_quality)}: {real_ber:.2%} | "
                    f"EMA: {real_ber_ema:.2%} | "
                    f"CleanEMA: {clean_ber_ema:.2%}"
                )
                
                score = real_ber + (p_loss.item() * 0.5)
                
                if epoch > 0 and real_ber < 0.15 and score < best_scores_per_tier[quality_tier_idx]:
                    best_scores_per_tier[quality_tier_idx] = score
                    
                    logger.info(f"   [SUCCESS] New Best Model for TIER {quality_tier_idx}! (Score: {score:.4f} | BER: {real_ber:.2%} | LPIPS: {p_loss.item():.4f})")
                    os.makedirs("saved_models/checkpoints", exist_ok=True)
                    
                    torch.save(encoder.state_dict(), f"saved_models/checkpoints/best_encoder_tier_{quality_tier_idx}.pth")
                    torch.save(decoder.state_dict(), f"saved_models/checkpoints/best_decoder_tier_{quality_tier_idx}.pth")
                    
                    torch.save(encoder.state_dict(), "saved_models/checkpoints/best_encoder_full.pth")
                    torch.save(decoder.state_dict(), "saved_models/checkpoints/best_decoder_full.pth")
                
                encoder.train()
                decoder.train()

                if epoch == 0 and manual_tier is None and (i + 1) >= CLEAN_WARMUP_STEPS:
                    logger.info(f"[WARMUP] Reached {CLEAN_WARMUP_STEPS} warmup steps. Ending warmup epoch early.")
                    break

                if (i + 1) >= steps_per_epoch:
                    logger.info(f"Reached {steps_per_epoch} steps. Ending Virtual Epoch {epoch+1}.")
                    break

        if (epoch + 1) % 1 == 0:
            os.makedirs("saved_models/checkpoints", exist_ok=True)
            torch.save(encoder.state_dict(), f"saved_models/checkpoints/encoder_epoch_{epoch+1}.pth")
            torch.save(decoder.state_dict(), f"saved_models/checkpoints/decoder_epoch_{epoch+1}.pth")
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

    print("\n" + "="*50)
    print("  MANUAL TIER OVERRIDE (OPTIONAL)")
    print("="*50)
    print("0: Warmup (Q80-95)")
    print("1: Tier 1  (Q70-90)")
    print("2: Tier 2  (Q60-85)")
    print("3: Tier 3  (Q55-80)")
    print("4: Tier 4  (Q50-80)")
    print("5: Tier 5  (Q45-75)")
    print("6: Tier 6  (Q40-70)")
    print("7: Tier 7  (Q35-90) [Max Difficulty]")
    
    tier_choice = input("\nEnter starting tier (0-7) or press Enter to use Automatic Curriculum: ").strip()
    
    manual_tier = None
    if tier_choice.isdigit() and 0 <= int(tier_choice) <= 7:
        manual_tier = int(tier_choice)
        print(f"-> Starting training locked at Manual Tier {manual_tier}")
    else:
        print("-> Using Automatic Curriculum")

    try:
        run_training_loop(resume_epoch=resume_epoch, manual_tier=manual_tier)
        logger.info("\n--- TRAINING COMPLETE ---")
        
    except Exception as e:
        logger.error("A FATAL ERROR OCCURRED DURING TRAINING!", exc_info=True)