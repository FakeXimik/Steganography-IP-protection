import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import uuid
import os

from models.encoder_dense import EncoderDense
from models.decoder_dense import DecoderDense
from utils.fec import RSCodecPipeline

# ==========================================
# DIRECTIONAL MASK - DYNAMIC RELATIVE UPGRADE
# ==========================================
class DirectionalTextureMask(nn.Module):
    def __init__(self):
        super().__init__()
        k1 = torch.tensor([[[[ 0.,  0.,  0.], [-1.,  2., -1.], [ 0.,  0.,  0.]]]]) 
        k2 = torch.tensor([[[[ 0., -1.,  0.], [ 0.,  2.,  0.], [ 0., -1.,  0.]]]]) 
        k3 = torch.tensor([[[[-1.,  0.,  0.], [ 0.,  2.,  0.], [ 0.,  0., -1.]]]]) 
        k4 = torch.tensor([[[[ 0., -1., -1.], [ 0.,  2.,  0.], [-1.,  0.,  0.]]]]) 
        self.register_buffer('filters', torch.cat([k1, k2, k3, k4], dim=0) / 4.0)

    def forward(self, img, strictness=1.0):
        img_01 = img * 0.5 + 0.5
        gray = 0.299 * img_01[:, 0:1] + 0.587 * img_01[:, 1:2] + 0.114 * img_01[:, 2:3]
        
        residuals = F.conv2d(gray, self.filters, padding=1)
        energy = torch.max(torch.abs(residuals), dim=1, keepdim=True)[0]
        energy = F.max_pool2d(energy, kernel_size=3, stride=1, padding=1)
        
        # DYNAMIC RELATIVE THRESHOLDING
        b, c, h, w = energy.shape
        energy_flat = energy.view(b, -1)
        mean_e = energy_flat.mean(dim=1).view(b, 1, 1, 1)
        std_e = energy_flat.std(dim=1).view(b, 1, 1, 1)
        
        # 'strictness' requires the texture to be above the image average to activate
        mask = (energy - mean_e) / (std_e * strictness + 1e-8)
        mask = torch.clamp(mask, 0.0, 1.0)
        mask = F.avg_pool2d(mask, 3, stride=1, padding=1)
        return mask

# ==========================================
# THE TILED INFERENCE ENGINE
# ==========================================
class SteganographyEngine:
    def __init__(self, encoder_weights, decoder_weights, data_depth=8):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_depth = data_depth
        self.stamp_size = 32 
        
        print(f"[SYSTEM] Initializing Dynamic Engine on {self.device.type.upper()}...")

        self.fec_pipeline = RSCodecPipeline(parity_symbols=10)
        self.payload_bits_count = (16 + 10) * 8 

        self.encoder = EncoderDense(data_depth=self.data_depth).to(self.device)
        self.decoder = DecoderDense(data_depth=self.data_depth).to(self.device)
        self.mask_module = DirectionalTextureMask().to(self.device)

        self.decoder.stn = nn.Identity()

        if not os.path.exists(encoder_weights) or not os.path.exists(decoder_weights):
            raise FileNotFoundError("Golden weights not found! Check your file paths.")

        self.encoder.load_state_dict(torch.load(encoder_weights, map_location=self.device, weights_only=True))
        self.decoder.load_state_dict(torch.load(decoder_weights, map_location=self.device, weights_only=True))

        self.encoder.eval()
        self.decoder.eval()
        self.mask_module.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        print("[SYSTEM] Engine Ready.")

    def _create_tiled_payload(self, bits_tensor, H, W):
        stamp = torch.zeros((self.data_depth, self.stamp_size, self.stamp_size)).to(self.device)
        flat_stamp = stamp.view(-1)
        flat_stamp[:len(bits_tensor)] = bits_tensor
        stamp = flat_stamp.view(self.data_depth, self.stamp_size, self.stamp_size)

        repeats_h = (H // self.stamp_size) + 1
        repeats_w = (W // self.stamp_size) + 1
        tiled = stamp.repeat(1, repeats_h, repeats_w)
        
        return tiled[:, :H, :W].unsqueeze(0)

    def embed_uuid(self, image_path, output_path, target_uuid=None):
        image = Image.open(image_path).convert('RGB')
        cover_tensor = self.transform(image).unsqueeze(0).to(self.device)
        _, _, H, W = cover_tensor.shape

        if target_uuid is None:
            raw_uuid = uuid.uuid4()
        elif isinstance(target_uuid, str):
            raw_uuid = uuid.UUID(target_uuid)
        else:
            raw_uuid = target_uuid
            
        encoded_bytes = self.fec_pipeline.rs.encode(raw_uuid.bytes)
        bits = [float((byte >> i) & 1) for byte in encoded_bytes for i in range(7, -1, -1)]
        bits_tensor = torch.tensor(bits).to(self.device)

        payload = self._create_tiled_payload(bits_tensor, H, W)

        with torch.no_grad():
            raw_stego = self.encoder(cover_tensor, payload)
            residual = raw_stego - cover_tensor
            
            mask = self.mask_module(cover_tensor, strictness=1.0)
            
            stego_tensor = torch.clamp(cover_tensor + (residual * mask), -1.0, 1.0)

        stego_01 = stego_tensor * 0.5 + 0.5
        transforms.ToPILImage()(stego_01.squeeze(0).cpu().clamp(0, 1)).save(output_path, format="PNG")
        return raw_uuid

    def extract_uuid(self, image_path):
        image = Image.open(image_path).convert('RGB')
        stego_tensor = self.transform(image).unsqueeze(0).to(self.device)
        _, _, H, W = stego_tensor.shape

        with torch.no_grad():
            decoded_logits = self.decoder(stego_tensor)
            
            # Use matching strictness to isolate the exact areas the encoder targeted
            mask = self.mask_module(stego_tensor, strictness=1.0)

        mask_active_pct = (mask > 0.5).float().mean().item()
        print(f"   [DEBUG] Forensic Mask Active: {mask_active_pct:.1%} of pixels")

        pad_h = (self.stamp_size - (H % self.stamp_size)) % self.stamp_size
        pad_w = (self.stamp_size - (W % self.stamp_size)) % self.stamp_size
        logits_padded = F.pad(decoded_logits, (0, pad_w, 0, pad_h))
        mask_padded = F.pad(mask, (0, pad_w, 0, pad_h))
        
        NH = logits_padded.shape[2] // self.stamp_size
        NW = logits_padded.shape[3] // self.stamp_size

        stamps_logits = logits_padded.view(self.data_depth, NH, self.stamp_size, NW, self.stamp_size)
        stamps_logits = stamps_logits.permute(1, 3, 0, 2, 4).reshape(-1, self.data_depth, self.stamp_size, self.stamp_size)
        
        stamps_mask = mask_padded.view(1, NH, self.stamp_size, NW, self.stamp_size)
        stamps_mask = stamps_mask.permute(1, 3, 0, 2, 4).reshape(-1, 1, self.stamp_size, self.stamp_size)

        weighted_sum = (stamps_logits * stamps_mask).sum(dim=0)
        total_weight = stamps_mask.sum(dim=0) + 1e-8
        final_stamp = weighted_sum / total_weight 

        predicted_bits = (torch.sigmoid(final_stamp.view(-1)) > 0.5).int()
        bits_list = predicted_bits[:self.payload_bits_count].tolist()

        recovered_bytes = bytearray()
        for i in range(0, len(bits_list), 8):
            byte_val = 0
            for bit in bits_list[i:i+8]: byte_val = (byte_val << 1) | bit
            recovered_bytes.append(byte_val)

        try:
            decoded_message, _, _ = self.fec_pipeline.rs.decode(recovered_bytes)
            return uuid.UUID(bytes=bytes(decoded_message))
        except Exception as e:
            return None