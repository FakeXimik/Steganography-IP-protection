import os
import torch
import torch.nn.functional as F
from PIL import Image
import uuid
import numpy as np

# Import the Tiled Stamp Engine
from models.stego_engine import SteganographyEngine

def run_debug_test():
    print("\n" + "="*60)
    print("  HiDDeN DEBUG PIPELINE — PINPOINTING ALIGNMENT")
    print("="*60)
    
    encoder_path = "saved_models/production/best_encoder_full.pth"
    decoder_path = "saved_models/production/best_decoder_full.pth"
    cover_image_path = "assets/test_photo.png"
    output_dir = "assets/output"
    os.makedirs(output_dir, exist_ok=True)

    # Initialize
    engine = SteganographyEngine(encoder_path, decoder_path)
    
    # Embed and Capture Raw Bits
    print("\n[STEP 1] Generating and Embedding Data...")
    stego_png_path = os.path.join(output_dir, "debug_stego.png")
    
    
    embedded_uuid = engine.embed_uuid(cover_image_path, stego_png_path)
    
    encoded_bytes = engine.fec_pipeline.rs.encode(embedded_uuid.bytes)
    original_bits = []
    for byte in encoded_bytes:
        for i in range(7, -1, -1):
            original_bits.append((byte >> i) & 1)
            
    print(f"   ✅ Saved Stego PNG. UUID: {embedded_uuid}")

    # Deep Extraction Analysis
    print("\n[STEP 2] Performing Forensic Extraction...")
    image = Image.open(stego_png_path).convert('RGB')
    stego_tensor = engine.transform(image).unsqueeze(0).to(engine.device)

    with torch.no_grad():
        decoded_logits = engine.decoder(stego_tensor)
        mask = engine.mask_module(stego_tensor)
        
    mask_coverage = (mask > 0.5).float().mean().item()
    avg_energy = mask.mean().item()
    print(f"   🔍 Mask Coverage: {mask_coverage:.1%} of image is 'Active'")
    print(f"   🔍 Average Mask Energy: {avg_energy:.4f}")
    
    if avg_energy < 0.001:
        print("   WARNING: Mask is almost entirely black! The 'energy cutoff' might be too high.")

    NH = decoded_logits.shape[2] // engine.stamp_size
    NW = decoded_logits.shape[3] // engine.stamp_size
    
    pad_h = (engine.stamp_size - (decoded_logits.shape[2] % engine.stamp_size)) % engine.stamp_size
    pad_w = (engine.stamp_size - (decoded_logits.shape[3] % engine.stamp_size)) % engine.stamp_size
    l_p = F.pad(decoded_logits, (0, pad_w, 0, pad_h))
    m_p = F.pad(mask, (0, pad_w, 0, pad_h))

    s_logits = l_p.view(engine.data_depth, NH+1 if pad_h else NH, engine.stamp_size, NW+1 if pad_w else NW, engine.stamp_size)
    s_logits = s_logits.permute(1, 3, 0, 2, 4).reshape(-1, engine.data_depth, engine.stamp_size, engine.stamp_size)
    
    s_mask = m_p.view(1, NH+1 if pad_h else NH, engine.stamp_size, NW+1 if pad_w else NW, engine.stamp_size)
    s_mask = s_mask.permute(1, 3, 0, 2, 4).reshape(-1, 1, engine.stamp_size, engine.stamp_size)

    weighted_logits = (s_logits * s_mask).sum(dim=0)
    final_stamp = weighted_logits / (s_mask.sum(dim=0) + 1e-8)
    
    # Compare raw predicted bits to original bits
    predicted_bits = (torch.sigmoid(final_stamp.view(-1)) > 0.5).int().cpu().tolist()
    compare_len = len(original_bits)
    matches = sum(1 for i in range(compare_len) if predicted_bits[i] == original_bits[i])
    raw_ber = 1.0 - (matches / compare_len)
    
    print(f"   📊 Raw Bit Accuracy (Before RS): {1.0-raw_ber:.2%}")
    print(f"   📊 Bit Errors: {compare_len - matches} out of {compare_len}")

    if raw_ber > 0.40:
        print("\n   FAILURE: BER is near 50% (Random Guessing).")
        print("   Cause: Spatial Mismatch. The bits are not where the decoder expects them.")
    elif raw_ber > 0.15:
        print("\n   FAILURE: Decoder sees the signal, but noise is too high for RS.")
    else:
        print("\n   SUCCESS: Signal is clean.")

    print("\n" + "="*60 + "\n")

    print("\n[PHASE 3] Starting JPEG Extraction Gauntlet...")
    qualities_to_test = [90, 75, 60, 50]
    
    for q in qualities_to_test:
        print(f"\n   --- Testing JPEG Quality: {q} ---")
        stego_jpg_path = os.path.join(output_dir, f"stego_full_Q{q}.jpg")
        
        # Save compressed version
        img = Image.open(stego_png_path).convert("RGB")
        img.save(stego_jpg_path, "JPEG", quality=q)
        
        # Attempt extraction
        recovered_uuid = engine.extract_uuid(stego_jpg_path)
        
        if recovered_uuid == embedded_uuid:
            print(f"   🟢 SUCCESS: Recovered at Q{q}!")
        else:
            print(f"   🔴 FAILED: Noise at Q{q} exceeded correction threshold.")

    print("\n" + "="*60)
    print("  STRESS TEST COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_debug_test()

