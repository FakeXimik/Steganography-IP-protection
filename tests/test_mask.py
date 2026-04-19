import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import os

# ==========================================
# OPTION 3: DIRECTIONAL RESIDUALS (S-UNIWARD Lite)
# ==========================================
class DirectionalTextureMask(nn.Module):
    def __init__(self):
        super().__init__()
        # 4 Directional High-Frequency Filters (Spatial Wavelets)
        # These strictly isolate high-frequency chaos and ignore smooth gradients.
        k1 = torch.tensor([[[[ 0.,  0.,  0.], [-1.,  2., -1.], [ 0.,  0.,  0.]]]]) # Horizontal
        k2 = torch.tensor([[[[ 0., -1.,  0.], [ 0.,  2.,  0.], [ 0., -1.,  0.]]]]) # Vertical
        k3 = torch.tensor([[[[-1.,  0.,  0.], [ 0.,  2.,  0.], [ 0.,  0., -1.]]]]) # Diagonal 1
        k4 = torch.tensor([[[[ 0.,  0., -1.], [ 0.,  2.,  0.], [-1.,  0.,  0.]]]]) # Diagonal 2

        self.register_buffer('filters', torch.cat([k1, k2, k3, k4], dim=0) / 4.0)

    def forward(self, img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        # 1. Get directional residuals (Output is 4 channels)
        residuals = F.conv2d(gray, self.filters, padding=1)

        # 2. Find the maximum texture energy across all 4 directions
        energy = torch.max(torch.abs(residuals), dim=1, keepdim=True)[0]

        # 3. Connect the dots: Small 3x3 Max Pool fills in the grass WITHOUT bleeding into the sky
        energy = F.max_pool2d(energy, kernel_size=3, stride=1, padding=1)

        # 4. The Noise Floor Cutoff
        # Subtracting 0.005 kills invisible JPEG artifacts in the sky dead to 0.0.
        # Multiplying by 100 boosts the remaining grass/texture to pure white (1.0).
        mask = torch.clamp((energy - 0.005) * 100.0, 0.0, 1.0)

        # 5. Gentle 3x3 blur so the noise seamlessly fades into the image
        mask = F.avg_pool2d(mask, 3, stride=1, padding=1)

        return mask

# ==========================================
# OPTION 1: LAPLACIAN HIGH-FREQUENCY MASK
# ==========================================
class LaplacianTextureMask(nn.Module):
    def __init__(self):
        super().__init__()
        # 2nd-order derivative kernel. 
        # Punishes smooth areas, ignores single edges, explodes on chaotic noise.
        laplacian = torch.tensor([[[[-1., -1., -1.],
                                    [-1.,  8., -1.],
                                    [-1., -1., -1.]]]]) / 8.0
        self.register_buffer('weight', laplacian)

    def forward(self, img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        
        # Extract high-frequency energy
        lap = F.conv2d(gray, self.weight, padding=1)
        energy = torch.abs(lap)
        
        # Smooth and boost
        energy = F.avg_pool2d(energy, 5, stride=1, padding=2)
        mask = torch.clamp(energy * 25.0, 0.0, 1.0)
        return mask

# ==========================================
# OPTION 2: MULTI-SCALE VARIANCE (SOTA)
# ==========================================
class MultiScaleVarianceMask(nn.Module):
    def __init__(self):
        super().__init__()

    def _local_variance(self, gray, kernel_size):
        pad = kernel_size // 2
        mean = F.avg_pool2d(gray, kernel_size, stride=1, padding=pad)
        sq_mean = F.avg_pool2d(gray**2, kernel_size, stride=1, padding=pad)
        var = F.relu(sq_mean - mean**2)
        return torch.sqrt(var + 1e-8)

    def forward(self, img):
        gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        # Calculate texture busyness at 3 different Zoom Levels
        v_micro = self._local_variance(gray, 3)   # Pores, single leaves
        v_medium = self._local_variance(gray, 7)  # Clumps of dirt/grass
        v_macro = self._local_variance(gray, 15)  # Overall foliage structure

        combined_std = (v_micro * v_medium * v_macro) ** (1.0/3.0)

        # Massive boost to turn the remaining deep textures into pure white
        mask = torch.clamp(combined_std * 100.0, 0.0, 1.0)
        
        # Soften the boundaries slightly so the stego noise fades in naturally
        mask = F.avg_pool2d(mask, 5, stride=1, padding=2)
        return mask

# ==========================================
# TEST RUNNER
# ==========================================
def run_test():
    image_path = "assets/test_photo.png"
    
    if not os.path.exists(image_path):
        print(f"Cannot find {image_path}!")
        return

    print(f"🔍 Analyzing textures in {image_path}...")
    image = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(image).unsqueeze(0) 

    # Test Option 1
    laplacian_layer = LaplacianTextureMask()
    with torch.no_grad():
        lap_tensor = laplacian_layer(img_tensor)
    transforms.ToPILImage()(lap_tensor.squeeze(0)).save("assets/output/mask_1_laplacian.png")
    print("Saved Option 1: mask_1_laplacian.png")

    # Test Option 2
    multi_layer = MultiScaleVarianceMask()
    with torch.no_grad():
        multi_tensor = multi_layer(img_tensor)
    transforms.ToPILImage()(multi_tensor.squeeze(0)).save("assets/output/mask_2_multiscale.png")
    print("Saved Option 2: mask_2_multiscale.png")

    # Test Option 3
    dir_layer = DirectionalTextureMask()
    with torch.no_grad():
        dir_tensor = dir_layer(img_tensor)
    transforms.ToPILImage()(dir_tensor.squeeze(0)).save("assets/output/mask_3_directional.png")
    print("Saved Option 3: mask_3_directional.png")

if __name__ == "__main__":
    run_test()