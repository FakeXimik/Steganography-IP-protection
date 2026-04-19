import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    """
    predicts an affine transformation matrix to un-warp the image 
    before handing it to the main Decoder block.
    """
    def __init__(self):
        super().__init__()
        
        # Extracts geometric features to figure out how the image is tilted/shifted
        self.localization = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            # forces the feature map to be exactly 4x4 
            nn.AdaptiveAvgPool2d((4, 4)) 
        )

        # Regressor - calculates the 6 numbers (2x3 Affine Matrix) needed to correct the geometry
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(True),
            nn.Linear(64, 6) # Outputs a 2x3 matrix
        )

        # Initialize the weights so the STN starts by doing absolutely nothing
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # Extract geometric features
        features = self.localization(x)
        features = features.view(-1, 32 * 4 * 4)
        
        # Calculate the Affine Matrix
        theta = self.fc_loc(features)
        theta = theta.view(-1, 2, 3)
        
        # Create a mathematical grid and warp the image back to straight
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        unwarped_x = F.grid_sample(x, grid, align_corners=False)
        
        return unwarped_x

class DecoderDense(nn.Module):
    """
    SteganoGAN DenseDecoder

    """
    def __init__(self, data_depth, hidden_size=32):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        self.stn = SpatialTransformer()

        # Layer 1: Takes 3 RGB channels of the Stego Image
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        # Layer 2: Takes Layer 1 Output
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        # Layer 3: Takes Layer 1 Output + Layer 2 Output
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 2, self.hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size)
        )

        # Layer 4: Takes L1 + L2 + L3 Outputs and generates the Spatial Message Tensor
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 3, self.data_depth, kernel_size=3, padding=1)
        )

    def forward(self, stego_image):
        aligned_image = self.stn(stego_image)

        # Init concatenation
        x1 = self.conv1(aligned_image)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3], dim=1))
        return x4

