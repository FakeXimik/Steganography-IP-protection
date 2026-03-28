import torch
import torch.nn as nn

class DecoderDense(nn.Module):
    """
    SteganoGAN DenseDecoder

    """
    def __init__(self, data_depth, hidden_size=32):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

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
        # Init concatenation
        x1 = self.conv1(stego_image)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x1, x2, x3], dim=1))
        
        # SPATIAL AGGREGATION 
        # x4 shape: (Batch_Size, Data_Depth, Height, Width)
        # We average across Height (dim=2) and Width (dim=3) to compress it
        # Final Output shape: (Batch_Size, Data_Depth)
        extracted_message = torch.mean(x4, dim=(2, 3))
        
        return extracted_message