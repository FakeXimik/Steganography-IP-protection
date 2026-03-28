import torch
import torch.nn as nn

class EncoderDense(nn.Module):
    """
    SteganoGAN DenseEncoder
    """
    def __init__(self, data_depth, hidden_size=32):
        super().__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        # Layer 1: takes 3 RGB channels 
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        
        # Layer 2: Takes Layer 1 Output + Message Tensor
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.hidden_size + self.data_depth, self.hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        
        # Layer 3: Takes Layer 1 Output + Layer 2 Output + Message Tensor
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 2 + self.data_depth, self.hidden_size, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        
        # Layer 4: Takes L1 + L2 + L3 + Message Tensor and outputs the noise map (3 RGB Channels)
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.hidden_size * 3 + self.data_depth, 3, kernel_size=3, padding=1)
        )

    def forward(self, image, message):
        # Expand 1D message to 2D image dim
        batch_size, _, height, width = image.shape
        message_expanded = message.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

        # Initialize layers
        x1 = self.conv1(image)
        x2 = self.conv2(torch.cat([x1, message_expanded], dim=1))
        x3 = self.conv3(torch.cat([x1, x2, message_expanded], dim=1))
        noise_map = self.conv4(torch.cat([x1, x2, x3, message_expanded], dim=1))

        # 3. Residual Connection (SteganoGAN's add_image = True)
        return image + noise_map