import torch
import torch.nn as nn

class DecoderCIFAR(nn.Module):
    def __init__(self, message_length):
        super(DecoderCIFAR, self).__init__()
        
        self.message_length = message_length
        
        # Conv layers to find the hidden patterns
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Linear layer to squish patterns down to the payload size
        # 64 channels * 32 height * 32 width = 65,536 data points per image.
        self.linear_layer = nn.Sequential(
            nn.Linear(64 * 32 * 32, message_length),
            nn.Sigmoid() # final output numbers between 0 and 1
        )

    def forward(self, stego_image):
        # Pass the noisy image through the conv layers to extract feature maps
        # Shape: (Batch, 64, 32, 32)
        features = self.conv_layers(stego_image)
        
        # Flatten the 3D grid into one massive flat line of numbers
        # Shape: (Batch, 65536)
        flattened_features = torch.flatten(features, start_dim=1)
        
        # Push the line through the Linear layer to guess the 1s and 0s
        # Shape: (Batch, message_length)
        extracted_message = self.linear_layer(flattened_features)
        
        return extracted_message