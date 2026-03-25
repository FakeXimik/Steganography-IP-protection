import torch
import torch.nn as nn

class DecoderCIFAR(nn.Module):
    def __init__(self, message_length):
        super(DecoderCIFAR, self).__init__()
        
        self.message_length = message_length
# Conv layers to find the hidden patterns
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # Linear layer to squish patterns down to the payload size
        # 64 channels * 32 height * 32 width = 65,536 data points per image.  
        self.linear_layer = nn.Sequential(
            nn.Linear(64 * 32 * 32, message_length)

        )

    def forward(self, stego_image):
        features = self.conv_layers(stego_image) #Extract feauture maps
        flattened_features = torch.flatten(features, start_dim=1) # Flatten 3D grid to 1dim
        extracted_message = self.linear_layer(flattened_features) # Push through linear layers
        return extracted_message