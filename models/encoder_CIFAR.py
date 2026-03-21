import torch
import torch.nn as nn

class EncoderCIFAR(nn.Module):
    def __init__(self, message_length):
        super(EncoderCIFAR, self).__init__()
        
        # CIFAR-100 has 3 color channels. Stack expanded message channels directly onto the image channels
        input_channels = 3 + message_length 
        
        # Convolutional layers to weave the data together
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(), #Activation func
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output of 3 channels to look like a normal RGB image
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        )

    def forward(self, image_tensor, message_tensor):
        # dim of the incoming image
        batch_size, channels, height, width = image_tensor.shape
        
        # Reshape
        # Cur shape: (Batch, message_length)
        # New shape: (Batch, message_length, 1, 1)
        message_reshaped = message_tensor.unsqueeze(-1).unsqueeze(-1)
        
        # Stretch  to match 32x32 size
        # New shape: (Batch, message_length, height, width)
        message_expanded = message_reshaped.expand(-1, -1, height, width)
        
        # Stack image and payload
        # New shape: (Batch, 3 + message_length, height, width)
        combined_tensor = torch.cat([image_tensor, message_expanded], dim=1)
        
        # Push through conv layers
        stego_image = self.conv_layers(combined_tensor)
        
        return stego_image
    
