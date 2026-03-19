import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import os

from models.noise import StandardNoiseLayer

# ==========================================
# 1. ACTUAL HIDDEN ARCHITECTURE 
# ==========================================
class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class HiddenEncoder(nn.Module):
    def __init__(self, payload_length):
        super().__init__()
        self.payload_length = payload_length
        self.features = nn.Sequential(*[ConvBNRelu(3 if i==0 else 64, 64) for i in range(4)])
        self.post_concat = ConvBNRelu(64 + payload_length + 3, 64)
        self.final_conv = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, img, message):
        img_features = self.features(img)
        B, _, H, W = img.shape
        msg_expanded = message.unsqueeze(-1).unsqueeze(-1).expand(B, self.payload_length, H, W)
        concat_features = torch.cat([img_features, msg_expanded, img], dim=1)
        stego_img = self.final_conv(self.post_concat(concat_features))
        return stego_img

class HiddenDecoder(nn.Module):
    def __init__(self, payload_length):
        super().__init__()
        self.convs = nn.Sequential(*[ConvBNRelu(3 if i==0 else 64, 64) for i in range(7)])
        self.final_conv = ConvBNRelu(64, payload_length)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(payload_length, payload_length)

    def forward(self, noised_img):
        x = self.final_conv(self.convs(noised_img))
        x = self.global_pool(x).view(x.size(0), -1)
        return self.linear(x)

class HiddenDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = nn.Sequential(*[ConvBNRelu(3 if i==0 else 64, 64) for i in range(3)])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 1) 

    def forward(self, img):
        x = self.convs(img)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.linear(x)

# ==========================================
# 2. LOSS FUNCTIONS & VISUAL TEST
# ==========================================
class HiDDeNLoss(nn.Module):
    def __init__(self, lambda_i=0.7, lambda_g=0.001):
        super().__init__()
        self.mse_loss = nn.MSELoss() 
        self.bce_with_logits = nn.BCEWithLogitsLoss() 
        self.lambda_i = lambda_i
        self.lambda_g = lambda_g

    def forward(self, cover, stego, msg_in, msg_out, discriminator_logits):
        l_i = self.mse_loss(stego, cover)
        l_m = self.bce_with_logits(msg_out, msg_in)
        real_labels = torch.ones_like(discriminator_logits)
        l_g = self.bce_with_logits(discriminator_logits, real_labels)
        total_loss = l_m + (self.lambda_i * l_i) + (self.lambda_g * l_g)
        return total_loss, l_i, l_m, l_g

def calculate_ber(decoded_logits, original_message):
    probabilities = torch.sigmoid(decoded_logits)
    predicted_bits = (probabilities > 0.5).float()
    errors = (predicted_bits != original_message).sum().item()
    return errors / original_message.numel()

def save_models_and_visual_test(encoder, decoder, device, payload_length):
    print("\n--- SAVING MODELS & GENERATING VISUAL TEST ---")
    os.makedirs("saved_models", exist_ok=True)
    
    torch.save(encoder.state_dict(), "saved_models/hidden_encoder.pth")
    torch.save(decoder.state_dict(), "saved_models/hidden_decoder.pth")
    print("Model weights saved to /saved_models/")

    encoder.eval() 
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    cover_image = test_dataset[0][0].unsqueeze(0).to(device) 
    
    test_payload = torch.randint(0, 2, (1, payload_length)).float().to(device)
    
    with torch.no_grad():
        stego_image = encoder(cover_image, test_payload)
        
    noise_pattern = torch.abs(cover_image - stego_image) * 15 
    comparison = torch.cat([cover_image, stego_image, noise_pattern], dim=3)
    save_image(comparison, "saved_models/visual_test.png")
    print("Visual test saved! Check 'saved_models/visual_test.png' to see the results.")

# ==========================================
# 3. THE MASTER TRAINING LOOP
# ==========================================
def run_training_loop():
    print("\n--- INITIALIZING HiDDeN PHASE 2 TRAINING PIPELINE ---")
    batch_size = 128 
    payload_length = 208 
    epochs = 5
    lr = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Ingesting CIFAR-10 Dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )
    
    encoder = HiddenEncoder(payload_length).to(device)
    decoder = HiddenDecoder(payload_length).to(device)
    discriminator = HiddenDiscriminator().to(device)
    noise_layer = StandardNoiseLayer().to(device)
    
    criterion = HiDDeNLoss(lambda_i=0.7, lambda_g=0.001).to(device)
    opt_enc_dec = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for i, (images, _) in enumerate(train_loader):
            cover_images = images.to(device)
            current_batch_size = cover_images.size(0)
            payloads = torch.randint(0, 2, (current_batch_size, payload_length)).float().to(device)
            
            # STEP A: Train Discriminator
            opt_disc.zero_grad()
            stego_images = encoder(cover_images, payloads).detach() 
            d_real_logits = discriminator(cover_images)
            d_fake_logits = discriminator(stego_images)
            d_loss_real = criterion.bce_with_logits(d_real_logits, torch.ones_like(d_real_logits))
            d_loss_fake = criterion.bce_with_logits(d_fake_logits, torch.zeros_like(d_fake_logits))
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            opt_disc.step()
            
            # STEP B: Train Encoder/Decoder
            opt_enc_dec.zero_grad()
            stego_images = encoder(cover_images, payloads)
            noisy_images = noise_layer(stego_images, cover_images)
            extracted_payloads = decoder(noisy_images)
            d_fake_logits_for_gen = discriminator(stego_images)
            
            loss, l2_loss, msg_loss, g_loss = criterion(cover_images, stego_images, payloads, extracted_payloads, d_fake_logits_for_gen)
            loss.backward()
            opt_enc_dec.step()
            
            ber = calculate_ber(extracted_payloads, payloads)
            
            if (i+1) % 100 == 0: 
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}] "
                      f"| Total Loss: {loss.item():.4f} | D_Loss: {d_loss.item():.4f} | BER: {ber:.2%}")

        print(f"--> Epoch {epoch+1} Completed!")

    # Exactly here: save everything after all epochs!
    save_models_and_visual_test(encoder, decoder, device, payload_length)

if __name__ == "__main__":
    run_training_loop()
