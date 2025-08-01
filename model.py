import torch
import torch.nn as nn
import torch.nn.functional as F

# === FiLM Layer ===
class FiLM(nn.Module):
    def __init__(self, feature_dim, condition_dim):
        super().__init__()
        self.film = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.LeakyReLU(0.1, inplace=True) #nn.ReLU()
        )

    def forward(self, x, cond):
        gamma_beta = self.film(cond)  # (B, 2 * C)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return gamma * x + beta

# === Downsample Block ===
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# === Bottleneck ResBlock with FiLM ===
class BottleneckResBlock(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm1 = nn.BatchNorm2d(channels)
        self.film = FiLM(channels, condition_dim)
        self.act = nn.LeakyReLU(0.1, inplace=True) #nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.norm2 = nn.BatchNorm2d(channels)

    def forward(self, x, cond):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.film(out, cond)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)
        return self.act(out + residual)

# === Upsample Block with FiLM ===
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, condition_dim):
        super().__init__()
        self.block = nn.Sequential(
            #nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True) #nn.ReLU(inplace=True)
        )
        self.film = FiLM(out_ch, condition_dim)

    def forward(self, x, cond):
        x = self.block(x)
        x = self.film(x, cond)
        return x

# === Main Conditional Autoencoder ===
class CCAE(nn.Module):
    def __init__(self, image_channels=1, condition_dim=4, base_channels=16):
        super().__init__()
        self.condition_dim = condition_dim
        self.register_buffer('global_step', torch.tensor(0, dtype=torch.long))  # for LPIPS tracking

        # === Condition embedding ===
        self.cond_embed = nn.Sequential(
            nn.Linear(condition_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(128, 128)
        )
        cond_latent_dim = 128

        # === Encoder ===
        chs = [base_channels * (2 ** i) for i in range(5)]  # [16, 32, 64, 128, 256]
        self.encoder_blocks = nn.ModuleList([
            DownBlock(in_ch, out_ch) for in_ch, out_ch in zip([image_channels] + chs[:-1], chs)
        ])
        self.encoder_conv = nn.Conv2d(chs[-1], 512, kernel_size=3, padding=1, padding_mode='reflect')

        # === Bottleneck ===
        self.bottleneck_blocks = nn.Sequential(
            BottleneckResBlock(512, cond_latent_dim),
            BottleneckResBlock(512, cond_latent_dim),
            BottleneckResBlock(512, cond_latent_dim),
        )

        # === Decoder ===
        self.decoder_blocks = nn.ModuleList([
            UpBlock(512, 256, cond_latent_dim),
            UpBlock(256, 128, cond_latent_dim),
            UpBlock(128, 64, cond_latent_dim),
            UpBlock(64, 32, cond_latent_dim),
            UpBlock(32, 16, cond_latent_dim)
        ])

        self.final_conv = nn.Conv2d(16, image_channels, kernel_size=3, padding=1, padding_mode='reflect')
        self.output_act = nn.Tanh()

    def encode(self, x):
        for down in self.encoder_blocks:
            x = down(x)
        x = self.encoder_conv(x)
        return x  # shape: (B, 512, 16, 16)

    def decode(self, bottleneck, cond):
        cond_emb = self.cond_embed(cond)        # (B, 128)
        cond_emb = F.normalize(cond_emb, dim=1) # optional normalization

        x = bottleneck
        for block in self.bottleneck_blocks:
            x = block(x, cond_emb)

        for up in self.decoder_blocks:
            x = up(x, cond_emb)

        x = self.final_conv(x)
        return self.output_act(x)

    def forward(self, x, cond):
        bottleneck = self.encode(x)
        output = self.decode(bottleneck, cond)
        return output, self.global_step
