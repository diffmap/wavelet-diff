import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class WaveletUNet(nn.Module):
    def __init__(
        self,
        model_dim: int,
        base_channels: int = 32,
        num_layers: int = 3,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(dropout)
        
        self.input_proj = nn.Conv1d(5, base_channels, 1)
        
        self.downs = nn.ModuleList()
        curr_channels = base_channels
        for _ in range(num_layers):
            next_channels = curr_channels * 2
            self.downs.append(ConvBlock(curr_channels, next_channels, kernel_size))
            curr_channels = next_channels
        
        self.bottleneck = ConvBlock(curr_channels, curr_channels, kernel_size)
        
        self.ups = nn.ModuleList()
        for i in reversed(range(num_layers)):
            upsample_channels = base_channels * (2 ** (i + 1))
            skip_channels     = upsample_channels
            out_channels      = base_channels * (2 ** i)
            
            self.ups.append(
                nn.ConvTranspose1d(upsample_channels, out_channels, kernel_size=2, stride=2)
            )
            self.ups.append(
                ConvBlock(out_channels + skip_channels, out_channels)
            )
        
        self.final_conv = nn.Conv1d(base_channels, 5, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        skips = []
        
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
            x = self.dropout(x)
        
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        for idx in range(0, len(self.ups), 2):
            up_conv = self.ups[idx]
            conv_block = self.ups[idx + 1]
            
            x = up_conv(x)
            skip = skips[-(idx//2 + 1)]
            
            if x.size(-1) != skip.size(-1):
                x = F.pad(x, (0, skip.size(-1) - x.size(-1)))
            
            x = torch.cat([x, skip], dim=1)
            x = conv_block(x)
            x = self.dropout(x)
        
        return self.final_conv(x)
    