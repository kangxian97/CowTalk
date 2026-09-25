import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import ConvBlock3d, ConvLayer3d


class ImplicitEncoder(nn.Module):
    """
    3D CNN encoder that processes volume data into a feature pyramid.
    
    Architecture:
    - Stem: Initial 7x7x7 convolution
    - Multiple ConvBlocks with downsampling via MaxPool3d
    - Returns a list of feature maps at different resolutions (feature pyramid)
    
    Args:
        in_channels: Number of input channels (typically 1 for volume)
        num_channels: List of channel counts for each stage [c0, c1, c2, c3]
        num_layers: Number of layers per ConvBlock
    """

    def __init__(self, in_channels, num_channels, num_layers):
        super().__init__()

        self.stem = ConvLayer3d(in_channels, num_channels[0], 7, True)
        self.conv_blocks = nn.ModuleList([ConvBlock3d(num_channels[i],
            num_channels[i + 1], 3, nn.InstanceNorm3d, num_layers, 0)
            for i in range(len(num_channels) - 1)])
        self.downsample = nn.MaxPool3d(2)

    def forward(self, x):
        """
        Forward pass through the encoder.
        
        Args:
            x: Input volume [B, C, D, H, W] (typically [B, 1, 128, 128, 128])
            
        Returns:
            features: List of feature maps at different resolutions
                - features[0]: Stem output [B, c0, D, H, W]
                - features[1]: After first ConvBlock + downsample [B, c1, D/2, H/2, W/2]
                - features[2]: After second ConvBlock + downsample [B, c2, D/4, H/4, W/4]
                - features[3]: After third ConvBlock + downsample [B, c3, D/8, H/8, W/8]
        """
        in_feature = self.stem(x)

        features = [in_feature]
        for i in range(len(self.conv_blocks)):
            out_feature = self.conv_blocks[i](in_feature)
            in_feature = self.downsample(out_feature)
            features.append(in_feature)

        return features
