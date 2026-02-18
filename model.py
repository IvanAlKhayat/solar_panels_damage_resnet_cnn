import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """
    Residual Block with skip connection.
    Structure: Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> (+ skip) -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the first convolution
        """
        super(ResBlock, self).__init__()

        # First convolution sequence: Conv -> BatchNorm -> ReLU
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        # Second convolution sequence: Conv -> BatchNorm
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity mapping with 1x1 conv if dimensions change)
        self.skip_connection = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # Final ReLU after adding skip connection
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass through the residual block"""
        # Store input for skip connection
        identity = x

        # First convolution sequence
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        # Second convolution sequence
        out = self.conv2(out)
        out = self.bn2(out)

        # Add skip connection
        identity = self.skip_connection(identity)
        out += identity

        # Final ReLU
        out = self.relu2(out)

        return out


class ResNet(nn.Module):
    """
    ResNet architecture for solar cell defect classification.
    Architecture follows Table 1 from the exercise description.
    """

    def __init__(self):
        super(ResNet, self).__init__()

        # Initial convolution block
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.resblock1 = ResBlock(64, 64, stride=1)
        self.resblock2 = ResBlock(64, 128, stride=2)
        self.resblock3 = ResBlock(128, 256, stride=2)
        self.resblock4 = ResBlock(256, 512, stride=2)

        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Flatten layer (handled in forward pass)

        # Fully connected layer
        self.fc = nn.Linear(512, 2)

        # Sigmoid activation for multi-label classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W)

        Returns:
            Output tensor of shape (batch_size, 2) with sigmoid activation
        """
        # Initial convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)

        # Global average pooling
        x = self.global_avg_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Fully connected layer
        x = self.fc(x)

        # Sigmoid activation
        x = self.sigmoid(x)

        return x


if __name__ == '__main__':
    # Test the model
    model = ResNet()

    # Create a dummy input (batch_size=4, channels=3, height=300, width=300)
    dummy_input = torch.randn(4, 3, 300, 300)

    # Forward pass
    output = model(dummy_input)

    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output values (should be between 0 and 1): {output}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")