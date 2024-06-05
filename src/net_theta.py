import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01)
        )

    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_blocks = nn.ModuleList([
            # ConvBlock(1, 16),
            # ConvBlock(16, 32),
            # ConvBlock(32, 64),
            ConvBlock(1, 64),  # NOTE: Imitating the original UNet
            ConvBlock(64, 128),  # NOTE: Trying to add one more block
            ConvBlock(128, 256),  # NOTE: Trying to add two more blocks
            # ConvBlock(256, 512),  # NOTE: Trying to add three more blocks
            # ConvBlock(512, 1024),  # NOTE: Imitating the original UNet
        ])
        self.pool = nn.MaxPool3d(kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        enc_features = []
        for block in self.enc_blocks:
            x = block(x)
            enc_features.append(x)
            x = self.pool(x)
        return enc_features

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.upconvs = nn.ModuleList([
            # nn.Conv3d(1024, 512, kernel_size=3, stride=1, padding=1),  # NOTE: Imitating the original UNet
            # nn.Conv3d(512, 256, kernel_size=3, stride=1, padding=1),  # NOTE: Trying to add three more blocks
            nn.Conv3d(256, 128, kernel_size=3, stride=1, padding=1),  # NOTE: Trying to add two more blocks
            nn.Conv3d(128, 64, kernel_size=3, stride=1, padding=1),  # NOTE: Trying to add one more block
            # nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
            # nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=1),
        ])
        self.dec_blocks = nn.ModuleList([
            # ConvBlock(1024, 512),  # NOTE: Imitating the original UNet
            # ConvBlock(512, 256),  # NOTE: Trying to add three more blocks
            ConvBlock(256, 128),  # NOTE: Trying to add two more blocks
            ConvBlock(128, 64),  # NOTE: Trying to add one more block
            # ConvBlock(64, 32),
            # ConvBlock(32, 16),
        ])

    def forward(self, x, enc_features):
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            x = torch.cat((x, enc_features[i]), dim=1)  # concatenate along the channel axis
            x = self.dec_blocks[i](x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        # self.c1x1 = nn.Conv3d(16, 2, kernel_size=1, stride=1, padding=0)
        self.c1x1 = nn.Conv3d(64, 2, kernel_size=1, stride=1, padding=0)    # NOTE: Imitating the original UNet

    def forward(self, x):
        enc_features = self.encoder(x)
        x = self.decoder(enc_features[-1], enc_features[::-1][1:]) # remove the last element and reverse the list
        x = self.c1x1(x)
        return x

def test_unet():
    # Example usage
    model = UNet()
    # input_tensor = torch.randn(1, 1, 64, 64, 64)  # batch size of 1, 1 channel, 64x64x64 volume
    input_tensor = torch.randn(1, 1, 512, 512, 1)  # batch size of 1, 1 channel, 64x64x64 volume
    output = model(input_tensor)
    # print(output.shape)  # should be (1, 2, 64, 64, 64)
    print(output.shape)  # should be (1, 2, 512, 512, 1)

    print(f"\n{model}")




def get_example_autoencoder(device):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(16, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),
        
        nn.Upsample(scale_factor=2),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(),

        nn.Upsample(scale_factor=2),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),

        nn.Upsample(scale_factor=2),
        nn.Conv2d(16, 3, 3, padding=1),
        nn.Sigmoid()
    )
    return model.to(device)