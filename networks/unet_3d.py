import torch
import torch.nn as nn

# Used https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py as a reference

class DoubleConv(nn.Module):
    def __init__(
            self, in_channels: int, out_channels: int, n_dimensions=3, activation="LeakyReLU"):
        super(DoubleConv, self).__init__()

        def get_conv(in_channels, out_channels):
            # 1-dimensional convolution is not supported
            if n_dimensions == 3:
                return nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 1), padding=(1, 1, 0))
            elif n_dimensions == 2:
                return nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
            else:
                raise ValueError(f"Unsupported number of dimensions: {n_dimensions}")

        def get_activation():
            if activation == "LeakyReLU":
                return nn.LeakyReLU(negative_slope=0.01, inplace=True)
            elif activation == "ReLU":
                return nn.ReLU(inplace=True)
            else:
                raise ValueError(f"Unsupported activation function: {activation}")

        self.conv_block = nn.Sequential(
            get_conv(in_channels, out_channels), get_activation(),
            get_conv(out_channels, out_channels), get_activation())

    def forward(self, x: torch.Tensor):
        return self.conv_block(x)
        

class EncodeBlock3d(nn.Module):
    def __init__(
            self, in_channels: int, n_dimensions=3,
            activation="LeakyReLU",
            downsampling_kernel=(2, 2, 1), downsampling_mode="max"):
        super(EncodeBlock3d, self).__init__()

        len = downsampling_kernel[0] # Assume kernel has shape (len, len, 1)
        assert downsampling_kernel == (len, len, 1), f"Expected a flat square kernel like {(len, len, 1)}, got {downsampling_kernel}"
        stride = (2, 2, 1) # Stride 2x2 to halve each side 
        padding = ((len-1)//2, (len-1)//2, 0) # Padding (len-1) // 2 to exactly halve each side 
        if downsampling_mode == "max_pool":
            self.pool = nn.MaxPool3d(
                kernel_size=downsampling_kernel, stride=stride, padding=padding)
        elif downsampling_mode == "avg_pool":
            self.pool = nn.AvgPool3d(
                kernel_size=downsampling_kernel, stride=stride, padding=padding)
        else:
            raise ValueError(f"Unknown pooling method: {downsampling_mode}")

        self.double_conv = DoubleConv(in_channels, in_channels * 2, n_dimensions, activation=activation)

    def forward(self, x: torch.Tensor):
        x = self.pool(x)
        x = self.double_conv(x)
        return x



class DecodeBlock3d(nn.Module):
    def __init__(
            self, in_channels: int, n_dimensions=3, 
            activation="LeakyReLU",
            upsampling_kernel=(2, 2, 1), upsampling_mode="linear_interpolation"):
        super(DecodeBlock3d, self).__init__()

        if upsampling_mode == "linear_interpolation":
            self.upsampling = nn.Sequential(
                nn.Upsample(
                    scale_factor=(2, 2, 1), # Assume the shape is (Nx, Ny, 1) where Nx is the image width and Ny is the image height.
                    mode='trilinear', align_corners=True), # What difference does it make in the end if align_corners is True or False? Preserving symmetry?
                # 1x1x1 convolution to reduce the number of channels while keeping the size the same
                nn.Conv3d(
                    in_channels, in_channels // 2, 
                    kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
            )
        elif upsampling_mode == "transposed_convolution":  
            len = upsampling_kernel[0] # Assume kernel has shape (len, len, 1)
            assert upsampling_kernel == (len, len, 1), f"Expected a flat square kernel like {(len, len, 1)}, got {upsampling_kernel}"
            stride = (2, 2, 1) # Stride 2x2 to double each side 
            padding = ((len-1)//2, (len-1)//2, 0) # Padding (len-1) // 2 to exactly double each side    
            self.upsampling = nn.ConvTranspose3d(
                in_channels, in_channels // 2, 
                kernel_size=upsampling_kernel, stride=stride, padding=padding, 
                output_padding=padding # TODO: Should this be the same as padding?
            )
        else:
            raise ValueError(f"Unsupported upsampling method: {upsampling_mode}")
        
        self.double_conv = DoubleConv(in_channels, in_channels // 2, n_dimensions, activation=activation)

    def forward(self, x: torch.Tensor, x_encoder_output: torch.Tensor):
        x = self.upsampling(x)
        x = torch.cat([x_encoder_output, x], dim=1)   # skip-connection. No cropping since we ensure that the size is the same.
        x = self.double_conv(x)
        return x



class UNet3d(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=2, init_filters=32, n_blocks=3,
            activation="LeakyReLU",
            downsampling_kernel=(2, 2, 1), downsampling_mode="max",
            upsampling_kernel=(2, 2, 1), upsampling_mode="linear_interpolation",
    ):
        """
        Assume that input is 5D tensor of shape (batch_size, channels, Nx, Ny, Nt)
        where Nx is the image width and Ny is the image height.
        Assume that batch_size = 1, channels = 1, Nx = Ny (square image), Nt = 1 (static image).
        NOTE: The convention used in pytorch documentation is (batch_size, channels, Nt, Ny, Nx).
        "channels" is equivalent to the number of filters or features.

        Our paper (figure 2):
            - in_channels = 1
            - out_channels = 2
            - init_filters = 32
            - n_blocks = 3
            - pooling: max pooling 2x2
            - pool padding = 1
                - 1 padding will keep the size of the "image" the same after each convolution. The skip-connection will NOT crop the encoder's output.
            - upsampling kernel: 2x2 ?
            - up_mode: linear interpolation

        U-Net paper (2015, Olaf Ronneberger https://arxiv.org/abs/1505.04597):
            - in_channels = 1
            - out_channels = 2
            - init_filters = 64
            - n_blocks = 4
            - pooling: max pooling 2x2
            - pool padding = 0
                - 0 padding will reduce the size of the "image" by 2 in each dimension after each convolution. The skip-connection will have to crop the encoder's output to match the decoder's input.
            - upsampling kernel: 2x2
            - up_mode: ? (linear interpolation or transposed convolution)
        """
        super(UNet3d, self).__init__()
        
        self.c0x0 = DoubleConv( # TODO: Find a better name
            in_channels=in_channels, 
            out_channels=init_filters,
            activation=activation
        )
        self.encoder = nn.ModuleList([
            EncodeBlock3d(
                in_channels=init_filters * 2**i,
                activation=activation,
                downsampling_kernel=downsampling_kernel,
                downsampling_mode=downsampling_mode
            ) for i in range(n_blocks)
        ])
        self.decoder = nn.ModuleList([
            DecodeBlock3d(
                in_channels=init_filters * 2**(n_blocks-i),
                activation=activation, 
                upsampling_kernel=upsampling_kernel,
                upsampling_mode=upsampling_mode
            ) for i in range(n_blocks)
        ])
        # 1x1x1 convo
        self.c1x1 = nn.Conv3d( # TODO: Find a better name
            in_channels=init_filters,
            out_channels=out_channels,
            kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)
        )

    def forward(self, x: torch.Tensor):
        # Assume that x is 5D tensor of shape (batch_size, channels, Nx, Ny, Nt)
        # where Nx is the image width and Ny is the image height.
        # Aslo assume that batch_size = 1, channels = 1, Nx = Ny (square image), Nt = 1 (static image).
        # NOTE: The convention used in pytorch documentation is (batch_size, channels, Nt, Ny, Nx).
        assert len(x.size()) == 5, f"Expected 5D tensor, got {x.size()}"
        batch_size, channels, Nx, Ny, Nt = x.size()
        assert channels == 1, f"Expected 1 channel, got {channels}" # TODO: Allow multiple channels (colour images)
        assert Nx == Ny, f"Expected square image, got ({Nx}, {Ny})" # TODO: Allow non-square images
        assert Nt == 1, f"Expected 1 time step, got {Nt}" # TODO: Allow multiple time steps (dynamic images, video)
        assert batch_size == 1, f"Expected batch size 1, got {batch_size}" # TODO: Might train with larger batch size

        n_blocks = len(self.encoder)
        assert Nx >= 2**n_blocks, f"Expected width (Nx) of at least {2**n_blocks}, got {Nx}"
        assert Ny >= 2**n_blocks, f"Expected height (Ny) of at least {2**n_blocks}, got {Ny}"

        x = self.c0x0(x)

        encoder_outputs = []
        for i, enc_block in enumerate(self.encoder):
            encoder_outputs.append(x)
            x = enc_block(x)
        for i, dec_block in enumerate(self.decoder):
            x = dec_block(x, encoder_outputs[-i-1]) # skip-connection inside
            
        x = self.c1x1(x)

        for enc_output in encoder_outputs:
            del enc_output
        del encoder_outputs

        return x