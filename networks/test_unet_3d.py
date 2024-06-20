import unittest
import torch
import torch.nn as nn

from networks.unet_3d import UNet3d, DoubleConv

def assert_and_clear_cuda(expected, actual):
    try:
        assert expected == actual
    except AssertionError:
        print(f"!!! ERROR !!! Expected: {expected}, got {actual}")
        with torch.no_grad():
            torch.cuda.empty_cache()
    
class TestUNet3d(unittest.TestCase):
    def test_unet_3d(self):
        input_tensor = torch.randn(1, 1, 512, 512, 1)  # batch size of 1, 1 channel, 512x512x1 volume
        
        config = {          
            "n_blocks": 4,
            "downsampling_mode": "max_pool",
            "upsampling_mode": "linear_interpolation",
        }

        # Example usage
        model = UNet3d(
            init_filters=32,
            n_blocks=config["n_blocks"],
            activation="ReLU",
            downsampling_kernel=(2, 2, 1),
            downsampling_mode=config["downsampling_mode"],
            upsampling_kernel=(2, 2, 1),
            upsampling_mode=config["upsampling_mode"],
        )
        output = model(input_tensor)
        print(f"UNet output shape: {output.shape}")
        assert_and_clear_cuda((1, 2, 512, 512, 1), output.shape)


        conv_3d = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)
        conv_3d_output = conv_3d(input_tensor)
        print(f"Conv3d output shape: {conv_3d_output.shape}")
        assert_and_clear_cuda((1, 64, 512, 512, 1), conv_3d_output.shape)


        double_conv_3d = DoubleConv(64, 128)
        double_conv_output = double_conv_3d(conv_3d_output)
        print(f"{DoubleConv.__name__} output shape: {double_conv_output.shape}")
        assert_and_clear_cuda((1, 128, 512, 512, 1), double_conv_output.shape)


        max_3d = nn.MaxPool3d((3, 3, 1), stride=(2, 2, 1), padding=(1, 1, 0))
        max_3d_output_1 = max_3d(input_tensor)
        print(f"MaxPool3d output 1 shape: {max_3d_output_1.shape}")
        assert_and_clear_cuda((1, 1, 256, 256, 1), max_3d_output_1.shape)

        max_3d_input = torch.randn(1, 128, 512, 512, 1)
        max_3d_output_2 = max_3d(max_3d_input)
        print(f"MaxPool3d output 2 shape: {max_3d_output_2.shape}")
        assert_and_clear_cuda((1, 128, 256, 256, 1), max_3d_output_2.shape)

        conv_transpose_3d = nn.ConvTranspose3d(
            1024, 512, 
            kernel_size=(3, 3, 1), 
            stride=(2, 2, 1), 
            padding=(1, 1, 0), 
            output_padding=(1, 1, 0)
        )
        conv_transpose_3d_input = torch.randn(1, 1024, 32, 32, 1)
        conv_transpose_3d_output = conv_transpose_3d(conv_transpose_3d_input)
        print(f"ConvTranspose3d output shape: {conv_transpose_3d_output.shape}")
        assert_and_clear_cuda((1, 512, 64, 64, 1), conv_transpose_3d_output.shape)


        up_sample = nn.Upsample(
            scale_factor=(2, 2, 1), 
            mode='trilinear', align_corners=True) # What difference does it make if align_corners is True or False?
        up_sample_output = up_sample(input_tensor)
        print(f"Upsample output shape: {up_sample_output.shape}")
        assert_and_clear_cuda((1, 1, 1024, 1024, 1), up_sample_output.shape)
                        

        # # print(f"\n{model}")

        with torch.no_grad():
            torch.cuda.empty_cache()

        # # Delete the model and the output tensor
        # del model
        # del output
        # torch.cuda.empty_cache()