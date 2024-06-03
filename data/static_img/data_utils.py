import torch

# TODO: Add documentation and tests to help others understand how this works!
def extract_patches_2d(x, kernel_size, padding=0, stride=1, dilation=1):
    """
    Extract small patches from a large image to make it fit into memory and speed up processing.

    Args:
        x (torch.Tensor): The input image.
        kernel_size (int): The size of the kernel.
        padding (int): The padding to add to the image.
        stride (int): The stride to use when extracting patches.
        dilation (int): The dilation to use when extracting patches.

    Returns:
        torch.Tensor: The extracted patches.
    """
    # TODO: What does this do?
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    # TODO: What does this do?
    if isinstance(padding, int):
        padding = (padding, padding)

    # TODO: What does this do?
    if isinstance(stride, int):
        stride = (stride, stride)

    # TODO: What does this do?
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    print(f"x.shape: {x.shape}")
    print(f"kernel_size: {kernel_size}")
    print(f"padding: {padding}")
    print(f"stride: {stride}")
    print(f"dilation: {dilation}")
    # x.shape: torch.Size([1, 1500, 2664, 1])
    # kernel_size: [192, 192, 32]
    # padding: (0, 0)
    # stride: [192, 192, 16]
    # dilation: (1, 1)

    # TODO: What does this do?
    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        # TODO: What is this formula?
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    # TODO: MAKE SURE THE INDEXES ARE CORRECT
    channels = x.shape[-3]  # TODO: What are channels? Surely not color channels, since the input is grayscale image, right?
    h_dim_in = x.shape[-2]  # TODO: Height?
    w_dim_in = x.shape[-1]  # TODO: Width?

    # TODO: MAKE SURE THE INDEXES ARE CORRECT
    h_dim_out_index = 0 # I made these variables so if I need to revert or change the index, I only need to change one place
    w_dim_out_index = 1

    # TODO: How does height-out differ from height-in? 
    h_dim_out = get_dim_blocks(
        h_dim_in, 
        kernel_size[h_dim_out_index], 
        padding[h_dim_out_index], 
        stride[h_dim_out_index], 
        dilation[h_dim_out_index]
    )
    # TODO: How does width-out differ from width-in?
    w_dim_out = get_dim_blocks(
        w_dim_in, 
        kernel_size[w_dim_out_index], 
        padding[w_dim_out_index], 
        stride[w_dim_out_index], 
        dilation[w_dim_out_index]
    )
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)
    print(f"h_dim_in = {h_dim_in}")
    print(f"w_dim_in = {w_dim_in}")
    print(f"h_dim_out = {h_dim_out}")
    print(f"w_dim_out = {w_dim_out}")
    # h_dim_in = 2664
    # w_dim_in = 1
    # h_dim_out = 13
    # w_dim_out = 0

    # (B, C, H, W)
    x = x.view(-1, 
               channels, 
               h_dim_in * w_dim_in)
    # (B, C, H * W)

    x = torch.nn.functional.unfold(x, 
                                   kernel_size=(kernel_size[0], 1), 
                                   padding=(padding[0], 0), 
                                   stride=(stride[0], 1), 
                                   dilation=(dilation[0], 1))                   
    # (B, C * kernel_size[0], d_dim_out * H * W)

    print(f"x.shape: {x.shape}")
    # x.shape: torch.Size([192, 18648])
    # 192 * 18,648 = 3,578,416
    print(f"channels = {channels}")
    print(f"kernel_size[0] = {kernel_size[0]}")
    print(f"h_dim_in = {h_dim_in}")
    print(f"w_dim_in = {w_dim_in}")
    print(f"channels * kernel_size[0] * h_dim_out * w_dim_out: {channels * kernel_size[0] * h_dim_out * w_dim_out}")
    # extracting patches of shape [192, 192, 32]; strides [192, 192, 16]
    # x.shape: torch.Size([192, 18648])
    # channels = 1500
    # kernel_size[0] = 192
    # h_dim_in = 2664
    # w_dim_in = 1
    # channels * kernel_size[0] * h_dim_out * w_dim_out: 0

    # RuntimeError: shape '[-1, 288000, 2664, 1]' is invalid for input of size 3580416
    # 288,000 * 2664 = 767,232,000
    x = x.view(-1, 
               channels * kernel_size[0], 
               h_dim_in, 
               w_dim_in)                                   
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, 
                                   kernel_size=(kernel_size[1], kernel_size[2]), 
                                   padding=(padding[1], padding[2]), 
                                   stride=(stride[1], stride[2]), 
                                   dilation=(dilation[1], dilation[2]))        
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

    x = x.view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x