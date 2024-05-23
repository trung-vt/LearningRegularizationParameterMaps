import torch


def extract_patches_3d(x, kernel_size, padding=0, stride=1, dilation=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    print(f"x.shape: {x.shape}")
    print(f"kernel_size: {kernel_size}")
    print(f"padding: {padding}")
    print(f"stride: {stride}")
    print(f"dilation: {dilation}")
    # x.shape: torch.Size([1, 1, 540, 960, 600])
    # kernel_size: [192, 192, 32]
    # padding: (0, 0, 0)
    # stride: [192, 192, 16]
    # dilation: (1, 1, 1)

    def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
        dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
        return dim_out

    channels = x.shape[-4]
    d_dim_in = x.shape[-3]
    h_dim_in = x.shape[-2]
    w_dim_in = x.shape[-1]
    d_dim_out = get_dim_blocks(d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
    h_dim_out = get_dim_blocks(h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])
    w_dim_out = get_dim_blocks(w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2])
    # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)
    print(f"d_dim_in = {d_dim_in}")
    print(f"h_dim_in = {h_dim_in}")
    print(f"w_dim_in = {w_dim_in}")
    print(f"d_dim_out = {d_dim_out}")
    print(f"h_dim_out = {h_dim_out}")
    print(f"w_dim_out = {w_dim_out}")
    # d_dim_in = 540
    # h_dim_in = 960
    # w_dim_in = 600
    # d_dim_out = 2
    # h_dim_out = 5
    # w_dim_out = 36

    # (B, C, D, H, W)
    x = x.view(-1, 
               channels, 
               d_dim_in, 
               h_dim_in * w_dim_in)                                                     
    # (B, C, D, H * W)

    x = torch.nn.functional.unfold(x, 
                                   kernel_size=(kernel_size[0], 1), 
                                   padding=(padding[0], 0), 
                                   stride=(stride[0], 1), 
                                   dilation=(dilation[0], 1))                   
    # (B, C * kernel_size[0], d_dim_out * H * W)

    print(f"x.shape: {x.shape}")
    # x.shape: torch.Size([1, 192, 1152000])
    # 1 * 192 * 1,152,000 = 221,184,000
    print(f"channels = {channels}")
    print(f"kernel_size[0] = {kernel_size[0]}")
    print(f"d_dim_out = {d_dim_out}")
    print(f"h_dim_in = {h_dim_in}")
    print(f"w_dim_in = {w_dim_in}")
    print(f"channels * kernel_size[0] * d_dim_out * h_dim_out * w_dim_out: {channels * kernel_size[0] * d_dim_out * h_dim_out * w_dim_out}")
    # extracting patches of shape [192, 192, 32]; strides [192, 192, 16]
    # x.shape: torch.Size([1, 192, 1152000])
    # channels = 1
    # kernel_size[0] = 192
    # d_dim_out = 2
    # h_dim_in = 960
    # w_dim_in = 600
    # channels * kernel_size[0] * d_dim_out * h_dim_out * w_dim_out: 69120

    x = x.view(-1, 
               channels * kernel_size[0] * d_dim_out, 
               h_dim_in, 
               w_dim_in)                                   
    # (B, C * kernel_size[0] * d_dim_out, H, W)

    x = torch.nn.functional.unfold(x, 
                                   kernel_size=(kernel_size[1], kernel_size[2]), 
                                   padding=(padding[1], padding[2]), 
                                   stride=(stride[1], stride[2]), 
                                   dilation=(dilation[1], dilation[2]))        
    # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

    x = x.view(-1, channels, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  
    # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)  

    x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
    # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

    x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2])
    # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

    return x