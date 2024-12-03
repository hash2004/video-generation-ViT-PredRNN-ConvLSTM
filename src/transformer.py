from functools import partial

import torch
from torch import nn, einsum

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

"""
This implementation is based on the paper:

    "MaxViT: Multi-Axis Vision Transformer." 
        Zhengzhong Tu1,2
        Hossein Talebi1
        Han Zhang1
        Feng Yang1
        Peyman Milanfar1
        Alan Bovik2
        Yinxiao Li1
        
It represents a modified version of the original approach, which was designed for image classification using ViT. 

The code has been adapted to handle video data, where the input is a sequence of frames and the output is another sequence of frames.

The original implementation, authored by Phil Wang, can be found at lucidrains' GitHub repository for ViT Pytorch: https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/max_vit.py

This modified version has been tailored specifically for video data and video prediction tasks.
"""


# helpers   

def exists(val):
    """
    Checks if a given value is not None.

    Arguments:
        val: The value to check.

    Returns:
        bool: True if `val` is not None, False otherwise.
    """
    return val is not None

def default(val, d):
    """
    Returns the value `val` if it exists; otherwise, returns a default value `d`.

    Arguments:
        val: The value to check.
        d: The default value to return if `val` is None.

    Returns:
        The original value `val` if it exists, otherwise the default value `d`.
    """
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    """
    Ensures that the input `val` is a tuple of a specified length.

    Arguments:
        val: The value to cast.
        length (int, optional): The desired length of the tuple. Defaults to 1.

    Returns:
        tuple: A tuple containing `val` repeated `length` times if `val` is not already a tuple.
               Otherwise, returns `val` as is.
    """
    return val if isinstance(val, tuple) else ((val,) * length)

# helper classes

class Residual(nn.Module):
    """
    Implements a residual connection wrapper for a given function or module.

    This class adds the input `x` to the output of the function `fn(x)`, facilitating the learning of residual mappings.

    Arguments:
        dim (int): The dimensionality of the input and output.
        fn (nn.Module): The function or module to wrap with a residual connection.

    Methods:
        forward (torch.Tensor): Applies the residual connection to the input tensor.
    """
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

class FeedForward(nn.Module):
    """
    Implements a feed-forward neural network with Layer Normalization, GELU activation, and Dropout.

    This module is typically used within transformer architectures to process each token independently.

    Arguments:
        dim (int): The input and output dimensionality.
        mult (int, optional): The expansion factor for the hidden layer. Defaults to 4.
        dropout (float, optional): The dropout probability. Defaults to 0.0.

    Methods:
        forward (torch.Tensor): Processes the input tensor through the feed-forward network.
    """
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# MBConv

class SqueezeExcitation(nn.Module):
    """
    Implements the Squeeze-and-Excitation (SE) block to recalibrate channel-wise feature responses.

    This module performs global average pooling followed by two fully connected layers with a non-linearity
    and sigmoid activation to generate channel-wise weights, which are then applied to the input tensor.

    Arguments:
        dim (int): The number of input channels.
        shrinkage_rate (float, optional): The reduction ratio for the hidden dimension. Defaults to 0.25.

    Methods:
        forward (torch.Tensor): Applies the SE block to the input tensor.
    """
    def __init__(self, dim, shrinkage_rate = 0.25):
        super().__init__()
        hidden_dim = int(dim * shrinkage_rate)

        self.gate = nn.Sequential(
            Reduce('b c h w -> b c', 'mean'),
            nn.Linear(dim, hidden_dim, bias = False),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim, bias = False),
            nn.Sigmoid(),
            Rearrange('b c -> b c 1 1')
        )

    def forward(self, x):
        return x * self.gate(x)


class MBConvResidual(nn.Module):
    """
    Wraps an MBConv block with a residual connection and applies dropout.

    This class adds the input `x` to the output of the MBConv block after applying dropout,
    facilitating better gradient flow and model generalization.

    Arguments:
        fn (nn.Module): The MBConv block to wrap.
        dropout (float, optional): The dropout probability. Defaults to 0.0.

    Methods:
        forward (torch.Tensor): Applies the residual connection and dropout to the input tensor.
    """
    def __init__(self, fn, dropout = 0.):
        super().__init__()
        self.fn = fn
        self.dropsample = Dropsample(dropout)

    def forward(self, x):
        out = self.fn(x)
        out = self.dropsample(out)
        return out + x

class Dropsample(nn.Module):
    """
    Implements the DropSample regularization technique.

    This module randomly drops entire samples (e.g., feature maps) with a specified probability during training.

    Arguments:
        prob (float, optional): The probability of dropping a sample. Defaults to 0.0.

    Methods:
        forward (torch.Tensor): Applies DropSample to the input tensor.
    """
    def __init__(self, prob = 0):
        super().__init__()
        self.prob = prob
  
    def forward(self, x):
        device = x.device

        if self.prob == 0. or (not self.training):
            return x

        keep_mask = torch.FloatTensor((x.shape[0], 1, 1, 1), device = device).uniform_() > self.prob
        return x * keep_mask / (1 - self.prob)

def MBConv(
    dim_in,
    dim_out,
    *,
    downsample,
    expansion_rate = 4,
    shrinkage_rate = 0.25,
    dropout = 0.
):
    """
    Constructs a Mobile Inverted Bottleneck Convolutional (MBConv) block.

    This block consists of a series of convolutional layers with batch normalization, GELU activation,
    Squeeze-and-Excitation, and optional residual connections with dropout. It is designed for efficient
    feature extraction in transformer architectures.

    Arguments:
        dim_in (int): The number of input channels.
        dim_out (int): The number of output channels.
        downsample (bool): Whether to apply downsampling via stride in the depthwise convolution.
        expansion_rate (int, optional): The expansion factor for the hidden dimension. Defaults to 4.
        shrinkage_rate (float, optional): The reduction ratio for the SE block. Defaults to 0.25.
        dropout (float, optional): The dropout probability for residual connections. Defaults to 0.0.

    Returns:
        nn.Sequential: The constructed MBConv block.
    """
    hidden_dim = int(expansion_rate * dim_out)
    stride = 2 if downsample else 1

    net = nn.Sequential(
        nn.Conv2d(dim_in, hidden_dim, 1),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        nn.Conv2d(hidden_dim, hidden_dim, 3, stride = stride, padding = 1, groups = hidden_dim),
        nn.BatchNorm2d(hidden_dim),
        nn.GELU(),
        SqueezeExcitation(hidden_dim, shrinkage_rate = shrinkage_rate),
        nn.Conv2d(hidden_dim, dim_out, 1),
        nn.BatchNorm2d(dim_out)
    )

    if dim_in == dim_out and not downsample:
        net = MBConvResidual(net, dropout = dropout)

    return net

# attention related classes

class Attention(nn.Module):
    """
    Implements multi-head self-attention with relative positional bias.

    This module performs self-attention on input tensors, incorporating relative positional embeddings
    to enhance the model's ability to capture spatial relationships within windows.

    Arguments:
        dim (int): The dimensionality of the input and output.
        dim_head (int, optional): The dimensionality of each attention head. Defaults to 32.
        dropout (float, optional): The dropout probability for attention weights. Defaults to 0.0.
        window_size (int, optional): The size of the attention window. Defaults to 4.

    Methods:
        forward (torch.Tensor): Applies self-attention to the input tensor.
    """
    def __init__(
        self,
        dim,
        dim_head = 32,
        dropout = 0.,
        window_size = 4
    ):
        super().__init__()
        assert (dim % dim_head) == 0, 'dimension should be divisible by dimension per head'

        self.heads = dim // dim_head
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)

        self.attend = nn.Sequential(
            nn.Softmax(dim = -1),
            nn.Dropout(dropout)
        )

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias = False),
            nn.Dropout(dropout)
        )

        # relative positional bias

        self.rel_pos_bias = nn.Embedding((2 * window_size - 1) ** 2, self.heads)

        pos = torch.arange(window_size)
        grid = torch.stack(torch.meshgrid(pos, pos, indexing = 'ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = rearrange(grid, 'i ... -> i 1 ...') - rearrange(grid, 'j ... -> 1 j ...')
        rel_pos += window_size - 1
        rel_pos_indices = (rel_pos * torch.tensor([2 * window_size - 1, 1])).sum(dim = -1)

        self.register_buffer('rel_pos_indices', rel_pos_indices, persistent = False)

    def forward(self, x):
        batch, height, width, window_height, window_width, _, device, h = *x.shape, x.device, self.heads

        x = self.norm(x)

        # flatten

        x = rearrange(x, 'b x y w1 w2 d -> (b x y) (w1 w2) d')

        # project for queries, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # sim

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add positional bias

        bias = self.rel_pos_bias(self.rel_pos_indices)
        sim = sim + rearrange(bias, 'i j h -> h i j')

        # attention

        attn = self.attend(sim)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h (w1 w2) d -> b w1 w2 (h d)', w1 = window_height, w2 = window_width)

        # combine heads out

        out = self.to_out(out)
        return rearrange(out, '(b x y) ... -> b x y ...', x = height, y = width)

class MaxViT(nn.Module):
    """
    Implements the MaxViT architecture for video generation tasks.

    This model extends the original MaxViT designed for image classification to handle video data by processing
    sequences of frames. It incorporates convolutional stems, multiple transformer-based stages with MBConv blocks,
    and a decoder to generate predicted video frames.

    Arguments:
        num_input_frames (int): The number of input frames in the video sequence.
        num_output_frames (int): The number of frames to generate in the output sequence.
        dim (int): The base dimensionality for the transformer blocks.
        depth (tuple of int): A tuple specifying the number of transformer blocks at each stage.
        dim_head (int, optional): The dimensionality of each attention head. Defaults to 32.
        dim_conv_stem (int, optional): The dimensionality of the convolutional stem. Defaults to `dim`.
        window_size (int, optional): The size of the attention window. Defaults to 4.
        mbconv_expansion_rate (int, optional): The expansion rate for MBConv blocks. Defaults to 4.
        mbconv_shrinkage_rate (float, optional): The shrinkage rate for SE blocks within MBConv. Defaults to 0.25.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        channels_per_frame (int, optional): The number of channels per video frame. Defaults to 3.

    Methods:
        forward (torch.Tensor): Processes the input video frames and generates predicted frames.
    """
    def __init__(
        self,
        *,
        num_input_frames,
        num_output_frames,
        dim,
        depth,
        dim_head=32,
        dim_conv_stem=None,
        window_size=4,  
        mbconv_expansion_rate=4,
        mbconv_shrinkage_rate=0.25,
        dropout=0.1,
        channels_per_frame=3
    ):
        super().__init__()
        assert isinstance(depth, tuple), 'depth needs to be a tuple of integers indicating the number of transformer blocks at each stage'

        # Calculate input and output channels
        input_channels = num_input_frames * channels_per_frame
        output_channels = num_output_frames * channels_per_frame

        dim_conv_stem = default(dim_conv_stem, dim)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(input_channels, dim_conv_stem, 3, stride=1, padding=1),  
            nn.Conv2d(dim_conv_stem, dim_conv_stem, 3, padding=1)
        )

        num_stages = len(depth)
        dims = tuple(map(lambda i: (2 ** i) * dim, range(num_stages)))
        dims = (dim_conv_stem, *dims)
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        self.layers = nn.ModuleList([])
        w = window_size

        for ind, ((layer_dim_in, layer_dim), layer_depth) in enumerate(zip(dim_pairs, depth)):
            for stage_ind in range(layer_depth):
                is_first = stage_ind == 0
                stage_dim_in = layer_dim_in if is_first else layer_dim

                downsample = is_first and (ind == 0)

                block = nn.Sequential(
                    MBConv(
                        stage_dim_in,
                        layer_dim,
                        downsample=downsample,  
                        expansion_rate=mbconv_expansion_rate,
                        shrinkage_rate=mbconv_shrinkage_rate
                    ),
                    Rearrange('b d (x w1) (y w2) -> b x y w1 w2 d', w1=w, w2=w),
                    Residual(layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
                    Residual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange('b x y w1 w2 d -> b d (x w1) (y w2)'),
                    Rearrange('b d (w1 x) (w2 y) -> b x y w1 w2 d', w1=w, w2=w),
                    Residual(layer_dim, Attention(dim=layer_dim, dim_head=dim_head, dropout=dropout, window_size=w)),
                    Residual(layer_dim, FeedForward(dim=layer_dim, dropout=dropout)),
                    Rearrange('b x y w1 w2 d -> b d (w1 x) (w2 y)'),
                )

                self.layers.append(block)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dims[-1], output_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.conv_stem(x)
        for stage in self.layers:
            x = stage(x)
        x = self.decoder(x)
        return x
