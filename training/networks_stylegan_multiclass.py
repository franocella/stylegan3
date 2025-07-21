# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is a modification of the original StyleGAN2 network architecture
# to include an auxiliary classification head in the Discriminator,
# enabling a multi-class classification task alongside the GAN training.

"""
Modified network architectures from the paper
"Analyzing and Improving the Image Quality of StyleGAN"
to support an auxiliary classification task.
"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import upfirdn2d

# These classes are imported from the original StyleGAN2 network script.
# Ensure that the original `networks_stylegan2.py` is in the `training` directory.
from training.networks_stylegan2 import Conv2dLayer, FullyConnectedLayer, MinibatchStdLayer, DiscriminatorBlock, MappingNetwork


@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    """
    Modified Discriminator Epilogue with an auxiliary classification head.

    This module processes the final feature map from the discriminator backbone
    and produces two separate outputs: one for the GAN's real/fake prediction
    and another for multi-class classification.
    """
    def __init__(self,
        in_channels,                    # Number of input channels.
        cmap_dim,                       # Dimensionality of mapped conditioning label, 0 = no label.
        resolution,                     # Resolution of this block.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        mbstd_group_size    = 4,        # Group size for the minibatch standard deviation layer.
        mbstd_num_channels  = 1,        # Number of features for the minibatch standard deviation layer.
        activation          = 'lrelu',  # Activation function.
        conv_clamp          = None,     # Clamp the output of convolution layers.
        num_classes         = 0,        # Number of classes for the auxiliary classifier. If 0, disabled.
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.architecture = architecture
        self.num_classes = num_classes

        # FromRGB layer for the 'skip' architecture.
        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)

        # Minibatch standard deviation layer to increase variety.
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        
        # Standard convolutional and fully-connected layers to process features.
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        
        # Output head for the GAN real/fake score.
        self.out_gan = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)
        
        # Output head for the multi-class classification task.
        # This head is only created if num_classes is greater than 0.
        self.out_class = None
        if num_classes > 0:
            self.out_class = FullyConnectedLayer(in_channels, num_classes)

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution])
        _ = force_fp32
        dtype = torch.float32
        memory_format = torch.contiguous_format  


        # Process input from the previous block and optionally from RGB.
        x = x.to(dtype=dtype, memory_format=memory_format)
        if self.architecture == 'skip':
            misc.assert_shape(img, [None, self.img_channels, self.resolution, self.resolution])
            img = img.to(dtype=dtype, memory_format=memory_format)
            x = x + self.fromrgb(img)

        # Apply main feature extraction layers.
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        features = self.fc(x.flatten(1))
        
        # Compute GAN logits, applying conditioning if provided.
        gan_logits = self.out_gan(features)
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            gan_logits = (gan_logits * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        # Compute classification logits if the classification head is enabled.
        class_logits = None
        if self.out_class is not None:
            class_logits = self.out_class(features)

        # Return both sets of logits.
        return gan_logits, class_logits

    def extra_repr(self):
        return f'resolution={self.resolution:d}, architecture={self.architecture:s}, num_classes={self.num_classes:d}'


@persistence.persistent_class
class Discriminator(torch.nn.Module):
    """
    Modified Discriminator network with an auxiliary classification head.

    This class constructs the discriminator by chaining together multiple
    `DiscriminatorBlock` modules and finishes with the modified `DiscriminatorEpilogue`.
    """
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label.
        num_classes         = 0,        # Number of classes for auxiliary classification.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        super().__init__()
        self.c_dim = c_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.num_classes = num_classes

        # Define network resolutions and channel counts.
        self.block_resolutions = [2 ** i for i in range(self.img_resolution_log2, 2, -1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions + [4]}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        # Set conditioning map dimensionality.
        if cmap_dim is None:
            cmap_dim = channels_dict[4]
        if c_dim == 0:
            cmap_dim = 0

        # Build the discriminator blocks.
        common_kwargs = dict(img_channels=img_channels, architecture=architecture, conv_clamp=conv_clamp)
        cur_layer_idx = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res] if res < img_resolution else 0
            tmp_channels = channels_dict[res]
            out_channels = channels_dict[res // 2]
            use_fp16 = (res >= fp16_resolution)
            block = DiscriminatorBlock(in_channels, tmp_channels, out_channels, resolution=res,
                first_layer_idx=cur_layer_idx, use_fp16=use_fp16, **block_kwargs, **common_kwargs)
            setattr(self, f'b{res}', block)
            cur_layer_idx += block.num_layers
        
        # Optional mapping network for conditioning labels.
        if c_dim > 0:
            self.mapping = MappingNetwork(z_dim=0, c_dim=c_dim, w_dim=cmap_dim, num_ws=None, w_avg_beta=None, **mapping_kwargs)
        
        # Instantiate the modified epilogue, passing the number of classes to it.
        self.b4 = DiscriminatorEpilogue(channels_dict[4], cmap_dim=cmap_dim, resolution=4, num_classes=num_classes, **epilogue_kwargs, **common_kwargs)

    def forward(self, img, c, update_emas=False, **block_kwargs):
        _ = update_emas # unused
        
        # Process image through the discriminator blocks.
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        # Get conditioning vector.
        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        
        # The epilogue now returns both GAN and classification logits.
        gan_logits, class_logits = self.b4(x, img, cmap)
        return gan_logits, class_logits

    def extra_repr(self):
        return f'c_dim={self.c_dim:d}, img_resolution={self.img_resolution:d}, img_channels={self.img_channels:d}, num_classes={self.num_classes:d}'