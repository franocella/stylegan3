# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is derived from an original work by NVIDIA CORPORATION & AFFILIATES.
# The modifications to implement the AC-GAN functionality are subject to the same
# license terms as the original work.

"""
Adapts the StyleGAN2 discriminator for an Auxiliary Classifier GAN (AC-GAN)
task by overriding the final network block to produce an additional output for
multi-class classification. This design maximizes code reuse and maintains
compatibility with both StyleGAN2 and StyleGAN3 generators.
"""

# --- Standard Library Imports ---
import numpy as np

# --- Third-party Imports ---
import torch
from torch_utils import misc
from torch_utils import persistence

# --- Local Application Imports ---
# Import base components from the original StyleGAN2 network module.
from training.networks_stylegan2 import Discriminator as StyleGAN2Discriminator, FullyConnectedLayer, Conv2dLayer, MinibatchStdLayer

# ============================================================================
# AC-GAN Discriminator Epilogue
# ============================================================================

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    """
    The final block of the Discriminator, modified for AC-GAN functionality.

    This epilogue features a dual-head architecture:
    1.  An 'adversarial head' for real-versus-fake discrimination.
    2.  A 'classification head' for predicting class labels.
    Both heads operate on a shared feature map from the discriminator's backbone.
    """
    def __init__(self,
        in_channels,
        cmap_dim,
        resolution,
        img_channels,
        num_classes,           
        architecture        = 'resnet',
        mbstd_group_size    = 4,
        mbstd_num_channels  = 1,
        activation          = 'lrelu',
        conv_clamp          = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution  
        self.num_classes = num_classes

        # Reuse standard StyleGAN2 layers for the shared backbone of the epilogue.
        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        
        # Define the two specialized output heads.

        # Head 1: Adversarial output for the main GAN task (real vs. fake).
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)
        
        # Head 2: Auxiliary Classifier (AC) output for the classification task.
        self.ac_out = FullyConnectedLayer(in_channels, self.num_classes) if self.num_classes > 0 else None

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) 
        # The final layers are explicitly cast to float32 to ensure numerical
        # stability during loss calculation, which is critical when using mixed precision.
        _ = force_fp32 # unused
        dtype = torch.float32
        memory_format = torch.contiguous_format
        x = x.to(dtype=dtype, memory_format=memory_format)
        
        # Process through the shared backbone.
        if hasattr(self, 'fromrgb'):
            img = img.to(torch.float32)
            x = x + self.fromrgb(img)
        if self.mbstd is not None:
            x = self.mbstd(x)
        x = self.conv(x)
        shared_features = self.fc(x.flatten(1))

        # Compute outputs from the two specialized heads.
        adv_logits = self.out(shared_features)
        ac_logits = self.ac_out(shared_features) if self.num_classes > 0 else None
        
        # Apply label conditioning to the adversarial output if provided.
        if self.cmap_dim > 0:
            misc.assert_shape(cmap, [None, self.cmap_dim])
            adv_logits = (adv_logits * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        # Return both sets of logits.
        return adv_logits, ac_logits

# ============================================================================
# Main AC-GAN Discriminator
# ============================================================================

@persistence.persistent_class
class Discriminator(StyleGAN2Discriminator):
    """
    The complete AC-GAN Discriminator, extending the original StyleGAN2 Discriminator.

    This class replaces the standard final block (the epilogue) with a new version
    that supports the dual adversarial/classification task. The implementation is
    designed to be robust and maintainable by inheriting from the original and
    overriding only the necessary components.
    """
    def __init__(self,
        c_dim,                          # Conditioning label (C) dimensionality.
        img_resolution,                 # Input resolution.
        img_channels,                   # Number of input color channels.
        num_classes         = 0,        # Number of classes for the auxiliary task.
        architecture        = 'resnet', # Architecture: 'orig', 'skip', 'resnet'.
        channel_base        = 32768,    # Overall multiplier for the number of channels.
        channel_max         = 512,      # Maximum number of channels in any layer.
        num_fp16_res        = 4,        # Use FP16 for the N highest resolutions.
        conv_clamp          = 256,      # Clamp the output of convolution layers to +-X, None = disable clamping.
        cmap_dim            = None,     # Dimensionality of mapped conditioning label, None = default.
        block_kwargs        = {},       # Arguments for DiscriminatorBlock.
        mapping_kwargs      = {},       # Arguments for MappingNetwork.
        epilogue_kwargs     = {},       # Arguments for DiscriminatorEpilogue.
    ):
        # Initialize the parent class to construct the standard StyleGAN2 discriminator.
        # This includes creating the original epilogue, which will be subsequently replaced.
        super().__init__(
            c_dim, img_resolution, img_channels, architecture,
            channel_base, channel_max, num_fp16_res, conv_clamp,
            cmap_dim, block_kwargs, mapping_kwargs, epilogue_kwargs
        )
        self.num_classes = num_classes

        # Dynamically inspect the original epilogue's parameters. This robustly
        # decouples the modification from the base implementation, ensuring
        # compatibility even if the underlying StyleGAN2 code changes.
        original_epilogue_params = dict(
            in_channels=self.b4.in_channels,
            cmap_dim=self.b4.cmap_dim,
            resolution=self.b4.resolution,
            img_channels=self.b4.img_channels,
        )

        # Overwrite the original epilogue (`self.b4`) with the new AC-GAN version.
        self.b4 = DiscriminatorEpilogue(
            **original_epilogue_params,
            num_classes=self.num_classes,
            **epilogue_kwargs
        )

    def forward(self, img, c, update_emas=False, force_fp32=False):
        _ = update_emas # Unused in the discriminator, accepted for compatibility.

        block_kwargs = dict(force_fp32=force_fp32)
        
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        
        adv_logits, ac_logits = self.b4(x, img, cmap)
        return adv_logits, ac_logits