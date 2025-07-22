# training/networks_stylegan_multiclass.py

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is derived from an original work by NVIDIA CORPORATION & AFFILIATES.
# The modifications to implement the AC-GAN functionality are subject to the same
# license terms as the original work.

"""
This module adapts the StyleGAN2 discriminator for an Auxiliary Classifier GAN
(AC-GAN) task. It is designed to be compatible with both StyleGAN2 and StyleGAN3
generators, as StyleGAN3 uses the StyleGAN2 discriminator architecture.

The adaptation is achieved by inheriting from the original StyleGAN2 components
and overriding only the final discriminator block (the "epilogue") to produce
an additional output for multi-class classification. This approach maximizes
code reuse and maintainability.
"""

import numpy as np
import torch
from torch_utils import misc
from torch_utils import persistence

# --- Import base components from the original StyleGAN2 network module ---
from .networks_stylegan2 import Discriminator as StyleGAN2Discriminator
from .networks_stylegan2 import FullyConnectedLayer, Conv2dLayer, MinibatchStdLayer

# ============================================================================
# Redefine the Discriminator's Epilogue for the AC-GAN Task
# ============================================================================

@persistence.persistent_class
class DiscriminatorEpilogue(torch.nn.Module):
    """
    The final block of the Discriminator, modified to support AC-GAN.

    This epilogue features a dual-head architecture:
    1.  An 'adversarial head' for real-vs-fake discrimination (the standard task).
    2.  A 'classification head' for predicting class labels (the auxiliary task).

    The module processes a shared feature map from the discriminator's backbone,
    allowing both tasks to benefit from a common feature representation.
    """
    def __init__(self,
        in_channels,
        cmap_dim,
        resolution,
        img_channels,
        num_classes,            # **NEW**: Number of classes for the auxiliary task.
        architecture        = 'resnet',
        mbstd_group_size    = 4,
        mbstd_num_channels  = 1,
        activation          = 'lrelu',
        conv_clamp          = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.cmap_dim = cmap_dim
        self.resolution = resolution  # **FIX**: Added this line to store the resolution.
        self.num_classes = num_classes

        # Reuse standard StyleGAN2 layers for the shared backbone of the epilogue.
        if architecture == 'skip':
            self.fromrgb = Conv2dLayer(img_channels, in_channels, kernel_size=1, activation=activation)
        self.mbstd = MinibatchStdLayer(group_size=mbstd_group_size, num_channels=mbstd_num_channels) if mbstd_num_channels > 0 else None
        self.conv = Conv2dLayer(in_channels + mbstd_num_channels, in_channels, kernel_size=3, activation=activation, conv_clamp=conv_clamp)
        self.fc = FullyConnectedLayer(in_channels * (resolution ** 2), in_channels, activation=activation)
        
        # --- Two separate, specialized output heads ---
        # Head 1: Adversarial output for the main GAN task (real vs. fake).
        self.out = FullyConnectedLayer(in_channels, 1 if cmap_dim == 0 else cmap_dim)
        
        # Head 2: Auxiliary Classifier (AC) output for the classification task.
        self.ac_out = FullyConnectedLayer(in_channels, self.num_classes) if self.num_classes > 0 else None

    def forward(self, x, img, cmap, force_fp32=False):
        misc.assert_shape(x, [None, self.in_channels, self.resolution, self.resolution]) # [NCHW]
        # Explicitly cast to float32 to ensure numerical stability in the final layers
        # and during loss calculation, which is crucial when using mixed precision.
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
# Redefine the Main Discriminator to Use the New Epilogue
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
        #    Initialize the parent class. This constructs the entire standard
        #    StyleGAN2 discriminator, including its original epilogue in `self.b4`.
        #    We pass all original arguments directly to the parent constructor.
        super().__init__(
            c_dim, img_resolution, img_channels, architecture,
            channel_base, channel_max, num_fp16_res, conv_clamp,
            cmap_dim, block_kwargs, mapping_kwargs, epilogue_kwargs
        )
        self.num_classes = num_classes

        #    Robustly replace the standard epilogue with our AC-GAN version.
        #    First, inspect the just-created epilogue to get its configuration dynamically.
        #    This avoids hardcoding values and makes our code resilient to changes
        #    in the base networks_stylegan2.py file.
        original_epilogue_params = dict(
            in_channels=self.b4.in_channels,
            cmap_dim=self.b4.cmap_dim,
            resolution=self.b4.resolution,
            img_channels=self.b4.img_channels,
        )

        #    Then, overwrite `self.b4` with our new epilogue, passing the inspected
        #    parameters along with our new `num_classes` parameter.
        self.b4 = DiscriminatorEpilogue(
            **original_epilogue_params,
            num_classes=self.num_classes,
            **epilogue_kwargs
        )

    def forward(self, img, c, update_emas=False, force_fp32=False):
        # We explicitly accept all expected keyword arguments to make the interface robust.
        # 'update_emas' is unused in the discriminator but accepted for compatibility with the training loop.
        _ = update_emas

        # Create a dictionary of keyword arguments to pass down to the blocks.
        # This ensures only known and expected parameters are propagated, preventing TypeErrors.
        block_kwargs = dict(force_fp32=force_fp32)
        
        # The forward pass through the main blocks remains unchanged from StyleGAN2.
        x = None
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x, img = block(x, img, **block_kwargs)

        # The mapping network for conditioning labels also remains unchanged.
        cmap = None
        if self.c_dim > 0:
            cmap = self.mapping(None, c)
        
        # Call our new epilogue (`self.b4`), which correctly returns two outputs.
        adv_logits, ac_logits = self.b4(x, img, cmap)
        return adv_logits, ac_logits