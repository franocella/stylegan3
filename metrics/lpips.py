"""Perceptual Loss (LPIPS) metric.

Reference:
"The Unreasonable Effectiveness of Deep Features as a Perceptual Metric"
by Zhang et al. (2018).
https://arxiv.org/abs/1801.03924
"""

import torch
import numpy as np
import lpips 
import dnnlib

#----------------------------------------------------------------------------

# Global variable to store the LPIPS model instance.
_lpips_model = None

def _get_lpips_model(device: torch.device) -> lpips.LPIPS:
    """
    Initializes and returns the LPIPS model.

    This function ensures the LPIPS model is loaded only once (singleton pattern)
    and placed on the specified device.

    Args:
        device: The PyTorch device (e.g., 'cuda', 'cpu') to load the model onto.

    Returns:
        An instance of the LPIPS model.
    """
    global _lpips_model
    if _lpips_model is None:
        # Initialize LPIPS with 'alex' net, which is commonly used.
        # spatial=False computes a single scalar distance per image pair.
        _lpips_model = lpips.LPIPS(net='alex', spatial=False).to(device)
        # Set the model to evaluation mode to disable dropout and batch norm updates.
        _lpips_model.eval()
    return _lpips_model

#----------------------------------------------------------------------------

def compute_lpips(opts: dnnlib.EasyDict, num_gen: int, num_real: int) -> float:
    """
    Computes the LPIPS (Learned Perceptual Image Patch Similarity) score.

    This metric quantifies the perceptual difference between two sets of images.

    Args:
        opts: MetricOptions object containing various options like device,
              batch size, and access to the generator/dataset iterators.
        num_gen: The total number of generated images to use for the calculation.
        num_real: The total number of real images to sample from the dataset.

    Returns:
        The mean LPIPS score as a float, or NaN if not calculated on rank 0.
    
    Raises:
        ValueError: If `num_gen` and `num_real` are not equal, as LPIPS
                    typically compares corresponding image pairs.
    """
    lpips_model = _get_lpips_model(opts.device)
    lpips_values = []

    # Use StyleGAN3's utility functions to efficiently iterate over real and
    # generated image minibatches. This handles multi-GPU distribution and
    # data loading seamlessly.

    # Iterator for real images from the dataset.
    real_images_iterator = opts.dataset_iterator(
        num_items=num_real,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        real_img_normalize=False, # We handle normalization explicitly below.
        xflip=False, # LPIPS typically does not use x-flips.
    )

    # Iterator for generated images from the generator (G_ema for evaluation).
    gen_images_iterator = opts.G_ema.generate_images_batch(
        num_items=num_gen,
        num_gpus=opts.num_gpus,
        rank=opts.rank,
        batch_size=opts.batch_size,
        # Other params like noise_mode, truncation_psi can be added if needed,
        # but typically default for metrics.
    )
    
    # Ensure that the number of generated and real images match for pairwise comparison.
    if num_gen != num_real:
        raise ValueError(
            f"For LPIPS calculation between generated and real images, "
            f"num_gen ({num_gen}) must be equal to num_real ({num_real})."
        )

    # Iterate through batches of real and generated images.
    for real_batch, gen_batch in zip(real_images_iterator, gen_images_iterator):
        # Image tensors from StyleGAN3 are typically in [0, 255] range (uint8 or float32).
        # LPIPS model expects float32 inputs normalized to [-1, 1].
        real_batch_norm = (real_batch / 127.5 - 1.0).to(torch.float32)
        gen_batch_norm = (gen_batch / 127.5 - 1.0).to(torch.float32)

        # Calculate LPIPS for the current batch.
        # The lpips_model typically returns a tensor of shape (batch_size, 1, 1, 1).
        lpips_batch_scores = lpips_model(gen_batch_norm, real_batch_norm)
        
        # Extract scalar scores and convert to a NumPy array for aggregation.
        lpips_values.extend(lpips_batch_scores.squeeze().cpu().numpy())

    # In a multi-GPU setup, the final aggregation (e.g., mean) should typically
    # happen only on rank 0 to avoid redundant calculations and ensure correctness
    # across distributed processes. If `metric_utils.iterate_minibatches` already
    # handles collection/broadcasting, this check might be more for robustness.
    if opts.rank != 0:
        return float('nan') # Return NaN for non-rank 0 processes.

    # Compute the mean LPIPS score across all samples.
    # LPIPS is commonly reported as a single mean value.
    lpips_mean = float(np.mean(lpips_values))
    return lpips_mean

#----------------------------------------------------------------------------