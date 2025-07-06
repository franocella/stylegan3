# dynamic_metrics.py

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
A module for computing dynamic GAN quality metrics.
Each function dynamically adjusts the number of samples based on the dataset size
to ensure that the number of real and generated images are identical, providing
a robust and statistically sound evaluation.
"""

import numpy as np
import scipy.linalg
import torch
import lpips
import dnnlib
import copy
from . import metric_utils

#----------------------------------------------------------------------------
#       Dynamic Frechet Inception Distance (FID)
#----------------------------------------------------------------------------

def compute_dynamic_fid(opts, max_real, num_gen):
    """
    Computes Frechet Inception Distance (FID), dynamically adjusting sample
    counts to ensure an equal number of real and generated images.
    """
    # --- Dynamic sample size logic ---
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    
    if max_real is None:
        num_real_to_use = len(dataset)
    else:
        num_real_to_use = min(max_real, len(dataset))

    num_to_compare = min(num_real_to_use, num_gen)

    if opts.rank == 0:
        if num_to_compare < num_gen:
            print(f"Warning: Dataset size limits the number of samples for FID.")
        elif max_real is not None and num_to_compare < max_real:
            print(f"Warning: Dataset size is smaller than the requested number of real images for FID.")
        
        if num_to_compare < num_gen or (max_real is not None and num_to_compare < max_real):
            print(f"FID will be computed using {num_to_compare} real and {num_to_compare} generated images.")
    # --- End of dynamic logic ---

    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_mean_cov=True, max_items=num_to_compare).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_mean_cov=True, max_items=num_to_compare).get_mean_cov()

    if opts.rank != 0:
        return float('nan')

    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
#       Dynamic Kernel Inception Distance (KID)
#----------------------------------------------------------------------------

def compute_dynamic_kid(opts, max_real, num_gen, num_subsets, max_subset_size):
    """
    Computes Kernel Inception Distance (KID), dynamically adjusting sample
    counts to ensure an equal number of real and generated images.
    """
    # --- Dynamic sample size logic ---
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    
    if max_real is None:
        num_real_to_use = len(dataset)
    else:
        num_real_to_use = min(max_real, len(dataset))

    num_to_compare = min(num_real_to_use, num_gen)

    if opts.rank == 0:
        if num_to_compare < num_gen:
            print(f"Warning: Dataset size limits the number of samples for KID.")
        elif max_real is not None and num_to_compare < max_real:
            print(f"Warning: Dataset size is smaller than the requested number of real images for KID.")
        
        if num_to_compare < num_gen or (max_real is not None and num_to_compare < max_real):
            print(f"KID will be computed using {num_to_compare} real and {num_to_compare} generated images.")
    # --- End of dynamic logic ---
    
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_all=True, max_items=num_to_compare).get_all()

    gen_features = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_all=True, max_items=num_to_compare).get_all()

    if opts.rank != 0:
        return float('nan')

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)

#----------------------------------------------------------------------------
#       Dynamic Learned Perceptual Image Patch Similarity (LPIPS)
#----------------------------------------------------------------------------

_lpips_model = None

def _get_lpips_model(device: torch.device) -> lpips.LPIPS:
    """Initializes and returns the LPIPS model."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex', spatial=False).to(device)
        _lpips_model.eval()
    return _lpips_model

def compute_dynamic_lpips(opts, num_gen, num_real, batch_size=64):
    """
    Computes LPIPS, dynamically adjusting sample counts to ensure an
    equal number of real and generated images.
    """
    lpips_model = _get_lpips_model(opts.device)
    lpips_values = []

    # --- Dynamic sample size logic ---
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    num_items_in_dataset = len(dataset)
    
    actual_num_to_compare = min(min(num_gen, num_real), num_items_in_dataset)

    if opts.rank == 0 and actual_num_to_compare < num_real:
        print(f"Warning: Dataset size ({num_items_in_dataset}) is smaller than the requested number of real images ({num_real}).")
        print(f"LPIPS will be computed using {actual_num_to_compare} real and {actual_num_to_compare} generated images.")
    # --- End of dynamic logic ---

    # --- Collect real images ---
    real_images_list = []
    item_subset_for_rank = [(i * opts.num_gpus + opts.rank) % actual_num_to_compare
                            for i in range((actual_num_to_compare - 1) // opts.num_gpus + 1)]
    progress_real = opts.progress.sub(tag='real images for LPIPS', num_items=actual_num_to_compare)
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset_for_rank, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        real_images_list.append(images.to(opts.device))
    all_real_images_local = torch.cat(real_images_list)
    progress_real.update(actual_num_to_compare)

    # --- Collect generated images ---
    G_ema = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    gen_images_list = []
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_size)
    progress_gen = opts.progress.sub(tag='gen images for LPIPS', num_items=actual_num_to_compare)
    num_local_gen_images = (actual_num_to_compare // opts.num_gpus)
    if opts.rank < (actual_num_to_compare % opts.num_gpus):
        num_local_gen_images += 1
    
    current_gen_items_local = 0
    while current_gen_items_local < num_local_gen_images:
        current_batch_size_local = min(batch_size, num_local_gen_images - current_gen_items_local)
        if current_batch_size_local == 0: break
        z = torch.randn([current_batch_size_local, G_ema.z_dim], device=opts.device)
        c = next(c_iter)[:current_batch_size_local]
        img = G_ema(z=z, c=c, **opts.G_kwargs)
        gen_images_list.append(img.to(torch.float32))
        current_gen_items_local += current_batch_size_local
    all_gen_images_local = torch.cat(gen_images_list)
    progress_gen.update(num_local_gen_images)

    # --- Combine and compute ---
    if opts.num_gpus > 1:
        # Complex multi-GPU gathering logic would be needed here
        pass

    lpips_mean = float('nan')
    if opts.rank == 0:
        real_images_normalized = (all_real_images_local.to(torch.float32) / 127.5 - 1.0) if all_real_images_local.dtype == torch.uint8 else all_real_images_local.to(torch.float32)
        gen_images_normalized = all_gen_images_local.to(torch.float32)
        
        assert real_images_normalized.shape == gen_images_normalized.shape

        for i in range(0, actual_num_to_compare, batch_size):
            end_idx = min(i + batch_size, actual_num_to_compare)
            real_batch = real_images_normalized[i:end_idx]
            gen_batch = gen_images_normalized[i:end_idx]
            lpips_values.extend(_get_lpips_model(opts.device)(gen_batch, real_batch).detach().squeeze().cpu().numpy())
        
        lpips_mean = float(np.mean(lpips_values))

    return lpips_mean

    """
    Computes LPIPS, dynamically adjusting sample counts to ensure an
    equal number of real and generated images.
    """
    lpips_model = _get_lpips_model(opts.device)
    lpips_values = []

    # --- Dynamic sample size logic ---
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    num_items_in_dataset = len(dataset)
    
    actual_num_to_compare = min(min(num_gen, num_real), num_items_in_dataset)

    if opts.rank == 0 and actual_num_to_compare < num_real:
        print(f"Warning: Dataset size ({num_items_in_dataset}) is smaller than the requested number of real images ({num_real}).")
        print(f"LPIPS will be computed using {actual_num_to_compare} real and {actual_num_to_compare} generated images.")
    # --- End of dynamic logic ---

    # --- Collect real images ---
    real_images_list = []
    item_subset_for_rank = [(i * opts.num_gpus + opts.rank) % actual_num_to_compare
                            for i in range((actual_num_to_compare - 1) // opts.num_gpus + 1)]
    progress_real = opts.progress.sub(tag='real images for LPIPS', num_items=actual_num_to_compare)
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    
    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset_for_rank, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        real_images_list.append(images.to(opts.device))
    all_real_images_local = torch.cat(real_images_list)
    progress_real.update(actual_num_to_compare)

    # --- Collect generated images ---
    G_ema = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    gen_images_list = []
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_size)
    progress_gen = opts.progress.sub(tag='gen images for LPIPS', num_items=actual_num_to_compare)
    num_local_gen_images = (actual_num_to_compare // opts.num_gpus)
    if opts.rank < (actual_num_to_compare % opts.num_gpus):
        num_local_gen_images += 1
    
    current_gen_items_local = 0
    while current_gen_items_local < num_local_gen_images:
        current_batch_size_local = min(batch_size, num_local_gen_images - current_gen_items_local)
        if current_batch_size_local == 0: break
        z = torch.randn([current_batch_size_local, G_ema.z_dim], device=opts.device)
        c = next(c_iter)[:current_batch_size_local]
        img = G_ema(z=z, c=c, **opts.G_kwargs)
        gen_images_list.append(img.to(torch.float32))
        current_gen_items_local += current_batch_size_local
    all_gen_images_local = torch.cat(gen_images_list)
    progress_gen.update(num_local_gen_images)

    # --- Combine and compute ---
    # This part assumes single GPU or that gathering logic is handled outside if needed.
    # For simplicity in this self-contained function, we handle the single GPU case.
    if opts.num_gpus > 1:
        # Complex multi-GPU gathering logic would be needed here for torch.distributed
        # For now, we proceed assuming rank 0 will do the final calculation.
        pass

    lpips_mean = float('nan')
    if opts.rank == 0:
        real_images_normalized = (all_real_images_local.to(torch.float32) / 127.5 - 1.0) if all_real_images_local.dtype == torch.uint8 else all_real_images_local.to(torch.float32)
        gen_images_normalized = all_gen_images_local.to(torch.float32)
        
        assert real_images_normalized.shape == gen_images_normalized.shape

        for i in range(0, actual_num_to_compare, batch_size):
            end_idx = min(i + batch_size, actual_num_to_compare)
            real_batch = real_images_normalized[i:end_idx]
            gen_batch = gen_images_normalized[i:end_idx]
            lpips_values.extend(_get_lpips_model(opts.device)(gen_batch, real_batch).squeeze().cpu().numpy())
        
        lpips_mean = float(np.mean(lpips_values))

    return lpips_mean