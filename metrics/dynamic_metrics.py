# dynamic_metrics.py

# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
A module for computing dynamic GAN quality metrics with caching for real dataset statistics.

Each function dynamically adjusts the number of samples based on the dataset size
to ensure that the number of real and generated images are identical, providing
a robust and statistically sound evaluation. The feature statistics of the real
dataset are cached on disk to accelerate subsequent evaluations.
"""

import os
import numpy as np
import scipy.linalg
import torch
import lpips
import dnnlib
import copy
import hashlib
import pickle
from pathlib import Path
from . import metric_utils

#----------------------------------------------------------------------------
#   Helper function for caching real dataset statistics
#----------------------------------------------------------------------------

def get_or_compute_real_stats(opts, metric_name, detector_url, detector_kwargs, capture_mean_cov, capture_all, max_items):
    """
    Checks for a cached version of the real dataset's feature statistics.

    If a cache file corresponding to the dataset and metric configuration exists,
    it loads the stats from the file. Otherwise, it computes the stats using
    the provided metric_utils function, saves them to a cache file for future
    use, and then returns them. This prevents re-computing real stats on every run.
    """
    # 1. Create a unique key for the current dataset and metric configuration.
    dataset_path = opts.dataset_kwargs.path
    # Use a hash of the config to create a unique, filesystem-safe filename.
    key_items = [dataset_path, metric_name, detector_url, str(detector_kwargs), capture_mean_cov, capture_all, max_items]
    key_str = "".join(map(str, key_items))
    key_hash = hashlib.md5(key_str.encode()).hexdigest()

    # 2. Define the cache file path, including the metric name for uniqueness.
    cache_dir = Path("./.cache/metric_stats")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize metric name (e.g., 'val/fid') for use in filename.
    sane_metric_name = metric_name.replace('/', '_')
    cache_file = cache_dir / f"{Path(dataset_path).stem}_{sane_metric_name}_{max_items}_{key_hash[:8]}.pkl"

    # 3. Load from cache if it exists.
    if cache_file.exists() and opts.rank == 0:
        print(f"Loading cached real feature stats from '{cache_file}'...")
        try:
            with open(cache_file, 'rb') as f:
                stats = pickle.load(f)
            return stats
        except Exception as e:
            print(f"Warning: Could not load cache file, recomputing. Error: {e}")

    # 4. If cache does not exist, compute the stats.
    if opts.rank == 0:
        print(f"Computing real feature stats for '{dataset_path}' (will be cached)...")

    stats = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_mean_cov=capture_mean_cov, capture_all=capture_all, max_items=max_items
    )

    # 5. Save the newly computed stats to the cache file for future runs.
    if opts.rank == 0 and stats is not None:
        print(f"Saving real feature stats to '{cache_file}'...")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(stats, f)
        except Exception as e:
            print(f"Warning: Could not save cache file. Error: {e}")

    return stats

#----------------------------------------------------------------------------
#   Dynamic Frechet Inception Distance (FID) with Caching
#----------------------------------------------------------------------------

def compute_dynamic_fid(opts, metric_name, max_real, num_gen):
    """
    Computes Frechet Inception Distance (FID), using a cache for the real
    dataset's statistics to accelerate computation.
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

    # Use the caching helper to load or compute real dataset statistics.
    real_stats = get_or_compute_real_stats(
        opts=opts, metric_name=metric_name, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_mean_cov=True, capture_all=False, max_items=num_to_compare
    )
    mu_real, sigma_real = real_stats.get_mean_cov()

    # Generator stats are always computed on the fly as the generator changes.
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
#   Dynamic Kernel Inception Distance (KID) with Caching
#----------------------------------------------------------------------------

def compute_dynamic_kid(opts, metric_name, max_real, num_gen, num_subsets, max_subset_size):
    """
    Computes Kernel Inception Distance (KID), using a cache for the real
    dataset's features to accelerate computation.
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

    # Use the caching helper to load or compute real dataset features.
    real_stats = get_or_compute_real_stats(
        opts=opts, metric_name=metric_name, detector_url=detector_url, detector_kwargs=detector_kwargs,
        capture_mean_cov=False, capture_all=True, max_items=num_to_compare
    )
    real_features = real_stats.get_all()

    # Generator features are always computed on the fly.
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
#   Dynamic Learned Perceptual Image Patch Similarity (LPIPS)
#----------------------------------------------------------------------------

_lpips_model = None

def _get_lpips_model(device: torch.device) -> lpips.LPIPS:
    """Initializes and returns the LPIPS model as a singleton."""
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex', spatial=False).to(device)
        _lpips_model.eval()
    return _lpips_model

def compute_dynamic_lpips(opts, num_gen, num_real, batch_size=64):
    """
    Computes LPIPS between pairs of real and generated images.
    This metric does not benefit from feature caching as it requires direct
    image-to-image comparison.
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
    item_subset = sorted(list(range(num_items_in_dataset)))[:actual_num_to_compare]
    item_subset_for_rank = [item for i, item in enumerate(item_subset) if i % opts.num_gpus == opts.rank]

    progress_real = opts.progress.sub(tag='real images for LPIPS', num_items=len(item_subset_for_rank))
    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    for images, _labels in torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset_for_rank, batch_size=batch_size, **data_loader_kwargs):
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        real_images_list.append(images.to(opts.device))

    all_real_images_local = torch.cat(real_images_list) if real_images_list else torch.empty(0, 3, opts.dataset_kwargs.resolution, opts.dataset_kwargs.resolution, device=opts.device)
    progress_real.update(len(item_subset_for_rank))

    # --- Collect generated images ---
    G_ema = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    gen_images_list = []
    c_iter = metric_utils.iterate_random_labels(opts=opts, batch_size=batch_size)

    num_local_gen_images = (actual_num_to_compare + opts.num_gpus - 1) // opts.num_gpus

    progress_gen = opts.progress.sub(tag='gen images for LPIPS', num_items=num_local_gen_images)

    for _i in range(0, num_local_gen_images, batch_size):
        z = torch.randn([batch_size, G_ema.z_dim], device=opts.device)
        c = next(c_iter)
        img = G_ema(z=z, c=c, **opts.G_kwargs)
        if img.shape[1] == 1:
            img = img.repeat([1, 3, 1, 1])
        gen_images_list.append(img.to(torch.float32))

    all_gen_images_local = torch.cat(gen_images_list)[:num_local_gen_images] if gen_images_list else torch.empty(0, 3, opts.dataset_kwargs.resolution, opts.dataset_kwargs.resolution, device=opts.device)
    progress_gen.update(num_local_gen_images)

    # --- Combine from all GPUs and compute ---
    if opts.num_gpus > 1:
        all_real_images_list = [torch.empty_like(all_real_images_local) for _ in range(opts.num_gpus)]
        all_gen_images_list = [torch.empty_like(all_gen_images_local) for _ in range(opts.num_gpus)]
        torch.distributed.all_gather(all_real_images_list, all_real_images_local)
        torch.distributed.all_gather(all_gen_images_list, all_gen_images_local)
        all_real_images = torch.cat(all_real_images_list)
        all_gen_images = torch.cat(all_gen_images_list)
    else:
        all_real_images = all_real_images_local
        all_gen_images = all_gen_images_local

    lpips_mean = float('nan')
    if opts.rank == 0:
        # Normalize images to [-1, 1] range for LPIPS model
        real_images_normalized = (all_real_images.to(torch.float32) / 127.5 - 1.0) if all_real_images.dtype == torch.uint8 else all_real_images
        gen_images_normalized = all_gen_images

        # Ensure tensor shapes match
        if real_images_normalized.shape[0] > actual_num_to_compare:
            real_images_normalized = real_images_normalized[:actual_num_to_compare]
        if gen_images_normalized.shape[0] > actual_num_to_compare:
            gen_images_normalized = gen_images_normalized[:actual_num_to_compare]
        assert real_images_normalized.shape == gen_images_normalized.shape

        # Compute LPIPS in batches
        for i in range(0, actual_num_to_compare, batch_size):
            end_idx = min(i + batch_size, actual_num_to_compare)
            real_batch = real_images_normalized[i:end_idx]
            gen_batch = gen_images_normalized[i:end_idx]
            lpips_values.extend(lpips_model(gen_batch, real_batch).detach().squeeze().cpu().numpy())

        lpips_mean = float(np.mean(lpips_values))

    return lpips_mean