# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
import dnnlib
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main

try:
    import wandb
except ImportError:
    wandb = None

#----------------------------------------------------------------------------

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]
    else:
        label_groups = dict()
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)
    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

#----------------------------------------------------------------------------

def run_validation(D, validation_loader, device, num_classes):
    classification_loss = torch.nn.CrossEntropyLoss()
    all_losses, all_accs, all_precs, all_recs = [], [], [], []
    D.eval()
    with torch.no_grad():
        for val_img, val_c in validation_loader:
            val_img = (val_img.to(device).to(torch.float32) / 127.5 - 1)
            val_c = val_c.to(device)
            val_c_indices = val_c.argmax(dim=1)
            _gan_logits, class_logits = D(val_img, val_c)
            if class_logits is None: continue
            loss = classification_loss(class_logits, val_c_indices)
            all_losses.append(loss)
            predicted_indices = class_logits.argmax(dim=1)
            accuracy = (predicted_indices == val_c_indices).float().mean()
            all_accs.append(accuracy)
            precisions, recalls = [], []
            for i in range(num_classes):
                tp = ((predicted_indices == i) & (val_c_indices == i)).sum()
                fp = ((predicted_indices == i) & (val_c_indices != i)).sum()
                fn = ((predicted_indices != i) & (val_c_indices == i)).sum()
                p = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=device)
                r = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=device)
                precisions.append(p)
                recalls.append(r)
            all_precs.append(torch.stack(precisions).mean())
            all_recs.append(torch.stack(recalls).mean())
    if not all_losses: return
    avg_loss = torch.stack(all_losses).mean()
    avg_acc = torch.stack(all_accs).mean()
    avg_prec = torch.stack(all_precs).mean()
    avg_rec = torch.stack(all_recs).mean()
    training_stats.report('Validation/Loss', avg_loss)
    training_stats.report('Validation/Accuracy', avg_acc)
    training_stats.report('Validation/Precision', avg_prec)
    training_stats.report('Validation/Recall', avg_rec)
    D.train()

#----------------------------------------------------------------------------

def training_loop(
    run_dir                 = '.',
    training_set_kwargs     = {},
    validation_set_kwargs   = None,
    val_interval            = 10,
    data_loader_kwargs      = {},
    G_kwargs                = {},
    D_kwargs                = {},
    G_opt_kwargs            = {},
    D_opt_kwargs            = {},
    augment_kwargs          = None,
    loss_kwargs             = {},
    metrics                 = [],
    random_seed             = 0,
    num_gpus                = 1,
    rank                    = 0,
    batch_size              = 4,
    batch_gpu               = 4,
    ema_kimg                = 10,
    ema_rampup              = 0.05,
    G_reg_interval          = None,
    D_reg_interval          = 16,
    augment_p               = 0,
    ada_target              = None,
    ada_interval            = 4,
    ada_kimg                = 500,
    total_kimg              = 25000,
    kimg_per_tick           = 4,
    image_snapshot_ticks    = 50,
    network_snapshot_ticks  = 50,
    resume_pkl              = None,
    resume_kimg             = 0,
    cudnn_benchmark         = True,
    abort_fn                = None,
    progress_fn             = None,
    wandb_log               = False,
    wandb_kwargs            = {},
):
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

    if rank == 0:
        print('Loading training set...')
    training_set = dnnlib.util.construct_class_by_name(**training_set_kwargs)
    training_set_sampler = misc.InfiniteSampler(dataset=training_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    training_set_iterator = iter(torch.utils.data.DataLoader(dataset=training_set, sampler=training_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))
    if rank == 0:
        print()
        print('Num images: ', len(training_set))
        print('Image shape:', training_set.image_shape)
        print('Label shape:', training_set.label_shape)
        print()
        
    validation_loader = None
    if validation_set_kwargs is not None and rank == 0:
        print('Loading validation set...')
        validation_set = dnnlib.util.construct_class_by_name(**validation_set_kwargs)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_set, batch_size=batch_size//num_gpus, drop_last=True, **data_loader_kwargs) # if the validation set is small, drop_last=True will ensure that the validation loop does not fail.

    if rank == 0:
        print('Constructing networks...')
    common_kwargs = dict(c_dim=training_set.label_dim, img_resolution=training_set.resolution, img_channels=training_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device)
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    for module in [G, D, G_ema, augment_pipe]:
        if module is not None and num_gpus > 1:
            for param in misc.params_and_buffers(module):
                torch.distributed.broadcast(param, src=0)

    if rank == 0:
        print('Setting up training phases...')
    loss = dnnlib.util.construct_class_by_name(device=device, G=G, D=D, augment_pipe=augment_pipe, **loss_kwargs)
    phases = []
    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'both', module=module, opt=opt, interval=1)]
        else:
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1)]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval)]
    for phase in phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    if rank == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=training_set)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        grid_z = torch.randn([labels.shape[0], G.z_dim], device=device).split(batch_gpu)
        grid_c = torch.from_numpy(labels).to(device).split(batch_gpu)
        images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)

    if rank == 0:
        print('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)
        if wandb is not None and wandb_log:
            print('Initializing W&B...')
            run_name = os.path.basename(run_dir)
            wandb.init(
                project=wandb_kwargs.get('project'),
                entity=wandb_kwargs.get('entity'),
                name=run_name,
                config=wandb_kwargs.get('opts', {})
            )

    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0
    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        with torch.autograd.profiler.record_function('data_fetch'):
            phase_real_img, phase_real_c = next(training_set_iterator)
            phase_real_img = (phase_real_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_c = phase_real_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [training_set.get_label(np.random.randint(len(training_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        for phase, phase_gen_z, phase_gen_c in zip(phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0: continue
            if phase.start_event is not None: phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)
            for real_img, real_c, gen_z, gen_c in zip(phase_real_img, phase_real_c, phase_gen_z, phase_gen_c):
                loss.accumulate_gradients(phase=phase.name, real_img=real_img, real_c=real_c, gen_z=gen_z, gen_c=gen_c, gain=phase.interval, cur_nimg=cur_nimg)
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                params = [param for param in phase.module.parameters() if param.grad is not None]
                if len(params) > 0:
                    flat = torch.cat([param.grad.flatten() for param in params])
                    if num_gpus > 1:
                        torch.distributed.all_reduce(flat)
                        flat /= num_gpus
                    misc.nan_to_num(flat, nan=0, posinf=1e5, neginf=-1e5, out=flat)
                    grads = flat.split([param.numel() for param in params])
                    for param, grad in zip(params, grads):
                        param.grad = grad.reshape(param.shape)
                phase.opt.step()
            if phase.end_event is not None: phase.end_event.record(torch.cuda.current_stream(device))

        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        cur_nimg += batch_size
        batch_idx += 1

        if (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000): continue

        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0: print(' '.join(fields))

        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0: print('\nAborting...')

        if (rank == 0) and (image_snapshot_ticks is not None) and (done or cur_tick % image_snapshot_ticks == 0):
            images = torch.cat([G_ema(z=z, c=c, noise_mode='const').cpu() for z, c in zip(grid_z, grid_c)]).numpy()
            save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)

        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or cur_tick % network_snapshot_ticks == 0):
            snapshot_data = dict(G=G, D=D, G_ema=G_ema, augment_pipe=augment_pipe, training_set_kwargs=dict(training_set_kwargs))
            for key, value in snapshot_data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    if num_gpus > 1:
                        misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                        for param in misc.params_and_buffers(value):
                            torch.distributed.broadcast(param, src=0)
                    snapshot_data[key] = value.cpu()
                del value
            snapshot_pkl = os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl')
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate validation loop metrics.
        if (rank == 0) and (validation_loader is not None) and (done or cur_tick % val_interval == 0):
            print('Running validation...')
            num_classes_d = getattr(D, 'num_classes', 0)
            if num_classes_d > 0:
                run_validation(D=D, validation_loader=validation_loader, device=device, num_classes=num_classes_d)
            else:
                print('Warning: Validation data provided, but model is not configured for classification.')
        
        # Evaluate snapshot metrics (FID, KID, etc.).
        if (snapshot_data is not None) and (len(metrics) > 0):
            if rank == 0:
                print('Evaluating metrics...')
            for metric in metrics:
                # Determine which dataset to use (train or val) based on the prefix.
                if metric.startswith('val/'):
                    if validation_set_kwargs is None:
                        if rank == 0: print(f'Warning: Skipping validation metric "{metric}" since --val-data is not set.')
                        continue
                    metric_dataset_kwargs = validation_set_kwargs
                    metric_name = metric[4:] # Use the base metric name for calculation
                else:
                    metric_dataset_kwargs = training_set_kwargs
                    metric_name = metric

                # Calculate the metric.
                result_dict = metric_main.calc_metric(metric=metric_name, G=snapshot_data['G_ema'],
                    dataset_kwargs=metric_dataset_kwargs, num_gpus=num_gpus, rank=rank, device=device)

                # If it was a validation metric, adjust the keys in the result dictionary
                # before passing it to any logging or reporting function.
                if metric.startswith('val/'):
                    result_dict['metric'] = metric
                    result_dict['results'] = {metric: value for _key, value in result_dict['results'].items()}

                # Report to the JSONL file (now works with prefixed names).
                if rank == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=snapshot_pkl)

                # Update the stats for TensorBoard and W&B.
                stats_metrics.update(result_dict['results'])
        del snapshot_data # conserve memory

        for phase in phases:
            value = []
            if (phase.start_event is not None) and (phase.end_event is not None):
                phase.end_event.synchronize()
                value = phase.start_event.elapsed_time(phase.end_event)
            training_stats.report0('Timing/' + phase.name, value)
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()

        # Create a unified dictionary for logging to TensorBoard and W&B.
        log_dict = dict()
        for name, value in stats_dict.items():
            log_dict[name] = value.mean
        for name, value in stats_metrics.items():
            # This will correctly handle names like 'val/fid'
            log_dict[f'Metrics/{name}'] = value

        # Log to TensorBoard.
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in log_dict.items():
                stats_tfevents.add_scalar(name, value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        
        # Log to Weights & Biases.
        if wandb is not None and wandb_log and rank == 0:
            wandb.log(log_dict, step=int(cur_nimg / 1e3))

        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done: break

    if rank == 0:
        print()
        print('Exiting...')
        if wandb is not None and wandb_log:
            wandb.finish()

#----------------------------------------------------------------------------