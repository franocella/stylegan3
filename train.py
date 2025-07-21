# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# (header del copyright)

"""
Train a GAN using the techniques described in the paper
"Alias-Free Generative Adversarial Networks".

This script has been modified to support an optional auxiliary classification
task by dynamically selecting the appropriate network and loss function,
and to include a validation loop and Weights & Biases logging.
"""

import os
import click
import re
import json
import tempfile
import torch

import dnnlib
from training import training_loop
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

#----------------------------------------------------------------------------

def subprocess_fn(rank, c, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    if c.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=c.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=c.num_gpus)
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'
    training_loop.training_loop(rank=rank, **c)

#----------------------------------------------------------------------------

def launch_training(c, desc, outdir, dry_run):
    dnnlib.util.Logger(should_flush=True)
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
    assert not os.path.exists(c.run_dir)

    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    if c.validation_set_kwargs is not None:
        print(f'Validation path:     {c.validation_set_kwargs.path}')
    print()

    if dry_run:
        print('Dry run; exiting.')
        return

    print('Creating output directory...')
    os.makedirs(c.run_dir)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(c, f, indent=2)

    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if c.num_gpus == 1:
            subprocess_fn(rank=0, c=c, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(c, temp_dir), nprocs=c.num_gpus)

#----------------------------------------------------------------------------

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs)
        dataset_kwargs.resolution = dataset_obj.resolution
        dataset_kwargs.use_labels = dataset_obj.has_labels
        dataset_kwargs.max_size = len(dataset_obj)
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

#----------------------------------------------------------------------------

def parse_comma_separated_list(s):
    if isinstance(s, list): return s
    if s is None or s.lower() == 'none' or s == '': return []
    return s.split(',')

#----------------------------------------------------------------------------

@click.command()
@click.option('--outdir',       help='Where to save the results', metavar='DIR', required=True)
@click.option('--cfg',          help='Base configuration', type=click.Choice(['stylegan3-t', 'stylegan3-r', 'stylegan2']), required=True)
@click.option('--data',         help='Training data', metavar='[ZIP|DIR]', type=str, required=True)
@click.option('--gpus',         help='Number of GPUs to use', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--batch',        help='Total batch size', metavar='INT', type=click.IntRange(min=1), required=True)
@click.option('--gamma',        help='R1 regularization weight', metavar='FLOAT', type=click.FloatRange(min=0), required=True)
@click.option('--val-data',     help='Validation data', metavar='[ZIP|DIR]', type=str, default=None)
@click.option('--val-interval', help='How often to run validation (in ticks)', metavar='INT', type=click.IntRange(min=1), default=10, show_default=True)
@click.option('--class-weight', help='Weight for the classification loss.', metavar='FLOAT', type=click.FloatRange(min=0), default=1.0, show_default=True)
@click.option('--cond',         help='Train conditional model', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--mirror',       help='Enable dataset x-flips', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--aug',          help='Augmentation mode', type=click.Choice(['noaug', 'ada', 'fixed']), default='ada', show_default=True)
@click.option('--resume',       help='Resume from given network pickle', metavar='[PATH|URL]', type=str)
@click.option('--freezed',      help='Freeze first layers of D', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--p',            help='Probability for --aug=fixed', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.2, show_default=True)
@click.option('--target',       help='Target value for --aug=ada', metavar='FLOAT', type=click.FloatRange(min=0, max=1), default=0.6, show_default=True)
@click.option('--batch-gpu',    help='Limit batch size per GPU', metavar='INT', type=click.IntRange(min=1))
@click.option('--cbase',        help='Capacity multiplier', metavar='INT', type=click.IntRange(min=1), default=32768, show_default=True)
@click.option('--cmax',         help='Max. feature maps', metavar='INT', type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--glr',          help='G learning rate', metavar='FLOAT', type=click.FloatRange(min=0))
@click.option('--dlr',          help='D learning rate', metavar='FLOAT', type=click.FloatRange(min=0), default=0.002, show_default=True)
@click.option('--map-depth',    help='Mapping network depth', metavar='INT', type=click.IntRange(min=1))
@click.option('--mbstd-group',  help='Minibatch std group size', metavar='INT', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--desc',         help='String to include in result dir name', metavar='STR', type=str)
@click.option('--metrics',      help='Quality metrics', metavar='[NAME|A,B,C|none]', type=parse_comma_separated_list, default='fid50k_full', show_default=True)
@click.option('--kimg',         help='Total training duration', metavar='KIMG', type=click.IntRange(min=1), default=25000, show_default=True)
@click.option('--tick',         help='How often to print progress', metavar='KIMG', type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--snap',         help='How often to save snapshots', metavar='TICKS', type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--seed',         help='Random seed', metavar='INT', type=click.IntRange(min=0), default=0, show_default=True)
@click.option('--fp32',         help='Disable mixed-precision', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--nobench',      help='Disable cuDNN benchmarking', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--workers',      help='DataLoader worker processes', metavar='INT', type=click.IntRange(min=1), default=3, show_default=True)
@click.option('-n','--dry-run', help='Print training options and exit', is_flag=True)
@click.option('--wandb-log',      help='Enable logging to Weights & Biases', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--wandb-project',  help='W&B project name', metavar='STR', type=str, default='stylegan-multiclass')
@click.option('--wandb-entity',   help='W&B entity name', metavar='STR', type=str, default=None)
# The --num-classes option is removed from here as it is now determined automatically from the dataset.
def main(**kwargs):
    opts = dnnlib.EasyDict(kwargs)
    c = dnnlib.EasyDict()
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0.0,0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)

    print('Determining number of classes from dataset...')
    temp_dataset = dnnlib.util.construct_class_by_name(**c.training_set_kwargs)
    num_classes = temp_dataset.label_dim
    del temp_dataset
    
    use_multiclass = (num_classes > 0)

    if use_multiclass:
        print(f"Dataset has {num_classes} classes. Enabling multi-class classification mode.")
        # The check for cfg=stylegan2 is removed to allow using a StyleGAN3 generator.
        c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan_multiclass.Discriminator', num_classes=num_classes, block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_multiclass.ACGANLoss', num_classes=num_classes)
    else:
        print("No labels found. Using standard unconditional training.")
        c.D_kwargs = dnnlib.EasyDict(class_name='training.networks_stylegan2.Discriminator', block_kwargs=dnnlib.EasyDict(), mapping_kwargs=dnnlib.EasyDict(), epilogue_kwargs=dnnlib.EasyDict())
        c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.StyleGAN2Loss')

    if use_multiclass and not opts.cond:
        print("Warning: Dataset has labels. Forcing --cond=True for classification.")
        opts.cond = True

    if opts.cond and not c.training_set_kwargs.use_labels:
        raise click.ClickException('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    c.validation_set_kwargs = None
    if opts.val_data is not None:
        print(f"Validation data specified at: {opts.val_data}")
        c.validation_set_kwargs, _ = init_dataset_kwargs(data=opts.val_data)
        if use_multiclass and not c.validation_set_kwargs.use_labels:
            raise click.ClickException('--val-data must have labels for classification task.')
        c.validation_set_kwargs.use_labels = c.training_set_kwargs.use_labels
    
    c.val_interval = opts.val_interval
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = c.D_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = c.D_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = (8 if opts.cfg == 'stylegan2' else 2) if opts.map_depth is None else opts.map_depth
    c.D_kwargs.block_kwargs.freeze_layers = opts.freezed
    c.D_kwargs.epilogue_kwargs.mbstd_group_size = opts.mbstd_group
    c.loss_kwargs.r1_gamma = opts.gamma
    c.loss_kwargs.class_weight = opts.class_weight
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.metrics = opts.metrics
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    c.wandb_log = opts.wandb_log
    c.wandb_kwargs = dnnlib.EasyDict()
    if c.wandb_log:
        c.wandb_kwargs.project = opts.wandb_project
        c.wandb_kwargs.entity = opts.wandb_entity
        c.wandb_kwargs.opts = dnnlib.EasyDict(kwargs)

    if c.batch_size % c.num_gpus != 0: raise click.ClickException('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0: raise click.ClickException('--batch must be a multiple of --gpus times --batch-gpu')
    if c.batch_gpu < c.D_kwargs.epilogue_kwargs.mbstd_group_size: raise click.ClickException('--batch-gpu cannot be smaller than --mbstd-group')
    if any(not metric_main.is_valid_metric(metric.split('/')[-1]) for metric in c.metrics):
        raise click.ClickException('\n'.join(['--metrics can only contain the following values:'] + metric_main.list_valid_metrics()))
    
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'training.networks_stylegan2.Generator'
        c.loss_kwargs.style_mixing_prob = 0.9
        c.loss_kwargs.pl_weight = 2
        c.G_reg_interval = 4
        c.G_kwargs.fused_modconv_default = 'inference_only'
        c.loss_kwargs.pl_no_weight_grad = True
    else:
        c.G_kwargs.class_name = 'training.networks_stylegan3.Generator'
        c.G_kwargs.magnitude_ema_beta = 0.5 ** (c.batch_size / (20 * 1e3))
        if opts.cfg == 'stylegan3-r':
            c.G_kwargs.conv_kernel = 1
            c.G_kwargs.channel_base *= 2
            c.G_kwargs.channel_max *= 2
            c.G_kwargs.use_radial_filters = True
            c.loss_kwargs.blur_init_sigma = 10
            c.loss_kwargs.blur_fade_kimg = c.batch_size * 200 / 32

    if opts.aug != 'noaug':
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1)
        if opts.aug == 'ada': c.ada_target = opts.target
        if opts.aug == 'fixed': c.augment_p = opts.p

    if opts.resume is not None:
        c.resume_pkl = opts.resume
        c.ada_kimg = 100
        c.ema_rampup = None
        c.loss_kwargs.blur_init_sigma = 0

    if opts.fp32:
        c.G_kwargs.num_fp16_res = c.D_kwargs.num_fp16_res = 0
        c.G_kwargs.conv_clamp = c.D_kwargs.conv_clamp = None
    if opts.nobench:
        c.cudnn_benchmark = False

    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}-gamma{c.loss_kwargs.r1_gamma:g}'
    if use_multiclass:
        desc += f'-ac{num_classes}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    launch_training(c=c, desc=desc, outdir=opts.outdir, dry_run=opts.dry_run)

#----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
#----------------------------------------------------------------------------