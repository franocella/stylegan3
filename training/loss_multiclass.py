# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work defines a loss function for an Auxiliary Classifier GAN (AC-GAN).
# It combines the standard adversarial loss with a multi-class
# cross-entropy loss for both the generator and the discriminator.

"""Loss functions for AC-GAN."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.loss import Loss

class ACGANLoss(Loss):
    """
    Auxiliary Classifier GAN (AC-GAN) loss function with a configurable weight
    for the classification component.

    This loss function comprises several components:
    - For the Generator (G):
        1. Adversarial Loss: Encourages G to generate images that the
           Discriminator (D) considers real.
        2. Classification Loss: Encourages G to generate images that D
           classifies into the correct class.
    - For the Discriminator (D):
        1. Adversarial Loss: Encourages D to distinguish between real and fake
           images.
        2. Classification Loss: Encourages D to correctly classify real images.
    - Regularization terms (R1, Path Length) are also included.
    """
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, num_classes=0, class_weight=1.0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.num_classes        = num_classes
        self.class_weight       = class_weight # Weight for the classification loss.
        self.classification_loss = torch.nn.CrossEntropyLoss()

    def run_G(self, z, c, update_emas=False):
        """Wrapper for running the generator."""
        ws = self.G.mapping(z, c, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        img = self.G.synthesis(ws, update_emas=update_emas)
        return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        """Wrapper for running the discriminator."""
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        # The modified discriminator returns both GAN and class logits.
        gan_logits, class_logits = self.D(img, c, update_emas=update_emas)
        return gan_logits, class_logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        """
        Calculates and accumulates gradients for a single training phase.
        'phase' can be one of 'Gmain', 'Dmain', 'Greg', 'Dreg', etc.
        """
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        # Disable regularization phases if their weights are zero.
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)

        # Calculate blur sigma for progressive blurring of images.
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Convert one-hot encoded labels to class indices for cross-entropy loss.
        real_c_indices = real_c.argmax(dim=1)
        gen_c_indices = gen_c.argmax(dim=1)

        # === GENERATOR TRAINING PHASES ===

        # Gmain: Main training phase for the generator.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # Generate images and run them through the discriminator.
                gen_img, _gen_ws = self.run_G(gen_z, gen_c)
                gen_gan_logits, gen_class_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                
                # Calculate adversarial loss: how well G fools D's realism judge.
                loss_Gmain_gan = torch.nn.functional.softplus(-gen_gan_logits)
                # Calculate classification loss: how well G fools D's classifier.
                loss_Gmain_class = self.classification_loss(gen_class_logits, gen_c_indices)
                
                # Report metrics with new, descriptive names.
                training_stats.report('Scores/fake', gen_gan_logits)
                training_stats.report('Signs/fake', gen_gan_logits.sign())
                training_stats.report('Loss/Generator/classification', loss_Gmain_class)
                training_stats.report('Loss/Generator/adversarial', loss_Gmain_gan)
                
                # Combine losses with the specified weight for the classification part.
                loss_Gmain = loss_Gmain_gan + loss_Gmain_class * self.class_weight
                training_stats.report('Loss/Generator/total', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Greg: Path length regularization for the generator.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/Penalties/pl', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/Generator/regularization', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # === DISCRIMINATOR TRAINING PHASES ===

        # Dmain: Main training phase for the discriminator.
        # Step 1: Loss on generated (fake) images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_gan_logits, _ = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Scores/fake', gen_gan_logits)
                training_stats.report('Signs/fake', gen_gan_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_gan_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain & Dreg: Loss on real images and R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_gan_logits, real_class_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                
                training_stats.report('Scores/real', real_gan_logits)
                training_stats.report('Signs/real', real_gan_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    # Adversarial loss for real images.
                    loss_Dreal_gan = torch.nn.functional.softplus(-real_gan_logits)
                    # Classification loss for real images.
                    loss_Dreal_class = self.classification_loss(real_class_logits, real_c_indices)
                    
                    # Report individual loss components.
                    training_stats.report('Loss/Discriminator/classification_real', loss_Dreal_class)
                    training_stats.report('Loss/Discriminator/adversarial_real', loss_Dreal_gan)
                    training_stats.report('Loss/Discriminator/adversarial_fake', loss_Dgen)
                    
                    # Calculate and report classification metrics on the training batch.
                    with torch.no_grad():
                        predicted_indices = real_class_logits.argmax(dim=1)
                        accuracy = (predicted_indices == real_c_indices).float().mean()
                        training_stats.report('Metrics/Accuracy/real', accuracy)
                        precisions, recalls = [], []
                        for i in range(self.num_classes):
                            tp = ((predicted_indices == i) & (real_c_indices == i)).sum()
                            fp = ((predicted_indices == i) & (real_c_indices != i)).sum()
                            fn = ((predicted_indices != i) & (real_c_indices == i)).sum()
                            p = tp / (tp + fp) if (tp + fp) > 0 else torch.tensor(0.0, device=self.device)
                            r = tp / (tp + fn) if (tp + fn) > 0 else torch.tensor(0.0, device=self.device)
                            precisions.append(p)
                            recalls.append(r)
                        macro_precision = torch.stack(precisions).mean()
                        macro_recall = torch.stack(recalls).mean()
                        training_stats.report('Metrics/Precision/real', macro_precision)
                        training_stats.report('Metrics/Recall/real', macro_recall)
                    
                    # Combine losses with the specified weight.
                    loss_Dreal = loss_Dreal_gan + loss_Dreal_class * self.class_weight
                    training_stats.report('Loss/Discriminator/total', loss_Dgen + loss_Dreal)

                # Dreg: R1 regularization.
                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_gan_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/Penalties/r1', r1_penalty)
                    training_stats.report('Loss/Discriminator/regularization', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()