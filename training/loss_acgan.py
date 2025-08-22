# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This work is derived from an original work by NVIDIA CORPORATION & AFFILIATES.
# The modifications to implement the AC-GAN functionality are subject to the same
# license terms as the original work.

"""
Implements the loss function for an Auxiliary Classifier GAN (AC-GAN),
which combines a non-saturating GAN loss with a cross-entropy classification
loss. The total loss is a weighted sum of these two components, providing
control over the trade-off between image quality and classification accuracy.
"""


# --- Third-party Imports ---
import torch
from torch.nn import functional as F
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix

# --- Local Application Imports ---
# Import the base loss class to inherit from.
from training.loss import StyleGAN2Loss

class ACGANLoss(StyleGAN2Loss):
    """
    An AC-GAN loss function that extends the StyleGAN2 loss with an auxiliary
    classification objective.
    """
    def __init__(self, device, G, D, class_weight=1.0, num_classes=0, grad_clip=1.0, **kwargs):
        super().__init__(device=device, G=G, D=D, **kwargs)
        self.class_weight = class_weight if num_classes > 0 else 0
        self.num_classes = num_classes
        self.grad_clip = grad_clip # Max norm for gradient clipping.

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg):
        if self.pl_weight == 0: phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0: phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # === Generator Main Loss (Gmain) ===
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _ = self.run_G(gen_z, gen_c)
                gen_adv_logits, gen_ac_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma)
                
                training_stats.report('Loss/scores/fake', gen_adv_logits)
                training_stats.report('Loss/signs/fake', gen_adv_logits.sign())

                # Original adversarial loss, reported 
                loss_Gmain = F.softplus(-gen_adv_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                
                # Additional classification loss.
                loss_G_class = 0
                if self.class_weight > 0:
                    class_labels = gen_c.argmax(dim=1)
                    loss_G_class = F.cross_entropy(gen_ac_logits, class_labels)
                    training_stats.report('Loss/G/class', loss_G_class)

                # Report the new total loss for monitoring.
                total_G_loss = loss_Gmain + loss_G_class * self.class_weight
                training_stats.report('Loss/G/total', total_G_loss)
            
            with torch.autograd.profiler.record_function('Gmain_backward'):
                total_G_loss.mean().mul(gain).backward()

        # === Generator Path Length Regularization (Greg) ===
        if phase in ['Greg', 'Gboth']:
            # This call automatically reports 'Loss/pl_penalty' and 'Loss/G/reg'.
            super().accumulate_gradients(phase='Greg', real_img=None, real_c=None, gen_z=gen_z, gen_c=gen_c, gain=gain, cur_nimg=cur_nimg)

        # === Discriminator Loss (Dmain & Dreg) ===
        
        # Dgen: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _ = self.run_G(gen_z, gen_c, update_emas=True)
                gen_adv_logits, _ = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_adv_logits)
                training_stats.report('Loss/signs/fake', gen_adv_logits.sign())
                loss_Dgen = F.softplus(gen_adv_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dreal + Dr1: Maximize logits for real images and apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal_Dr1' if phase == 'Dboth' else 'Dreal' if phase == 'Dmain' else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(self.r1_gamma > 0)
                real_adv_logits, real_ac_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                
                training_stats.report('Loss/scores/real', real_adv_logits)
                training_stats.report('Loss/signs/real', real_adv_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = F.softplus(-real_adv_logits)
                    # Report total adversarial D loss 
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)
                
                # Additional classification loss for real images.
                loss_D_class = 0
                if self.class_weight > 0:
                    class_labels = real_c.argmax(dim=1)
                    loss_D_class = F.cross_entropy(real_ac_logits, class_labels)
                    training_stats.report('Loss/D/class', loss_D_class)

                # R1 regularization penalizes the discriminator's gradient with respect to real inputs, which helps stabilize training. This is applied to both adversarial and classification outputs.
                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with conv2d_gradfix.no_weight_gradients():
                        # Calculate the R1 gradient for the adversarial logits
                        r1_grads_adv = torch.autograd.grad(
                            outputs=[real_adv_logits.sum()], 
                            inputs=[real_img_tmp], 
                            create_graph=True, 
                            only_inputs=True
                        )[0]
                        r1_penalty_adv = r1_grads_adv.square().sum([1,2,3])
                        # Calculate the R1 gradients for the classification logits
                        r1_penalty_ac = 0
                        if self.class_weight > 0: # if not we aren't using classification
                            r1_grads_ac = torch.autograd.grad(
                                outputs=[real_ac_logits.sum()], 
                                inputs=[real_img_tmp], 
                                create_graph=True, 
                                only_inputs=True
                            )[0]
                            r1_penalty_ac = r1_grads_ac.square().sum([1,2,3])

                        # Combine R1 penalties, weighting the classification one if specified
                        r1_penalty = r1_penalty_adv + (r1_penalty_ac * self.class_weight)
                        
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    
                    # Report R1 penalty and D regularization loss
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)
            
            with torch.autograd.profiler.record_function(name + '_backward'):
                # The total discriminator loss includes the adversarial, classification, and R1 terms.

                total_D_loss = loss_Dreal + (loss_D_class * self.class_weight) + loss_Dr1
                total_D_loss.mean().mul(gain).backward()

                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.D.parameters(), self.grad_clip)