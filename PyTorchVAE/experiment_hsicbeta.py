import os
import math
import torch
from torch import optim
from PyTorchVAE.models import BaseVAE
from PyTorchVAE.models.types_ import *
from PyTorchVAE.utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import pdb
from loss_capacity.functions import HSIC
from loss_capacity.utils import get_representations_data_split, hsic_batch

class VAEXperiment_hsicbeta(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment_hsicbeta, self).__init__()
        self.save_hyperparameters()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False

        self.hsic_every_epoch = self.params.hsic_every_epoch if hasattr(self.params, 'hsic_every_epoch') else False
        self.s_x = self.params.s_x if hasattr(self.params, 's_x') else 1
        self.s_y = self.params.s_y if hasattr(self.params, 's_y') else 1
        self.num_samples_hisc = self.params.num_samples_hisc if hasattr(self.params, 'num_samples_hisc') else 512
        self.num_sample_reparam = self.params.num_sample_reparam if hasattr(self.params, 'num_sample_reparam') else 1
        
        try:
            self.hold_graph = self.params.retain_first_backpass
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        
        train_loss = self.model.loss_function(*results,
                M_N = self.params.kld_weight, #al_img.shape[0]/ self.num_train_imgs,
                optimizer_idx=optimizer_idx,
                batch_idx = batch_idx,
                loss_type='l2',
                s_x = self.s_x,
                s_y = self.s_y,
                num_sample_reparam = self.num_sample_reparam,
                hsic_reg_version = self.params.hsic_reg_version
                )

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device
        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                optimizer_idx = optimizer_idx,
                batch_idx = batch_idx,
                loss_type='l2',
                s_x = self.s_x,
                s_y = self.s_y,
                num_sample_reparam = self.num_sample_reparam,
                hsic_reg_version = self.params.hsic_reg_version
                )
        # if self.hsic_every_epoch:
        #     hsic_score = hsic_batch(real_img, self, s_x=self.s_x * self.model.latent_dim, s_y=self.s_y * self.model.latent_dim, device='cuda', batch_size=512)
        #     self.log("val_hsic", hsic_score.item())
        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)
        

        
    def on_validation_end(self) -> None:
        self.sample_images(stage='val', loader=self.trainer.datamodule.val_dataloader())
    
    

    def test_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                optimizer_idx = optimizer_idx,
                batch_idx = batch_idx,
                loss_type='l2',
                s_x = self.s_x,
                s_y = self.s_y,
                num_sample_reparam = self.num_sample_reparam,
                hsic_reg_version = self.params.hsic_reg_version
                )

        self.log_dict({f"test_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        
    def on_test_end(self) -> None:
        self.sample_images(stage='test', loader=self.trainer.datamodule.test_dataloader())
        
    def sample_images(self, stage='val', loader=None):
        # Get sample reconstruction image            
        # test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input, test_label = next(iter(loader))

        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       f"Reconstructions_{stage}", 
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        vutils.save_image(test_input,
                                os.path.join(self.logger.log_dir , 
                                            f"InputImages_{stage}", 
                                            f"input_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                                normalize=True,
                                nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir , 
                                           f"Samples_{stage}",      
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params.LR,
                               weight_decay=self.params.weight_decay)
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params.LR_2 is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params.submodel).parameters(),
                                        lr=self.params.LR_2)
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params.scheduler_gamma is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params.scheduler_gamma)
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params.scheduler_gamma_2 is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params.scheduler_gamma_2)
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
    
    def hsic(self, dataset, s_x=1, s_y=1, device='cuda', batch_size=512):
        ### create a dataloader using dataset. Not fit for a batch data, for example in validating_step.
        all_inputs, all_feats, all_outputs = get_representations_data_split(self.model, dataset, device,  num_samples=self.num_samples_hisc,
                     shuffle=True, batch_size = batch_size)
        all_inputs = all_inputs.to(device)
        all_feats = all_feats.to(device)
        all_outputs = all_outputs.to(device)
        hsic_score = HSIC(all_feats, all_inputs - all_outputs, s_x, s_y)

        return hsic_score