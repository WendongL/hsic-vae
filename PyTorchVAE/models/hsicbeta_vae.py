import torch
from PyTorchVAE.models import BaseVAE
from torch import bernoulli, nn
from torch.nn import functional as F
from .types_ import *
import pdb
from loss_capacity.functions import HSIC, HSIC_
from loss_capacity.utils import hsic_batch, hsic_batch_v2

class SmallHsicBetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha: float = 0.0,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 recons_type: str = 'l2',
                 **kwargs) -> None:
        super(SmallHsicBetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha # constant for hsic regularization
        self.beta = beta # constant for beta-vae regularization
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.recons_type = recons_type

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,  out_channels=32, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=256), nn.ReLU(inplace=True) # should we use relu here?
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_var = nn.Linear(256, latent_dim)


        # Build Decoder
        # modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        # hidden_dims.reverse()

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )



        # self.decoder = nn.Sequential(*modules)


        self.decoder = nn.Sequential(
            nn.Linear(in_features=latent_dim, out_features=256),  nn.ReLU(inplace=True),
            nn.Linear(in_features=256,         out_features=1024), nn.ReLU(inplace=True),
            nn.Unflatten(dim=1, unflattened_size=[64, 4, 4]),
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=in_channels,  kernel_size=4, stride=2, padding=1), 
            nn.Sigmoid()
        )


        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
        #                               kernel_size= 3, padding= 1),
        #                     # nn.Tanh()
        #                     nn.Sigmoid()
        #                     )

    def encode(self, input: Tensor):
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return z
        # z, mu, log_var = self.encode(input)
    def encode_(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        # result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = self.decoder(z)
        # result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough to compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        s_x = kwargs['s_x']
        s_y = kwargs['s_y']
        num_sample_reparam = kwargs['num_sample_reparam']
        hsic_reg_version = kwargs['hsic_reg_version']

        reduction = 'sum'
        l2_loss = F.mse_loss(recons, input, reduction=reduction)
        bce_loss =  F.binary_cross_entropy(recons, input,reduction=reduction)

        if hsic_reg_version == 'v1':
            hsic_loss = self.hsic_reg(input,s_x = s_x, s_y = s_y, num_sample_reparam = num_sample_reparam)
        elif hsic_reg_version == 'v2':
            hsic_loss = self.hsic_reg_v2(input,s_x = s_x, s_y = s_y, num_sample_reparam = num_sample_reparam)
        
        if self.recons_type == 'l2':
            recons_loss = l2_loss
        elif self.recons_type == 'bce':
            recons_loss = bce_loss
        else:
            print('select recons_type from: l2 or bce')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss + self.alpha * hsic_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs() + self.alpha * hsic_loss
            # loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs() 
        else:
            raise ValueError('Undefined loss type.')
        # torch.cuda.empty_cache()
        # print('done')
        return {'loss': loss, 
            'Reconstruction_Loss':recons_loss, 
            'KLD':kld_loss,
            'l2_loss' : l2_loss,
            'bce_loss' : bce_loss,
            'hsic_loss' : hsic_loss
            }

    def hsic_reg(self, images, s_x=1, s_y=1, device='cuda', num_sample_reparam = 1):
                # num_sample_reparam: num of times of reparametrization.
        flat = torch.nn.Flatten()
        hsic_score = 0
        for i in range(num_sample_reparam):
            inputs = images.to(device)
            feats = self.encode(inputs).to(device)
            outputs = self.decode(feats).to(device)
            inputs = flat(inputs)
            feats = flat(feats)
            outputs = flat(outputs)
            # inputs = inputs.detach()
            # feats = feats.detach()
            # outputs = outputs.detach()
            hsic_score += HSIC(feats, inputs - outputs, s_x* self.latent_dim, s_y* self.latent_dim, no_grad = False)
        hsic_score /= num_sample_reparam
        return hsic_score

    def hsic_reg_v2(self, images, s_x=1, s_y=1, device='cuda', num_sample_reparam = 1):
        # correspond to utils.hsic_batch_v2
        # num_sample_reparam: num of samples of reparamatrizing z using mu and sigma
        flat = torch.nn.Flatten(start_dim=-3, end_dim=-1)
        batch_size = images.shape[0]
        # print('images', images.shape)

        inputs = images.to(device)
        feats = torch.zeros(batch_size, num_sample_reparam, self.latent_dim)
        outputs = torch.zeros(batch_size, num_sample_reparam, inputs.shape[1], inputs.shape[2], inputs.shape[3])
        # print('outputs',outputs.shape)
        # pdb.set_trace()
        feats = feats.to(device)
        outputs = outputs.to(device)
        for i in range(num_sample_reparam):
            feats[:, i, :] = self.encode(inputs)
            # print('outputs[:, i, :, :]',outputs[:, i, :, :].shape)
            # print('experiment.model.decode(torch.squeeze(feats[:, i, :]))',experiment.model.decode(torch.squeeze(feats[:, i, :])).shape)
            outputs[:, i, :, :, :] = self.decode(feats[:, i, :])##########
        
        # pdb.set_trace()
        inputs = inputs.unsqueeze(1).repeat(1, num_sample_reparam, 1, 1, 1) # repeat batch_size times H
        inputs = flat(inputs)
        outputs = flat(outputs)
        inputs = inputs.detach()
        feats = feats.detach()
        outputs = outputs.detach()
        
        
        hsic_score = HSIC_(feats, inputs - outputs, s_x, s_y, no_grad = False)

        return hsic_score
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def sample_interpolate(self,
               num_samples:int,
               current_device: int,
               
                **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]





class HsicBetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 alpha: float = 0.0,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 recons_type: str = 'l2',
                 **kwargs) -> None:
        super(HsicBetaVAE, self).__init__()

        self.latent_dim = latent_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.recons_type = recons_type
        self.input_channels = in_channels
        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )



        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= self.input_channels,
                                      kernel_size= 3, padding= 1),
                            # nn.Tanh()
                            nn.Sigmoid()
                            )

    def encode(self, input: Tensor):
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return z
        # z, mu, log_var = self.encode(input)
    def encode_(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode_(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        s_x = kwargs['s_x']
        s_y = kwargs['s_y']
        num_sample_reparam = kwargs['num_sample_reparam']

        reduction = 'sum'
        l2_loss = F.mse_loss(recons, input, reduction=reduction)
        bce_loss =  F.binary_cross_entropy(recons, input,reduction=reduction)
        if hsic_reg_version == 'v1':
            hsic_loss = self.hsic_reg(input,s_x = s_x, s_y = s_y, num_sample_reparam = num_sample_reparam)
        elif hsic_reg_version == 'v3':
            hsic_loss = self.hsic_reg_v3(input,s_x = s_x, s_y = s_y, num_sample_reparam = num_sample_reparam)
        
        if self.recons_type == 'l2':
            recons_loss = l2_loss
        elif self.recons_type == 'bce':
            recons_loss = bce_loss
        else:
            print('select recons_type from: l2 or bce')
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss + self.alpha * hsic_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs() + self.alpha * hsic_loss
        else:
            raise ValueError('Undefined loss type.')
        
        return {'loss': loss, 
            'Reconstruction_Loss':recons_loss, 
            'KLD':kld_loss,
            'l2_loss' : l2_loss,
            'bce_loss' : bce_loss,
            'hsic_loss' : hsic_loss
            }

    def hsic_reg(self, images, s_x=1, s_y=1, device='cuda', num_sample_reparam = 1):
                # num_sample_reparam: num of samples of reparamatrizing z using mu and sigma
        flat = torch.nn.Flatten()
        hsic_score = 0
        for i in range(num_sample_reparam):
            inputs = images.to(device)
            feats = self.encode(inputs).to(device)
            outputs = self.decode(feats).to(device)
            inputs = flat(inputs)
            feats = flat(feats)
            outputs = flat(outputs)
            inputs = inputs.detach()
            feats = feats.detach()
            outputs = outputs.detach()
            hsic_score += HSIC(feats, inputs - outputs, s_x* self.latent_dim, s_y* self.latent_dim, no_grad = False)
        hsic_score /= num_sample_reparam
        return hsic_score
    def hsic_reg_v2(self, images, s_x=1, s_y=1, device='cuda', num_sample_reparam = 1):
        # correspond to utils.hsic_batch_v2
        # num_sample_reparam: num of samples of reparamatrizing z using mu and sigma
        flat = torch.nn.Flatten(start_dim=-3, end_dim=-1)
        batch_size = images.shape[0]
        # print('images', images.shape)

        inputs = images.to(device)
        feats = torch.zeros(batch_size, num_sample_reparam, self.latent_dim)
        outputs = torch.zeros(batch_size, num_sample_reparam, inputs.shape[1], inputs.shape[2], inputs.shape[3])
        # print('outputs',outputs.shape)
        # pdb.set_trace()
        feats = feats.to(device)
        outputs = outputs.to(device)
        for i in range(num_sample_reparam):
            feats[:, i, :] = self.encode(inputs)
            # print('outputs[:, i, :, :]',outputs[:, i, :, :].shape)
            # print('experiment.model.decode(torch.squeeze(feats[:, i, :]))',experiment.model.decode(torch.squeeze(feats[:, i, :])).shape)
            outputs[:, i, :, :, :] = self.decode(feats[:, i, :])##########
        
        # pdb.set_trace()
        inputs = inputs.unsqueeze(1).repeat(1, num_sample_reparam, 1, 1, 1) # repeat batch_size times H
        inputs = flat(inputs)
        outputs = flat(outputs)
        inputs = inputs.detach()
        feats = feats.detach()
        outputs = outputs.detach()
        
        
        hsic_score = HSIC_(feats, inputs - outputs, s_x, s_y, no_grad = False)

        return hsic_score
        
    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]