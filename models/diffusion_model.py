from utils import model_utils
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from tqdm import tqdm
from einops import rearrange, reduce
from random import random
from typing import Tuple


class DiffusionModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        in_channels: int,
        timesteps: int,
        loss_type: str = "l2",
        beta_schedule: str = "linear",
    ):
        super().__init__()
        self.config = {
            "image_size": image_size,
            "in_channels": in_channels,
            "timesteps": timesteps,
            "loss_type": loss_type,
            "beta_schedule": beta_schedule,
        }
        self.model = model
        self.timesteps = timesteps
        self.image_size = image_size
        # self.in_channels = in_channels
        self.in_channels = self.model.channels

        if loss_type == "l1":
            self.loss_func = F.l1_loss
        elif loss_type == "l2":
            self.loss_func = F.mse_loss
        elif loss_type == "huber":
            self.loss_func = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        if beta_schedule == "linear":
            betas = model_utils.linear_beta_schedule(timesteps, beta_end=0.02)
        elif beta_schedule == "cosine":
            # cosine better: Improved Denoising Diffusion Probabilistic Models https://arxiv.org/abs/2102.09672
            betas = model_utils.cosine_beta_schedule(timesteps, s=0.008)
        elif beta_schedule == "sigmoid":
            betas = model_utils.sigmoid_beta_schedule(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # sigma of q
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        ema_model: nn.Module = None,
        classes: torch.Tensor = None,
        cond_weight: float = 1,
        use_ddim: bool = False,
        eta: float = 1 # for ddim sampling
    ) -> torch.Tensor:
        """
        Generates samples denoised (images)

        Args:
            classes (_type_): _description_
            shape (_type_): _description_
            cond_weight (_type_): _description_

        Returns:
            _type_: _description_
        """
        # sampling with ema_model
        if ema_model is not None:
            unet_model = self.model
            self.model = ema_model

        self.model.eval()

        device = next(self.model.parameters()).device
        shape = (n_samples, self.in_channels, self.image_size, self.image_size)

        # start from pure noise (for each example in the batch)
        # img = x_t
        img = torch.randn(shape, device=device)

        if classes is not None:
            n_sample = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_sample:] = 0.0  # makes second half of batch context free

            if use_ddim:
                print("DDIM Sample Guided")
                sampling_fn = partial(
                    self.p_ddim_sample_guided,
                    classes=classes,
                    context_mask=context_mask,
                    eta=eta,
                    temp=1,
                    cond_weight=cond_weight,
                )
            else:
                print("DDPM Sample Guided")
                sampling_fn = partial(
                    self.p_sample_guided,
                    classes=classes,
                    cond_weight=cond_weight,
                    context_mask=context_mask,
                )
            """
            sampling_fn = partial(
                self.p_sample_guided2, classes=classes, cond_weight=cond_weight
            )
            """
        else:
            if use_ddim:
                print("DDIM Sample")
                sampling_fn = partial(self.p_ddim_sample, eta=eta, temp=1)
            else:
                print("DDPM Sample")
                sampling_fn = partial(self.p_sample)

        # from timesteps - 1 to 0
        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling Time Step:"):
            img = sampling_fn(
                x=img,
                t=torch.full((n_samples,), i, device=device, dtype=torch.long),
                t_index=i,
            )

        # img.clamp_(-1.0, 1.0)
        # img = model_utils.unnormalize_to_zero_to_one(img)

        # if i want to train again: set self.model back to Unet

        if ema_model is not None:
            self.model = unet_model
        return img

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """
        Generates samples after DDPM Paper

        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_

        Returns:
            _type_: _description_
        """
        # self.model.eval()

        betas_t = model_utils.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = model_utils.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, time=t) / sqrt_one_minus_alphas_cumprod_t
        )

        # self.model.train()
        # I actually dont need this if => if alpha_prev == alpha_0 => posterior_variance = 0
        # but I could define posterior_variance in another way
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = model_utils.extract(
                self.posterior_variance, t, x.shape
            )
            # posterior_variance_t = betas_t
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_guided(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor,
        t_index: int,
        context_mask,
        cond_weight: float = 1,
    ) -> torch.Tensor:
        """
        Generates guided samples adapted from: https://openreview.net/pdf?id=qw8AKxfYbI

        Args:
            x (torch.Tensor): _description_
            classes (int): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            context_mask (_type_): _description_
            cond_weight (float, optional): _description_. Defaults to 0.0.

        Returns:
            torch.Tensor: _description_
        """

        batch_size = x.shape[0]
        # double to do guidance with
        t_double = t.repeat(2)
        x_double = x.repeat(2, 1, 1, 1)
        betas_t = model_utils.extract(self.betas, t_double, x_double.shape)
        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape
        )
        sqrt_recip_alphas_t = model_utils.extract(
            self.sqrt_recip_alphas, t_double, x_double.shape
        )

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)
        # first half is gui, second
        preds = self.model(x_double, time=t_double, classes=classes_masked)
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        # predicted noise
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x
            - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = model_utils.extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            #noise = torch.zeros_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_guided2(
        self,
        x: torch.Tensor,
        classes: int,
        t: torch.Tensor,
        t_index: int,
        cond_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        More intuitive implementation

        Args:
            x (torch.Tensor): _description_
            classes (int): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            cond_weight (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """

        betas_t = model_utils.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = model_utils.extract(self.sqrt_recip_alphas, t, x.shape)
        pred_noise = self.model(x, t, classes)

        if cond_weight > 0:
            uncond_pred_noise = self.model(x, t, None)
            pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cond_weight)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = model_utils.extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_ddim_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        eta: float = 0,
        temp: float = 1.0,
    ) -> torch.Tensor:
        """
        Generates samples after DDIM Paper


        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            eta (int, optional): _description_. Defaults to 0.
            temp (float, optional): _description_. Defaults to 1.0.

        Returns:
            torch.Tensor: _description_
        """
        alphas_cumprod_t = model_utils.extract(self.alphas_cumprod, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )

        # predict x0 with prediction of noise
        pred_noise = self.model(x, time=t)
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * pred_noise) / (
            alphas_cumprod_t**0.5
        )

        #print("pred_x0", pred_x0.shape)

        alpha_prev_t = model_utils.extract(self.alphas_cumprod_prev, t, x.shape)
        #print("alpha_prev_t", alpha_prev_t.shape)

        # eta = 1: Generative process becomes DDPM
        # sigma = 0: Deterministic forward process: Generative process becomes DDIM
        sigma = (
            eta
            * (
                (1 - alpha_prev_t)
                / (1 - alphas_cumprod_t)
                * (1 - alphas_cumprod_t / alpha_prev_t)
            )
            ** 0.5
        )
        #print("sigma", sigma.shape)
        # Direction from equation 12: added here classes
        # dir_xt = (1.0 - alpha_prev_t - sigma**2).sqrt() * self.model(x, time=t)
        dir_xt = (1.0 - alpha_prev_t - sigma**2).sqrt() * pred_noise
        #print("dir_xt", dir_xt.shape)

        if (sigma == 0.0).all():
            noise = 0.0
        else:
            #noise = torch.randn((1, x.shape[1:]))
            #noise = torch.randn((1, *x.shape[1:]))
            noise = torch.randn_like(x)
        noise *= temp

        # prediction of x_{t-1}: Equation 12 in Paper
        #if t_index == 0:
        #    noise = torch.randn_like(pred_x0) 
        #    x_prev = pred_x0 + sigma * noise 

        #else:
        # t == 0 => alpha_prev = alpha_0 = 1 => sigma = 0 => dir_xt = 0 => x_prev = x_0 = pred_x0 =
        x_prev = (alpha_prev_t**0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev

    @torch.no_grad()
    def p_ddim_sample_guided(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        t_index: int,
        classes: torch.Tensor,
        context_mask: torch.Tensor,
        eta: float = 0,
        temp: float = 1.0,
        cond_weight: float = 2.0,
    ) -> torch.Tensor:
        """
        Generates samples after DDIM Paper


        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            eta (int, optional): _description_. Defaults to 0.
            temp (float, optional): _description_. Defaults to 1.0.

        Returns:
            torch.Tensor: _description_
        """
        batch_size = x.shape[0]
        # double to do guidance with
        t_double = t.repeat(2)
        x_double = x.repeat(2, 1, 1, 1)

        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape
        )

        alphas_cumprod_t = model_utils.extract(
            self.alphas_cumprod, t_double, x_double.shape
        )

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)
        # first half is gui, second
        preds = self.model(x_double, time=t_double, classes=classes_masked)
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        # predicted noise
        pred_noise = eps1 - eps2

        # predict x0 with prediction of noise
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t[:batch_size] * pred_noise) / (
            alphas_cumprod_t[:batch_size] ** 0.5
        )

        alpha_prev_t = model_utils.extract(self.alphas_cumprod_prev, t, x.shape)

        # eta = 1: Generative process becomes DDPM
        # sigma = 0: Deterministic forward process: Generative process becomes DDIM
        sigma = (
            eta
            * (
                (1 - alpha_prev_t)
                / (1 - alphas_cumprod_t)
                * (1 - alphas_cumprod_t / alpha_prev_t)
            )
            ** 0.5
        )
        # Direction from equation 12: added here classes: why predict again
        # dir_xt = (1.0 - alpha_prev_t - sigma**2).sqrt() * self.model(x, time=t, classes=classes)
        dir_xt = (1.0 - alpha_prev_t - sigma**2).sqrt() * pred_noise

        if sigma == 0.0:
            noise = 0.0
        else:
            noise = torch.randn((1, x.shape[1:]))
        noise *= temp

        # prediction of x_{t-1}: Equation 12 in Paper
        x_prev = (alpha_prev_t**0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev

    def forward(
        self,
        x: torch.Tensor,
        classes: torch.Tensor = None,
        p_uncond: float = 0.1,
    ):
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        # x = model_utils.normalize_to_neg_one_to_one(x)

        noise = torch.randn_like(x)

        # q_sample: noise the input image/data
        # with autocast(enabled=False):
        x_noisy = (
            model_utils.extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + model_utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            * noise
        )

        # setting some class labels with probability of p_uncond to 0
        # Question to this in todo.txt
        """
        if random() < p_uncond:
            classes = None
        """

        if classes is not None:
            context_mask = torch.bernoulli(
                torch.zeros(classes.shape[0]) + (1 - p_uncond)
            ).to(device)
            # mask for unconditinal guidance
            classes = classes * context_mask
            classes = classes.type(torch.long)  # multiplication changes type

        pred_noise = self.model(x=x_noisy, time=t, classes=classes)

        loss = self.loss_func(noise, pred_noise)
        return loss


class DiffusionModelExtended(DiffusionModel):
    """
    Extendend DiffusionModel class by self-conditioning, loss weighing and offset-noise:

    Activate self-conditioning by setting self_condition in Unet constructor to True
    Activate loss weighing by setting loss_weighing=True https://arxiv.org/abs/2303.09556
    Activate offset-noise by setting offset_noise_strength > 0 https://www.crosslabs.org/blog/diffusion-with-offset-noise

    Args:
        DiffusionModel (_type_): _description_
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 32,
        in_channels: int = 1,
        timesteps: int = 1000,
        loss_type: str = "l2",
        beta_schedule: str = "linear",
        loss_weighing: bool = False,  # if False => Loss weigth is simply 1 if we predict noise
        min_snr_gamma: int = 5,  # default in the paper
        offset_noise_strength: float = 0.0,
    ):
        super().__init__(
            model=model,
            image_size=image_size,
            in_channels=in_channels,
            timesteps=timesteps,
            loss_type=loss_type,
            beta_schedule=beta_schedule,
        )
        self.config = {
            "image_size": image_size,
            "in_channels": in_channels,
            "timesteps": timesteps,
            "loss_type": loss_type,
            "beta_schedule": beta_schedule,
            "loss_weighing": loss_weighing,
            "min_snr_gamma": min_snr_gamma,
            "offset_noise_strength": offset_noise_strength,
        }

        self.offset_noise_strength = offset_noise_strength
        self.self_condition = self.model.self_condition

        # to predict x_0 from noise in self-conditioning step
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1)
        )

        # predicting noise ε is mathematically equivalent to predicting x0 by intrinsically involving
        # Signal-to-Noise Ratio as a weight factor, thus we divide the SNR term in practice
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if loss_weighing:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        self.register_buffer("loss_weight", maybe_clipped_snr / snr)

    def forward(
        self,
        x: torch.Tensor,
        classes: torch.Tensor = None,
        p_uncond: float = 0.1,
        clip_x_start: bool = False,
    ) -> float:
        """
        Calculate the loss conditioned and noise injected.

        Args:
            x_start (torch.Tensor): _description_
            classes (torch.Tensor, optional): torch.tensor with size of batch_size=x_start.shape[0] with the class labels
            noise (torch.Tensor, optional): _description_. Defaults to None.
            loss_type (str, optional): _description_. Defaults to "l2".
            p_uncond (float, optional): _description_. Defaults to 0.1.

        Raises:
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """
        # self.model.train()
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        # x = model_utils.normalize_to_neg_one_to_one(x)
        noise = torch.randn_like(x)

        # offset noise
        if self.offset_noise_strength > 0.0:
            offset_noise = torch.randn(x.shape[:2], device=device)
            noise += self.offset_noise_strength * rearrange(
                offset_noise, "b c -> b c 1 1"
            )

        # q_sample: noise the input image/data
        # with autocast(enabled=False):
        x_noisy = (
            model_utils.extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + model_utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            * noise
        )

        x_self_cond_x0 = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                # could include predicted noise but in the paper they include previous predicted x0
                # but in Bit-Diffusion they include previously predicted noise but they also give log(beta_t) as input in the forward function of the unet instead of t
                # x_self_cond_noise = self._model_predictions(x=x_noisy, t=t, classes=classes, cond_weight=1)

                x_self_cond_noise = self.model(x=x_noisy, time=t, classes=classes)
                # x_self_cond_noise = self.model(x=x_noisy, t=t, classes=classes, x_self_cond=None)
                x_self_cond_noise.detach_()

                # predict x_start from noise: same as func: predict x_start_from_noise
                maybe_clip = (
                    partial(torch.clamp, min=-1.0, max=1.0)
                    if clip_x_start
                    else model_utils.identity
                )

                x_self_cond_x0 = (
                    x_noisy
                    - model_utils.extract(
                        self.sqrt_one_minus_alphas_cumprod, t, x_noisy.shape
                    )
                    * x_self_cond_noise
                ) / (model_utils.extract(self.alphas_cumprod, t, x.shape) ** 0.5)
                # x_self_cond_x0 = (model_utils.extract(self.sqrt_recip_alphas_cumprod, t, x_noisy.shape) * x_noisy - model_utils.extract(self.sqrt_recipm1_alphas_cumprod, t, x_noisy.shape) * x_self_cond_noise)
                x_self_cond_x0 = maybe_clip(x_self_cond_x0)

        # predict and take gradient step
        """
        if random() < p_uncond:
            classes = None
        """

        if classes is not None:
            context_mask = torch.bernoulli(
                torch.zeros(classes.shape[0]) + (1 - p_uncond)
            ).to(device)

            # mask for unconditinal guidance
            classes = classes * context_mask
            classes = classes.type(torch.long)  # multiplication changes type

        pred_noise = self.model(
            x=x_noisy, time=t, classes=classes, x_self_cond=x_self_cond_x0
        )

        loss = self.loss_func(noise, pred_noise, reduction="none")
        loss = reduce(loss, "b ... -> b (...)", "mean")
        loss = loss * model_utils.extract(a=self.loss_weight, t=t, x_shape=loss.shape)

        return loss.mean()

    def _model_predictions(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor = None,
        cond_weight: float = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if classes is not None:
            device = x.device
            n_samples = classes.shape[0]
            context_mask = torch.ones_like(classes).to(device)
            # make 0 index unconditional
            # double the batch
            classes = classes.repeat(2)
            context_mask = context_mask.repeat(2)
            context_mask[n_samples:] = 0.0

            batch_size = x.shape[0]
            # double to do guidance with
            t_double = t.repeat(2)
            x_double = x.repeat(2, 1, 1, 1)

            classes_masked = classes * context_mask
            classes_masked = classes_masked.type(torch.long)
            # first half is gui, second
            preds = self.model(
                x=x_double,
                time=t_double,
                classes=classes_masked,
                x_self_cond=None,
            )
            eps1 = (1 + cond_weight) * preds[:batch_size]
            eps2 = cond_weight * preds[batch_size:]
            pred_noise = eps1 - eps2

        else:
            pred_noise = self.model(x=x, time=t, classes=None, x_self_cond=None)

        return pred_noise


class DiffusionModelTest(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        image_size: int,
        in_channels: int,
        timesteps: int,
        loss_type: str = "l2",
        beta_schedule: str = "linear",
        use_cfg_me: bool = True,
    ):
        super().__init__()
        self.config = {
            "image_size": image_size,
            "in_channels": in_channels,
            "timesteps": timesteps,
            "loss_type": loss_type,
            "beta_schedule": beta_schedule,
            "use_cfg_me": use_cfg_me,
        }
        self.model = model
        self.timesteps = timesteps
        self.image_size = image_size
        # self.in_channels = in_channels
        self.in_channels = self.model.channels
        self.use_cfg_me = use_cfg_me

        if loss_type == "l1":
            self.loss_func = F.l1_loss
        elif loss_type == "l2":
            self.loss_func = F.mse_loss
        elif loss_type == "huber":
            self.loss_func = F.smooth_l1_loss
        else:
            raise NotImplementedError()

        if beta_schedule == "linear":
            betas = model_utils.linear_beta_schedule(timesteps, beta_end=0.02)
        elif beta_schedule == "cosine":
            # cosine better: Improved Denoising Diffusion Probabilistic Models https://arxiv.org/abs/2102.09672
            betas = model_utils.cosine_beta_schedule(timesteps, s=0.008)
        elif beta_schedule == "sigmoid":
            betas = model_utils.sigmoid_beta_schedule(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # Store as buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        # sigma of q
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        ema_model: nn.Module = None,
        classes: torch.Tensor = None,
        cond_weight: float = 1,
    ) -> torch.Tensor:
        """
        Generates samples denoised (images)

        Args:
            classes (_type_): _description_
            shape (_type_): _description_
            cond_weight (_type_): _description_

        Returns:
            _type_: _description_
        """
        if ema_model is not None:
            unet_model = self.model
            self.model = ema_model

        self.model.eval()

        device = next(self.model.parameters()).device
        shape = (n_samples, self.in_channels, self.image_size, self.image_size)

        # start from pure noise (for each example in the batch)
        # img = x_t
        img = torch.randn(shape, device=device)

        if classes is not None:
            if self.use_cfg_me:
                sampling_fn = partial(
                    self.p_sample_guided2, classes=classes, cond_weight=cond_weight
                )

            else:
                n_sample = classes.shape[0]
                context_mask = torch.ones_like(classes).to(device)
                # make 0 index unconditional
                # double the batch
                classes = classes.repeat(2)
                context_mask = context_mask.repeat(2)
                context_mask[n_sample:] = 0.0  # makes second half of batch context free
                sampling_fn = partial(
                    self.p_sample_guided,
                    classes=classes,
                    cond_weight=cond_weight,
                    context_mask=context_mask,
                )

        else:
            sampling_fn = partial(self.p_sample)

        for i in tqdm(reversed(range(0, self.timesteps)), desc="Sampling Time Step:"):
            img = sampling_fn(
                x=img,
                t=torch.full((n_samples,), i, device=device, dtype=torch.long),
                t_index=i,
            )

        if ema_model is not None:
            self.model = unet_model

        # img.clamp_(-1.0, 1.0)
        # img = model_utils.unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """
        Generates samples after DDPM Paper

        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_

        Returns:
            _type_: _description_
        """
        # self.model.eval()

        betas_t = model_utils.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = model_utils.extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * self.model(x, time=t) / sqrt_one_minus_alphas_cumprod_t
        )
        # self.model.train()
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = model_utils.extract(
                self.posterior_variance, t, x.shape
            )
            # posterior_variance_t = betas_t
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_ddim_sample(
        self, x: torch.Tensor, t: torch.Tensor, t_index: int, eta=0, temp=1.0
    ) -> torch.Tensor:
        """
        Generates samples after DDIM Paper


        Args:
            x (torch.Tensor): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            eta (int, optional): _description_. Defaults to 0.
            temp (float, optional): _description_. Defaults to 1.0.

        Returns:
            torch.Tensor: _description_
        """
        alpha_t = model_utils.extract(self.alphas_cumprod, t, x.shape)
        alpha_prev_t = model_utils.extract(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * ((1 - alpha_prev_t) / (1 - alpha_t) * (1 - alpha_t / alpha_prev_t)) ** 0.5
        )
        sqrt_one_minus_alphas_cumprod = model_utils.extract(
            sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod * self.model(x, time=t)) / (
            alpha_t**0.5
        )
        dir_xt = (1.0 - alpha_prev_t - sigma**2).sqrt() * self.model(x, time=t)
        if sigma == 0.0:
            noise = 0.0
        else:
            noise = torch.randn((1, x.shape[1:]))
        noise *= temp

        x_prev = (alpha_prev_t**0.5) * pred_x0 + dir_xt + sigma * noise

        return x_prev

    @torch.no_grad()
    def p_sample_guided(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        classes: torch.Tensor,
        t_index: int,
        context_mask,
        cond_weight: float = 1,
    ) -> torch.Tensor:
        """
        Generates guided samples adapted from: https://openreview.net/pdf?id=qw8AKxfYbI

        Args:
            x (torch.Tensor): _description_
            classes (int): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            context_mask (_type_): _description_
            cond_weight (float, optional): _description_. Defaults to 0.0.

        Returns:
            torch.Tensor: _description_
        """

        batch_size = x.shape[0]
        # double to do guidance with
        t_double = t.repeat(2)
        x_double = x.repeat(2, 1, 1, 1)
        betas_t = model_utils.extract(self.betas, t_double, x_double.shape)
        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t_double, x_double.shape
        )
        sqrt_recip_alphas_t = model_utils.extract(
            self.sqrt_recip_alphas, t_double, x_double.shape
        )

        # classifier free sampling interpolates between guided and non guided using `cond_weight`
        classes_masked = classes * context_mask
        classes_masked = classes_masked.type(torch.long)
        # first half is gui, second
        preds = self.model(x_double, time=t_double, classes=classes_masked)
        eps1 = (1 + cond_weight) * preds[:batch_size]
        eps2 = cond_weight * preds[batch_size:]
        # predicted noise
        x_t = eps1 - eps2

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t[:batch_size] * (
            x
            - betas_t[:batch_size] * x_t / sqrt_one_minus_alphas_cumprod_t[:batch_size]
        )

        if t_index <= 1:
            return model_mean
        else:
            posterior_variance_t = model_utils.extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_guided2(
        self,
        x: torch.Tensor,
        classes: int,
        t: torch.Tensor,
        t_index: int,
        cond_weight: float = 1.0,
    ) -> torch.Tensor:
        """
        More intuitive implementation

        Args:
            x (torch.Tensor): _description_
            classes (int): _description_
            t (torch.Tensor): _description_
            t_index (int): _description_
            cond_weight (float, optional): _description_. Defaults to 0.0.

        Returns:
            _type_: _description_
        """

        betas_t = model_utils.extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = model_utils.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = model_utils.extract(self.sqrt_recip_alphas, t, x.shape)
        pred_noise = self.model(x, t, classes)

        if cond_weight > 0:
            uncond_pred_noise = self.model(x, t, None)
            pred_noise = torch.lerp(uncond_pred_noise, pred_noise, cond_weight)

        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t
        )

        # if t_index == 0:
        if t_index <= 1:
            return model_mean
        else:
            posterior_variance_t = model_utils.extract(
                self.posterior_variance, t, x.shape
            )
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    def forward(
        self,
        x: torch.Tensor,
        classes: torch.Tensor = None,
        p_uncond: float = 0.1,
    ):
        device = x.device
        t = torch.randint(0, self.timesteps, (x.shape[0],), device=device).long()
        # x = model_utils.normalize_to_neg_one_to_one(x)

        noise = torch.randn_like(x)

        # q_sample: noise the input image/data
        # with autocast(enabled=False):
        x_noisy = (
            model_utils.extract(self.sqrt_alphas_cumprod, t, x.shape) * x
            + model_utils.extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            * noise
        )

        # setting some class labels with probability of p_uncond to 0
        # Question to this in todo.txt

        # from Dome272

        if classes is not None:
            if self.use_cfg_me:
                if random() < 0.1:
                    classes = None
            else:
                context_mask = torch.bernoulli(
                    torch.zeros(classes.shape[0]) + (1 - p_uncond)
                ).to(device)

                # mask for unconditinal guidance
                classes = classes * context_mask
                classes = classes.type(torch.long)  # multiplication changes type

        pred_noise = self.model(x=x_noisy, time=t, classes=classes)

        loss = self.loss_func(noise, pred_noise)
        return loss
