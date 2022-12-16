# Copyright (c) OpenMMLab. All rights reserved.
from collections import OrderedDict

import torch
import torch.distributions as D
from torch.nn.parallel.distributed import _find_tensors

from mmgen.models.common import set_requires_grad
from mmgen.models.builder import MODELS, build_module
from mmgen.models.translation_models.cyclegan import CycleGAN


@MODELS.register_module()
class UnpairedCycleR2R(CycleGAN):
    """CycleGAN model for unpaired image-to-image translation.

    Ref:
    Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial
    Networks
    """

    def __init__(self, *args, **kwargs):
        self.lambda_bright_diversity = kwargs.pop('lambda_bright_diversity', None)
        self.lambda_color_diversity = kwargs.pop('lambda_color_diversity', None)
        self.c_condition = 2

        pretrained = kwargs.pop('pretrained', None)
        kwargs['pretrained'] = None
        super().__init__(*args, **kwargs)
        self.l1 = torch.nn.L1Loss()
        del self.generators
        torch.cuda.empty_cache()
        self.generator = build_module(kwargs['generator'])
        if 'pretrained' in kwargs['generator'].keys():
            self.generator.init_weights(kwargs['generator']['pretrained'])

        if pretrained is not None:
            self.init_weights(pretrained)

    def init_weights(self, pretrained):
        """Placeholder for init weights"""
        if pretrained is not None:
            state_dict = torch.load(pretrained, map_location='cpu')
            if 'state_dict' in state_dict.keys():
                state_dict = state_dict['state_dict']
                print('Direct loading from {}'.format(pretrained))
            try:
                self.load_state_dict(state_dict, strict=True)
            except:
                pass
            updated_state_dict = OrderedDict()
            cache_init_keys = []
            non_same = {}
            for k in self.state_dict():
                if k not in state_dict.keys():
                    non_same[k] = (None, self.state_dict()[k].shape)
                elif self.state_dict()[k].shape == state_dict[k].shape:
                    if 'init' not in k:
                        updated_state_dict[k] = state_dict[k]
                    else:
                        cache_init_keys.append(k)
                else:
                    non_same[k] = (state_dict[k].shape, self.state_dict()[k].shape)
            if len(non_same.keys()) == 0:
                for k in cache_init_keys:
                    updated_state_dict[k] = state_dict[k]
            else:
                print('Not all state dict shape same')
                for k, v in non_same.items():
                    print('{}: {} -> {}'.format(k, v[0], v[1]))
            self.load_state_dict(updated_state_dict, strict=False)
        else:
            # todo add default init weights
            pass

    def translation(self, image, condition, target_domain=None, **kwargs):
        """Translation Image to target style.

        Args:
            image (tensor): Image tensor with a shape of (N, C, H, W).
            target_domain (str, optional): Target domain of output image.
                Default to None.

        Returns:
            dict: Image tensor of target style.
        """
        if target_domain is None:
            target_domain = self._default_domain
        if target_domain == 'raw':
            outputs = self.generator(image, condition=condition, rev=False)
        else:
            outputs = self.generator(image, condition=condition, rev=True)
        return outputs

    def _get_target_generator(self, domain):
        raise NotImplementedError

    def _get_condition(self, img):
        # condition = torch.randn([img.shape[0], self.c_condition]).to(img.device).detach()
        mean_var = self.generator.color_condition_gen(img)
        m = D.Normal(mean_var[:, 0], torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        color_condition = m.sample()
        mean_var = self.generator.bright_condition_gen(img)
        m = D.Normal(mean_var[:, 0], torch.clamp_min(torch.abs(mean_var[:, 1]), 1e-6))
        bright_condition = m.sample()
        condition = torch.cat([color_condition[:, None], bright_condition[:, None]], 1)
        return condition

    def forward_train(self, img, target_domain, **kwargs):
        condition = self._get_condition(img)
        target= self.translation(img, condition, target_domain=target_domain, **kwargs)
        results = dict(source=img, target=target, condition=condition)
        return results

    def forward_test(self, img, target_domain, **kwargs):
        # This is a trick for CycleGAN
        # ref: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/e1bdf46198662b0f4d0b318e24568205ec4d7aee/test.py#L54 # noqa
        self.train()
        condition = self._get_condition(img).detach()
        target = self.translation(img, condition, target_domain=target_domain, **kwargs)
        results = dict(source=img.cpu(), target=target.cpu(), condition=condition.cpu())
        return results

    def train_step(self,
                   data_batch,
                   optimizer,
                   ddp_reducer=None,
                   running_status=None):
        """Training step function.

        Args:
            data_batch (dict): Dict of the input data batch.
            optimizer (dict[torch.optim.Optimizer]): Dict of optimizers for
                the generators and discriminators.
            ddp_reducer (:obj:`Reducer` | None, optional): Reducer from ddp.
                It is used to prepare for ``backward()`` in ddp. Defaults to
                None.
            running_status (dict | None, optional): Contains necessary basic
                information for training, e.g., iteration number. Defaults to
                None.

        Returns:
            dict: Dict of loss, information for logger, the number of samples\
                and results for visualization.
        """
        # get running status
        if running_status is not None:
            curr_iter = running_status['iteration']
        else:
            # dirty walkround for not providing running status
            if not hasattr(self, 'iteration'):
                self.iteration = 0
            curr_iter = self.iteration

        # forward generators
        outputs = dict()
        for target_domain in self._reachable_domains:
            # fetch data by domain
            source_domain = self.get_other_domains(target_domain)[0]
            img = data_batch[f'img_{source_domain}']
            # translation process
            results = self(img, test_mode=False, target_domain=target_domain)
            outputs[f'real_{source_domain}'] = results['source']
            outputs[f'fake_{target_domain}'] = results['target']
            outputs[f'fake_{target_domain}_condition'] = results['condition']
            # cycle process
            results = self(
                results['target'],
                test_mode=False,
                target_domain=source_domain)
            outputs[f'cycle_{source_domain}'] = results['target']
        results = self(data_batch['img_rgb'], test_mode=False, target_domain='raw')
        outputs['fake_raw_diversity'] = results['target']
        outputs['fake_raw_diversity_condition'] = results['condition']

        log_vars = dict()

        # discriminators
        set_requires_grad(self.discriminators, True)
        # optimize
        optimizer['discriminators'].zero_grad()
        loss_d, log_vars_d = self._get_disc_loss(outputs)
        log_vars.update(log_vars_d)
        if ddp_reducer is not None:
            ddp_reducer.prepare_for_backward(_find_tensors(loss_d))
        loss_d.backward(retain_graph=True)
        optimizer['discriminators'].step()

        # generators, no updates to discriminator parameters.
        if (curr_iter % self.disc_steps == 0
                and curr_iter >= self.disc_init_steps):
            set_requires_grad(self.discriminators, False)
            # optimize
            optimizer['generator'].zero_grad()
            loss_g, log_vars_g = self._get_gen_loss(outputs)
            log_vars.update(log_vars_g)
            if ddp_reducer is not None:
                ddp_reducer.prepare_for_backward(_find_tensors(loss_g))
            loss_g.backward(retain_graph=True)
            optimizer['generator'].step()

        if hasattr(self, 'iteration'):
            self.iteration += 1

        image_results = dict()
        for domain in self._reachable_domains:
            image_results[f'real_{domain}'] = outputs[f'real_{domain}'].cpu()
            image_results[f'fake_{domain}'] = outputs[f'fake_{domain}'].cpu()
            image_results[f'cycle_{domain}'] = outputs[f'cycle_{domain}'].cpu()
        image_results['fake_raw_diversity'] = outputs['fake_raw_diversity'].cpu()
        results = dict(
            log_vars=log_vars,
            num_samples=len(outputs[f'real_{domain}']),
            results=image_results)

        return results

    def _get_gen_loss(self, outputs):
        """Backward function for the generators.

        Args:
            outputs (dict): Dict of forward results.

        Returns:
            dict: Generators' loss and loss dict.
        """
        discriminators = self.get_module(self.discriminators)

        losses = dict()
        for domain in self._reachable_domains:
            # Identity reconstruction for generators
            # rev = True if domain == 'rgb' else False
            # outputs[f'identity_{domain}'] = self.generator(outputs[f'real_{domain}'], rev=rev)
            # GAN loss for generators
            fake_pred = discriminators[domain](outputs[f'fake_{domain}'])
            losses[f'loss_gan_g_{domain}'] = self.gan_loss(
                fake_pred, target_is_real=True, is_disc=False)

        # gen auxiliary loss
        if self.with_gen_auxiliary_loss:
            for loss_module in self.gen_auxiliary_losses:
                loss_ = loss_module(outputs)
                if loss_ is None:
                    continue
                # the `loss_name()` function return name as 'loss_xxx'
                if loss_module.loss_name() in losses:
                    losses[loss_module.loss_name(
                    )] = losses[loss_module.loss_name()] + loss_
                else:
                    losses[loss_module.loss_name()] = loss_

        if self.lambda_color_diversity is not None:
            b = outputs['fake_raw'].shape[0]

            im1, im2 = outputs['fake_raw'], outputs['fake_raw_diversity']
            l1_bright = torch.abs(im1[:, 1]-im2[:, 1]).reshape([b, -1]).mean(1)
            bright_condition = torch.abs(
                outputs['fake_raw_condition'][:, 1] - outputs['fake_raw_diversity_condition'][:, 1])
            _loss = torch.abs((l1_bright/bright_condition)-0.2).mean()
            losses['loss_bright_diversity'] = self.lambda_bright_diversity * _loss

            # im1, im2 = im1.mean(-1).mean(-1), im2.mean(-1).mean(-1)
            im1, im2 = torch.clamp_min(im1, 1e-6), torch.clamp(im2, 1e-6)
            u1, v1 = im1[:, 1] / im1[:, 0], im1[:, 1] / im1[:, 2]  # [b, h, w]
            u2, v2 = im2[:, 1] / im2[:, 0], im2[:, 1] / im2[:, 2]  # [b, h, w]
            u1, v1 = torch.log(u1), torch.log(v1)
            u2, v2 = torch.log(u2), torch.log(v2)
            l1_uv = torch.abs(u1-u2) + torch.abs(v1-v2)
            l1_uv = l1_uv.view([b, -1])/2
            color_condition = torch.abs(
                outputs['fake_raw_condition'][:, 0] - outputs['fake_raw_diversity_condition'][:, 0])[:, None]
            _loss = torch.abs((l1_uv/color_condition)-2).mean()
            losses['loss_color_diversity'] = self.lambda_color_diversity * _loss

        loss_g, log_vars_g = self._parse_losses(losses)
        log_vars_g['diversity'] = float(self.l1(outputs['fake_raw'], outputs['fake_raw_diversity']))
        return loss_g, log_vars_g
