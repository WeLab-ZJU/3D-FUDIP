import os.path as osp

import torch
from contextlib import nullcontext
from collections import OrderedDict
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite_volume, tensor2volume, volume2preview
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class FUDIPModel(BaseModel):
    """BasicSR model wrapper for 3D-FUDIP networks with stage-wise outputs."""

    def __init__(self, opt):
        super(FUDIPModel, self).__init__(opt)

        # Stage loss weights ('sum' as default): sum | uniform | log | log_soft | no_mid
        self.stage_weight_mode = opt['train'].get('stage_weight_mode', 'log_soft')
        self.use_SAM = opt['network_g'].get('use_SAM', True)
        logger = get_root_logger()
        if not self.use_SAM:
            logger.info(
                'use_SAM is False: training loss is applied to the final output only '
                '(no intermediate stage supervision).')
        elif self.stage_weight_mode == 'no_mid':
            logger.info(
                'stage_weight_mode is no_mid: intermediate stage losses are disabled; '
                'SAM module remains enabled.')

        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        bf16_phases = []
        if train_opt.get('use_bf16', False):
            bf16_phases.append('train')
        if self.opt.get('val', {}).get('use_bf16', False):
            bf16_phases.append('val')
        if bf16_phases:
            logger = get_root_logger()
            logger.info(f'BF16 autocast enabled for {"/".join(bf16_phases)} forward pass.')

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)

            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()

        if train_opt.get('char_opt'):
            self.cri_char = build_loss(train_opt['char_opt']).to(self.device)
        else:
            self.cri_char = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('edge_opt'):
            self.cri_edge = build_loss(train_opt['edge_opt']).to(self.device)
        else:
            self.cri_edge = None

        if train_opt.get('freq_opt'):
            self.cri_freq = build_loss(train_opt['freq_opt']).to(self.device)
        else:
            self.cri_freq = None

        if self.cri_char is None and self.cri_ssim is None and self.cri_edge is None and self.cri_freq is None:
            raise ValueError('char_opt, ssim_opt, edge_opt and freq_opt are all None.')

        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_g_opt = train_opt['optim_g']

        # nn.ParameterList attributes holding the FUDIP physical hyper-parameters:
        # para0 = GDM step size; para1/para2/para3 = Z1/Z2/Z3 softplus weights.
        physics_group_names = ('para0', 'para1', 'para2', 'para3')
        normal_params, no_decay_params, physics_params = [], [], []
        for name, param in self.net_g.named_parameters():
            if not param.requires_grad:
                logger = get_root_logger()
                logger.warning(f'Params {name} will not be optimized.')
                continue

            name_parts = name.split('.')
            param_name = name_parts[-1]
            # ParameterList members are named '<owner>.<index>' (or 'module.<owner>.<index>' under DDP).
            owner_name = name_parts[-2] if len(name_parts) >= 2 else param_name

            if owner_name in physics_group_names:
                physics_params.append(param)
            elif param.dim() == 1 or param_name == 'bias' or param_name in ('beta', 'gamma'):
                no_decay_params.append(param)
            else:
                normal_params.append(param)

        base_lr = optim_g_opt['lr']
        physics_lr_mult = optim_g_opt.pop('physics_lr_mult', 10.0)
        physics_lr = optim_g_opt.pop('physics_lr', base_lr * physics_lr_mult)

        optim_params = [
            {'params': normal_params},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        self._has_physics_group = bool(physics_params)
        if self._has_physics_group:
            optim_params.append({'params': physics_params, 'lr': physics_lr, 'weight_decay': 0.0})
            logger = get_root_logger()
            logger.info(
                f'Physics parameters (para0-para3): {len(physics_params)} scalars, '
                f'lr={physics_lr:g} (base lr={base_lr:g}, x{physics_lr / base_lr:.3g}).')

        optim_type = optim_g_opt.pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **optim_g_opt)
        self.optimizers.append(self.optimizer_g)

    def get_current_learning_rate(self):
        param_groups = self.optimizers[0].param_groups
        lrs = [param_groups[0]['lr']]
        if getattr(self, '_has_physics_group', False):
            lrs.append(param_groups[-1]['lr'])
        return lrs

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    @staticmethod
    def _as_stage_list(restored):
        if isinstance(restored, (list, tuple)):
            if len(restored) == 0:
                raise ValueError('The network returned an empty stage output list.')
            return list(restored)
        return [restored]

    def _stage_weights(self, num_stages, dtype):
        mode = self.stage_weight_mode
        if mode == 'sum':
            return torch.ones((num_stages,), device=self.device, dtype=dtype)
        if mode == 'uniform':
            return torch.full(
                (num_stages,), 1.0 / num_stages, device=self.device, dtype=dtype)
        if mode == 'log':
            return torch.logspace(-1, 0, steps=num_stages, device=self.device, dtype=dtype)
        if mode == 'log_soft':
            return torch.logspace(-0.5, 0, steps=num_stages, device=self.device, dtype=dtype)
        if mode == 'no_mid':
            weights = torch.zeros((num_stages,), device=self.device, dtype=dtype)
            weights[-1] = 1.0
            return weights
        raise ValueError(
            f'Unknown stage_weight_mode: {mode!r}. Use sum, uniform, log, log_soft, or no_mid.')

    def _bf16_autocast(self, opt_key):
        use_bf16 = self.opt.get(opt_key, {}).get('use_bf16', False)
        if not use_bf16:
            return nullcontext()
        if self.device.type != 'cuda':
            raise RuntimeError(f'{opt_key}.use_bf16 requires CUDA, but the model device is {self.device}.')
        if not hasattr(torch, 'autocast'):
            raise RuntimeError(f'{opt_key}.use_bf16 requires torch.autocast support.')

        is_bf16_supported = getattr(torch.cuda, 'is_bf16_supported', None)
        if is_bf16_supported is not None and not is_bf16_supported():
            raise RuntimeError(f'{opt_key}.use_bf16 requires a CUDA device with BF16 support.')
        return torch.autocast(device_type='cuda', dtype=torch.bfloat16)

    @staticmethod
    def _stage_list_to_float(restored):
        return [stage.float() for stage in restored]

    def _accumulate_loss_dict(self, loss_dict, accumulation_step):
        if accumulation_step == 0 or not hasattr(self, '_accumulated_loss_dict'):
            self._accumulated_loss_dict = OrderedDict()

        for key, value in loss_dict.items():
            if key not in self._accumulated_loss_dict:
                self._accumulated_loss_dict[key] = value.detach()
            else:
                self._accumulated_loss_dict[key] = self._accumulated_loss_dict[key] + value.detach()

    def optimize_parameters(self, current_iter, accumulation_step=0, should_update=True):
        accumulate_grad_batches = int(self.opt['train'].get('accumulate_grad_batches', 1))
        if accumulate_grad_batches < 1:
            raise ValueError(f'accumulate_grad_batches must be >= 1, but got {accumulate_grad_batches}.')
        if accumulation_step < 0 or accumulation_step >= accumulate_grad_batches:
            raise ValueError(
                f'accumulation_step must be in [0, {accumulate_grad_batches}), but got {accumulation_step}.')

        if accumulation_step == 0:
            self.optimizer_g.zero_grad()

        with self._bf16_autocast('train'):
            self.restored = self._as_stage_list(self.net_g(self.lq))
        self.restored = self._stage_list_to_float(self.restored)
        self.output = self.restored[-1]

        l_total = self.lq.new_tensor(0.)
        loss_dict = OrderedDict()

        if self.use_SAM:
            stage_weights = self._stage_weights(len(self.restored), self.gt.dtype)
            if self.cri_char:
                l_char_total = self.lq.new_tensor(0.)
                for weight, stage_output in zip(stage_weights, self.restored):
                    l_char_total = l_char_total + weight * self.cri_char(stage_output, self.gt)
                l_total = l_total + l_char_total
                loss_dict['l_char'] = l_char_total

            if self.cri_ssim:
                l_ssim_total = self.lq.new_tensor(0.)
                for weight, stage_output in zip(stage_weights, self.restored):
                    l_ssim_total = l_ssim_total + weight * self.cri_ssim(stage_output, self.gt)
                l_total = l_total + l_ssim_total
                loss_dict['l_ssim'] = l_ssim_total

            if self.cri_edge:
                l_edge_total = self.lq.new_tensor(0.)
                for weight, stage_output in zip(stage_weights, self.restored):
                    l_edge_total = l_edge_total + weight * self.cri_edge(stage_output, self.gt)
                l_total = l_total + l_edge_total
                loss_dict['l_edge'] = l_edge_total

            if self.cri_freq:
                l_freq_total = self.lq.new_tensor(0.)
                for weight, stage_output in zip(stage_weights, self.restored):
                    l_freq_total = l_freq_total + weight * self.cri_freq(stage_output, self.gt)
                l_total = l_total + l_freq_total
                loss_dict['l_freq'] = l_freq_total
        else:
            if self.cri_char:
                l_char_total = self.cri_char(self.output, self.gt)
                l_total = l_total + l_char_total
                loss_dict['l_char'] = l_char_total

            if self.cri_ssim:
                l_ssim_total = self.cri_ssim(self.output, self.gt)
                l_total = l_total + l_ssim_total
                loss_dict['l_ssim'] = l_ssim_total

            if self.cri_edge:
                l_edge_total = self.cri_edge(self.output, self.gt)
                l_total = l_total + l_edge_total
                loss_dict['l_edge'] = l_edge_total

            if self.cri_freq:
                l_freq_total = self.cri_freq(self.output, self.gt)
                l_total = l_total + l_freq_total
                loss_dict['l_freq'] = l_freq_total

        loss_dict['l_total'] = l_total
        self._accumulate_loss_dict(loss_dict, accumulation_step)

        (l_total / accumulate_grad_batches).backward()
        if should_update:
            use_grad_clip = self.opt['train'].get('use_grad_clip', True)
            if use_grad_clip:
                grad_clip_max_norm = self.opt['train'].get('grad_clip_max_norm', 1.0)
                torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), grad_clip_max_norm)
            self.optimizer_g.step()

            log_loss_dict = OrderedDict()
            for key, value in self._accumulated_loss_dict.items():
                log_loss_dict[key] = value / accumulate_grad_batches
            self.log_dict = self.reduce_loss_dict(log_loss_dict)
            del self._accumulated_loss_dict

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                with self._bf16_autocast('val'):
                    self.restored = self._as_stage_list(self.net_g_ema(self.lq))
        else:
            self.net_g.eval()
            with torch.no_grad():
                with self._bf16_autocast('val'):
                    self.restored = self._as_stage_list(self.net_g(self.lq))
            if self.is_train:
                self.net_g.train()
        self.restored = self._stage_list_to_float(self.restored)
        self.output = self.restored[-1]

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='volume')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            if with_metrics:
                if not hasattr(self, 'gt'):
                    raise ValueError('GT is required for validation metrics.')
                metric_data = {
                    'img': self.output.detach(),
                    'img2': self.gt.detach(),
                }
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if save_img:
                volumes = tensor2volume(self.output)
                for batch_idx, volume in enumerate(volumes):
                    batch_suffix = f'_b{batch_idx}' if len(volumes) > 1 else ''
                    if self.opt['is_train']:
                        save_img_path = osp.join(
                            self.opt['path']['visualization'],
                            img_name,
                            f'{img_name}_{current_iter}{batch_suffix}.tif')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(
                                self.opt['path']['visualization'],
                                dataset_name,
                                f'{img_name}_{self.opt["val"]["suffix"]}{batch_suffix}.tif')
                        else:
                            save_img_path = osp.join(
                                self.opt['path']['visualization'],
                                dataset_name,
                                f'{img_name}_{self.opt["name"]}{batch_suffix}.tif')
                    imwrite_volume(volume, save_img_path)

            del self.lq
            del self.output
            del self.restored
            if hasattr(self, 'gt'):
                del self.gt
            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Val {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        else:
            logger = get_root_logger()
            logger.info(
                f'Validation {dataset_name}: completed {len(dataloader)} 3D volume batches at iter {current_iter}.')

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = volume2preview(self.lq)
        out_dict['result'] = volume2preview(self.output)
        if hasattr(self, 'gt'):
            out_dict['gt'] = volume2preview(self.gt)
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
