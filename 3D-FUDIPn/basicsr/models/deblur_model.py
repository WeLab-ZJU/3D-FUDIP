import torch
from collections import OrderedDict

from basicsr.utils.registry import MODEL_REGISTRY
from .fudip_model import FUDIPModel


@MODEL_REGISTRY.register()
class DeblurModel(FUDIPModel):
    """BasicSR model wrapper for single-output 3D deblurring networks (U-Net, RCAN, CARE, etc.)."""

    @staticmethod
    def _forward_output(net, lq):
        """Normalize network output to a single tensor."""
        output = net(lq)
        if isinstance(output, (list, tuple)):
            if len(output) == 0:
                raise ValueError('The network returned an empty output list.')
            return output[-1]
        return output

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        self.output = self._forward_output(self.net_g, self.lq)
        self.restored = [self.output]

        l_total = self.lq.new_tensor(0.)
        loss_dict = OrderedDict()

        if self.cri_char:
            l_char = self.cri_char(self.output, self.gt)
            l_total = l_total + l_char
            loss_dict['l_char'] = l_char

        if self.cri_ssim:
            l_ssim = self.cri_ssim(self.output, self.gt)
            l_total = l_total + l_ssim
            loss_dict['l_ssim'] = l_ssim

        loss_dict['l_total'] = l_total
        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            grad_clip_max_norm = self.opt['train'].get('grad_clip_max_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), grad_clip_max_norm)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self._forward_output(self.net_g_ema, self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self._forward_output(self.net_g, self.lq)
            if self.is_train:
                self.net_g.train()
        self.restored = [self.output]
