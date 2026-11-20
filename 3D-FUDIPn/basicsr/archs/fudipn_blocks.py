import torch
import torch.nn as nn
import torch.nn.functional as F

from basicsr.archs.NAFblock import SCAB


################################################
# Basic modules
################################################
def conv(in_channels, out_channels, kernel_size, bias=True, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size,
                     padding=(kernel_size // 2), stride=stride, bias=bias)


def make_shallow_feature(in_channels, n_feat, kernel_size, bias, use_SCAB=True):
    modules = [conv(in_channels, n_feat, kernel_size, bias=bias)]
    if use_SCAB:
        modules.append(SCAB(n_feat))
    return nn.Sequential(*modules)


def init_isff_zero(model):
    for name, m in model.named_modules():
        if name.endswith("isff_enc") or name.endswith("isff_dec"):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def _inv_softplus(y):
    # Numerically stable inverse of softplus for positive scalar initial values.
    return y + torch.log(-torch.expm1(-y))


# Using Res-block to simulate the degradation
class ResBlock(nn.Module):
    def __init__(self, in_channels=1, mid_channels=32, k_size=3, relu_slope=0.2):
        super(ResBlock, self).__init__()

        self.conv1 = conv(in_channels, mid_channels, kernel_size=k_size, stride=1)
        self.conv2 = conv(mid_channels, in_channels, kernel_size=k_size, stride=1)
        self.relu1 = nn.LeakyReLU(relu_slope, inplace=False)

    def forward(self, x):
        # Step 1: Extract features and apply non-linear activation
        out = self.relu1(self.conv1(x))

        # Step 2: Map the feature dimensions back to the input channels space
        out = self.conv2(out)

        # Step 3: Form the residual connection
        return out + x


# Resizing module
class ResizeDownsample3d(nn.Module):
    """
    Downsamples 3D spatial dimensions using trilinear interpolation 
    followed by a standard convolution.
    """

    def __init__(self, in_channels, out_channels):
        super(ResizeDownsample3d, self).__init__()
        self.down = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=False)
        self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x


class PoolDownsample3d(nn.Module):
    """
    Downsamples 3D spatial dimensions using pooling operations 
    followed by a standard convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        pool_type (str): Type of pooling operation. Valid options are 'avg' or 'max'.
    """

    def __init__(self, in_channels, out_channels, pool_type='avg'):
        super().__init__()
        if pool_type == 'max':
            self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        elif pool_type == 'avg':
            self.down = nn.AvgPool3d(kernel_size=2, stride=2)
        else:
            raise ValueError("pool_type must be 'max' or 'avg'")
        self.conv = conv(in_channels, out_channels, kernel_size=3, stride=1)

    def forward(self, x):
        return self.conv(self.down(x))


class DWConvDownsample3d(nn.Module):
    """
    Downsamples 3D spatial dimensions using a depthwise separable strided convolution.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the pointwise convolution.
        kernel_size (int): Size of the depthwise convolving kernel. Defaults to 3.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size,
                                   stride=2, padding=kernel_size // 2,
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class Downsample3d(nn.Module):
    """
    Unified Factory Module for 3D Downsampling.
    This module serves as a single entry point to instantiate various downsampling strategies.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        mode (str): The downsampling strategy to employ.
                    Valid options: 'max_pool', 'avg_pool', 'dw_conv', 'resize_conv'.
    """

    def __init__(self, in_channels: int, out_channels: int, mode: str = 'dw_conv', **kwargs):
        super().__init__()

        if mode == 'max_pool':
            self.block = PoolDownsample3d(in_channels, out_channels, pool_type='max')
        elif mode == 'avg_pool':
            self.block = PoolDownsample3d(in_channels, out_channels, pool_type='avg')
        elif mode == 'dw_conv':
            self.block = DWConvDownsample3d(in_channels, out_channels, **kwargs)
        elif mode == 'resize_conv':
            self.block = ResizeDownsample3d(in_channels, out_channels, **kwargs)
        else:
            raise NotImplementedError(
                f"Downsample mode '{mode}' is not supported. "
                "Available modes: 'max_pool', 'avg_pool', 'dw_conv', 'resize_conv'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Routes the input tensor through the instantiated downsample block.
        """
        return self.block(x)


################################################
# Supervised Attention Module
################################################

class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)
        self.conv3 = conv(1, n_feat, kernel_size, bias=bias)

    def forward(self, feat_in, x_img):
        feat_proj = self.conv1(feat_in)
        img_out = self.conv2(feat_in) + x_img
        feat_mask = torch.sigmoid(self.conv3(img_out))
        feat_out = feat_proj * feat_mask + feat_in
        return feat_out, img_out


class NoSAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(NoSAM, self).__init__()

    def forward(self, feat_in, img_raw):
        # Remove attention feature modulation.
        feat_out = feat_in
        img_out = img_raw
        return feat_out, img_out


class MergeBlock(nn.Module):
    """
    Content-adaptive affine merge for cross-stage feature fusion.
    """

    def __init__(self, n_feat, kernel_size=3, bias=False):
        super(MergeBlock, self).__init__()
        self.gamma = conv(n_feat * 2, n_feat, 1, bias=bias)
        self.beta = conv(n_feat * 2, n_feat, 1, bias=bias)
        self.fuse = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        if self.gamma.bias is not None:
            nn.init.zeros_(self.gamma.bias)
        if self.beta.bias is not None:
            nn.init.zeros_(self.beta.bias)

    def forward(self, x, bridge):
        cond = torch.cat([x, bridge], dim=1)
        scale = 1 + self.gamma(cond)
        bridge = bridge * scale + self.beta(cond)
        out = self.fuse(torch.cat([x, bridge], dim=1)) + x
        return out


################################################
# UNet blocks
################################################
class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope=0.2, use_isff=False,
                 isff_mode='add', use_HIN=False, down_mode='dw_conv'):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv3d(in_size, out_size, 1, 1, 0)
        self.use_isff = use_isff
        self.isff_mode = isff_mode
        if use_isff and isff_mode not in ('add', 'gated'):
            raise ValueError(f"isff_mode must be 'add' or 'gated', got {isff_mode}.")

        self.conv_1 = nn.Conv3d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv3d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_isff:
            self.isff_enc = nn.Conv3d(out_size, out_size, 3, 1, 1)
            self.isff_dec = nn.Conv3d(in_size, out_size, 3, 1, 1)
            if isff_mode == 'gated':
                self.isff_phi = nn.Conv3d(out_size, out_size, 3, 1, 1)
                self.isff_gamma = nn.Conv3d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm3d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = Downsample3d(out_size, out_size, mode=down_mode)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)

        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)

        if enc is not None and dec is not None:
            assert self.use_isff
            skip = self.isff_enc(enc) + self.isff_dec(dec)
            if self.isff_mode == 'add':
                out = out + skip
            else:
                skip = F.leaky_relu(skip, 0.1, inplace=True)
                out = out * torch.sigmoid(self.isff_phi(skip)) + self.isff_gamma(skip) + out

        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class Encoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, use_ISFF, depth=5, use_HIN=True, isff_mode='add'):
        super(Encoder, self).__init__()
        self.body = nn.ModuleList()
        self.depth = depth
        for i in range(depth - 1):
            self.body.append(
                UNetConvBlock(in_size=n_feat + scale_unetfeats * i,
                              out_size=n_feat + scale_unetfeats * (i + 1),
                              downsample=True, relu_slope=0.2, use_isff=use_ISFF,
                              isff_mode=isff_mode, use_HIN=use_HIN))
        self.body.append(UNetConvBlock(in_size=n_feat + scale_unetfeats * (depth - 1),
                                       out_size=n_feat + scale_unetfeats * (depth - 1), downsample=False,
                                       relu_slope=0.2, use_isff=use_ISFF,
                                       isff_mode=isff_mode, use_HIN=use_HIN))

    def forward(self, x, encoder_outs=None, decoder_outs=None):
        res = []
        if encoder_outs is not None and decoder_outs is not None:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x, encoder_outs[i], decoder_outs[-i - 1])
                    res.append(x_up)
                else:
                    x = down(x)
        else:
            for i, down in enumerate(self.body):
                if (i + 1) < self.depth:
                    x, x_up = down(x)
                    res.append(x_up)
                else:
                    x = down(x)
        return res, x


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class Decoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, depth=5):
        super(Decoder, self).__init__()

        self.body = nn.ModuleList()
        self.skip_conv = nn.ModuleList()
        for i in range(depth - 1):
            self.body.append(UNetUpBlock(in_size=n_feat + scale_unetfeats * (depth - i - 1) +
                                                 n_feat + scale_unetfeats * (depth - i - 2),
                                         out_size=n_feat + scale_unetfeats * (depth - i - 2),
                                         relu_slope=0.2))
            self.skip_conv.append(
                nn.Conv3d(n_feat + scale_unetfeats * (depth - i - 1),
                          n_feat + scale_unetfeats * (depth - i - 2), 3, 1, 1))

    def forward(self, x, bridges):
        res = []
        for i, up in enumerate(self.body):
            x = up(x, self.skip_conv[i](bridges[-i - 1]))
            res.append(x)

        return res


################################################
# Update Data Transformation Module
################################################
class Z1_block(nn.Module):
    def __init__(self):
        super(Z1_block, self).__init__()

    def forward(self, Hx, f, q_1, para_1):
        para_1 = F.softplus(para_1)
        temp_ = Hx + q_1 - para_1
        z1 = 0.5 * (temp_ + torch.sqrt(F.relu(temp_ ** 2 + 4 * f * para_1) + 1e-6))
        return z1


class Z2_block(nn.Module):
    def __init__(self):
        super(Z2_block, self).__init__()

    def forward(self, x, g, q2, para_2):
        para_2 = F.softplus(para_2)
        z2 = para_2 * g + (x + q2)
        z2 /= para_2 + 1
        return z2


class Z3_block(nn.Module):
    def __init__(self):
        super(Z3_block, self).__init__()

    def forward(self, Wx, q3, para_3):
        para_3 = F.softplus(para_3)
        z3 = torch.abs(Wx + q3) - para_3
        z3 = torch.sign(Wx + q3) * F.relu(z3)
        return z3
