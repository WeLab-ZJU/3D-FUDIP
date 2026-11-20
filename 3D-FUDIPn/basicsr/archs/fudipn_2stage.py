import torch
import torch.nn as nn

from basicsr.archs.fudipn_blocks import (
    Decoder,
    Encoder,
    MergeBlock,
    NoSAM,
    ResBlock,
    SAM,
    Z1_block,
    Z2_block,
    Z3_block,
    _inv_softplus,
    conv,
    init_isff_zero,
    make_shallow_feature,
)


################################################
# 2-stage 3D-FUDIPn
################################################
class FUDIP_Net(nn.Module):
    def __init__(self, kernel_size=3, n_feat=16, scale_unetfeat=32, unet_depth=4, bias=False,
                 use_SCAB=True, use_HIN=True, use_ISFF=True,
                 isff_mode='add', use_SAM=True):
        super(FUDIP_Net, self).__init__()

        stages = 2
        sam_cls = SAM if use_SAM else NoSAM
        self.use_ISFF = use_ISFF

        # Stage 1 -- kernels
        self.H1 = ResBlock(in_channels=1, mid_channels=16)
        self.Ht1 = ResBlock(in_channels=1, mid_channels=16)
        self.W1 = ResBlock(in_channels=1, mid_channels=4)
        self.Wt1 = ResBlock(in_channels=1, mid_channels=4)
        # Stage 2 -- kernels
        self.H2 = ResBlock(in_channels=1, mid_channels=16)
        self.Ht2 = ResBlock(in_channels=1, mid_channels=16)
        self.W2 = ResBlock(in_channels=1, mid_channels=4)
        self.Wt2 = ResBlock(in_channels=1, mid_channels=4)

        # Data transformation module
        self.Z1 = Z1_block()
        self.Z2 = Z2_block()
        self.Z3 = Z3_block()

        # Hyperparameters
        self.para0 = nn.ParameterList(
            [nn.Parameter(torch.Tensor([0.1])) for _ in range(stages)]
        )
        self.para1 = nn.ParameterList(
            [nn.Parameter(_inv_softplus(torch.tensor([1.0]))) for _ in range(stages)]
        )
        self.para2 = nn.ParameterList(
            [nn.Parameter(_inv_softplus(torch.tensor([0.1]))) for _ in range(stages)]
        )
        self.para3 = nn.ParameterList(
            [nn.Parameter(_inv_softplus(torch.tensor([0.01]))) for _ in range(stages)]
        )

        self.shallow_feat1 = make_shallow_feature(1, n_feat, kernel_size, bias, use_SCAB)
        self.shallow_feat2 = make_shallow_feature(1, n_feat, kernel_size, bias, use_SCAB)

        self.encoder0 = Encoder(n_feat, scale_unetfeat, use_ISFF=False, depth=unet_depth, use_HIN=use_HIN)
        self.decoder0 = Decoder(n_feat, scale_unetfeat, depth=unet_depth)
        self.encoder = Encoder(n_feat, scale_unetfeat, use_ISFF=use_ISFF, depth=unet_depth,
                               use_HIN=use_HIN, isff_mode=isff_mode)
        self.decoder = Decoder(n_feat, scale_unetfeat, depth=unet_depth)

        self.sam1 = sam_cls(n_feat, kernel_size=1, bias=bias)

        self.merge12 = MergeBlock(n_feat, kernel_size=3, bias=bias)
        self.final_conv = conv(n_feat, 1, kernel_size=3, stride=1)
        init_isff_zero(self)

    def forward(self, f):
        res = []
        x = f.clone()
        g1 = x.clone().detach()

        q1 = torch.zeros_like(f)
        q2 = torch.zeros_like(f)
        q3 = torch.zeros_like(f)

        ###################
        # stage-1
        ###################
        Hx = self.H1(x)
        Wx = self.W1(x)

        z1 = self.Z1(Hx, f, q1, self.para1[0])
        z2 = self.Z2(x, g1, q2, self.para2[0])
        z3 = self.Z3(Wx, q3, self.para3[0])

        x = x - self.para0[0] * (self.Ht1(Hx - z1 + q1) +
                                 (x - z2 + q2) +
                                 self.Wt1(Wx - z3 + q3)
                                 )

        x_feats = self.shallow_feat1(x)
        enc_out, feat_enc = self.encoder0(x_feats)
        dec_out = self.decoder0(feat_enc, enc_out)
        g1_samfeats, g1 = self.sam1(dec_out[-1], x)

        res.append(g1)

        q1 += self.H1(x) - z1
        q2 += x - z2
        q3 += self.W1(x) - z3

        ###################
        # stage-2
        ###################
        Hx = self.H2(x)
        Wx = self.W2(x)

        z1 = self.Z1(Hx, f, q1, self.para1[1])
        z2 = self.Z2(x, g1, q2, self.para2[1])
        z3 = self.Z3(Wx, q3, self.para3[1])

        x = x - self.para0[1] * (self.Ht2(Hx - z1 + q1) +
                                 (x - z2 + q2) +
                                 self.Wt2(Wx - z3 + q3)
                                 )

        x_feats = self.shallow_feat2(x)
        x_cat = self.merge12(x_feats, g1_samfeats)
        if self.use_ISFF:
            enc_out, feat_enc = self.encoder(x_cat, enc_out, dec_out)
        else:
            enc_out, feat_enc = self.encoder(x_cat)
        dec_out = self.decoder(feat_enc, enc_out)
        output = self.final_conv(dec_out[-1]) + x

        res.append(output)

        return res
