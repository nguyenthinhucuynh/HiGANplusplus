import numpy as np
import torch
from torch import nn
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.utils import _len2mask, init_weights
from .masking import StyleMasking


class StyleBackbone(nn.Module):
    def __init__(self, resolution=16, max_dim=256, in_channel=1, init='N02', dropout=0.0, norm='bn'):
        super(StyleBackbone, self).__init__()
        self.reduce_len_scale = 16
        nf = resolution

        cnn_f = [
            nn.ConstantPad2d(2, -1),
            Conv2dBlock(in_channel, nf, 5, 2, 0, norm='none', activation='none')
        ]

        for i in range(2):
            nf_out = min(int(nf * 2), max_dim)
            cnn_f += [
                ActFirstResBlock(nf, nf, None, 'relu', norm, 'zero', dropout=dropout / 2),
                nn.ZeroPad2d((1, 1, 0, 0)),
                ActFirstResBlock(nf, nf_out, None, 'relu', norm, 'zero', dropout=dropout / 2),
                nn.ZeroPad2d(1),
                nn.MaxPool2d(3, 2)
            ]
            nf = nf_out

        df = nf
        for i in range(2):
            df_out = min(int(df * 2), max_dim)
            cnn_f += [
                ActFirstResBlock(df, df, None, 'relu', norm, 'zero', dropout=dropout),
                ActFirstResBlock(df, df_out, None, 'relu', norm, 'zero', dropout=dropout)
            ]
            if i < 1:
                cnn_f += [nn.MaxPool2d(3, 2)]
            else:
                cnn_f += [nn.ZeroPad2d((1, 1, 0, 0))]
            df = df_out

        self.cnn_backbone = nn.Sequential(*cnn_f)

        self.cnn_ctc = nn.Sequential(
            nn.ReLU(),
            Conv2dBlock(df, df, 3, 1, 0, norm=norm, activation='relu')
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, x, ret_feats=False):
        feats = []
        for layer in self.cnn_backbone:
            x = layer(x)
            if ret_feats:
                feats.append(x)

        out = self.cnn_ctc(x).squeeze(-2)
        return out, feats


class StyleEncoder(nn.Module):
    def __init__(self, style_dim=32, in_dim=256, init='N02'):
        super(StyleEncoder, self).__init__()
        self.style_dim = style_dim

        self.masking = StyleMasking()

        # Shared MLP
        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )

        # Fusion layer (NEW)
        self.fusion = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(in_dim, style_dim)
        self.logvar = nn.Linear(in_dim, style_dim)

        if init != 'none':
            init_weights(self, init)

    def masked_pool(self, feat, img_len):
        mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        return (feat * mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        feat, all_feats = cnn_backbone(img, ret_feats)

        img_len = img_len // cnn_backbone.reduce_len_scale

        # ===== ORIGINAL GLOBAL STYLE =====
        style_global = self.masked_pool(feat, img_len)

        # ===== VERTICAL MASKING =====
        feat_v = self.masking.vertical_mask(feat)
        style_vertical = self.masked_pool(feat_v, img_len)

        # ===== HORIZONTAL MASKING =====
        feat_h = self.masking.horizontal_mask(feat)
        style_horizontal = self.masked_pool(feat_h, img_len)

        # ===== FUSION =====
        style = torch.cat([style_global, style_vertical, style_horizontal], dim=1)
        style = self.fusion(style)

        style = self.linear_style(style)

        mu = self.mu(style)

        if vae_mode:
            logvar = self.logvar(style)
            style = self.reparameterize(mu, logvar)
            style = (style, mu, logvar)
        else:
            style = mu

        if ret_feats:
            return style, all_feats
        else:
            return style

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

class WriterIdentifier(nn.Module):
    def __init__(self, n_writer=372, in_dim=256, init='N02'):
        super(WriterIdentifier, self).__init__()
        self.reduce_len_scale = 32

        self.linear_wid = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, n_writer),
        )

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone, ret_feats=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        img_len = img_len // cnn_backbone.reduce_len_scale

        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()
        wid_feat = (feat * img_len_mask).sum(dim=-1) / (img_len.unsqueeze(1).float() + 1e-8)

        wid_logits = self.linear_wid(wid_feat)

        if ret_feats:
            return wid_logits, all_feats
        else:
            return wid_logits
