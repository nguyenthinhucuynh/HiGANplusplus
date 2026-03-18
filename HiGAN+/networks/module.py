import numpy as np
import torch
from torch import nn
from networks.block import Conv2dBlock, ActFirstResBlock, DeepBLSTM, DeepGRU, DeepLSTM, Identity
from networks.utils import _len2mask, init_weights


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

        self.linear_style = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
            nn.Linear(in_dim, in_dim),
            nn.LeakyReLU(),
        )

        self.mu_v = nn.Linear(in_dim, style_dim // 2)
        self.mu_h = nn.Linear(in_dim, style_dim // 2)

        self.logvar_v = nn.Linear(in_dim, style_dim // 2)
        self.logvar_h = nn.Linear(in_dim, style_dim // 2)

        if init != 'none':
            init_weights(self, init)

    def forward(self, img, img_len, cnn_backbone=None, ret_feats=False, vae_mode=False):
        feat, all_feats = cnn_backbone(img, ret_feats)
        # feat: (B, C, W)

        img_len = img_len // cnn_backbone.reduce_len_scale
        img_len_mask = _len2mask(img_len, feat.size(-1)).unsqueeze(1).float().detach()

        B, C, W = feat.shape

        # mask theo chiều width
        mask_line = (torch.rand(B, 1, W, device=feat.device) > 0.3).float()
        mask_line = mask_line * img_len_mask

        feat_v = feat * mask_line
        feat_h = feat * (img_len_mask - mask_line)

        len_v = mask_line.sum(dim=-1) + 1e-8
        len_h = (img_len_mask - mask_line).sum(dim=-1) + 1e-8

        style_v_base = feat_v.sum(dim=-1) / len_v
        style_h_base = feat_h.sum(dim=-1) / len_h

        style_v = self.linear_style(style_v_base)
        style_h = self.linear_style(style_h_base)

        style_v = style_v + 0.01 * torch.randn_like(style_v)
        style_h = style_h + 0.01 * torch.randn_like(style_h)

        mu_v = self.mu_v(style_v)
        mu_h = self.mu_h(style_h)

        mu = 0.5 * (mu_v + mu_h)

        if vae_mode:
            logvar_v = self.logvar_v(style_v)
            logvar_h = self.logvar_h(style_h)
            logvar = logvar_v + logvar_h

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
