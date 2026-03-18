import torch

class StyleMasking:
    def __init__(self):
        pass

    @staticmethod
    def vertical_mask(feat, ratio=0.5):
        """
        Mask along WIDTH → preserve vertical info
        feat: [B, C, W]
        """
        B, C, W = feat.shape
        keep = int(W * ratio)

        noise = torch.rand(B, W, device=feat.device)
        ids = torch.argsort(noise, dim=1)[:, :keep]

        mask = torch.zeros(B, W, device=feat.device)
        mask.scatter_(1, ids, 1.0)

        return feat * mask.unsqueeze(1)

    @staticmethod
    def horizontal_mask(feat, ratio=0.5):
        """
        Mask along CHANNEL → simulate height masking
        (since HiGAN+ collapsed H dimension)
        """
        B, C, W = feat.shape
        keep = int(C * ratio)

        noise = torch.rand(B, C, device=feat.device)
        ids = torch.argsort(noise, dim=1)[:, :keep]

        mask = torch.zeros(B, C, device=feat.device)
        mask.scatter_(1, ids, 1.0)

        return feat * mask.unsqueeze(-1)
