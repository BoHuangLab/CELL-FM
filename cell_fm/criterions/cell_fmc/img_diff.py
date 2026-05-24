import torch.nn as nn


class ImgDiffCriterion(nn.Module):
    def __init__(self, args, reduction="none"):
        super().__init__()
        self.args = args
        self.img_diff_loss_coeff = args.img_diff_loss_coeff

    def forward(self, batch_data, model_output):
        img_diff_loss = model_output[0] * self.img_diff_loss_coeff
        log_loss = {
            'total_loss': img_diff_loss.item(),
            'img_diff_loss': img_diff_loss.item() / self.img_diff_loss_coeff,
        }
        return img_diff_loss, log_loss
