"""Two-channel ResNet34 U-Net; no external segmentation dependency."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU(),
        nn.Conv2d(outputs, outputs, 3, padding=1, bias=False), nn.GroupNorm(8, outputs), nn.ReLU())


class InstanceUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet34(weights=None, norm_layer=lambda c: nn.GroupNorm(8, c))
        self.encoder.conv1 = nn.Conv2d(2, 64, 7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()
        self.shallow = block(2, 32)
        self.decode = nn.ModuleList([block(768, 256), block(384, 128), block(192, 64),
                                    block(128, 64), block(96, 32)])
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e = self.encoder
        s0 = self.shallow(x)
        s1 = e.relu(e.bn1(e.conv1(x)))
        s2 = e.layer1(e.maxpool(s1))
        s3 = e.layer2(s2)
        s4 = e.layer3(s3)
        x = e.layer4(s4)
        for decoder, skip in zip(self.decode, [s4, s3, s2, s1, s0]):
            x = decoder(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear',
                                                 align_corners=False), skip], dim=1))
        return self.head(x)


def mask_loss(logits, target, valid):
    logits, target, valid = logits.float(), target.float(), valid.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*valid).sum()/valid.sum()
    p = logits.sigmoid()*valid
    target = target*valid
    dims = tuple(range(1, p.ndim))
    dice = (2*(p*target).sum(dims)+1)/(p.sum(dims)+target.sum(dims)+1)
    return bce + (1-dice).mean()
