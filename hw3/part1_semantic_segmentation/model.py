import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


class ConvBNRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ConvBNRelu, self).__init__(*layers)


class DownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        layers = [
            ConvBNRelu(in_channels, out_channels),
            ConvBNRelu(out_channels, out_channels)
        ]
        super(DownBlock, self).__init__(*layers)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 3, 2, 1, 1)
        self.basic_block = nn.Sequential(
            nn.Dropout(p=0.5),
            DownBlock(2 * in_channels, out_channels)
        )

    def forward(self, left, right):
        right = self.upsample(right)
        out = torch.cat([left, right], dim=1)
        out = self.basic_block(out)

        return out


class UNet(nn.Module):
    """
    TODO: 8 points

    A standard UNet network (with padding in covs).

    For reference, see the scheme in materials/unet.png
    - Use batch norm between conv and relu
    - Use max pooling for downsampling
    - Use conv transpose with kernel size = 3, stride = 2, padding = 1, and output padding = 1 for upsampling
    - Use 0.5 dropout after concat

    Args:
      - num_classes: number of output classes
      - min_channels: minimum number of channels in conv layers
      - max_channels: number of channels in the bottleneck block
      - num_down_blocks: number of blocks which end with downsampling

    The full architecture includes downsampling blocks, a bottleneck block and upsampling blocks

    You also need to account for inputs which size does not divide 2**num_down_blocks:
    interpolate them before feeding into the blocks to the nearest size which divides 2**num_down_blocks,
    and interpolate output logits back to the original shape
    """
    def __init__(self, 
                 num_classes,
                 min_channels=32,
                 max_channels=512, 
                 num_down_blocks=4):

        super(UNet, self).__init__()
        self.down_path = nn.ModuleList([nn.Sequential(ConvBNRelu(3, min_channels), ConvBNRelu(min_channels, 2 * min_channels))])
        self.down_path.extend([DownBlock(min_channels * 2 ** i, min_channels * 2 ** (i + 1)) for i in range(1, num_down_blocks)])
        self.up_path = nn.ModuleList([UpBlock(max_channels // 2 ** i, max_channels // 2 ** (i + 1)) for i in range(num_down_blocks)])
        self.bridge = DownBlock(max_channels, max_channels)
        self.num_classes = num_classes
        self.conv = nn.Conv2d(min_channels, num_classes, 1, 1, 0)
        self.num_down_blocks = num_down_blocks

    def forward(self, inputs):
        shape = inputs.shape[2:]
        x = F.interpolate(inputs, self.find_shape(shape), mode='bilinear', align_corners=False)

        down = []
        for layer in self.down_path:
            x = layer(x)
            down.append(x)
            x = F.max_pool2d(x, 2, 2)

        right = self.bridge(x)

        for layer, left in zip(self.up_path, reversed(down)):
            right = layer(left, right)

        x = self.conv(right)
        logits = F.interpolate(x, shape, mode='bilinear', align_corners=False) # TODO

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits

    def find_shape(self, shape):
        n = 2 ** self.num_down_blocks
        h, w = shape
        h = h if h % n == 0 else ((h // n) + 1) * n
        w = w if w % n == 0 else ((w // n) + 1) * n
        return h, w


class DeepLab(nn.Module):
    """
    TODO: 6 points

    (simplified) DeepLab segmentation network.
    
    Args:
      - backbone: ['resnet18', 'vgg11_bn', 'mobilenet_v3_small'],
      - aspp: use aspp module
      - num classes: num output classes

    During forward pass:
      - Pass inputs through the backbone to obtain features
      - Apply ASPP (if needed)
      - Apply head
      - Upsample logits back to the shape of the inputs
    """
    def __init__(self, backbone, aspp, num_classes):
        super(DeepLab, self).__init__()
        self.backbone = backbone
        self.init_backbone()

        if aspp:
            self.aspp = ASPP(self.out_features, 256, [12, 24, 36])

        self.head = DeepLabHead(self.out_features, num_classes)
        self.num_classes = num_classes

    def init_backbone(self):
        # TODO: initialize an ImageNet-pretrained backbone
        if self.backbone == 'resnet18':
            self.out_features = 512 # TODO: number of output features in the backbone
            self.backbone_ = models.resnet18(pretrained=True)

        elif self.backbone == 'vgg11_bn':
            self.out_features = 512 # TODO
            self.backbone_ = models.vgg11_bn(pretrained=True)

        elif self.backbone == 'mobilenet_v3_small':
            self.out_features = 576  # TODO
            self.backbone_ = models.mobilenet_v3_small(pretrained=True)

    def _forward(self, x):
        # TODO: forward pass through the backbone
        backbone = self.backbone_
        if self.backbone == 'resnet18':
            x = backbone.conv1(x)
            x = backbone.bn1(x)
            x = backbone.relu(x)
            x = backbone.maxpool(x)

            x = backbone.layer1(x)
            x = backbone.layer2(x)
            x = backbone.layer3(x)
            x = backbone.layer4(x)

        elif self.backbone == 'vgg11_bn':
            x = backbone.features(x)

        elif self.backbone == 'mobilenet_v3_small':
            x = backbone.features(x)

        return x

    def forward(self, inputs):
        x = self._forward(inputs) # TODO
        if hasattr(self, 'aspp'):
            x = self.aspp(x)
        x = self.head(x)
        logits = F.interpolate(x, inputs.shape[2:], mode='bilinear', align_corners=False)

        assert logits.shape == (inputs.shape[0], self.num_classes, inputs.shape[2], inputs.shape[3]), 'Wrong shape of the logits'
        return logits


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, 1)
        )


class ASPP(nn.Module):
    """
    TODO: 8 points

    Atrous Spatial Pyramid Pooling module
    with given atrous_rates and out_channels for each head
    Description: https://paperswithcode.com/method/aspp
    
    Detailed scheme: materials/deeplabv3.png
      - "Rates" are defined by atrous_rates
      - "Conv" denotes a Conv-BN-ReLU block
      - "Image pooling" denotes a global average pooling, followed by a 1x1 "conv" block and bilinear upsampling
      - The last layer of ASPP block should be Dropout with p = 0.5

    Args:
      - in_channels: number of input and output channels
      - num_channels: number of output channels in each intermediate "conv" block
      - atrous_rates: a list with dilation values
    """
    def __init__(self, in_channels, num_channels, atrous_rates):
        super(ASPP, self).__init__()
        modules = list()
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU())
        )

        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, num_channels, rate))

        modules.append(ASPPPooling(in_channels, num_channels))

        self.convs = nn.ModuleList(modules)

        self.head = nn.Sequential(
            nn.Conv2d(len(self.convs) * num_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # TODO: forward pass through the ASPP module
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        res = self.head(res)

        assert res.shape[1] == x.shape[1], 'Wrong number of output channels'
        assert res.shape[2] == x.shape[2] and res.shape[3] == x.shape[3], 'Wrong spatial size'
        return res


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        layers = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*layers)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        for layer in self:
            x = layer(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
