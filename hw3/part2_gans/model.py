import torch
from torch import nn
from torch.nn import functional as F
import functools
import math
from torch.nn.utils import spectral_norm
from torch.nn.utils.spectral_norm import SpectralNorm


class AdaptiveBatchNorm(nn.BatchNorm2d):
    """
    Adaptive batch normalization layer (4 points)

    Args:
        num_features: number of features in batch normalization layer
        embed_features: number of features in embeddings

    The base layer (BatchNorm2d) is applied to "inputs" with affine = False

    After that, the "embeds" are linearly mapped to "gamma" and "bias"
    
    These "gamma" and "bias" are applied to the outputs like in batch normalization
    with affine = True (see definition of batch normalization for reference)
    """
    def __init__(self, num_features: int, embed_features: int):
        super(AdaptiveBatchNorm, self).__init__(num_features, affine=False)
        self.adaptive_gamma = nn.Linear(embed_features, num_features)
        self.adaptive_bias = nn.Linear(embed_features, num_features)

    def forward(self, inputs, embeds):
        gamma = self.adaptive_gamma(embeds) # TODO
        bias = self.adaptive_bias(embeds) # TODO

        assert gamma.shape[0] == inputs.shape[0] and gamma.shape[1] == inputs.shape[1]
        assert bias.shape[0] == inputs.shape[0] and bias.shape[1] == inputs.shape[1]

        outputs = super().forward(inputs) # TODO: apply batchnorm

        return outputs * gamma[..., None, None] + bias[..., None, None]


class PreActResBlock(nn.Module):
    """
    Pre-activation residual block (6 points)

    Paper: https://arxiv.org/pdf/1603.05027.pdf
    Scheme: materials/preactresblock.png
    Review: https://towardsdatascience.com/resnet-with-identity-mapping-over-1000-layers-reached-image-classification-bb50a42af03e

    Args:
        in_channels: input number of channels
        out_channels: output number of channels
        batchnorm: this block is with/without adaptive batch normalization
        upsample: use nearest neighbours upsampling at the beginning
        downsample: use average pooling after the end

    in_channels != out_channels:
        - first conv: in_channels -> out_channels
        - second conv: out_channels -> out_channels
        - use 1x1 conv in skip connection

    in_channels == out_channels: skip connection is without a conv
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int = None,
                 batchnorm: bool = False,
                 upsample: bool = False,
                 downsample: bool = False):
        super(PreActResBlock, self).__init__()
        # TODO: define pre-activation residual block
        # TODO: apply spectral normalization to conv layers
        # Don't forget that activation after residual sum cannot be inplace!

        self.norm1 = AdaptiveBatchNorm(in_channels, embed_channels) if batchnorm else None
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=batchnorm)
        self.norm2 = AdaptiveBatchNorm(out_channels, embed_channels) if batchnorm else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=batchnorm)

        self.batchnorm = batchnorm
        self.upsample = upsample
        self.downsample = downsample
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0) if in_channels != out_channels else None

    def forward(self, 
                inputs, # regular features 
                embeds=None): # embeds used in adaptive batch norm

        size = torch.tensor(inputs.shape[2:])
        if self.upsample:
            inputs = F.interpolate(inputs, tuple(2 * size), mode='nearest')

        out = inputs
        if self.batchnorm:
            out = self.norm1(out, embeds)

        out = F.relu(out)
        out = self.conv1(out)
        if self.batchnorm:
            out = self.norm2(out, embeds)

        out = F.relu(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.adaptive_avg_pool2d(out, size)

        shortcut = self.shortcut(inputs) if self.shortcut is not None else inputs
        out = out + shortcut

        if self.downsample:
            out = F.adaptive_avg_pool2d(out, size // 2)

        outputs = out
        return outputs


class Generator(nn.Module):
    """
    Generator network (8 points)
    
    TODO:

      - Implement an option to condition the synthesis on trainable class embeddings
        (use nn.Embedding module with noise_channels as the size of each embed)

      - Concatenate input noise with class embeddings (if use_class_condition = True) to obtain input embeddings

      - Linearly map input embeddings into input tensor with the following dims: max_channels x 4 x 4

      - Forward an input tensor through a convolutional part, 
        which consists of num_blocks PreActResBlocks and performs upsampling by a factor of 2 in each block

      - Each PreActResBlock is additionally conditioned on the input embeddings (via adaptive batch normalization)

      - At the end of the convolutional part apply regular BN, ReLU and Conv as an image prediction head

      - Apply spectral norm to all conv and linear layers (not the embedding layer)

      - Use Sigmoid at the end to map the outputs into an image

    Notes:

      - The last convolutional layer should map min_channels to 3. With each upsampling you should decrease
        the number of channels by a factor of 2

      - Class embeddings are only used and trained if use_class_condition = True
    """    
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 noise_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_class_condition: bool):
        super(Generator, self).__init__()
        self.output_size = 4 * 2**num_blocks
        # TODO
        self.embedding = nn.Embedding(num_classes, noise_channels) if use_class_condition else None
        self.map = nn.Linear(2 * noise_channels if use_class_condition else noise_channels, max_channels * 4 * 4)
        self.max_channels = max_channels
        embed_channels = 2 * noise_channels if use_class_condition else None
        self.conv_part = nn.ModuleList([
            PreActResBlock(max_channels // 2 ** i, max_channels // 2 ** (i + 1),
                           embed_channels=embed_channels, batchnorm=use_class_condition, upsample=True)
            for i in range(num_blocks)]
        )

        self.head = nn.Sequential(
            nn.BatchNorm2d(max_channels // 2 ** num_blocks),
            nn.ReLU(),
            nn.Conv2d(min_channels, 3, 3, 1, 1),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                SpectralNorm.apply(m, name='weight', n_power_iterations=1, eps=1e-12, dim=0)

    def forward(self, noise, labels):
        input_embedding = torch.cat([noise, self.embedding(labels)], dim=1) if self.embedding is not None else noise
        out = self.map(input_embedding)
        out = out.view(noise.shape[0], self.max_channels, 4, 4)
        for block in self.conv_part:
            out = block(out, input_embedding)

        out = self.head(out)
        outputs = torch.sigmoid(out)

        assert outputs.shape == (noise.shape[0], 3, self.output_size, self.output_size)
        return outputs


class Discriminator(nn.Module):
    """
    Discriminator network (8 points)

    TODO:
    
      - Define a convolutional part of the discriminator similarly to
        the generator blocks, but in the inverse order, with downsampling, and
        without batch normalization
    
      - At the end of the convolutional part apply ReLU and sum pooling
    
    TODO: implement projection discriminator head (https://arxiv.org/abs/1802.05637)
    
    Scheme: materials/prgan.png
    
    Notation:
    
      - phi is a convolutional part of the discriminator
    
      - psi is a vector
    
      - y is a class embedding
    
    Class embeddings matrix is similar to the generator, shape: num_classes x max_channels

    Discriminator outputs a B x 1 matrix of realism scores

    Apply spectral norm for all layers (conv, linear, embedding)
    """
    def __init__(self, 
                 min_channels: int, 
                 max_channels: int,
                 num_classes: int,
                 num_blocks: int,
                 use_projection_head: bool):
        super(Discriminator, self).__init__()
        # TODO

        layers = [
            nn.Conv2d(3, min_channels, 3, 1, 1),
            nn.BatchNorm2d(min_channels),
            nn.ReLU()
        ]
        layers.extend([PreActResBlock(min_channels * 2 ** i, min_channels * 2 ** (i + 1),
                                      embed_channels=None, downsample=True) for i in range(num_blocks)])
        self.phi = nn.Sequential(*layers)
        self.psi = nn.Linear(max_channels, 1)
        self.use_projection_head = use_projection_head
        self.embedding = nn.Embedding(num_classes, max_channels) if use_projection_head else None
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding)):
                SpectralNorm.apply(m, name='weight', n_power_iterations=1, eps=1e-12, dim=0)

    def forward(self, inputs, labels):
        out = F.relu(self.phi(inputs)).sum(dim=(2, 3)) # TODO
        scores = self.psi(out)
        if self.use_projection_head:
            scores *= (self.embedding(labels) * out).sum(dim=1, keepdims=True) # TODO

        scores = scores.squeeze(1)
        assert scores.shape == (inputs.shape[0],)
        return scores