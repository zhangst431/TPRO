import torch
import torch.nn as nn
import torch.nn.functional as F

from seg_network.segformer_head import SegFormerHead
from seg_network import mix_transformer


class SegNetwork(nn.Module):
    def __init__(self, backbone, seg_num_classes=None, embedding_dim=256, stride=None, pretrained=None, pooling="gap"):
        super().__init__()
        self.seg_num_classes = seg_num_classes
        self.embedding_dim = embedding_dim
        self.feature_strides = [4, 8, 16, 32]
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        ## initilize encoder
        if pretrained:
            state_dict = torch.load('./pretrained/'+backbone+'.pth', map_location="cpu")
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            state_dict = {k: v for k, v in state_dict.items() if k in self.encoder.state_dict().keys()}
            self.encoder.load_state_dict(state_dict, strict=False)

        if pooling=="gmp":
            self.pooling = F.adaptive_max_pool2d
        elif pooling=="gap":
            self.pooling = F.adaptive_avg_pool2d

        self.decoder = SegFormerHead(feature_strides=self.feature_strides, in_channels=self.in_channels, embedding_dim=self.embedding_dim, num_classes=self.seg_num_classes)

    def get_param_groups(self):
        regularized = []
        not_regularized = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # we do not regularize biases nor Norm parameters
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            else:
                regularized.append(param)
        return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

    def forward(self, x):
        _x, _attns = self.encoder(x)

        seg = self.decoder(_x)
        # seg = self.decoder(_x4)

        return None, seg, _attns
    