import pickle as pkl
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from cls_network import mix_transformer
from cls_network.attention import Block



class AdaptiveLayer(nn.Module):
    def __init__(self, in_dim, n_ratio, out_dim):
        super().__init__()
        hidden_dim = int(in_dim * n_ratio)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ClsNetwork(nn.Module):
    def __init__(self,
                 backbone='mit_b1',
                 cls_num_classes=4,
                 stride=[4, 2, 2, 1],
                 pretrained=True,
                 n_ratio=0.5,
                 k_fea_path=None,
                 l_fea_path=None):
        super().__init__()
        self.cls_num_classes = cls_num_classes
        self.stride = stride

        self.encoder = getattr(mix_transformer, backbone)(stride=self.stride)
        self.in_channels = self.encoder.embed_dims

        # initialize encoder
        if pretrained:
            state_dict = torch.load('./pretrained/'+backbone+'.pth', map_location="cpu")
            state_dict.pop('head.weight')
            state_dict.pop('head.bias')
            state_dict = {k: v for k, v in state_dict.items() if k in self.encoder.state_dict().keys()}
            self.encoder.load_state_dict(state_dict, strict=False)

        self.pooling = F.adaptive_avg_pool2d

        ## medclip
        self.l_fc1 = AdaptiveLayer(512, n_ratio, self.in_channels[0])
        self.l_fc2 = AdaptiveLayer(512, n_ratio, self.in_channels[1])
        self.l_fc3 = AdaptiveLayer(512, n_ratio, self.in_channels[2])
        self.l_fc4 = AdaptiveLayer(512, n_ratio, self.in_channels[3])

        with open("./text&features/text_features/{}.pkl".format(l_fea_path), "rb") as lf:
            self.l_fea = pkl.load(lf).cpu()
        self.logit_scale1 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale2 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale3 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)
        self.logit_scale4 = nn.parameter.Parameter(torch.ones([1]) * 1 / 0.07)

        ## knowledge attention
        with open("./text&features/text_features/{}.pkl".format(k_fea_path), "rb") as kf:
            self.k_fea = pkl.load(kf).cpu()
        self.k_fc4 = AdaptiveLayer(self.k_fea.shape[-1], n_ratio, self.in_channels[3])
        self.ka4 = nn.ModuleList([Block(self.in_channels[3], 8, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop=0,
                                        attn_drop=0, drop_path=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6)) for _ in range(2)])

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

        logit_scale1 = self.logit_scale1
        logit_scale2 = self.logit_scale2
        logit_scale3 = self.logit_scale3
        logit_scale4 = self.logit_scale4

        imshape = [_.shape for _ in _x]
        image_features = [_.permute(0, 2, 3, 1).reshape(-1, _.shape[1]) for _ in _x]   
        _x1, _x2, _x3, _x4 = image_features
        l_fea = self.l_fea.to(x.device)
        l_fea1 = self.l_fc1(l_fea)
        l_fea2 = self.l_fc2(l_fea)
        l_fea3 = self.l_fc3(l_fea)
        l_fea4 = self.l_fc4(l_fea)
        _x1 = _x1 / _x1.norm(dim=-1, keepdim=True)
        logits_per_image1 = logit_scale1 * _x1 @ l_fea1.t().float() 
        out1 = logits_per_image1.view(imshape[0][0], imshape[0][2], imshape[0][3], -1).permute(0, 3, 1, 2) 
        cam1 = out1.clone().detach()
        cls1 = self.pooling(out1, (1, 1)).view(-1, l_fea1.shape[0]) 

        _x2 = _x2 / _x2.norm(dim=-1, keepdim=True)
        logits_per_image2 = logit_scale2 * _x2 @ l_fea2.t().float() 
        out2 = logits_per_image2.view(imshape[1][0], imshape[1][2], imshape[1][3], -1).permute(0, 3, 1, 2) 
        cam2 = out2.clone().detach()
        cls2 = self.pooling(out2, (1, 1)).view(-1, l_fea2.shape[0]) 

        _x3 = _x3 / _x3.norm(dim=-1, keepdim=True)
        logits_per_image3 = logit_scale3 * _x3 @ l_fea3.t().float() 
        out3 = logits_per_image3.view(imshape[2][0], imshape[2][2], imshape[2][3], -1).permute(0, 3, 1, 2) 
        cam3 = out3.clone().detach()
        cls3 = self.pooling(out3, (1, 1)).view(-1, l_fea3.shape[0]) 

        k_fea = self.k_fea.to(x.device)
        k_fea4 = self.k_fc4(k_fea)
        k_fea4 = k_fea4.reshape(1, k_fea4.shape[0], k_fea4.shape[1])
        k_fea4 = k_fea4.repeat(imshape[3][0], 1, 1)
        _x4 = _x4.reshape(imshape[3][0], -1, imshape[3][1])
        _z4 = torch.cat((_x4, k_fea4), dim=1)
        for blk in self.ka4:
            _z4, attn = blk(_z4, imshape[3][2], imshape[3][3])
        _x4 = _z4[:, :imshape[3][2] * imshape[3][3], :]
        _x4 = _x4.reshape(-1, imshape[3][1])
        _x4 = _x4 / _x4.norm(dim=-1, keepdim=True)
        logits_per_image4 = logit_scale4 * _x4 @ l_fea4.t().float() 
        out4 = logits_per_image4.view(imshape[3][0], imshape[3][2], imshape[3][3], -1).permute(0, 3, 1, 2) 
        cam4 = out4.clone().detach()
        cls4 = self.pooling(out4, (1, 1)).view(-1, l_fea4.shape[0]) 

        return cls1, cam1, cls2, cam2, cls3, cam3, cls4, cam4, _attns
