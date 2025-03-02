import json
import torch.nn as nn
import torch
from models import bricks
import torch.nn.functional as F
from timm.models import register_model
from models.mscan import MSCAN


"""
[batch_size, in_channels, height, width] -> [batch_size, out_channels, height // 4, width // 4]
"""



class Hamburger(nn.Module):

    def __init__(
            self,
            hamburger_channels=256,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(Hamburger, self).__init__()
        self.ham_in = bricks.ConvModule(hamburger_channels, hamburger_channels, bias=True, num_groups=0)


        self.ham = bricks.NMF2D(args=nmf2d_config)

        self.ham_out = bricks.ConvModule(hamburger_channels, hamburger_channels)


    def forward(self, x):
        out = self.ham_in(x)
        out = F.relu(out, inplace=True)
        out = self.ham(out)
        out = self.ham_out(out)
        out = F.relu(x + out, inplace=True)
        return out


class LightHamHead(nn.Module):

    def __init__(
            self,
            in_channels_list=[64, 160, 256],
            hidden_channels=256,
            out_channels=256,
            num_classes=150,
            drop_prob=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(LightHamHead, self).__init__()

        self.conv_seg = nn.Conv2d(
                in_channels=out_channels,
                out_channels=num_classes,
                kernel_size=(1, 1))

        self.squeeze = bricks.ConvModule(
                                        in_channels=sum(in_channels_list),
                                        out_channels=hidden_channels
        )


        self.hamburger = Hamburger(
            hamburger_channels=hidden_channels,
            nmf2d_config=nmf2d_config
        )

        self.align = bricks.ConvModule(
                                        in_channels=hidden_channels,
                                        out_channels=out_channels
        )


    # inputs: [x, x_1, x_2, x_3]
    # x: [batch_size, channels, height, width]
    def forward(self, inputs):
        assert len(inputs) >= 2
        o = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)
        inputs = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]

        # x: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        x = torch.cat(inputs, dim=1)
        # out: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        out = self.squeeze(x)
        out = self.hamburger(out)
        out = self.align(out)

        # out: [batch_size, num_classes, standard_height, standard_width]
        out = self.conv_seg(out)
        _, _, original_height, original_width = o.shape
        # out: [batch_size, num_classes, original_height, original_width]
        out = F.interpolate(
            input=out,
            size=(original_height, original_width),
            mode="bilinear",
            align_corners=False
        )
        # print('*********************')
        # print(out.view(batch_size, -1, original_height * original_width).shape)
        # out = torch.transpose(out.view(batch_size, -1, original_height * original_width), -2, -1)
        # out = out.view(batch_size, -1, original_height, original_width)
        return out


class SegNeXt(nn.Module):

    def __init__(
            self,
            embed_dims=[32, 64, 160, 256],
            expand_rations=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob_of_encoder=0.1,
            drop_path_prob=0.1,
            hidden_channels=256,
            out_channels=256,
            num_classes=19,
            drop_prob_of_decoder=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            ),
            **kwargs
    ):
        super(SegNeXt, self).__init__()

        self.backbone = MSCAN(
            in_chans=3,
            embed_dims=embed_dims,
            mlp_ratios=expand_rations,
            depths=depths,
            drop_rate=drop_prob_of_encoder,
            drop_path_rate=drop_path_prob
        )

        self.decode_head = LightHamHead(
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_classes=num_classes,
            drop_prob=drop_prob_of_decoder,
            nmf2d_config=nmf2d_config
        )

    def forward(self, x):
        out = self.backbone(x)
        out = self.decode_head(out)
        out = F.interpolate(out, size=x.size()[-2:], mode='bilinear', align_corners=True)
        return out


@register_model
def SegNeXt_T(num_classes, pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    embed_dims = [32, 64, 160, 256]
    expand_rations = [8, 8, 4, 4]
    depths = [3, 3, 5, 2]
    hidden_channels = 256
    out_channels = 256

    net = SegNeXt(embed_dims=embed_dims, expand_rations=expand_rations,
                  depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                  num_classes=num_classes, **kwargs)
    return net


@register_model
def SegNeXt_S(num_classes, pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    embed_dims = [64, 128, 320, 512]
    expand_rations = [8, 8, 4, 4]
    depths = [2, 2, 4, 2]
    hidden_channels = 256
    out_channels = 256

    net = SegNeXt(embed_dims=embed_dims, expand_rations=expand_rations,
                  depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                  num_classes=num_classes, **kwargs)
    return net

@register_model
def SegNeXt_B(num_classes, pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    embed_dims = [64, 128, 320, 512]
    expand_rations = [8, 8, 4, 4]
    depths = [3, 3, 12, 3]
    hidden_channels = 512
    out_channels = 512

    net = SegNeXt(embed_dims=embed_dims, expand_rations=expand_rations,
                  depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                  num_classes=num_classes, **kwargs)
    return net


@register_model
def SegNeXt_L(num_classes, pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    embed_dims = [64, 128, 320, 512]
    expand_rations = [8, 8, 4, 4]
    depths = [3, 5, 27, 3]
    hidden_channels = 1024
    out_channels = 1024

    net = SegNeXt(embed_dims=embed_dims, expand_rations=expand_rations,
                  depths=depths, hidden_channels=hidden_channels, out_channels=out_channels,
                  num_classes=num_classes, **kwargs)
    return net

# if __name__ == '__main__':
#     from torchinfo import summary
#     net = SegNeXt_S(num_classes=19).cuda()
#     x = torch.randn(2, 3, 1024, 1024).cuda()
#     y = net(x)
#     print(y.shape)
    # summary(net, input_size=(1, 3, 512, 512))
    # for name, param in net.named_parameters():
    #     print(name)

