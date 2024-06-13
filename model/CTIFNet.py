import torch
from functools import partial
import torch.nn as nn
from model.VisionTransformer import VisionTransformer
import math
import torchvision
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class conv_upsample(nn.Module):
    def __init__(self, channel,outchannel,kernel_size=1):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, outchannel, kernel_size)

    def forward(self, x, targetSize):

        if type(targetSize) == list:
            if x.size()[2:] != targetSize[2:]:
                # print("targetSize is ",targetSize)
                x = self.conv(F.interpolate(x, size=targetSize[2:], mode='bilinear', align_corners=True))
        else:
            if x.size()[2:] != targetSize.size()[2:]:
                 x = self.conv(F.interpolate(x, size=targetSize.size()[2:], mode='bilinear', align_corners=True))

        return x


# simple feature pyramid from Exploring Plain Vision Transformer Backbones for Object Detection
class  Plain_FPN(nn.Module):

    def __init__(self,inchannel=768,outchannel=768):
        super(Plain_FPN, self).__init__()
        self.conv1 = BasicConv2d(inchannel,outchannel,kernel_size=3,stride=2,padding=1)
        self.conv2 = BasicConv2d(inchannel,outchannel,kernel_size=1,stride=1)
        self.up_conv3 = conv_upsample(inchannel,outchannel)
        self.up_conv4 = conv_upsample(inchannel,outchannel)

    def forward(self,x):

        b,c,h,w = x.shape
        x4 = self.conv1(x)
        x3 = self.conv2(x)
        x2 = self.up_conv3(x,[b,c,2*h,2*w])
        x1 = self.up_conv4(x,[b,c,4*h,4*w])

        return x1,x2,x3,x4


class UNetLikeConcat(nn.Module):
    def __init__(self, channel,return_middle_layer=False):
        super(UNetLikeConcat, self).__init__()
        self.conv_upsample1 = conv_upsample(channel, channel,kernel_size=1)
        self.conv_upsample2 = conv_upsample(channel, channel,kernel_size=1)
        self.conv_upsample3 = conv_upsample(channel, channel,kernel_size=1)


        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2 * channel, 2 * channel, 3, padding=1),
            BasicConv2d(2 * channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )
        self.return_middle_layer = return_middle_layer

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(x4,x3)), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(x3,x2)), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(x2,x1)), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)

        if self.return_middle_layer:
            return x, x1, x2, x3, x4
        else:
            return x


class StructureFeatureEnhancementModule(nn.Module):
    def __init__(self,channel):
        super(StructureFeatureEnhancementModule, self).__init__()
        self.conv_upsample1 = conv_upsample(channel,channel)
        self.conv_upsample2 = conv_upsample(channel,channel)
        self.conv_upsample3 = conv_upsample(channel,channel)
        self.conv_upsample4 = conv_upsample(channel,channel)
        self.conv_upsample5 = conv_upsample(channel,channel)
        self.conv_upsample6 = conv_upsample(channel,channel)
        self.conv_upsample7 = conv_upsample(channel,channel)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(5 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(4 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(3 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(2 * channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self,t_1, t_2, t_3, t_4, f_1, f_2, f_3, f_4):
        t1_1 = t_1 + self.conv_f1(torch.cat((t_1, f_1,
                                               self.conv_upsample1(f_2, t_1),
                                               self.conv_upsample2(f_3, t_1),
                                               self.conv_upsample3(f_4, t_1)), 1))
        t1_2 = t_2 + self.conv_f2(torch.cat((t_2, f_2,
                                               self.conv_upsample4(f_3, t_2),
                                               self.conv_upsample5(f_4, t_2)), 1))
        t1_3 = t_3 + self.conv_f3(torch.cat((t_3, f_3,
                                               self.conv_upsample6(f_4, t_3)), 1))
        t1_4 = t_4 + self.conv_f4(torch.cat((f_4, t_4), 1))

        return t1_1,t1_2,t1_3,t1_4

class GlobalContextEnhancementModule(nn.Module):
    def __init__(self,channel):
        super(GlobalContextEnhancementModule, self).__init__()
        self.conv_upsample1 = conv_upsample(channel,channel)
        self.conv_upsample2 = conv_upsample(channel,channel)
        self.conv_upsample3 = conv_upsample(channel,channel)
        self.conv_upsample4 = conv_upsample(channel,channel)
        self.conv_upsample5 = conv_upsample(channel,channel)
        self.conv_upsample6 = conv_upsample(channel,channel)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
    def forward(self,t1_1, t1_2, t1_3, t1_4, f_1, f_2, f_3, f_4):

        f1_1 = f_1 + self.conv_f1(f_1 * t1_1 *
                                    self.conv_upsample1(t1_2, f_1) *
                                    self.conv_upsample2(t1_3, f_1) *
                                    self.conv_upsample3(t1_4, f_1))

        f1_2 = f_2 + self.conv_f2(f_2 * t1_2 *
                                    self.conv_upsample4(t1_3, f_2) *
                                    self.conv_upsample5(t1_4, f_2))

        f1_3 = f_3 + self.conv_f3(f_3 * t1_3 *
                                    self.conv_upsample6(t1_4, f_3))

        f1_4 = f_4 + self.conv_f4(f_4 * t1_4)

        return f1_1,f1_2,f1_3,f1_4

class FeatureEnhancementUnit(nn.Module):

    def __init__(self, channel):
        super(FeatureEnhancementUnit, self).__init__()
        self.sfem =  StructureFeatureEnhancementModule(channel)
        self.gcem =  GlobalContextEnhancementModule(channel)

    def forward(self, t_1, t_2, t_3, t_4, f_1, f_2, f_3, f_4):
        t1_1,t1_2,t1_3,t1_4 = self.sfem(t_1, t_2, t_3, t_4, f_1, f_2, f_3, f_4)

        f1_1, f1_2, f1_3, f1_4 = self.gcem(t1_1,t1_2,t1_3,t1_4, f_1, f_2, f_3, f_4)

        return t1_1, t1_2, t1_3, t1_4, f1_1, f1_2, f1_3, f1_4


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)

class TransformerConvertor(nn.Module):
    def __init__(self,inputChannel,outputChannel):
        super(TransformerConvertor, self).__init__()
        self.conv = BasicConv2d(inputChannel, outputChannel, kernel_size=1)
        self.plain_fpn = Plain_FPN(inchannel=outputChannel, outchannel=outputChannel)

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def forward(self,t):
        outs = self.conv(self.to_2D(t))
        t0_1, t0_2, t0_3, t0_4 = self.plain_fpn(outs)
        return t0_1, t0_2, t0_3, t0_4

class CNNConvertor(nn.Module):
    def __init__(self,channel):
        super(CNNConvertor, self).__init__()
        self.reduce_f1 = Reduction(256, channel)
        self.reduce_f2 = Reduction(512, channel)
        self.reduce_f3 = Reduction(1024, channel)
        self.reduce_f4 = Reduction(2048, channel)

    def forward(self,t1,t2,t3,t4):
        t0_1 = self.reduce_f1(t1)
        t0_2 = self.reduce_f2(t2)
        t0_3 = self.reduce_f3(t3)
        t0_4 = self.reduce_f4(t4)
        return t0_1,t0_2,t0_3,t0_4


class CTIFNet(nn.Module):

    def __init__(self,args):
        super(CTIFNet, self).__init__()

        embed_dim = 1024
        # init and load vit_model
        vit_model = VisionTransformer(
            patch_size=16, embed_dim=embed_dim, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
            num_classes=embed_dim,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), fully_conn_output=False)
        vit_model.cuda()
        # load pre-trained model
        print("Load pre-trained checkpoint from: %s" % args.mae_model_path)
        checkpoint = torch.load(args.mae_model_path)
        checkpoint_model = checkpoint['model']
        vit_model.load_state_dict(checkpoint_model, strict=False)

        self.vit_model = vit_model
        channel = 32
        fpn_output_channel = 32

        self.cnc = CNNConvertor(channel)
        self.tnc = TransformerConvertor(embed_dim, channel)

        self.resnet = torchvision.models.resnet50(pretrained=True)

        self.feu1 = FeatureEnhancementUnit(channel)
        self.feu2 = FeatureEnhancementUnit(channel)
        self.feu3 = FeatureEnhancementUnit(channel)
        self.feu4 = FeatureEnhancementUnit(channel)

        self.finalConcat_res = UNetLikeConcat(fpn_output_channel)
        self.finalConcat_tran = UNetLikeConcat(fpn_output_channel)

    def forward(self, x):

        """extract feature by Transformer"""
        x_trans = self.vit_model(x)
        t0_1, t0_2, t0_3, t0_4 =  self.tnc(x_trans)

        """extract feature by CNN"""
        x_orignal = x
        size = x.size()[2:]
        res_x = self.resnet.conv1(x_orignal)
        res_x = self.resnet.bn1(res_x)
        res_x = self.resnet.relu(res_x)
        res_x = self.resnet.maxpool(res_x)

        f1 = self.resnet.layer1(res_x)
        f2 = self.resnet.layer2(f1)
        f3 = self.resnet.layer3(f2)
        f4 = self.resnet.layer4(f3)

        f0_1,f0_2,f0_3,f0_4 = self.cnc(f1,f2,f3,f4)


        """feature fusion"""
        t1_1, t1_2, t1_3, t1_4, f1_1, f1_2, f1_3, f1_4 = self.feu1(t0_1,t0_2,t0_3,t0_4,
                                                                                 f0_1, f0_2, f0_3, f0_4)

        t2_1, t2_2, t2_3, t2_4, f2_1, f2_2, f2_3, f2_4 = self.feu2(t1_1, t1_2, t1_3, t1_4,
                                                                                  f1_1, f1_2, f1_3, f1_4)

        t3_1, t3_2, t3_3, t3_4, f3_1, f3_2, f3_3, f3_4 = self.feu3(t2_1, t2_2, t2_3, t2_4,
                                                                                   f2_1, f2_2, f2_3, f2_4)

        t4_1, t4_2, t4_3, t4_4, f4_1, f4_2, f4_3, f4_4 = self.feu4(t3_1, t3_2, t3_3, t3_4,
                                                                                    f3_1, f3_2, f3_3, f3_4)

        y_res = self.finalConcat_res(f4_1, f4_2, f4_3, f4_4)
        y_tran = self.finalConcat_tran(t4_1, t4_2, t4_3, t4_4)

        y_res = F.interpolate(y_res, size=size, mode='bilinear', align_corners=True)
        y_tran = F.interpolate(y_tran, size=size, mode='bilinear', align_corners=True)
        return y_res, y_tran





def vit_large_patch16(pre_model_path="",map_location=torch.device('cpu'),**kwargs):

    model = CTIFNet(vit_model, dim=embed_dim)

    return model




