import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# MobileNetV3模型定义
class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, False, 1),  # 0: 16通道
            Block(3, 16, 64, 24, nn.ReLU, False, 2),  # 1: 24通道
            Block(3, 24, 72, 24, nn.ReLU, False, 1),  # 2: 24通道
            Block(5, 24, 72, 40, nn.ReLU, True, 2),  # 3: 40通道
            Block(5, 40, 120, 40, nn.ReLU, True, 1),  # 4: 40通道
            Block(5, 40, 120, 40, nn.ReLU, True, 1),  # 5: 40通道
            Block(3, 40, 240, 80, act, False, 2),  # 6: 80通道
            Block(3, 80, 200, 80, act, False, 1),  # 7: 80通道
            Block(3, 80, 184, 80, act, False, 1),  # 8: 80通道
            Block(3, 80, 184, 80, act, False, 1),  # 9: 80通道
            Block(3, 80, 480, 112, act, True, 1),  # 10: 112通道
            Block(3, 112, 672, 112, act, True, 1),  # 11: 112通道
            Block(5, 112, 672, 160, act, True, 2),  # 12: 160通道
            Block(5, 160, 672, 160, act, True, 1),  # 13: 160通道
            Block(5, 160, 960, 160, act, True, 1),  # 14: 160通道
        )

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.classifier = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.global_pool(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        return self.classifier(out)


# 注意力模块定义
class DualCoordAttentionGate(nn.Module):
    def __init__(self, inp, oup, groups=32, reduction=4, use_residual=True):
        super(DualCoordAttentionGate, self).__init__()
        self.use_residual = use_residual

        # 水平与垂直方向的平均/最大池化
        self.pool_h_mean = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_mean = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))

        # 压缩中间维度 mip
        mip = max(8, inp // groups)

        # 平均和最大通路共享的卷积模块（降低通道维度）
        self.shared_conv1 = nn.Conv2d(inp, mip, kernel_size=1)
        self.shared_bn = nn.BatchNorm2d(mip)
        self.relu = nn.ReLU(inplace=True)

        # 分别用于水平和垂直方向的卷积恢复通道维度
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

        # 可学习门控机制，用于自适应融合 mean 与 max 注意力
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inp, inp // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp // reduction, 2, 1),  # 输出两个权重：mean 和 max
            nn.Softmax(dim=1)  # 对两个权重进行归一化
        )

        # 通道注意力模块（SE 类似）增强融合后的特征响应
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(oup, oup // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(oup // reduction, oup, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # 平均池化路径
        x_h_mean = self.pool_h_mean(x)
        x_w_mean = self.pool_w_mean(x).permute(0, 1, 3, 2)
        y_mean = torch.cat([x_h_mean, x_w_mean], dim=2)
        y_mean = self.shared_conv1(y_mean)
        y_mean = self.shared_bn(y_mean)
        y_mean = self.relu(y_mean)
        x_h_mean, x_w_mean = torch.split(y_mean, [h, w], dim=2)
        x_w_mean = x_w_mean.permute(0, 1, 3, 2)
        x_h_mean = self.conv_h(x_h_mean).sigmoid()
        x_w_mean = self.conv_w(x_w_mean).sigmoid()
        attn_mean = x_h_mean * x_w_mean

        # 最大池化路径
        x_h_max = self.pool_h_max(x)
        x_w_max = self.pool_w_max(x).permute(0, 1, 3, 2)
        y_max = torch.cat([x_h_max, x_w_max], dim=2)
        y_max = self.shared_conv1(y_max)
        y_max = self.shared_bn(y_max)
        y_max = self.relu(y_max)
        x_h_max, x_w_max = torch.split(y_max, [h, w], dim=2)
        x_w_max = x_w_max.permute(0, 1, 3, 2)
        x_h_max = self.conv_h(x_h_max).sigmoid()
        x_w_max = self.conv_w(x_w_max).sigmoid()
        attn_max = x_h_max * x_w_max

        # 可学习权重融合 mean 和 max 路径
        gate_weights = self.gate(identity)  # 输出维度为 [B, 2, 1, 1]
        mean_weight = gate_weights[:, 0:1]
        max_weight = gate_weights[:, 1:2]
        attn = attn_mean * mean_weight + attn_max * max_weight

        # 应用注意力
        out = identity * attn

        # 通道注意力增强输出特征
        scale = self.channel_att(out)
        out = out * scale

        # 可选残差连接
        if self.use_residual:
            out = out + identity
        return out


# 带注意力机制的MobileNetV3
class MobileNetV3WithAttention(nn.Module):
    """带注意力机制的MobileNetV3"""

    def __init__(self, num_classes=1000):
        super(MobileNetV3WithAttention, self).__init__()
        # 原始的MobileNetV3骨干网络
        self.backbone = MobileNetV3_Large(num_classes=num_classes)

        # 为每个特征层添加对应的注意力模块，使用优化后的参数配置
        self.attention_c2 = DualCoordAttentionGate(
            inp=24, oup=24, groups=8, reduction=4, use_residual=True  # bneck[1]后，24通道
        )
        self.attention_c3 = DualCoordAttentionGate(
            inp=40, oup=40, groups=8, reduction=4, use_residual=True  # bneck[3]后，40通道
        )
        self.attention_c4 = DualCoordAttentionGate(
            inp=80, oup=80, groups=16, reduction=4, use_residual=True  # bneck[6]后，80通道
        )
        self.attention_c5 = DualCoordAttentionGate(
            inp=160, oup=160, groups=32, reduction=4, use_residual=True  # 最后，160通道
        )

    def extract_backbone_features(self, x):
        """提取带注意力增强的骨干特征"""
        x = self.backbone.hs1(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.bneck[0](x)
        x = self.backbone.bneck[1](x)
        c2_feat = self.attention_c2(x)  # 24通道，应用注意力

        x = self.backbone.bneck[2](c2_feat)
        x = self.backbone.bneck[3](x)
        c3_feat = self.attention_c3(x)  # 40通道，应用注意力

        for i in range(4, 7):
            x = self.backbone.bneck[i](x)
        c4_feat = self.attention_c4(x)  # 80通道，应用注意力

        for i in range(7, len(self.backbone.bneck)):
            x = self.backbone.bneck[i](x)
        c5_feat = self.attention_c5(x)  # 160通道，应用注意力

        return [c2_feat, c3_feat, c4_feat, c5_feat]

    def forward(self, x):
        """前向传播，保持与原始MobileNetV3相同的接口"""
        # 使用带注意力的特征提取
        x = self.backbone.hs1(self.backbone.bn1(self.backbone.conv1(x)))
        x = self.backbone.bneck[0](x)
        x = self.backbone.bneck[1](x)
        x = self.attention_c2(x)  # c2层加注意力

        x = self.backbone.bneck[2](x)
        x = self.backbone.bneck[3](x)
        x = self.attention_c3(x)  # c3层加注意力

        for i in range(4, 7):
            x = self.backbone.bneck[i](x)
        x = self.attention_c4(x)  # c4层加注意力

        for i in range(7, len(self.backbone.bneck)):
            x = self.backbone.bneck[i](x)
        x = self.attention_c5(x)  # c5层加注意力

        # 继续后续的全局池化和分类层
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.hs2(x)
        x = self.backbone.global_pool(x).flatten(1)
        x = self.backbone.drop(self.backbone.hs3(self.backbone.bn3(self.backbone.linear3(x))))
        x = self.backbone.classifier(x)

        return x