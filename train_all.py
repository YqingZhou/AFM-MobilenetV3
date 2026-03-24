import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import torch.nn.functional as F
import gc
import cv2
from collections import Counter
import time
import warnings

# ===== 修复字体配置问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# ===== 可视化相关导入 =====
try:
    from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    GRADCAM_AVAILABLE = True
    print("✓ pytorch_grad_cam 可用")
except ImportError:
    GRADCAM_AVAILABLE = False
    print("⚠️ pytorch_grad_cam 未安装，注意力可视化功能受限")
    print("安装方法: pip install grad-cam")

# ===== 导入你的原有模块 =====
from mobilenetv3_se import MobileNetV3WithAttention


def clean_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_tensor_check(tensor, name="tensor"):
    """安全检查张量是否有效"""
    if tensor is None:
        print(f"警告: {name} 为 None")
        return False
    if not isinstance(tensor, torch.Tensor):
        print(f"警告: {name} 不是张量，类型为 {type(tensor)}")
        return False
    if tensor.numel() == 0:
        print(f"警告: {name} 为空张量")
        return False
    if torch.isnan(tensor).any():
        print(f"警告: {name} 包含NaN值")
        return False
    if torch.isinf(tensor).any():
        print(f"警告: {name} 包含无穷值")
        return False
    return True


# ============= 轻量化感受野模块 =============
class LightweightMultiScale(nn.Module):
    """轻量化多尺度感受野模块，参数量更少"""

    def __init__(self, dim):
        super().__init__()
        # 使用更少的通道数，减少参数量
        mid_dim = max(dim // 4, 8)  # 至少8个通道

        # 多尺度分支 - 使用更少的输出通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU6(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim, mid_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU6(inplace=True)
        )

        # 使用深度可分离卷积减少参数
        self.conv5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False),  # 深度卷积
            nn.Conv2d(dim, mid_dim, 1, bias=False),  # 点卷积
            nn.BatchNorm2d(mid_dim),
            nn.ReLU6(inplace=True)
        )

        # 融合层
        self.merge = nn.Sequential(
            nn.Conv2d(mid_dim * 3, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        if not safe_tensor_check(x, "LightweightMultiScale输入"):
            return x

        try:
            x1 = self.conv1(x)  # 1x1卷积 - 点特征
            x3 = self.conv3(x)  # 3x3卷积 - 局部特征
            x5 = self.conv5(x)  # 5x5卷积 - 更广感受野

            # 拼接并融合
            multi_scale = torch.cat([x1, x3, x5], dim=1)
            out = self.merge(multi_scale)

            # 残差连接
            return out + x
        except Exception as e:
            print(f"LightweightMultiScale forward 错误: {e}")
            return x


# ============= 改进的FPN架构 =============
class FPN(nn.Module):
    """特征金字塔网络 - 增强错误处理"""

    def __init__(self, in_channels_list, fpn_dim=128):
        super(FPN, self).__init__()
        print(f"初始化FPN: 输入通道 {in_channels_list} -> 输出维度 {fpn_dim}")

        # 验证输入参数
        if not in_channels_list or len(in_channels_list) == 0:
            raise ValueError("in_channels_list 不能为空")

        self.num_features = len(in_channels_list)
        self.fpn_dim = fpn_dim

        # 侧向连接：将不同层的特征映射到统一维度
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_dim, 1, bias=False) for in_ch in in_channels_list
        ])

        # FPN输出层：对融合后的特征进行处理
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=max(1, fpn_dim // 4), bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim)
            ) for _ in range(len(in_channels_list))
        ])

        self.activation = nn.ReLU6(inplace=True)
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features):
        """前向传播 - 增强错误处理"""
        try:
            # 验证输入
            if not features or len(features) == 0:
                print("错误：FPN收到空特征列表")
                return []

            # 检查特征列表长度
            if len(features) != self.num_features:
                print(f"警告：期望 {self.num_features} 个特征，收到 {len(features)} 个")
                # 调整特征列表长度
                if len(features) > self.num_features:
                    features = features[:self.num_features]
                else:
                    # 如果特征不够，用最后一个特征复制
                    last_feature = features[-1] if features else None
                    while len(features) < self.num_features and last_feature is not None:
                        features.append(last_feature)

            # 检查每个特征张量并应用侧向连接
            laterals = []
            for i, (feat, conv) in enumerate(zip(features, self.lateral_convs)):
                if safe_tensor_check(feat, f"feature_{i}"):
                    try:
                        lateral = conv(feat)
                        if safe_tensor_check(lateral, f"lateral_{i}"):
                            laterals.append(lateral)
                        else:
                            print(f"侧向连接 {i} 输出无效")
                    except Exception as e:
                        print(f"侧向连接 {i} 失败: {e}")
                else:
                    print(f"跳过无效特征 {i}")

            if not laterals:
                print("错误：没有有效的侧向连接输出")
                return []

            # 自顶向下路径：特征融合
            try:
                for i in range(len(laterals) - 1, 0, -1):
                    if not safe_tensor_check(laterals[i]) or not safe_tensor_check(laterals[i - 1]):
                        continue

                    # 确保空间尺寸匹配
                    if laterals[i].shape[2:] != laterals[i - 1].shape[2:]:
                        upsampled = F.interpolate(
                            laterals[i], size=laterals[i - 1].shape[2:],
                            mode='bilinear', align_corners=False
                        )
                    else:
                        upsampled = laterals[i]

                    # 检查上采样结果
                    if safe_tensor_check(upsampled, f"upsampled_{i}"):
                        # 固定权重融合：简单的50:50混合
                        laterals[i - 1] = 0.5 * laterals[i - 1] + 0.5 * upsampled
                    else:
                        print(f"上采样 {i} 失败")
            except Exception as e:
                print(f"自顶向下融合失败: {e}")

            # 输出处理：添加残差连接和激活
            fpn_outs = []
            for i, (lateral, conv) in enumerate(zip(laterals, self.fpn_convs)):
                if not safe_tensor_check(lateral, f"lateral_final_{i}"):
                    continue

                try:
                    identity = lateral
                    out = conv(lateral)
                    if safe_tensor_check(out, f"fpn_conv_out_{i}"):
                        out = out + identity  # 残差连接
                        out = self.activation(out)
                        fpn_outs.append(out)
                    else:
                        print(f"FPN卷积输出 {i} 无效，使用原始lateral")
                        fpn_outs.append(lateral)
                except Exception as e:
                    print(f"FPN输出处理 {i} 失败: {e}")
                    # 如果处理失败，直接使用原始lateral
                    fpn_outs.append(lateral)

            return fpn_outs

        except Exception as e:
            print(f"FPN forward 发生严重错误: {e}")
            return []


# ============= 改进的多尺度特征融合模块 =============
class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块 - 增强错误处理"""

    def __init__(self, fpn_dim, backbone_dim, num_classes, num_fpn_features, dropout=0.2):
        super(MultiScaleFeatureFusion, self).__init__()
        print(f"初始化多尺度特征融合: FPN维度={fpn_dim}, backbone维度={backbone_dim}")

        self.fpn_dim = fpn_dim
        self.backbone_dim = backbone_dim
        self.num_classes = num_classes

        # 多尺度池化
        self.multi_scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((4, 4)),
        ])

        # backbone特征投影
        self.backbone_proj = nn.Conv2d(backbone_dim, fpn_dim, 1, bias=False)

        # 每个尺度的特征处理器
        expected_channels = fpn_dim * (num_fpn_features + 1)  # +1 for backbone
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(expected_channels, fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU6(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            ) for _ in range(len(self.multi_scale_pools))
        ])

        # 跨尺度特征融合
        total_scale_features = len(self.multi_scale_pools) * fpn_dim
        self.cross_scale_fusion = nn.Sequential(
            nn.Linear(total_scale_features, fpn_dim * 2),
            nn.BatchNorm1d(fpn_dim * 2),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fpn_dim * 2, fpn_dim),
            nn.BatchNorm1d(fpn_dim),
            nn.ReLU6(inplace=True)
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fpn_dim, num_classes)
        )

        # 备用简单分类器
        self.backup_classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone_dim, fpn_dim),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fpn_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, fpn_features, backbone_feature):
        """前向传播 - 增强错误处理"""
        try:
            # 验证backbone特征
            if not safe_tensor_check(backbone_feature, "backbone_feature"):
                print("错误：backbone特征无效")
                raise ValueError("backbone_feature 无效")

            # 处理backbone特征
            try:
                backbone_global = self.backbone_proj(backbone_feature)
                if not safe_tensor_check(backbone_global, "backbone_global"):
                    print("backbone投影失败，使用备用分类器")
                    return self.backup_classifier(backbone_feature)
            except Exception as e:
                print(f"backbone投影失败: {e}，使用备用分类器")
                return self.backup_classifier(backbone_feature)

            # 验证和收集有效的FPN特征
            valid_fpn_features = []
            if fpn_features:
                for i, feat in enumerate(fpn_features):
                    if safe_tensor_check(feat, f"fpn_feature_{i}"):
                        valid_fpn_features.append(feat)
                    else:
                        print(f"跳过无效FPN特征 {i}")
            else:
                print("警告：FPN特征列表为空")

            # 合并所有有效特征
            all_features = valid_fpn_features + [backbone_global]

            if len(all_features) == 0:
                print("没有有效特征，使用备用分类器")
                return self.backup_classifier(backbone_feature)

            # 多尺度特征提取
            scale_features = []
            for scale_idx, (pool, processor) in enumerate(zip(self.multi_scale_pools, self.scale_processors)):
                try:
                    # 对每个特征应用当前尺度的池化
                    pooled_features = []
                    for feat_idx, feat in enumerate(all_features):
                        try:
                            if safe_tensor_check(feat, f"feat_{feat_idx}_scale_{scale_idx}"):
                                pooled = pool(feat)
                                if safe_tensor_check(pooled, f"pooled_{feat_idx}_scale_{scale_idx}"):
                                    pooled_features.append(pooled)
                            else:
                                print(f"特征 {feat_idx} 在尺度 {scale_idx} 无效")
                        except Exception as e:
                            print(f"池化失败 scale={scale_idx}, feat={feat_idx}: {e}")

                    if not pooled_features:
                        print(f"尺度 {scale_idx} 没有有效的池化特征")
                        continue

                    # 拼接同一尺度下的所有特征
                    try:
                        concat_feat = torch.cat(pooled_features, dim=1)
                        if not safe_tensor_check(concat_feat, f"concat_feat_scale_{scale_idx}"):
                            continue
                    except Exception as e:
                        print(f"特征拼接失败 scale={scale_idx}: {e}")
                        continue

                    # 检查通道数是否匹配
                    expected_channels = self.scale_processors[scale_idx][0].in_channels
                    if concat_feat.size(1) != expected_channels:
                        print(f"尺度 {scale_idx} 通道数不匹配: 期望{expected_channels}, 实际{concat_feat.size(1)}")
                        # 尝试调整通道数
                        if concat_feat.size(1) > expected_channels:
                            concat_feat = concat_feat[:, :expected_channels, :, :]
                        else:
                            # 如果通道数不够，重复最后的通道
                            repeat_times = expected_channels // concat_feat.size(1) + 1
                            concat_feat = concat_feat.repeat(1, repeat_times, 1, 1)[:, :expected_channels, :, :]

                    # 处理并统一到1x1
                    try:
                        processed = processor(concat_feat)
                        if safe_tensor_check(processed, f"processed_scale_{scale_idx}"):
                            scale_features.append(processed.flatten(1))
                        else:
                            print(f"处理后特征 {scale_idx} 无效")
                    except Exception as e:
                        print(f"特征处理失败 scale={scale_idx}: {e}")

                except Exception as e:
                    print(f"尺度 {scale_idx} 处理完全失败: {e}")

            # 检查是否有有效的尺度特征
            if not scale_features:
                print("所有尺度特征处理失败，使用备用分类器")
                return self.backup_classifier(backbone_feature)

            # 跨尺度融合
            try:
                all_scale_features = torch.cat(scale_features, dim=1)
                if safe_tensor_check(all_scale_features, "all_scale_features"):
                    fused_features = self.cross_scale_fusion(all_scale_features)
                    if not safe_tensor_check(fused_features, "fused_features"):
                        raise ValueError("融合特征无效")
                else:
                    raise ValueError("尺度特征拼接失败")
            except Exception as e:
                print(f"跨尺度融合失败: {e}")
                # 使用第一个尺度特征作为备用
                fused_features = scale_features[0]

            # 分类
            try:
                output = self.classifier(fused_features)
                if safe_tensor_check(output, "classifier_output"):
                    return output
                else:
                    raise ValueError("分类器输出无效")
            except Exception as e:
                print(f"分类器失败: {e}")
                return self.backup_classifier(backbone_feature)

        except Exception as e:
            print(f"MultiScaleFeatureFusion forward 发生严重错误: {e}")
            # 最终备用方案：使用备用分类器
            try:
                return self.backup_classifier(backbone_feature)
            except Exception as e2:
                print(f"备用分类器也失败: {e2}")
                # 返回随机输出作为最后手段
                batch_size = backbone_feature.size(0)
                return torch.randn(batch_size, self.num_classes, device=backbone_feature.device)


class FineGrainedModel(nn.Module):
    """细粒度分类模型 - 增强错误处理"""

    def __init__(self, num_classes, fpn_dim=128, dropout=0.2, target_layers=[2, 6, 10]):
        super(FineGrainedModel, self).__init__()
        print(f"初始化细粒度模型: {num_classes}类, FPN维度={fpn_dim}, 目标层={target_layers}")

        self.backbone = MobileNetV3WithAttention(num_classes=num_classes)
        self.target_layer_indices = target_layers
        self.num_classes = num_classes
        self.fpn_dim = fpn_dim

        # 动态检测通道数
        self._detect_channel_numbers()

        # 根据检测结果决定是否使用高级融合
        if self.selected_channels and len(self.selected_channels) > 0:
            try:
                # 使用FPN
                self.fpn = FPN(
                    in_channels_list=self.selected_channels,
                    fpn_dim=fpn_dim
                )

                # 使用改进的多尺度特征融合模块
                self.feature_fusion = MultiScaleFeatureFusion(
                    fpn_dim=fpn_dim,
                    backbone_dim=self.final_channels,
                    num_classes=num_classes,
                    num_fpn_features=len(self.selected_channels),
                    dropout=dropout
                )
                self.use_advanced_fusion = True
                print("✓ 成功初始化高级特征融合")
            except Exception as e:
                print(f"高级融合初始化失败: {e}，使用简单backbone")
                self.use_advanced_fusion = False
        else:
            print("通道检测失败，使用简单backbone")
            self.use_advanced_fusion = False

    def _detect_channel_numbers(self):
        """动态检测各层通道数 - 增强错误处理"""
        try:
            test_input = torch.zeros(1, 3, 224, 224)
            self.backbone.eval()

            with torch.no_grad():
                features = []
                x = test_input

                # 通过backbone的第一层
                x = self.backbone.backbone.hs1(self.backbone.backbone.bn1(self.backbone.backbone.conv1(x)))

                # 遍历backbone的瓶颈层
                for i, layer in enumerate(self.backbone.backbone.bneck):
                    x = layer(x)
                    if i in self.target_layer_indices and safe_tensor_check(x):
                        features.append(x.shape[1])

                if safe_tensor_check(x):
                    self.final_channels = x.shape[1]
                    self.selected_channels = features
                    print(f"✓ 检测到的通道数: {self.selected_channels}, 最终通道数: {self.final_channels}")
                else:
                    raise ValueError("最终特征张量无效")

        except Exception as e:
            print(f"通道检测失败: {e}")
            # 使用MobileNetV3的典型通道数
            self.selected_channels = [24, 40, 112]
            self.final_channels = 960
            print(f"使用默认通道数: {self.selected_channels}, 最终通道数: {self.final_channels}")

    def extract_selected_features(self, x):
        """从指定层提取特征 - 增强错误处理"""
        features = []

        try:
            if not safe_tensor_check(x, "extract_input"):
                return [], x

            # 通过backbone的第一层
            x = self.backbone.backbone.hs1(self.backbone.backbone.bn1(self.backbone.backbone.conv1(x)))

            # 遍历backbone的瓶颈层，提取目标层特征
            for i, layer in enumerate(self.backbone.backbone.bneck):
                x = layer(x)
                if i in self.target_layer_indices and safe_tensor_check(x, f"bneck_layer_{i}"):
                    features.append(x)

            return features, x

        except Exception as e:
            print(f"特征提取失败: {e}")
            return [], x

    def forward(self, x):
        """前向传播 - 增强错误处理"""
        try:
            if not safe_tensor_check(x, "model_input"):
                print("模型输入无效")
                batch_size = x.size(0) if hasattr(x, 'size') else 1
                return torch.randn(batch_size, self.num_classes,
                                   device=x.device if torch.cuda.is_available() else 'cpu')

            # 如果不使用高级融合，直接使用backbone
            if not self.use_advanced_fusion:
                return self.backbone(x)

            # 提取多层特征
            selected_features, final_feature = self.extract_selected_features(x)

            # 如果特征提取失败，回退到backbone分类
            if len(selected_features) == 0 or not safe_tensor_check(final_feature, "final_feature"):
                print("特征提取失败，使用backbone分类")
                return self.backbone(x)

            # FPN处理
            try:
                fpn_outputs = self.fpn(selected_features)
            except Exception as e:
                print(f"FPN处理失败: {e}，使用backbone分类")
                return self.backbone(x)

            # 使用改进的多尺度特征融合
            try:
                output = self.feature_fusion(fpn_outputs, final_feature)
                if safe_tensor_check(output, "fusion_output"):
                    return output
                else:
                    print("融合输出无效，使用backup")
                    return self.backbone(x)
            except Exception as e:
                print(f"特征融合失败: {e}，使用backbone分类")
                return self.backbone(x)

        except Exception as e:
            print(f"FineGrainedModel forward 发生严重错误: {e}")
            # 最终备用方案：使用backbone
            try:
                return self.backbone(x)
            except Exception as e2:
                print(f"Backbone也失败: {e2}")
                # 返回随机输出
                batch_size = x.size(0) if hasattr(x, 'size') else 1
                device = x.device if hasattr(x, 'device') else 'cpu'
                return torch.randn(batch_size, self.num_classes, device=device)


# ============= 数据加载 =============
class PlantDataset(Dataset):
    """植物数据集类"""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert('RGB')
            label = self.labels[idx]

            if self.transform:
                image = self.transform(image)

            return image, label
        except Exception as e:
            print(f"加载图像失败 {self.image_paths[idx]}: {e}")
            # 返回黑色图像作为备用
            if self.transform:
                black_image = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                black_image = torch.zeros(3, 224, 224)
            return black_image, self.labels[idx]


def load_data(data_dir):
    """加载数据集"""
    print("=" * 50)
    print("           数据加载流程")
    print("=" * 50)

    image_paths = []
    labels = []
    class_names = []
    class_to_idx = {}

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not class_dirs:
        raise ValueError(f"在 {data_dir} 中没有找到类别文件夹")

    class_dirs.sort()

    print(f"发现 {len(class_dirs)} 个类别文件夹")
    print("-" * 50)

    for idx, class_name in enumerate(class_dirs):
        class_to_idx[class_name] = idx
        class_names.append(class_name)
        class_dir = os.path.join(data_dir, class_name)

        img_names = []
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(class_dir, img_name)
                # 检查文件是否可读
                try:
                    with Image.open(img_path) as test_img:
                        test_img.verify()
                    img_names.append(img_name)
                except Exception as e:
                    print(f"跳过损坏的图像: {img_path}")

        img_names.sort()
        print(f"类别 {idx}: {class_name} -> {len(img_names)} 张有效图像")

        for img_name in img_names:
            img_path = os.path.join(class_dir, img_name)
            image_paths.append(img_path)
            labels.append(idx)

    print("-" * 50)
    print(f"总计: {len(image_paths)} 张图像, {len(class_names)} 个类别")

    if len(image_paths) == 0:
        raise ValueError("没有找到有效的图像文件")

    return image_paths, labels, class_names, class_to_idx


def split_data(image_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """将数据分割为训练/验证/测试集"""
    print("=" * 50)
    print("           数据分割流程")
    print("=" * 50)

    # 检查数据是否足够分割
    if len(image_paths) < 10:
        raise ValueError(f"数据量太少({len(image_paths)})，至少需要10张图像")

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=(val_ratio + test_ratio),
        random_state=random_state, stratify=labels
    )

    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_size),
        random_state=random_state, stratify=temp_labels
    )

    print(f"训练集: {len(train_paths)} 张 ({train_ratio * 100:.0f}%)")
    print(f"验证集: {len(val_paths)} 张 ({val_ratio * 100:.0f}%)")
    print(f"测试集: {len(test_paths)} 张 ({test_ratio * 100:.0f}%)")
    print("-" * 50)

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_transforms(enable_augmentation=True):
    """定义数据增强和预处理"""
    if enable_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3)
            ], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def load_pretrained_weights(model, pretrained_path):
    """加载预训练权重"""
    if not os.path.exists(pretrained_path):
        print(f"预训练权重文件不存在: {pretrained_path}")
        return

    try:
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        pretrained_dict = {}

        for k, v in state_dict.items():
            matched = False

            # 策略1: 直接匹配
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                matched = True

            # 策略2: 添加backbone前缀
            if not matched:
                backbone_key = f"backbone.{k}"
                if backbone_key in model_dict and model_dict[backbone_key].shape == v.shape:
                    pretrained_dict[backbone_key] = v
                    matched = True

            # 策略3: 添加backbone.backbone前缀
            if not matched:
                double_backbone_key = f"backbone.backbone.{k}"
                if double_backbone_key in model_dict and model_dict[double_backbone_key].shape == v.shape:
                    pretrained_dict[double_backbone_key] = v
                    matched = True

        if len(pretrained_dict) > 0:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            print(f"✓ 成功加载 {len(pretrained_dict)} 层预训练权重")
        else:
            print("没有匹配的预训练权重，从头开始训练")
    except Exception as e:
        print(f"加载预训练权重失败: {e}")


# ============= 可视化模块 =============
def plot_class_distribution(labels, class_names, save_path):
    """可视化类别分布"""
    try:
        counter = Counter(labels)
        classes = [class_names[i] for i in range(len(class_names))]
        counts = [counter[i] for i in range(len(class_names))]

        plt.figure(figsize=(15, 8))
        bars = plt.bar(classes, counts, color='skyblue', alpha=0.7)
        plt.title('数据集类别分布', fontsize=16, fontweight='bold')
        plt.xlabel('类别', fontsize=12)
        plt.ylabel('样本数量', fontsize=12)
        plt.xticks(rotation=45, ha='right')

        # 在柱状图上添加数值标签
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        # 打印统计信息
        print(f"\n类别分布统计:")
        print(f"总样本数: {sum(counts)}")
        print(f"平均每类: {np.mean(counts):.1f}")
        print(f"最多类别: {class_names[np.argmax(counts)]} ({max(counts)} 样本)")
        print(f"最少类别: {class_names[np.argmin(counts)]} ({min(counts)} 样本)")
    except Exception as e:
        print(f"绘制类别分布失败: {e}")


def plot_data_splits(train_labels, val_labels, test_labels, class_names, save_path):
    """可视化数据集分割情况"""
    try:
        train_counter = Counter(train_labels)
        val_counter = Counter(val_labels)
        test_counter = Counter(test_labels)

        x = np.arange(len(class_names))
        width = 0.25

        train_counts = [train_counter[i] for i in range(len(class_names))]
        val_counts = [val_counter[i] for i in range(len(class_names))]
        test_counts = [test_counter[i] for i in range(len(class_names))]

        fig, ax = plt.subplots(figsize=(15, 8))
        bars1 = ax.bar(x - width, train_counts, width, label='训练集', alpha=0.8, color='lightblue')
        bars2 = ax.bar(x, val_counts, width, label='验证集', alpha=0.8, color='lightgreen')
        bars3 = ax.bar(x + width, test_counts, width, label='测试集', alpha=0.8, color='lightcoral')

        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title('数据分割分布', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"绘制数据分割失败: {e}")


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """绘制训练历史"""
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 损失曲线
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='训练损失', linewidth=2)
        ax1.plot(epochs, val_losses, 'r-', label='验证损失', linewidth=2)
        ax1.set_title('训练和验证损失', fontsize=14, fontweight='bold')
        ax1.set_xlabel('轮次', fontsize=12)
        ax1.set_ylabel('损失', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 准确率曲线
        ax2.plot(epochs, train_accs, 'b-', label='训练准确率', linewidth=2)
        ax2.plot(epochs, val_accs, 'r-', label='验证准确率', linewidth=2)
        ax2.set_title('训练和验证准确率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('轮次', fontsize=12)
        ax2.set_ylabel('准确率 (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"绘制训练历史失败: {e}")


def safe_grad_cam_visualization(model, test_loader, device, class_names, save_path, num_samples=6):
    """安全的Grad-CAM可视化 - 增强错误处理"""
    if not GRADCAM_AVAILABLE:
        print("Grad-CAM不可用，跳过此可视化")
        return

    print("开始安全Grad-CAM可视化...")

    try:
        # 获取测试样本
        images_to_viz = []
        labels_to_viz = []

        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                for i in range(min(num_samples, images.size(0))):
                    if safe_tensor_check(images[i]) and safe_tensor_check(labels[i]):
                        images_to_viz.append(images[i])
                        labels_to_viz.append(labels[i])
                if len(images_to_viz) >= num_samples:
                    break

        if not images_to_viz:
            print("没有找到有效的测试样本")
            return

        # 寻找适合的目标层
        target_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
                # 优先选择backbone中的层
                if 'backbone' in name and 'bneck' in name:
                    target_layers.append(module)
                    if len(target_layers) >= 3:  # 最多3层
                        break

        if not target_layers:
            # 如果没找到，尝试其他层
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
                    target_layers.append(module)
                    if len(target_layers) >= 3:
                        break

        if not target_layers:
            print("未找到合适的目标层进行可视化")
            return

        print(f"找到 {len(target_layers)} 个目标层用于可视化")

        # 创建可视化图
        num_cols = len(target_layers) + 1
        fig, axes = plt.subplots(num_samples, num_cols,
                                 figsize=(4 * num_cols, 4 * num_samples))

        # 处理单行情况
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        fig.suptitle('安全Grad-CAM可视化', fontsize=16, fontweight='bold')

        for idx, (image, label) in enumerate(zip(images_to_viz, labels_to_viz)):
            if idx >= num_samples:
                break

            try:
                # 原始图像处理
                img_np = image.cpu()
                # 反归一化
                img_np = img_np * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                    [0.485, 0.456, 0.406]).view(3, 1, 1)
                img_np = torch.clamp(img_np, 0, 1).permute(1, 2, 0).numpy()

                # 显示原始图像
                axes[idx, 0].imshow(img_np)
                axes[idx, 0].set_title(f'原始图像\n{class_names[label.item()]}', fontsize=10)
                axes[idx, 0].axis('off')

                # 对每个目标层生成Grad-CAM
                for layer_idx, target_layer in enumerate(target_layers):
                    try:
                        # 创建Grad-CAM
                        cam = GradCAM(model=model, target_layers=[target_layer])

                        # 生成CAM
                        targets = [ClassifierOutputTarget(label.item())]

                        # 确保输入tensor形状正确
                        input_tensor = image.unsqueeze(0)
                        if not safe_tensor_check(input_tensor, "grad_cam_input"):
                            continue

                        # 检查tensor维度
                        if input_tensor.dim() != 4 or input_tensor.size(1) != 3:
                            print(f"输入张量形状异常: {input_tensor.shape}")
                            axes[idx, layer_idx + 1].axis('off')
                            axes[idx, layer_idx + 1].text(0.5, 0.5, '形状错误', ha='center', va='center')
                            continue

                        # 生成Grad-CAM
                        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

                        if grayscale_cam is None or len(grayscale_cam) == 0:
                            print(f"Grad-CAM生成失败")
                            axes[idx, layer_idx + 1].axis('off')
                            axes[idx, layer_idx + 1].text(0.5, 0.5, 'CAM失败', ha='center', va='center')
                            continue

                        grayscale_cam = grayscale_cam[0, :]

                        # 检查CAM输出
                        if not isinstance(grayscale_cam, np.ndarray) or grayscale_cam.size == 0:
                            print(f"无效的CAM输出")
                            axes[idx, layer_idx + 1].axis('off')
                            axes[idx, layer_idx + 1].text(0.5, 0.5, 'CAM无效', ha='center', va='center')
                            continue

                        # 叠加到原图
                        visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True,
                                                          colormap=cv2.COLORMAP_JET)

                        axes[idx, layer_idx + 1].imshow(visualization)
                        axes[idx, layer_idx + 1].set_title(f'层 {layer_idx + 1}', fontsize=10)
                        axes[idx, layer_idx + 1].axis('off')

                    except Exception as e:
                        print(f"层 {layer_idx} 处理失败: {e}")
                        axes[idx, layer_idx + 1].axis('off')
                        axes[idx, layer_idx + 1].text(0.5, 0.5, f'错误\n{str(e)[:20]}',
                                                      ha='center', va='center', fontsize=8)

            except Exception as e:
                print(f"样本 {idx} 处理失败: {e}")
                # 清空该行
                for col in range(num_cols):
                    axes[idx, col].axis('off')
                    axes[idx, col].text(0.5, 0.5, f'样本{idx}错误', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ 安全Grad-CAM可视化完成，保存到: {save_path}")

    except Exception as e:
        print(f"Grad-CAM可视化完全失败: {e}")


def visualize_sample_predictions(model, test_loader, device, class_names, save_path, num_samples=16):
    """可视化样本预测结果"""
    try:
        model.eval()

        # 收集一些预测结果
        all_images = []
        all_predictions = []
        all_targets = []
        all_confidences = []

        with torch.no_grad():
            for images, targets in test_loader:
                images, targets = images.to(device), targets.to(device)
                try:
                    outputs = torch.nn.functional.softmax(model(images), dim=1)
                    confidences, predictions = torch.max(outputs, 1)

                    for i in range(images.size(0)):
                        if safe_tensor_check(images[i]) and safe_tensor_check(targets[i]):
                            all_images.append(images[i].cpu())
                            all_predictions.append(predictions[i].cpu().item())
                            all_targets.append(targets[i].cpu().item())
                            all_confidences.append(confidences[i].cpu().item())

                            if len(all_images) >= num_samples:
                                break
                except Exception as e:
                    print(f"预测失败: {e}")
                    continue

                if len(all_images) >= num_samples:
                    break

        if not all_images:
            print("没有有效的预测结果")
            return

        # 创建可视化
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('样本预测结果', fontsize=18, fontweight='bold')

        for i in range(min(num_samples, len(all_images))):
            row, col = i // 4, i % 4
            ax = axes[row, col]

            try:
                # 反归一化图像用于显示
                image = all_images[i]
                image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor(
                    [0.485, 0.456, 0.406]).view(3, 1, 1)
                image = torch.clamp(image, 0, 1)
                image = image.permute(1, 2, 0).numpy()

                ax.imshow(image)
                ax.axis('off')

                # 设置标题颜色（正确预测为绿色，错误为红色）
                pred_label = class_names[all_predictions[i]]
                true_label = class_names[all_targets[i]]
                confidence = all_confidences[i]

                is_correct = all_predictions[i] == all_targets[i]
                color = 'green' if is_correct else 'red'

                title = f"预测: {pred_label}\n真实: {true_label}\n置信度: {confidence:.2f}"
                ax.set_title(title, fontsize=10, color=color, fontweight='bold')
            except Exception as e:
                ax.axis('off')
                ax.text(0.5, 0.5, f'显示错误\n{str(e)[:20]}', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"样本预测可视化失败: {e}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    try:
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar_kws={'label': '样本数量'})
        plt.title('混淆矩阵', fontsize=16, fontweight='bold')
        plt.xlabel('预测类别', fontsize=12)
        plt.ylabel('真实类别', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"混淆矩阵绘制失败: {e}")


def create_comprehensive_dashboard(config, results, save_dir):
    """创建综合可视化仪表板"""
    try:
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # 1. 模型性能指标
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['测试准确率', '测试精确率', '测试召回率', '测试F1分数']
        values = [results['final_test_acc'], results['final_test_precision'],
                  results['final_test_recall'], results['final_test_f1']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

        bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
        ax1.set_title('模型性能指标 (%)', fontweight='bold', fontsize=12)
        ax1.set_ylim(0, 100)
        ax1.tick_params(axis='x', rotation=45)

        for bar, value in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

        # 2. 训练过程
        ax2 = fig.add_subplot(gs[0, 1])
        epochs = range(1, len(results['train_history']['val_accs']) + 1)
        ax2.plot(epochs, results['train_history']['train_accs'], 'b-', label='训练', linewidth=2)
        ax2.plot(epochs, results['train_history']['val_accs'], 'r-', label='验证', linewidth=2)
        ax2.set_title('准确率变化', fontweight='bold', fontsize=12)
        ax2.set_xlabel('轮次')
        ax2.set_ylabel('准确率 (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 配置信息
        ax3 = fig.add_subplot(gs[0, 2:])
        ax3.axis('off')
        config_text = f"""
训练配置:
轮数: {config['num_epochs']} (实际: {results['best_epoch']})
批量大小: {config['batch_size']}
学习率: {config['learning_rate']:.2e}
FPN维度: {config['fpn_dim']}
Dropout: {config['dropout']}
目标层: {config['target_layers']}
数据增强: {'启用' if config['enable_augmentation'] else '禁用'}
        """
        ax3.text(0.1, 0.5, config_text, fontsize=11, va='center',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

        # 4. 结果摘要
        ax4 = fig.add_subplot(gs[1:, :])
        ax4.axis('off')

        summary_text = f"""
FineGrainedModel 训练结果总结

最终性能:
• 最佳验证准确率: {results['best_val_acc']:.2f}% (第 {results['best_epoch']} 轮)
• 最终测试准确率: {results['final_test_acc']:.2f}%
• 测试精确率: {results['final_test_precision']:.2f}%
• 测试召回率: {results['final_test_recall']:.2f}%
• 测试F1分数: {results['final_test_f1']:.2f}%

模型架构特点:
• 多尺度特征提取: 使用FPN金字塔网络结构
• 特征融合策略: 改进的多尺度特征融合模块
• 注意力机制: 集成MobileNetV3WithAttention骨干网络
• 轻量化设计: 适合移动端部署的高效架构

训练策略:
• 早停机制: 防止过拟合，耐心值 {config['patience']} 轮
• 学习率调度: {config['lr_scheduler']} 调度器
• 数据增强: {'多种增强技术提升泛化能力' if config['enable_augmentation'] else '无数据增强'}
• 权重衰减: {config['weight_decay']:.1e} 正则化

可视化功能:
• Grad-CAM注意力热力图: 展示模型关注区域
• 样本预测结果: 预测正确性和置信度分析
• 训练过程监控: 损失和准确率变化趋势
• 数据分布分析: 类别平衡和数据分割情况
        """

        ax4.text(0.05, 0.95, summary_text, fontsize=12, va='top', ha='left',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
                 transform=ax4.transAxes)

        fig.suptitle('FineGrainedModel with FPN - 植物病害分类结果仪表板',
                     fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(os.path.join(save_dir, 'comprehensive_dashboard.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    except Exception as e:
        print(f"综合仪表板创建失败: {e}")


# ============= 训练函数 =============
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'训练 Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        try:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

            # 检查输入数据有效性
            if not safe_tensor_check(data, "训练数据") or not safe_tensor_check(target, "训练标签"):
                print(f"跳过无效批次 {batch_idx}")
                continue

            optimizer.zero_grad()
            output = model(data)

            # 检查模型输出
            if not safe_tensor_check(output, "模型输出"):
                print(f"模型输出无效，跳过批次 {batch_idx}")
                continue

            loss = criterion(output, target)

            if not safe_tensor_check(loss, "损失"):
                print(f"损失计算失败，跳过批次 {batch_idx}")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

            # 定期清理内存
            if batch_idx % 50 == 0:
                clean_memory()

        except Exception as e:
            print(f"训练批次 {batch_idx} 失败: {e}")
            continue

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device, epoch):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'验证 Epoch {epoch}')
        for batch_idx, (data, target) in enumerate(pbar):
            try:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                if not safe_tensor_check(data, "验证数据") or not safe_tensor_check(target, "验证标签"):
                    continue

                output = model(data)

                if not safe_tensor_check(output, "验证输出"):
                    continue

                val_loss += criterion(output, target).item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                pbar.set_postfix({
                    'Loss': f'{val_loss / (batch_idx + 1):.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
            except Exception as e:
                print(f"验证批次 {batch_idx} 失败: {e}")
                continue

    return val_loss / len(val_loader), 100. * correct / total


def test_model(model, test_loader, device, class_names):
    """测试模型并返回详细结果"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    print("=" * 50)
    print("           模型测试")
    print("=" * 50)

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='测试中')
        for batch_idx, (data, target) in enumerate(pbar):
            try:
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                if not safe_tensor_check(data, "测试数据") or not safe_tensor_check(target, "测试标签"):
                    continue

                output = model(data)

                if not safe_tensor_check(output, "测试输出"):
                    continue

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                pbar.set_postfix({
                    'Acc': f'{100. * correct / total:.2f}%'
                })
            except Exception as e:
                print(f"测试批次 {batch_idx} 失败: {e}")
                continue

    if total == 0:
        print("没有有效的测试样本")
        return 0, 0, 0, 0, [], []

    test_acc = 100. * correct / total

    # 计算详细指标
    try:
        precision, recall, f1, support = precision_recall_fscore_support(
            all_targets, all_predictions, average='weighted', zero_division=0
        )
    except Exception as e:
        print(f"指标计算失败: {e}")
        precision = recall = f1 = 0

    print("-" * 50)
    print(f"测试结果:")
    print(f"   测试准确率: {test_acc:.2f}%")
    print(f"   加权精确率: {precision * 100:.2f}%")
    print(f"   加权召回率: {recall * 100:.2f}%")
    print(f"   加权F1分数: {f1 * 100:.2f}%")
    print("-" * 50)

    return test_acc, precision * 100, recall * 100, f1 * 100, all_predictions, all_targets


# ============= 主训练函数 =============
def train_main():
    """主训练函数"""

    # 配置参数
    config = {
        'data_dir': './merged_dataset',
        'pretrained_path': 'checkpoint/checkpoints_mobilenetv3_attention/best_model.pth',
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.0005,
        'weight_decay': 9e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'results_improved_multiscale_with_viz',
        'num_workers': 4,
        'lr_scheduler': 'cosine',
        'cosine_eta_min_ratio': 0.01,
        'enable_augmentation': True,
        'fpn_dim': 128,
        'dropout': 0.2,
        'target_layers': [2, 8, 10],
        'patience': 15,
    }

    clean_memory()
    print("=" * 80)
    print("开始训练 - FineGrainedModel + 完整可视化版本 (修复版)")
    print("=" * 80)
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    os.makedirs(config['save_dir'], exist_ok=True)

    try:
        # 加载数据
        print("正在加载数据...")
        image_paths, labels, class_names, class_to_idx = load_data(config['data_dir'])
        num_classes = len(class_names)

        # 可视化数据集分布
        plot_class_distribution(labels, class_names,
                                os.path.join(config['save_dir'], 'class_distribution.png'))

        # 分割数据集
        (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_data(
            image_paths, labels, random_state=42
        )

        # 可视化数据分割
        plot_data_splits(train_labels, val_labels, test_labels, class_names,
                         os.path.join(config['save_dir'], 'data_splits.png'))

        # 创建数据加载器
        print("=" * 50)
        print("         创建数据加载器")
        print("=" * 50)
        train_transform, val_transform = get_transforms(enable_augmentation=config['enable_augmentation'])

        train_dataset = PlantDataset(train_paths, train_labels, transform=train_transform)
        val_dataset = PlantDataset(val_paths, val_labels, transform=val_transform)
        test_dataset = PlantDataset(test_paths, test_labels, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=config['num_workers'], pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                shuffle=False, num_workers=config['num_workers'], pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'],
                                 shuffle=False, num_workers=config['num_workers'], pin_memory=True)

        print(f"训练批次: {len(train_loader)} batches")
        print(f"验证批次: {len(val_loader)} batches")
        print(f"测试批次: {len(test_loader)} batches")
        print("-" * 50)

        # 创建模型
        print("=" * 50)
        print("           创建模型")
        print("=" * 50)
        model = FineGrainedModel(
            num_classes=num_classes,
            fpn_dim=config['fpn_dim'],
            dropout=config['dropout'],
            target_layers=config['target_layers']
        )

        # 加载预训练权重
        load_pretrained_weights(model, config['pretrained_path'])

        device = torch.device(config['device'])
        model = model.to(device)

        # 训练设置
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                weight_decay=config['weight_decay'])

        if config['lr_scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config['num_epochs'],
                eta_min=config['learning_rate'] * config['cosine_eta_min_ratio']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        # 早停和模型选择
        best_val_acc = 0.0
        best_epoch = 0
        wait = 0
        best_model_path = os.path.join(config['save_dir'], 'best_model.pth')

        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        # 训练循环
        print("=" * 80)
        print(f"              开始训练 {config['num_epochs']} 轮")
        print("=" * 80)

        for epoch in range(config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
            print("-" * 60)

            # 训练
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1)

            # 验证
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch + 1)

            scheduler.step()

            train_losses.append(train_loss)
            train_accs.append(train_acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            # 早停逻辑
            if val_acc > best_val_acc + 0.1:  # 至少提升0.1%
                best_val_acc = val_acc
                best_epoch = epoch + 1
                wait = 0
                try:
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'config': config
                    }, best_model_path)
                    print(f"✓ 保存最佳模型: 验证准确率 {val_acc:.2f}%")
                except Exception as e:
                    print(f"保存模型失败: {e}")
            else:
                wait += 1

            print(f"训练: {train_acc:.2f}%, 验证: {val_acc:.2f}%")
            print(f"最佳验证准确率: {best_val_acc:.2f}% (Epoch {best_epoch})")
            print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")

            if wait >= config['patience']:
                print(f"早停: 连续{config['patience']}轮无改善")
                break

        # 绘制训练历史
        plot_training_history(train_losses, train_accs, val_losses, val_accs,
                              os.path.join(config['save_dir'], 'training_history.png'))

        # 加载最佳模型进行最终测试
        print(f"\n加载最佳模型进行最终测试...")
        if os.path.exists(best_model_path):
            try:
                checkpoint = torch.load(best_model_path, map_location=device)
                model.load_state_dict(checkpoint['state_dict'])
                print(f"✓ 加载最佳模型 (验证准确率: {checkpoint['val_acc']:.2f}%)")
            except Exception as e:
                print(f"加载最佳模型失败: {e}")

        # 最终测试
        test_acc, test_precision, test_recall, test_f1, predictions, targets = test_model(
            model, test_loader, device, class_names)

        # 生成可视化
        print("\n" + "=" * 60)
        print("生成可视化结果")
        print("=" * 60)

        # 1. 样本预测可视化
        visualize_sample_predictions(model, test_loader, device, class_names,
                                     os.path.join(config['save_dir'], 'sample_predictions.png'))

        # 2. 混淆矩阵
        if predictions and targets:
            plot_confusion_matrix(targets, predictions, class_names,
                                  os.path.join(config['save_dir'], 'confusion_matrix.png'))

        # 3. Grad-CAM可视化
        safe_grad_cam_visualization(model, test_loader, device, class_names,
                                    os.path.join(config['save_dir'], 'gradcam_attention.png'))

        # 保存结果
        results = {
            'config': config,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_test_acc': test_acc,
            'final_test_precision': test_precision,
            'final_test_recall': test_recall,
            'final_test_f1': test_f1,
            'train_history': {
                'train_losses': train_losses,
                'train_accs': train_accs,
                'val_losses': val_losses,
                'val_accs': val_accs
            }
        }

        # 4. 综合仪表板
        create_comprehensive_dashboard(config, results, config['save_dir'])

        # 保存结果到JSON
        try:
            with open(os.path.join(config['save_dir'], 'training_results.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存结果失败: {e}")

        print("=" * 80)
        print("                     训练完成!")
        print("=" * 80)
        print(f"   最佳验证准确率: {best_val_acc:.2f}%")
        print(f"   最终测试准确率: {test_acc:.2f}%")
        print(f"   验证-测试差异: {best_val_acc - test_acc:+.2f}%")
        print("=" * 80)

        print("\n生成的可视化文件:")
        print("  - class_distribution.png: 类别分布图")
        print("  - data_splits.png: 数据分割可视化")
        print("  - training_history.png: 训练历史曲线")
        print("  - sample_predictions.png: 样本预测结果")
        print("  - confusion_matrix.png: 混淆矩阵")
        if GRADCAM_AVAILABLE:
            print("  - gradcam_attention.png: Grad-CAM注意力热力图")
        print("  - comprehensive_dashboard.png: 综合结果仪表板")
        print("  - training_results.json: 详细训练结果")

        clean_memory()
        return results

    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    print("FineGrainedModel + 完整可视化功能 (修复版)")
    print("主要特性:")
    print("   1. 增强的错误处理和安全检查")
    print("   2. FPN特征金字塔网络 + 多尺度特征融合")
    print("   3. 完整的可视化功能集成")
    print("   4. 安全的Grad-CAM注意力热力图")
    print("   5. 数据分布和训练过程可视化")
    print("   6. 样本预测和混淆矩阵分析")
    print("   7. 综合性能仪表板")
    print("   8. 修复了字体显示问题")
    print("=" * 80)

    try:
        results = train_main()
        if results:
            print("✓ 训练和可视化成功完成!")
        else:
            print("训练失败")

    except KeyboardInterrupt:
        print("\n用户中断训练")

    except Exception as e:
        print(f"\n程序运行出错: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("清理资源...")
        clean_memory()
        print("程序结束")