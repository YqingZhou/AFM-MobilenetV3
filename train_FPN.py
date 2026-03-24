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
from tqdm import tqdm
import json
import time
import torch.nn.functional as F
import gc
from thop import profile

# 导入原始MobileNetV3模型
from mobilenetv3_original import MobileNetV3_Large

plt.rcParams['font.family'] = 'SimHei'

def clean_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_and_clear_gpu():
    """检查GPU状态并清理显存"""
    if torch.cuda.is_available():
        print(f"🖥️ GPU设备: {torch.cuda.get_device_name()}")
        print(f"💾 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 清理显存
        torch.cuda.empty_cache()

        # 检查当前显存使用
        memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
        memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3

        print(f"📊 已分配显存: {memory_allocated:.2f} GB")
        print(f"📦 已保留显存: {memory_reserved:.2f} GB")

        if memory_reserved > 0.1:  # 如果有较多保留显存
            print("🔄 清理GPU缓存...")
            torch.cuda.empty_cache()
            print("✅ GPU缓存已清理")
    else:
        print("❌ GPU不可用，将使用CPU")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='训练中')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='验证中')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().itemimport
            os




plt.rcParams['font.family'] = 'SimHei'


def clean_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def check_and_clear_gpu():
    """检查GPU状态并清理显存"""
    if torch.cuda.is_available():
        print(f"🖥️ GPU设备: {torch.cuda.get_device_name()}")
        print(f"💾 显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

        # 清理显存
        torch.cuda.empty_cache()

        # 检查当前显存使用
        memory_allocated = torch.cuda.memory_allocated() / 1024 ** 3
        memory_reserved = torch.cuda.memory_reserved() / 1024 ** 3

        print(f"📊 已分配显存: {memory_allocated:.2f} GB")
        print(f"📦 已保留显存: {memory_reserved:.2f} GB")

        if memory_reserved > 0.1:  # 如果有较多保留显存
            print("🔄 清理GPU缓存...")
            torch.cuda.empty_cache()
            print("✅ GPU缓存已清理")
    else:
        print("❌ GPU不可用，将使用CPU")


class ImprovedLightweightFPN(nn.Module):
    """改进的轻量级FPN"""

    def __init__(self, in_channels_list, fpn_dim=128):
        super(ImprovedLightweightFPN, self).__init__()

        # 动态创建侧向连接
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, fpn_dim, 1, bias=False) for in_ch in in_channels_list
        ])

        # 动态创建输出层
        self.fpn_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, groups=fpn_dim // 4, bias=False),
                nn.BatchNorm2d(fpn_dim),
                nn.ReLU6(inplace=True),
                nn.Conv2d(fpn_dim, fpn_dim, 1, bias=False),
                nn.BatchNorm2d(fpn_dim)
            ) for _ in range(len(in_channels_list))
        ])

        self.activation = nn.ReLU6(inplace=True)
        self.feature_weights = nn.Parameter(torch.ones(len(in_channels_list)))

    def forward(self, features):
        # 侧向连接
        laterals = [conv(feat) for conv, feat in zip(self.lateral_convs, features)]

        # 自顶向下路径
        for i in range(len(laterals) - 1, 0, -1):
            if laterals[i].shape[2:] != laterals[i - 1].shape[2:]:
                upsampled = F.interpolate(
                    laterals[i], size=laterals[i - 1].shape[2:], mode='bilinear', align_corners=False
                )
            else:
                upsampled = laterals[i]

            # 加权融合
            weight_i = torch.sigmoid(self.feature_weights[i])
            weight_i_1 = torch.sigmoid(self.feature_weights[i - 1])
            laterals[i - 1] = weight_i_1 * laterals[i - 1] + weight_i * upsampled

        # 输出处理
        fpn_outs = []
        for i, (conv, lateral) in enumerate(zip(self.fpn_convs, laterals)):
            identity = lateral
            out = conv(lateral)
            out = out + identity
            out = self.activation(out)
            fpn_outs.append(out)

        return fpn_outs


class MultiScaleFeatureFusion(nn.Module):
    """多尺度特征融合模块"""

    def __init__(self, fpn_dim, backbone_dim, num_classes, num_fpn_features, dropout=0.2):
        super(MultiScaleFeatureFusion, self).__init__()

        # 多尺度池化
        self.multi_scale_pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.AdaptiveAvgPool2d((4, 4)),
        ])

        self.backbone_proj = nn.Conv2d(backbone_dim, fpn_dim, 1, bias=False)

        # 每个尺度的特征处理
        self.scale_processors = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim * (num_fpn_features + 1), fpn_dim, 1, bias=False),
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

    def forward(self, fpn_features, backbone_feature):
        # 处理骨干网络特征
        backbone_global = self.backbone_proj(backbone_feature)

        # 合并所有特征
        all_features = fpn_features + [backbone_global]

        # 多尺度特征提取
        scale_features = []
        for pool, processor in zip(self.multi_scale_pools, self.scale_processors):
            # 对每个特征应用当前尺度的池化
            pooled_features = []
            for feat in all_features:
                pooled = pool(feat)
                pooled_features.append(pooled)

            # 拼接同一尺度下的所有特征
            concat_feat = torch.cat(pooled_features, dim=1)

            # 处理并统一到1x1
            processed = processor(concat_feat)
            scale_features.append(processed.flatten(1))

        # 跨尺度融合
        all_scale_features = torch.cat(scale_features, dim=1)
        fused_features = self.cross_scale_fusion(all_scale_features)

        # 分类
        output = self.classifier(fused_features)

        return output


class ImprovedFineGrainedModel(nn.Module):
    """3层FPN细粒度分类模型"""

    def __init__(self, num_classes, fpn_dim=128, dropout=0.2, target_layers=[2, 6, 10]):
        super(ImprovedFineGrainedModel, self).__init__()

        self.backbone = MobileNetV3_Large(num_classes=num_classes)
        self.target_layer_indices = target_layers

        # 动态检测通道数
        self._detect_channel_numbers()

        # 创建FPN和融合模块
        self.fpn = ImprovedLightweightFPN(
            in_channels_list=self.selected_channels, fpn_dim=fpn_dim
        )

        self.feature_fusion = MultiScaleFeatureFusion(
            fpn_dim=fpn_dim,
            backbone_dim=self.final_channels,
            num_classes=num_classes,
            num_fpn_features=len(self.selected_channels),
            dropout=dropout
        )

    def _detect_channel_numbers(self):
        """动态检测各层通道数"""
        test_input = torch.randn(1, 3, 224, 224)
        self.backbone.eval()

        with torch.no_grad():
            features = []
            x = test_input

            x = self.backbone.hs1(self.backbone.bn1(self.backbone.conv1(x)))

            for i, layer in enumerate(self.backbone.bneck):
                x = layer(x)
                if i in self.target_layer_indices:
                    features.append(x.shape[1])

            self.final_channels = x.shape[1]
            self.selected_channels = features

    def extract_selected_features(self, x):
        """提取指定层特征"""
        features = []
        x = self.backbone.hs1(self.backbone.bn1(self.backbone.conv1(x)))

        for i, layer in enumerate(self.backbone.bneck):
            x = layer(x)
            if i in self.target_layer_indices:
                features.append(x)

        final_feature = x
        return features, final_feature

    def forward(self, x):
        selected_features, final_feature = self.extract_selected_features(x)
        fpn_outputs = self.fpn(selected_features)
        output = self.feature_fusion(fpn_outputs, final_feature)
        return output


class PlantDocDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def load_plantdoc_data(data_dir):
    image_paths = []
    labels = []
    class_names = []
    class_to_idx = {}

    # 获取所有类别文件夹
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs.sort()

    for idx, class_name in enumerate(class_dirs):
        class_to_idx[class_name] = idx
        class_names.append(class_name)
        class_dir = os.path.join(data_dir, class_name)

        # 获取该类别下的所有图片
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append(img_path)
                labels.append(idx)

    return image_paths, labels, class_names, class_to_idx


def split_data(image_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
    """按8:1:1比例分割数据"""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    # 先分出训练集和临时集(验证集+测试集)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=(val_ratio + test_ratio),
        random_state=random_state, stratify=labels
    )

    # 再从临时集中分出验证集和测试集
    val_size = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=(1 - val_size),
        random_state=random_state, stratify=temp_labels
    )

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def get_transforms(enable_augmentation=True):
    """定义数据增强和预处理"""
    if enable_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # 增加平移
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
    if os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        model_dict = model.state_dict()
        pretrained_dict = {}

        for k, v in state_dict.items():
            if k in model_dict and model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                backbone_key = f"backbone.{k}"
                if backbone_key in model_dict and model_dict[backbone_key].shape == v.shape:
                    pretrained_dict[backbone_key] = v

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"成功加载 {len(pretrained_dict)} 层权重")
    else:
        print(f"预训练权重文件不存在: {pretrained_path}")


def debug_model_forward(model, sample_input):
    """调试模型前向传播"""
    model.eval()
    with torch.no_grad():
        try:
            features, final_feat = model.extract_selected_features(sample_input)
            fpn_outputs = model.fpn(features)
            output = model(sample_input)
            print(f"模型前向传播成功，输出形状: {output.shape}")
        except Exception as e:
            print(f"前向传播失败: {e}")
            raise e


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='训练中')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='验证中')
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({
                'Loss': f'{val_loss / (len(pbar)):.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })

    return val_loss / len(val_loader), 100. * correct / total


def save_checkpoint(model, optimizer, epoch, loss, acc, filepath):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'accuracy': acc
    }
    torch.save(checkpoint, filepath)


def evaluate_model(model, test_loader, device, class_names):
    """评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_time = 0
    num_samples = 0

    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='评估中'):
            data, target = data.to(device), target.to(device)

            start_time = time.time()
            output = model(data)
            end_time = time.time()

            total_time += (end_time - start_time)
            num_samples += data.size(0)

            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy = 100.0 * np.mean(all_predictions == all_targets)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    precision *= 100
    recall *= 100
    f1 *= 100

    avg_latency = (total_time / num_samples) * 1000

    report = classification_report(
        all_targets, all_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'latency_ms': avg_latency,
        'classification_report': report,
        'predictions': all_predictions,
        'targets': all_targets
    }


def calculate_model_size_and_flops(model, input_size=(1, 3, 224, 224)):
    """计算模型大小和FLOPs"""
    # 保存原始设备
    original_device = next(model.parameters()).device

    # 计算模型大小
    param_size = 0
    buffer_size = 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_size_mb = (param_size + buffer_size) / (1024 * 1024)

    # 计算FLOPs
    dummy_input = torch.randn(input_size)
    model_copy = model.cpu()  # 创建CPU副本用于计算FLOPs
    flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
    flops_g = flops / (1e9)  # 转换为GFLOPs

    # 将模型移回原始设备
    model.to(original_device)

    return model_size_mb, flops_g


def save_evaluation_results(results, model_size_mb, flops_g, save_path):
    """保存评估结果到文件"""
    eval_summary = {
        'Performance Metrics': {
            'Accuracy (%)': f"{results['accuracy']:.2f}",
            'Precision (%)': f"{results['precision']:.2f}",
            'Recall (%)': f"{results['recall']:.2f}",
            'F1-score (%)': f"{results['f1_score']:.2f}"
        },
        'Model Efficiency': {
            'Model Size (MB)': f"{model_size_mb:.2f}",
            'FLOPs (G)': f"{flops_g:.2f}",
            'Latency (ms)': f"{results['latency_ms']:.2f}"
        },
        'Detailed Classification Report': results['classification_report']
    }

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, indent=2, ensure_ascii=False)

    return eval_summary


def print_evaluation_summary(results, model_size_mb, flops_g):
    """打印评估结果"""
    print("\n" + "=" * 50)
    print("           模型评估结果")
    print("=" * 50)
    print(f"准确率:      {results['accuracy']:.2f}%")
    print(f"精确率:      {results['precision']:.2f}%")
    print(f"召回率:      {results['recall']:.2f}%")
    print(f"F1分数:      {results['f1_score']:.2f}%")
    print(f"模型大小:    {model_size_mb:.2f} MB")
    print(f"计算量:      {flops_g:.2f} G")
    print(f"推理延迟:    {results['latency_ms']:.2f} ms")
    print("=" * 50)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label='训练损失', color='blue')
    ax1.plot(val_losses, label='验证损失', color='red')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('轮次')
    ax1.set_ylabel('损失')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_accs, label='训练准确率', color='blue')
    ax2.plot(val_accs, label='验证准确率', color='red')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('轮次')
    ax2.set_ylabel('准确率 (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def main():
    # 配置参数
    config = {
        'data_dir': './merged_dataset',
        'pretrained_path': '450_act3_mobilenetv3_large.pth',
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.0005,
        'weight_decay': 9e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_3layers_mobilenetv3_fpn_all1',
        'num_workers': 4,
        'lr_scheduler': 'cosine',
        'cosine_eta_min_ratio': 0.01,
        'enable_augmentation': True,
        'fpn_dim': 128,
        'dropout': 0,
        'target_layers': [2, 8, 10],
    }

    clean_memory()
    print("=" * 60)
    print("    3层FPN MobileNetV3细粒度分类模型")
    print("=" * 60)

    check_and_clear_gpu()
    os.makedirs(config['save_dir'], exist_ok=True)

    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("加载数据集...")
    image_paths, labels, class_names, class_to_idx = load_plantdoc_data(config['data_dir'])
    num_classes = len(class_names)

    print(f"总样本数: {len(image_paths)}, 类别数: {num_classes}")

    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_data(
        image_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42
    )

    print(f"训练集: {len(train_paths)}, 验证集: {len(val_paths)}, 测试集: {len(test_paths)}")

    class_info = {
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_classes': num_classes
    }
    with open(os.path.join(config['save_dir'], 'class_info.json'), 'w') as f:
        json.dump(class_info, f, indent=2)

    train_transform, val_transform = get_transforms(enable_augmentation=config['enable_augmentation'])

    train_dataset = PlantDocDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantDocDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = PlantDocDataset(test_paths, test_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=True
    )

    print("创建模型...")
    model = ImprovedFineGrainedModel(
        num_classes=num_classes,
        fpn_dim=config['fpn_dim'],
        dropout=config['dropout'],
        target_layers=config['target_layers']
    )

    load_pretrained_weights(model, config['pretrained_path'])

    device = torch.device(config['device'])
    model = model.to(device)

    print(f"使用设备: {device}")
    print(f"模型参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    sample_input = torch.randn(1, 3, 224, 224).to(device)
    debug_model_forward(model, sample_input)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * config['cosine_eta_min_ratio']
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    print("\n开始训练...")
    for epoch in range(config['num_epochs']):
        print(f"\n轮次 {epoch + 1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"训练 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(config['save_dir'], 'best_model.pth')
            )
            print(f"保存最佳模型! 验证准确率: {val_acc:.2f}%")

        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
            )

    save_checkpoint(
        model, optimizer, config['num_epochs'] - 1, val_loss, val_acc,
        os.path.join(config['save_dir'], 'final_model.pth')
    )

    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        os.path.join(config['save_dir'], 'training_history.png')
    )

    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")

    print("\n加载最佳模型进行评估...")
    best_checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(best_checkpoint['state_dict'])

    print("计算模型大小和FLOPs...")
    model_size_mb, flops_g = calculate_model_size_and_flops(model)

    test_results = evaluate_model(model, test_loader, device, class_names)
    print_evaluation_summary(test_results, model_size_mb, flops_g)

    eval_summary = save_evaluation_results(
        test_results, model_size_mb, flops_g,
        os.path.join(config['save_dir'], 'evaluation_results.json')
    )

    plot_confusion_matrix(
        test_results['targets'], test_results['predictions'], class_names,
        os.path.join(config['save_dir'], 'confusion_matrix.png')
    )

    with open(os.path.join(config['save_dir'], 'classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write("3层FPN MobileNetV3分类报告\n")
        f.write("=" * 40 + "\n")
        f.write(f"准确率: {test_results['accuracy']:.2f}%\n")
        f.write(f"精确率: {test_results['precision']:.2f}%\n")
        f.write(f"召回率: {test_results['recall']:.2f}%\n")
        f.write(f"F1分数: {test_results['f1_score']:.2f}%\n")
        f.write(f"模型大小: {model_size_mb:.2f} MB\n")
        f.write(f"计算量: {flops_g:.2f} G\n")
        f.write(f"推理延迟: {test_results['latency_ms']:.2f} ms\n\n")
        f.write("详细分类报告:\n")
        f.write(test_results['classification_report'])

    print(f"\n结果已保存到: {config['save_dir']}")

    return model, config, test_results


if __name__ == '__main__':
    main()