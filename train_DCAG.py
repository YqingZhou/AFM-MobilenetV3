import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from collections import defaultdict, Counter
import shutil
import time
from thop import profile, clever_format
import cv2

# 注意力可视化相关导入
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

# 导入你的MobileNetV3模型
from mobilenetv3_se import MobileNetV3WithAttention

# 全局变量用于存储随机种子
GLOBAL_SEED = 42


def worker_init_fn(worker_id):
    """DataLoader的worker初始化函数，确保每个worker有不同的随机种子"""
    np.random.seed(GLOBAL_SEED + worker_id)
    random.seed(GLOBAL_SEED + worker_id)


def set_seed(seed=42):
    """设置所有随机种子以确保结果可复现"""
    global GLOBAL_SEED
    GLOBAL_SEED = seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 设置CuDNN为确定性模式（可能会降低性能）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 设置Python哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"已设置随机种子为: {seed}")


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


def plot_class_distribution(labels, class_names, save_path):
    """可视化类别分布"""
    counter = Counter(labels)
    classes = [class_names[i] for i in range(len(class_names))]
    counts = [counter[i] for i in range(len(class_names))]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(classes, counts, color='skyblue', alpha=0.7)
    plt.title('Class Distribution in Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')

    # 在柱状图上添加数值标签
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(count), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 打印统计信息
    print(f"\n类别分布统计:")
    print(f"总样本数: {sum(counts)}")
    print(f"平均每类: {np.mean(counts):.1f}")
    print(f"最多类别: {class_names[np.argmax(counts)]} ({max(counts)} 样本)")
    print(f"最少类别: {class_names[np.argmin(counts)]} ({min(counts)} 样本)")


def plot_data_splits(train_labels, val_labels, test_labels, class_names, save_path):
    """可视化数据集分割情况"""
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    test_counter = Counter(test_labels)

    x = np.arange(len(class_names))
    width = 0.25

    train_counts = [train_counter[i] for i in range(len(class_names))]
    val_counts = [val_counter[i] for i in range(len(class_names))]
    test_counts = [test_counter[i] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, train_counts, width, label='Train', alpha=0.8, color='lightblue')
    bars2 = ax.bar(x, val_counts, width, label='Validation', alpha=0.8, color='lightgreen')
    bars3 = ax.bar(x + width, test_counts, width, label='Test', alpha=0.8, color='lightcoral')

    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('Data Split Distribution by Class', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_learning_rate_schedule(learning_rates, save_path):
    """可视化学习率变化"""
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, linewidth=2, color='orange')
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 修复后的注意力可视化模块 ====================

def diagnose_model_structure(model):
    """诊断模型结构，查找可能的注意力模块"""
    print("=" * 60)
    print("模型结构诊断")
    print("=" * 60)

    attention_modules = []
    conv_modules = []
    se_modules = []

    for name, module in model.named_modules():
        module_type = module.__class__.__name__

        # 查找注意力相关模块
        if any(keyword in name.lower() for keyword in ['attention', 'att', 'gate']):
            attention_modules.append((name, module_type))
        elif any(keyword in module_type.lower() for keyword in ['attention', 'att', 'gate']):
            attention_modules.append((name, module_type))

        # 查找SE模块
        elif any(keyword in name.lower() for keyword in ['se', 'squeeze', 'excitation']):
            se_modules.append((name, module_type))

        # 查找重要的卷积层
        elif 'conv' in module_type.lower() and len(list(module.children())) == 0:
            conv_modules.append((name, module_type))

    print(f"找到 {len(attention_modules)} 个注意力模块:")
    for name, mtype in attention_modules[:5]:
        print(f"  - {name}: {mtype}")

    print(f"\n找到 {len(se_modules)} 个SE模块:")
    for name, mtype in se_modules[:5]:
        print(f"  - {name}: {mtype}")

    print(f"\n找到 {len(conv_modules)} 个卷积层 (显示前5个):")
    for name, mtype in conv_modules[:5]:
        print(f"  - {name}: {mtype}")

    # 推荐用于可视化的层
    recommended_layers = []

    # 优先推荐注意力模块
    if attention_modules:
        recommended_layers.extend([name for name, _ in attention_modules[:3]])
    elif se_modules:
        recommended_layers.extend([name for name, _ in se_modules[:3]])
    else:
        # 选择一些关键的卷积层
        important_convs = [name for name, _ in conv_modules if
                           any(keyword in name for keyword in ['features', 'layer', 'block'])]
        recommended_layers.extend(important_convs[:3])

    print(f"\n推荐用于可视化的层:")
    for layer in recommended_layers:
        print(f"  - {layer}")

    return recommended_layers


def simple_grad_cam_visualization(model, test_loader, device, class_names, save_path, num_samples=8):
    """使用Grad-CAM进行注意力可视化"""
    if not GRADCAM_AVAILABLE:
        print("❌ Grad-CAM不可用，跳过此可视化")
        return

    print("使用Grad-CAM进行可视化...")

    # 诊断模型结构
    recommended_layers = diagnose_model_structure(model)

    if not recommended_layers:
        print("未找到合适的可视化层，使用默认策略...")
        # 尝试找到最后几个卷积层
        conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
                conv_layers.append(name)
        recommended_layers = conv_layers[-3:] if conv_layers else []

    if not recommended_layers:
        print("❌ 无法找到合适的层进行可视化")
        return

    # 获取测试样本
    images_to_viz = []
    labels_to_viz = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            for i in range(min(num_samples, images.size(0))):
                images_to_viz.append(images[i])
                labels_to_viz.append(labels[i])
            if len(images_to_viz) >= num_samples:
                break

    if not images_to_viz:
        print("没有找到测试样本")
        return

    # 选择要可视化的层（最多3个）
    target_layers_names = recommended_layers[:3]

    # 创建可视化
    num_layers = len(target_layers_names)
    fig, axes = plt.subplots(num_samples, num_layers + 1, figsize=(4 * (num_layers + 1), 4 * num_samples))
    fig.suptitle('Grad-CAM Attention Visualization', fontsize=16, fontweight='bold')

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_layers + 1 == 1:
        axes = axes.reshape(-1, 1)

    for idx, (image, label) in enumerate(zip(images_to_viz, labels_to_viz)):
        if idx >= num_samples:
            break

        # 原始图像
        img_np = image.cpu()
        img_np = img_np * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(
            3, 1, 1)
        img_np = torch.clamp(img_np, 0, 1).permute(1, 2, 0).numpy()

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f'Original\nClass: {class_names[label.item()]}', fontsize=10)
        axes[idx, 0].axis('off')

        # 对每个推荐层生成Grad-CAM
        for layer_idx, layer_name in enumerate(target_layers_names):
            try:
                # 获取目标层
                target_layer = None
                for name, module in model.named_modules():
                    if name == layer_name:
                        target_layer = module
                        break

                if target_layer is None:
                    axes[idx, layer_idx + 1].axis('off')
                    axes[idx, layer_idx + 1].text(0.5, 0.5, f'Layer not found:\n{layer_name}',
                                                  ha='center', va='center',
                                                  transform=axes[idx, layer_idx + 1].transAxes)
                    continue

                # 创建Grad-CAM
                cam = GradCAM(model=model, target_layers=[target_layer])

                # 生成CAM
                targets = [ClassifierOutputTarget(label.item())]
                grayscale_cam = cam(input_tensor=image.unsqueeze(0), targets=targets)
                grayscale_cam = grayscale_cam[0, :]

                # 叠加到原图
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

                axes[idx, layer_idx + 1].imshow(visualization)
                short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name[:15]
                axes[idx, layer_idx + 1].set_title(f'Grad-CAM\n{short_name}', fontsize=10)
                axes[idx, layer_idx + 1].axis('off')

            except Exception as e:
                print(f"处理层 {layer_name} 时出错: {e}")
                axes[idx, layer_idx + 1].axis('off')
                axes[idx, layer_idx + 1].text(0.5, 0.5, f'Error:\n{str(e)[:20]}...',
                                              ha='center', va='center', transform=axes[idx, layer_idx + 1].transAxes)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Grad-CAM可视化完成，保存到: {save_path}")


def feature_map_visualization(model, test_loader, device, class_names, save_path, num_samples=4):
    """修复版特征图可视化 - 为每个样本单独生成特征图"""
    print("生成特征图可视化（修复版：每个样本独立计算）...")

    # 获取测试样本
    images_to_viz = []
    labels_to_viz = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            for i in range(min(num_samples, images.size(0))):
                images_to_viz.append(images[i])
                labels_to_viz.append(labels[i])
            if len(images_to_viz) >= num_samples:
                break

    if not images_to_viz:
        print("没有找到测试样本")
        return

    # 找到要可视化的卷积层
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
            target_layers.append((name, module))
            if len(target_layers) >= 3:  # 只取前3层
                break

    if not target_layers:
        print("未找到合适的卷积层")
        return

    print(f"将可视化这些层: {[name for name, _ in target_layers]}")

    # 为每个样本分别生成特征图
    all_feature_maps = []

    for sample_idx, (image, label) in enumerate(zip(images_to_viz, labels_to_viz)):
        print(f"处理样本 {sample_idx + 1}/{len(images_to_viz)}: {class_names[label.item()]}")

        sample_feature_maps = {}

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                if len(output.shape) == 4:  # [B, C, H, W]
                    B, C, H, W = output.shape
                    if H >= 7 and W >= 7 and C <= 512:  # 选择合适大小的特征图
                        # 计算每个通道的平均激活强度
                        channel_importance = torch.mean(torch.abs(output), dim=[2, 3])  # [B, C]
                        top_channels = torch.topk(channel_importance[0], k=min(6, C)).indices

                        sample_feature_maps[layer_name] = {
                            'features': output[0][top_channels].detach().cpu(),  # 只保存重要通道
                            'shape': (H, W)
                        }

            return hook_fn

        # 为当前样本注册hooks
        hooks = []
        for layer_name, module in target_layers:
            hook = module.register_forward_hook(create_hook_fn(layer_name))
            hooks.append(hook)

        # 为当前样本单独进行前向传播（关键修复点）
        model.eval()
        with torch.no_grad():
            _ = model(image.unsqueeze(0))  # 使用当前样本而不是第一个样本

        # 移除hooks
        for hook in hooks:
            hook.remove()

        all_feature_maps.append(sample_feature_maps)

    # 检查是否成功提取到特征图
    if not all_feature_maps or not all_feature_maps[0]:
        print("未能提取到特征图")
        return

    # 创建可视化
    num_layers = len(target_layers)
    fig, axes = plt.subplots(num_samples, num_layers + 1, figsize=(4 * (num_layers + 1), 4 * num_samples))
    fig.suptitle('Feature Maps Visualization (Fixed)', fontsize=16, fontweight='bold')

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_layers + 1 == 1:
        axes = axes.reshape(-1, 1)

    layer_names = [name for name, _ in target_layers]

    for idx, (image, label) in enumerate(zip(images_to_viz, labels_to_viz)):
        if idx >= num_samples:
            break

        # 显示原始图像
        img_np = image.cpu()
        img_np = img_np * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(
            3, 1, 1)
        img_np = torch.clamp(img_np, 0, 1).permute(1, 2, 0).numpy()

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f'Original\nClass: {class_names[label.item()]}', fontsize=10)
        axes[idx, 0].axis('off')

        # 显示对应样本的特征图
        current_feature_maps = all_feature_maps[idx]

        for layer_idx, layer_name in enumerate(layer_names):
            if layer_name in current_feature_maps:
                features = current_feature_maps[layer_name]['features']  # [6, H, W]

                # 将多个通道合并为一个可视化图
                combined_feature = torch.mean(torch.abs(features), dim=0)  # [H, W]

                # 上采样到输入图像大小
                combined_feature = torch.nn.functional.interpolate(
                    combined_feature.unsqueeze(0).unsqueeze(0),
                    size=(224, 224), mode='bilinear', align_corners=False
                )[0, 0]

                # 归一化到0-1范围
                combined_feature = (combined_feature - combined_feature.min()) / (
                        combined_feature.max() - combined_feature.min() + 1e-8)

                # 叠加到原图
                axes[idx, layer_idx + 1].imshow(img_np)
                axes[idx, layer_idx + 1].imshow(combined_feature.numpy(), alpha=0.6, cmap='jet')

                short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name[:15]
                axes[idx, layer_idx + 1].set_title(f'Features\n{short_name}', fontsize=10)
                axes[idx, layer_idx + 1].axis('off')
            else:
                # 如果没有特征图，显示错误信息
                axes[idx, layer_idx + 1].axis('off')
                axes[idx, layer_idx + 1].text(0.5, 0.5, 'No Features', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 特征图可视化完成，保存到: {save_path}")
    print("现在每个样本都有自己独特的特征图了！")


def improved_feature_visualization(model, test_loader, device, class_names, save_path, num_samples=4):
    """改进的特征图可视化 - 使用全局归一化避免伪影"""
    print("生成改进的特征图可视化（全局归一化）...")

    images_to_viz = []
    labels_to_viz = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            for i in range(min(num_samples, images.size(0))):
                images_to_viz.append(images[i])
                labels_to_viz.append(labels[i])
            if len(images_to_viz) >= num_samples:
                break

    if not images_to_viz:
        print("没有找到测试样本")
        return

    # 找到要可视化的卷积层
    target_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and len(list(module.children())) == 0:
            target_layers.append((name, module))
            if len(target_layers) >= 3:
                break

    if not target_layers:
        print("未找到合适的卷积层")
        return

    print(f"将可视化这些层: {[name for name, _ in target_layers]}")

    # 收集所有样本的特征图统计信息
    all_feature_maps = []
    all_feature_stats = []

    for sample_idx, (image, label) in enumerate(zip(images_to_viz, labels_to_viz)):
        sample_feature_maps = {}

        def create_hook_fn(layer_name):
            def hook_fn(module, input, output):
                if len(output.shape) == 4:
                    B, C, H, W = output.shape
                    if H >= 7 and W >= 7 and C <= 512:
                        channel_importance = torch.mean(torch.abs(output), dim=[2, 3])
                        top_channels = torch.topk(channel_importance[0], k=min(6, C)).indices

                        features = output[0][top_channels].detach().cpu()
                        sample_feature_maps[layer_name] = {
                            'features': features,
                            'shape': (H, W)
                        }

                        # 记录统计信息用于全局归一化
                        all_feature_stats.append({
                            'layer': layer_name,
                            'min': features.min().item(),
                            'max': features.max().item(),
                            'mean': features.mean().item()
                        })

            return hook_fn

        hooks = []
        for layer_name, module in target_layers:
            hook = module.register_forward_hook(create_hook_fn(layer_name))
            hooks.append(hook)

        model.eval()
        with torch.no_grad():
            _ = model(image.unsqueeze(0))

        for hook in hooks:
            hook.remove()

        all_feature_maps.append(sample_feature_maps)

    # 计算每层的全局统计量
    layer_global_stats = {}
    for stat in all_feature_stats:
        layer_name = stat['layer']
        if layer_name not in layer_global_stats:
            layer_global_stats[layer_name] = {'min': [], 'max': [], 'mean': []}
        layer_global_stats[layer_name]['min'].append(stat['min'])
        layer_global_stats[layer_name]['max'].append(stat['max'])
        layer_global_stats[layer_name]['mean'].append(stat['mean'])

    # 计算全局归一化参数（使用分位数避免极端值）
    global_norm_params = {}
    for layer_name, stats in layer_global_stats.items():
        global_min = np.percentile(stats['min'], 5)
        global_max = np.percentile(stats['max'], 95)
        global_norm_params[layer_name] = {
            'min': global_min,
            'max': global_max
        }
        print(f"  {layer_name}: min={global_min:.4f}, max={global_max:.4f}")

    # 创建可视化
    num_layers = len(target_layers)
    fig, axes = plt.subplots(num_samples, num_layers + 1,
                             figsize=(4 * (num_layers + 1), 4 * num_samples))
    fig.suptitle('Improved Feature Maps (Global Normalization)',
                 fontsize=16, fontweight='bold')

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_layers + 1 == 1:
        axes = axes.reshape(-1, 1)

    layer_names = [name for name, _ in target_layers]

    for idx, (image, label) in enumerate(zip(images_to_viz, labels_to_viz)):
        if idx >= num_samples:
            break

        # 显示原始图像
        img_np = image.cpu()
        img_np = img_np * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
                 torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        img_np = torch.clamp(img_np, 0, 1).permute(1, 2, 0).numpy()

        axes[idx, 0].imshow(img_np)
        axes[idx, 0].set_title(f'Original\n{class_names[label.item()]}', fontsize=10)
        axes[idx, 0].axis('off')

        # 显示特征图
        current_feature_maps = all_feature_maps[idx]

        for layer_idx, layer_name in enumerate(layer_names):
            if layer_name in current_feature_maps:
                features = current_feature_maps[layer_name]['features']

                # 合并通道
                combined_feature = torch.mean(torch.abs(features), dim=0)

                # 上采样
                combined_feature = torch.nn.functional.interpolate(
                    combined_feature.unsqueeze(0).unsqueeze(0),
                    size=(224, 224), mode='bilinear', align_corners=False
                )[0, 0]

                # 使用全局归一化参数
                norm_params = global_norm_params[layer_name]
                combined_feature = torch.clamp(combined_feature,
                                               norm_params['min'],
                                               norm_params['max'])
                combined_feature = (combined_feature - norm_params['min']) / \
                                   (norm_params['max'] - norm_params['min'] + 1e-8)

                # 单独显示特征图（不叠加原图）
                im = axes[idx, layer_idx + 1].imshow(combined_feature.numpy(),
                                                     cmap='jet', vmin=0, vmax=1)

                short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name[:15]
                axes[idx, layer_idx + 1].set_title(f'{short_name}\nGlobal Norm', fontsize=10)
                axes[idx, layer_idx + 1].axis('off')

                # 添加颜色条
                plt.colorbar(im, ax=axes[idx, layer_idx + 1], fraction=0.046, pad=0.04)
            else:
                axes[idx, layer_idx + 1].axis('off')
                axes[idx, layer_idx + 1].text(0.5, 0.5, 'No Features',
                                              ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 改进的特征图可视化完成: {save_path}")


def comprehensive_attention_visualization(model, test_loader, device, class_names, save_dir):
    """综合的注意力和特征可视化"""
    print("\n" + "=" * 60)
    print("开始综合注意力可视化（修复版）")
    print("=" * 60)

    try:
        # 方法1: Grad-CAM可视化
        simple_grad_cam_visualization(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            save_path=os.path.join(save_dir, "gradcam_attention.png"),
            num_samples=6
        )
    except Exception as e:
        print(f"Grad-CAM可视化失败: {e}")

    try:
        # 方法2: 修复后的特征图可视化
        feature_map_visualization(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            save_path=os.path.join(save_dir, "feature_maps_fixed.png"),
            num_samples=4
        )
    except Exception as e:
        print(f"特征图可视化失败: {e}")

    try:
        # 方法3: 改进的全局归一化特征图
        improved_feature_visualization(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            save_path=os.path.join(save_dir, "feature_maps_improved.png"),
            num_samples=4
        )
    except Exception as e:
        print(f"改进特征图可视化失败: {e}")

    print("综合可视化完成!")


# ==================== 其他可视化函数 ====================

def visualize_sample_predictions(model, test_loader, device, class_names, save_path, num_samples=16):
    """可视化样本预测结果"""
    model.eval()

    # 收集一些预测结果
    all_images = []
    all_predictions = []
    all_targets = []
    all_confidences = []

    with torch.no_grad():
        for images, targets in test_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = torch.nn.functional.softmax(model(images), dim=1)
            confidences, predictions = torch.max(outputs, 1)

            for i in range(images.size(0)):
                all_images.append(images[i].cpu())
                all_predictions.append(predictions[i].cpu().item())
                all_targets.append(targets[i].cpu().item())
                all_confidences.append(confidences[i].cpu().item())

                if len(all_images) >= num_samples:
                    break
            if len(all_images) >= num_samples:
                break

    # 创建可视化
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

    for i in range(min(num_samples, len(all_images))):
        row, col = i // 4, i % 4
        ax = axes[row, col]

        # 反归一化图像用于显示
        image = all_images[i]
        image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(3,
                                                                                                                     1,
                                                                                                                     1)
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

        title = f"Pred: {pred_label}\nTrue: {true_label}\nConf: {confidence:.2f}"
        ax.set_title(title, fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_comprehensive_visualization_dashboard(config, results, model_size_mb, flops_g, save_dir):
    """创建综合可视化仪表板"""
    fig = plt.figure(figsize=(20, 12))

    # 创建子图布局
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # 1. 模型性能指标 (左上)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    values = [results['accuracy'], results['precision'], results['recall'], results['f1_score']]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_title('Performance Metrics (%)', fontweight='bold', fontsize=12)
    ax1.set_ylim(0, 100)

    for bar, value in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. 模型效率指标 (右上)
    ax2 = fig.add_subplot(gs[0, 1])
    efficiency_metrics = ['Size (MB)', 'FLOPs (G)', 'Latency (ms)']
    efficiency_values = [model_size_mb, flops_g, results['latency_ms']]

    bars2 = ax2.bar(efficiency_metrics, efficiency_values, color=['#FFA07A', '#98D8C8', '#F7DC6F'], alpha=0.8)
    ax2.set_title('Model Efficiency', fontweight='bold', fontsize=12)

    for bar, value in zip(bars2, efficiency_values):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(efficiency_values) * 0.01,
                 f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. 配置信息 (左中到右中)
    ax3 = fig.add_subplot(gs[0, 2:])
    ax3.axis('off')
    config_text = f"""
Training Configuration:
• Epochs: {config['num_epochs']}
• Batch Size: {config['batch_size']}
• Learning Rate: {config['learning_rate']:.2e}
• Scheduler: {config['lr_scheduler']}
• Attention: {config['use_attention']}
• Seed: {config['seed']}
• Data Augmentation: {config['enable_augmentation']}
    """
    ax3.text(0.1, 0.5, config_text, fontsize=11, va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    # 4. 训练设备和环境信息
    device_info = f"""
Environment:
• Device: {config['device']}
• Deterministic: {config['deterministic']}
• Workers: {config['num_workers']}
    """
    ax3.text(0.6, 0.5, device_info, fontsize=11, va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))

    # 5. 性能总结 (底部全宽)
    ax4 = fig.add_subplot(gs[1:, :])
    ax4.axis('off')

    summary_text = f"""
Model Performance Summary:

✓ Classification Accuracy: {results['accuracy']:.2f}% 
✓ Model Size: {model_size_mb:.2f} MB (Efficient for mobile deployment)
✓ Inference Speed: {results['latency_ms']:.2f} ms per sample
✓ FLOPs: {flops_g:.2f} G (Computational efficiency)

Attention Mechanism Impact:
{'• Enhanced feature representation with attention mechanism' if config['use_attention'] else '• Standard MobileNetV3 without attention'}
{'• Spatial and channel attention fusion' if config['use_attention'] else ''}
{'• Learnable gating mechanism for adaptive feature selection' if config['use_attention'] else ''}

Visualization Improvements:
• Fixed: Each sample now uses its own feature maps (not shared)
• Added: Global normalization (5%-95% percentile) to avoid artifacts
• Improved: Separate visualization for feature maps and attention mechanisms
• Enhanced: Color bars showing absolute activation strength

Training Strategy:
• {config['lr_scheduler'].title()} learning rate scheduling for optimal convergence
• {'Cosine annealing with smooth decay' if config['lr_scheduler'] == 'cosine' else 'Step-wise learning rate reduction'}
• {'Data augmentation enabled for better generalization' if config['enable_augmentation'] else 'No data augmentation applied'}
• {'Deterministic mode for reproducible results' if config['deterministic'] else 'Non-deterministic training mode'}
    """

    ax4.text(0.05, 0.95, summary_text, fontsize=12, va='top', ha='left',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
             transform=ax4.transAxes)

    fig.suptitle('MobileNetV3 with Attention - Training Results Dashboard (Fixed)',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig(os.path.join(save_dir, 'comprehensive_dashboard.png'),
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# ==================== 训练相关函数 ====================

def load_pretrained_weights(model, pretrained_path, skip_se=True):
    """加载预训练权重，可选择跳过SE层"""
    if os.path.exists(pretrained_path):
        print(f"加载预训练权重: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')

        # 如果保存的是完整模型状态
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # 获取当前模型的状态字典
        model_dict = model.state_dict()

        print(f"预训练权重包含 {len(state_dict)} 层")
        print(f"当前模型包含 {len(model_dict)} 层")

        # 调试：打印一些层名称以检查匹配情况
        print("\n预训练权重前5层:")
        for i, key in enumerate(list(state_dict.keys())[:5]):
            print(f"  {key}: {state_dict[key].shape}")

        print("\n当前模型前5层:")
        for i, key in enumerate(list(model_dict.keys())[:5]):
            print(f"  {key}: {model_dict[key].shape}")

        # 尝试直接匹配层名称
        pretrained_dict = {}
        skipped_se_layers = []
        skipped_attention_layers = []
        shape_mismatch_layers = []

        for k, v in state_dict.items():
            # 检查层名称是否存在于当前模型中
            if k in model_dict:
                # 检查形状是否匹配
                if model_dict[k].shape == v.shape:
                    # 如果需要跳过SE层
                    if skip_se and any(
                            se_keyword in k.lower() for se_keyword in ['se.', '.se.', 'squeeze', 'excitation']):
                        skipped_se_layers.append(k)
                        continue
                    pretrained_dict[k] = v
                else:
                    shape_mismatch_layers.append((k, model_dict[k].shape, v.shape))
            else:
                # 尝试映射到backbone
                backbone_key = f"backbone.{k}"
                if backbone_key in model_dict and model_dict[backbone_key].shape == v.shape:
                    if skip_se and any(
                            se_keyword in k.lower() for se_keyword in ['se.', '.se.', 'squeeze', 'excitation']):
                        skipped_se_layers.append(backbone_key)
                        continue
                    pretrained_dict[backbone_key] = v
                    print(f"映射权重: {k} -> {backbone_key}")

        # 检查注意力层（这些层在预训练权重中不存在，这是正常的）
        for k in model_dict.keys():
            if 'attention' in k.lower():
                skipped_attention_layers.append(k)

        # 更新模型权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"\n权重加载结果:")
        print(f"  成功加载: {len(pretrained_dict)}/{len(model_dict)} 层权重")
        print(f"  跳过的SE层: {len(skipped_se_layers)}")
        print(f"  新增的注意力层: {len(skipped_attention_layers)}")
        print(f"  形状不匹配层: {len(shape_mismatch_layers)}")

        if shape_mismatch_layers:
            print("\n形状不匹配的层:")
            for layer, model_shape, pretrained_shape in shape_mismatch_layers[:3]:  # 只显示前3个
                print(f"  {layer}: 模型 {model_shape} vs 预训练 {pretrained_shape}")

        if len(pretrained_dict) == 0:
            print("\n⚠️ 警告：没有成功加载任何预训练权重！")
            print("可能的原因：")
            print("1. 预训练权重文件格式与当前模型不兼容")
            print("2. 层名称完全不匹配")
            print("3. 模型结构发生了重大变化")
            print("模型将使用随机初始化的权重进行训练。")
        else:
            print(f"✓ 成功加载了 {len(pretrained_dict)} 个层的预训练权重")

    else:
        print(f"预训练权重文件不存在: {pretrained_path}")
        print("模型将使用随机初始化的权重进行训练。")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
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

        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss / (batch_idx + 1):.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
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

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


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
    """全面评估模型性能"""
    model.eval()
    all_predictions = []
    all_targets = []
    total_time = 0
    num_samples = 0

    print("正在评估模型...")
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc='Evaluating'):
            data, target = data.to(device), target.to(device)

            # 测量推理时间
            start_time = time.time()
            output = model(data)
            end_time = time.time()

            total_time += (end_time - start_time)
            num_samples += data.size(0)

            _, predicted = output.max(1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # 计算各项指标
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 准确率
    accuracy = 100.0 * np.mean(all_predictions == all_targets)

    # 精确率、召回率、F1分数
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    precision *= 100
    recall *= 100
    f1 *= 100

    # 平均推理时间 (毫秒)
    avg_latency = (total_time / num_samples) * 1000

    # 生成详细的分类报告
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
    """打印评估结果摘要"""
    print("\n" + "=" * 60)
    print("                    模型评估结果")
    print("=" * 60)
    print(f"Accuracy (%):      {results['accuracy']:.2f}")
    print(f"Precision (%):     {results['precision']:.2f}")
    print(f"Recall (%):        {results['recall']:.2f}")
    print(f"F1-score (%):      {results['f1_score']:.2f}")
    print(f"Model Size (MB):   {model_size_mb:.2f}")
    print(f"FLOPs (G):         {flops_g:.2f}")
    print(f"Latency (ms):      {results['latency_ms']:.2f}")
    print("=" * 60)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)

    # 添加数值标注
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 损失曲线
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)

    # 准确率曲线
    ax2.plot(train_accs, label='Train Acc', color='blue')
    ax2.plot(val_accs, label='Val Acc', color='red')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ==================== 主函数 ====================

def main():
    # 配置参数
    config = {
        'data_dir': './PlantDoc',  # 请修改为你的数据集路径
        'pretrained_path': '450_act3_mobilenetv3_large.pth',
        'batch_size': 32,
        'num_epochs': 1,
        'learning_rate': 0.0002,
        'weight_decay': 5e-5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_attention_fixed',  #1 修改保存目录名
        'num_workers': 0 if os.name == 'nt' else 4,
        'lr_scheduler': 'cosine',
        'cosine_eta_min_ratio': 0.01,
        'skip_se_weights': False,
        'seed': 42,
        'enable_augmentation': True,
        'deterministic': True,
        'use_attention': True
    }

    # Windows环境提示
    if os.name == 'nt' and config['num_workers'] == 0:
        print("检测到Windows环境，已自动设置num_workers=0以避免多进程问题")

    # 设置随机种子以确保结果可复现
    if config['deterministic']:
        set_seed(config['seed'])
        print("注意：确定性模式已启用，这会确保结果可复现但可能影响性能")

    # 创建保存目录
    os.makedirs(config['save_dir'], exist_ok=True)

    # 保存配置信息
    with open(os.path.join(config['save_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print("正在加载数据集...")
    # 加载数据
    image_paths, labels, class_names, class_to_idx = load_plantdoc_data(config['data_dir'])
    num_classes = len(class_names)

    print(f"数据集信息:")
    print(f"  总样本数: {len(image_paths)}")
    print(f"  类别数: {num_classes}")
    print(f"  类别名称: {class_names}")

    # 可视化数据集分布
    plot_class_distribution(labels, class_names,
                            os.path.join(config['save_dir'], 'class_distribution.png'))

    # 分割数据集
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = split_data(
        image_paths, labels, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=config['seed']
    )

    print(f"数据分割:")
    print(f"  训练集: {len(train_paths)} 样本")
    print(f"  验证集: {len(val_paths)} 样本")
    print(f"  测试集: {len(test_paths)} 样本")

    # 可视化数据分割情况
    plot_data_splits(train_labels, val_labels, test_labels, class_names,
                     os.path.join(config['save_dir'], 'data_splits.png'))

    # 保存类别信息
    class_info = {
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'num_classes': num_classes
    }
    with open(os.path.join(config['save_dir'], 'class_info.json'), 'w') as f:
        json.dump(class_info, f, indent=2)

    # 创建数据变换
    train_transform, val_transform = get_transforms(enable_augmentation=config['enable_augmentation'])

    # 创建数据集和数据加载器
    train_dataset = PlantDocDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantDocDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = PlantDocDataset(test_paths, test_labels, transform=val_transform)

    # 在Windows下处理多进程数据加载
    use_multiprocessing = config['num_workers'] > 0 and config['deterministic']

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'], pin_memory=True,
        worker_init_fn=worker_init_fn if use_multiprocessing else None
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=True,
        worker_init_fn=worker_init_fn if use_multiprocessing else None
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, num_workers=config['num_workers'], pin_memory=True,
        worker_init_fn=worker_init_fn if use_multiprocessing else None
    )

    # 创建模型
    print(f"创建模型... 使用注意力机制: {config['use_attention']}")
    model = MobileNetV3WithAttention(num_classes=num_classes)

    # 加载预训练权重
    load_pretrained_weights(model, config['pretrained_path'], skip_se=config['skip_se_weights'])

    # 将模型移动到设备
    device = torch.device(config['device'])
    model = model.to(device)

    print(f"使用设备: {device}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"学习率调度器: {config['lr_scheduler']}")
    if config['lr_scheduler'] == 'cosine':
        eta_min = config['learning_rate'] * config['cosine_eta_min_ratio']
        print(f"余弦调度器 - 初始LR: {config['learning_rate']:.2e}, 最小LR: {eta_min:.2e}")

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度器
    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * config['cosine_eta_min_ratio']
        )
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9, gamma=0.5)

    # 训练历史记录
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    learning_rates = []
    best_val_acc = 0.0

    print("开始训练...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)

        # 记录当前学习率
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # 显示余弦调度器的学习率变化信息
        if config['lr_scheduler'] == 'cosine':
            progress = (epoch + 1) / config['num_epochs']
            print(
                f"训练进度: {progress:.1%}, LR衰减进度: {1 - optimizer.param_groups[0]['lr'] / config['learning_rate']:.1%}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(config['save_dir'], 'best_model.pth')
            )
            print(f"保存最佳模型! Val Acc: {val_acc:.2f}%")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_loss, val_acc,
                os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch + 1}.pth')
            )

    # 保存最终模型
    save_checkpoint(
        model, optimizer, config['num_epochs'] - 1, val_loss, val_acc,
        os.path.join(config['save_dir'], 'final_model.pth')
    )

    # 绘制训练历史
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        os.path.join(config['save_dir'], 'training_history.png')
    )

    # 绘制学习率变化
    plot_learning_rate_schedule(learning_rates,
                                os.path.join(config['save_dir'], 'learning_rate_schedule.png'))

    print(f"\n训练完成!")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")

    # 加载最佳模型进行最终评估
    print("\n加载最佳模型进行最终评估...")
    best_checkpoint = torch.load(os.path.join(config['save_dir'], 'best_model.pth'))
    model.load_state_dict(best_checkpoint['state_dict'])

    # 计算模型大小和FLOPs
    print("计算模型大小和FLOPs...")
    model_size_mb, flops_g = calculate_model_size_and_flops(model)

    # 在测试集上评估模型
    test_results = evaluate_model(model, test_loader, device, class_names)

    # 打印评估结果摘要
    print_evaluation_summary(test_results, model_size_mb, flops_g)

    # 保存详细评估结果
    eval_summary = save_evaluation_results(
        test_results, model_size_mb, flops_g,
        os.path.join(config['save_dir'], 'evaluation_results.json')
    )

    # 绘制混淆矩阵
    plot_confusion_matrix(
        test_results['targets'], test_results['predictions'], class_names,
        os.path.join(config['save_dir'], 'confusion_matrix.png')
    )

    # 可视化样本预测结果
    visualize_sample_predictions(model, test_loader, device, class_names,
                                 os.path.join(config['save_dir'], 'sample_predictions.png'))

    # 综合注意力可视化（修复版）
    if config['use_attention']:
        comprehensive_attention_visualization(
            model=model,
            test_loader=test_loader,
            device=device,
            class_names=class_names,
            save_dir=config['save_dir']
        )

    # 创建综合仪表板
    create_comprehensive_visualization_dashboard(config, test_results, model_size_mb, flops_g, config['save_dir'])

    # 保存详细的分类报告
    with open(os.path.join(config['save_dir'], 'classification_report.txt'), 'w') as f:
        f.write("Classification Report - MobileNetV3 with Attention (Fixed)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Attention Mechanism: {config['use_attention']}\n")
        f.write(f"Bug Fixes:\n")
        f.write(f"  - Fixed: Each sample now uses its own feature maps\n")
        f.write(f"  - Added: Global normalization to avoid brightness artifacts\n")
        f.write(f"  - Improved: Separate visualizations for features and attention\n\n")
        f.write(f"Learning Rate Scheduler: {config['lr_scheduler']}\n")
        if config['lr_scheduler'] == 'cosine':
            f.write(
                f"Cosine LR - Initial: {config['learning_rate']:.2e}, Min: {config['learning_rate'] * config['cosine_eta_min_ratio']:.2e}\n")
        f.write(f"Accuracy: {test_results['accuracy']:.2f}%\n")
        f.write(f"Precision: {test_results['precision']:.2f}%\n")
        f.write(f"Recall: {test_results['recall']:.2f}%\n")
        f.write(f"F1-score: {test_results['f1_score']:.2f}%\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")
        f.write(f"FLOPs: {flops_g:.2f} G\n")
        f.write(f"Latency: {test_results['latency_ms']:.2f} ms\n\n")
        f.write("Detailed Report:\n")
        f.write(test_results['classification_report'])

    print(f"\n所有结果已保存到: {config['save_dir']}")
    print("\n生成的文件包括:")
    print("  - best_model.pth: 最佳模型权重")
    print("  - evaluation_results.json: 评估结果JSON")
    print("  - classification_report.txt: 详细分类报告")
    print("  - confusion_matrix.png: 混淆矩阵图")
    print("  - training_history.png: 训练历史曲线")
    print("  - learning_rate_schedule.png: 学习率变化曲线")
    print("  - class_distribution.png: 类别分布图")
    print("  - data_splits.png: 数据分割可视化")
    print("  - sample_predictions.png: 样本预测可视化")
    if GRADCAM_AVAILABLE:
        print("  - gradcam_attention.png: Grad-CAM注意力热力图")
    print("  - feature_maps_fixed.png: 修复后的特征图可视化（每个样本独立）")
    print("  - feature_maps_improved.png: 改进的特征图可视化（全局归一化）")
    print("  - comprehensive_dashboard.png: 综合仪表板")
    print("  - class_info.json: 类别信息")
    print("  - config.json: 实验配置")

    print("\n=== Bug修复说明 ===")
    print("✓ 修复：每个样本现在使用自己的特征图，而不是共享第一个样本的特征图")
    print("✓ 改进：添加了全局归一化版本，避免局部归一化导致的伪影")
    print("✓ 增强：提供了两个版本的特征图可视化供对比")
    print("  - feature_maps_fixed.png: 修复了样本匹配问题")
    print("  - feature_maps_improved.png: 使用全局归一化，避免亮度伪影")


if __name__ == '__main__':
    main()