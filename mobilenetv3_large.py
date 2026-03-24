import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import time
from thop import profile

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


def load_plantdoc_dataset(data_dir):
    """
    加载已经分好训练集和测试集的PlantDoc-Dataset数据集
    预期数据集结构:
    PlantDoc-Dataset/
    ├── train/
    │   ├── class1/
    │   ├── class2/
    │   └── ...
    └── test/
        ├── class1/
        ├── class2/
        └── ...
    """
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise ValueError(f"数据集目录结构不正确。请确保 {data_dir} 包含 'train' 和 'test' 子目录")

    # 获取类别名称（从训练集目录获取）
    class_names = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_names.sort()
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}

    def load_split_data(split_dir):
        image_paths = []
        labels = []

        for class_name in class_names:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 在 {split_dir} 中未找到类别 {class_name}")
                continue

            class_idx = class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    image_paths.append(img_path)
                    labels.append(class_idx)

        return image_paths, labels

    # 加载训练集和测试集
    train_paths, train_labels = load_split_data(train_dir)
    test_paths, test_labels = load_split_data(test_dir)

    return (train_paths, train_labels), (test_paths, test_labels), class_names, class_to_idx


def split_validation_from_train(train_paths, train_labels, val_ratio=0.2, random_state=42):
    """从训练集中分出验证集"""
    train_paths_new, val_paths, train_labels_new, val_labels = train_test_split(
        train_paths, train_labels, test_size=val_ratio,
        random_state=random_state, stratify=train_labels
    )
    return (train_paths_new, train_labels_new), (val_paths, val_labels)


def get_transforms(enable_augmentation=True):
    """定义数据增强和预处理"""
    if enable_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
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


def create_mobilenet_model(num_classes, model_type='mobilenet_v3_large', pretrained=True):
    """创建MobileNetV3模型

    Args:
        num_classes: 分类数量
        model_type: MobileNetV3模型类型，可选 'mobilenet_v3_large', 'mobilenet_v3_small'
        pretrained: 是否使用预训练权重
    """
    print(f"创建{model_type.upper()}模型，类别数: {num_classes}")

    # 选择MobileNetV3模型类型
    if model_type == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(pretrained=pretrained)
        model_name = "MobileNetV3-Large"
    elif model_type == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=pretrained)
        model_name = "MobileNetV3-Small"
    else:
        raise ValueError(f"不支持的MobileNetV3模型类型: {model_type}")

    if pretrained:
        print("✓ 加载ImageNet预训练权重")
    else:
        print("使用随机初始化权重")

    # 修改分类器的最后一层以适应新的类别数
    # MobileNetV3的分类器结构: classifier[3] 是最后的Linear层
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, num_classes)

    print(f"模型创建完成，最后一层: {num_ftrs} -> {num_classes}")
    print(f"{model_name} 特征提取器层数: {len(model.features)}")
    print(f"{model_name} 分类器层数: {len(model.classifier)}")

    # 打印模型架构信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    return model


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
    try:
        flops, params = profile(model_copy, inputs=(dummy_input,), verbose=False)
        flops_g = flops / (1e9)  # 转换为GFLOPs
    except Exception as e:
        print(f"FLOPs计算失败: {e}")
        flops_g = 0.0

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
            'FLOPs (G)': f"{flops_g:.2f}" if flops_g > 0 else "N/A",
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
    if flops_g > 0:
        print(f"FLOPs (G):         {flops_g:.2f}")
    else:
        print(f"FLOPs (G):         N/A")
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
    plt.show()


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
    plt.show()


def main():
    # 配置参数
    config = {
        'data_dir': './PlantDoc-Dataset',  # 请修改为你的数据集路径
        'model_type': 'mobilenet_v3_large',  # 选择MobileNetV3模型: 'mobilenet_v3_large', 'mobilenet_v3_small'
        'batch_size': 32,
        'num_epochs': 30,
        'learning_rate': 0.0005,  # MobileNetV3可以使用相对较高的学习率
        'weight_decay': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': 'checkpoints_mobilenetv3_large',  # 会根据model_type自动调整
        'num_workers': 4 if os.name != 'nt' else 0,  # Windows下使用单进程
        'lr_scheduler': 'cosine',  # 使用余弦调度器
        'cosine_eta_min_ratio': 0.01,  # 最小学习率比例
        'seed': 42,  # 随机种子
        'enable_augmentation': True,  # 是否启用数据增强
        'deterministic': True,  # 是否使用确定性模式
        'pretrained': True,  # 是否使用ImageNet预训练权重
        'val_ratio': 0.1,  # 从训练集中分出的验证集比例
    }

    # 根据模型类型调整保存目录
    config['save_dir'] = f"checkpoints_{config['model_type']}"

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

    print("正在加载PlantDoc-Dataset数据集...")

    try:
        # 加载已经分好的训练集和测试集
        (train_paths, train_labels), (test_paths, test_labels), class_names, class_to_idx = load_plantdoc_dataset(
            config['data_dir'])
        num_classes = len(class_names)

        print(f"数据集信息:")
        print(f"  训练集样本数: {len(train_paths)}")
        print(f"  测试集样本数: {len(test_paths)}")
        print(f"  类别数: {num_classes}")
        print(f"  类别名称: {class_names}")

        # 从训练集中分出验证集
        (train_paths, train_labels), (val_paths, val_labels) = split_validation_from_train(
            train_paths, train_labels, val_ratio=config['val_ratio'], random_state=config['seed']
        )

        print(f"数据分割:")
        print(f"  训练集: {len(train_paths)} 样本")
        print(f"  验证集: {len(val_paths)} 样本")
        print(f"  测试集: {len(test_paths)} 样本")

    except ValueError as e:
        print(f"数据集加载失败: {e}")
        print("请确保数据集目录结构正确:")
        print("PlantDoc-Dataset/")
        print("├── train/")
        print("│   ├── class1/")
        print("│   ├── class2/")
        print("│   └── ...")
        print("└── test/")
        print("    ├── class1/")
        print("    ├── class2/")
        print("    └── ...")
        return

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

    # 创建MobileNetV3模型
    model = create_mobilenet_model(num_classes, config['model_type'], pretrained=config['pretrained'])

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
    best_val_acc = 0.0

    print("开始训练...")
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        print("-" * 50)

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

    # 保存详细的分类报告
    with open(os.path.join(config['save_dir'], 'classification_report.txt'), 'w') as f:
        f.write(f"Classification Report - {config['model_type'].upper()}\n")
        f.write("=" * 50 + "\n")
        f.write(f"Model Type: {config['model_type'].upper()}\n")
        f.write(f"Pretrained: {'Yes (ImageNet)' if config['pretrained'] else 'No'}\n")
        f.write(f"Learning Rate Scheduler: {config['lr_scheduler']}\n")
        if config['lr_scheduler'] == 'cosine':
            f.write(
                f"Cosine LR - Initial: {config['learning_rate']:.2e}, Min: {config['learning_rate'] * config['cosine_eta_min_ratio']:.2e}\n")
        f.write(f"Accuracy: {test_results['accuracy']:.2f}%\n")
        f.write(f"Precision: {test_results['precision']:.2f}%\n")
        f.write(f"Recall: {test_results['recall']:.2f}%\n")
        f.write(f"F1-score: {test_results['f1_score']:.2f}%\n")
        f.write(f"Model Size: {model_size_mb:.2f} MB\n")
        if flops_g > 0:
            f.write(f"FLOPs: {flops_g:.2f} G\n")
        else:
            f.write(f"FLOPs: N/A\n")
        f.write(f"Latency: {test_results['latency_ms']:.2f} ms\n\n")
        f.write("Detailed Report:\n")
        f.write(test_results['classification_report'])

    print(f"\n所有结果已保存到: {config['save_dir']}")
    print("生成的文件包括:")
    print("  - best_model.pth: 最佳模型权重")
    print("  - evaluation_results.json: 评估结果JSON")
    print("  - classification_report.txt: 详细分类报告")
    print("  - confusion_matrix.png: 混淆矩阵图")
    print("  - training_history.png: 训练历史曲线")
    print("  - class_info.json: 类别信息")
    print("  - config.json: 实验配置")

    # 可复现性说明
    print("\n=== 可复现性说明 ===")
    print(f"随机种子: {config['seed']}")
    print(f"确定性模式: {config['deterministic']}")
    print(f"数据增强: {config['enable_augmentation']}")
    print(f"预训练权重: {'ImageNet' if config['pretrained'] else '随机初始化'}")

    if config['deterministic']:
        print("\n✓ 确定性模式已启用")
        print("  - 相同环境下多次运行将得到相同结果")
        print("  - 模型仍会正常训练，准确率不是固定值")
        print("  - 消除了随机性带来的结果差异")
    else:
        print("\n注意：如需完全可复现的结果，请设置 deterministic=True")

    # MobileNetV3模型特性说明
    print(f"\n=== {config['model_type'].upper()}模型说明 ===")
    print(f"✓ 使用{config['model_type'].upper()}架构")

    # MobileNetV3模型信息
    mobilenet_info = {
        'mobilenet_v3_large': {
            'params': '~5.4M',
            'flops': '~219M',
            'description': 'MobileNetV3-Large，平衡精度和效率',
            'features': ['更大的模型容量', '更高的准确率', '适合对精度要求较高的场景']
        },
        'mobilenet_v3_small': {
            'params': '~2.9M',
            'flops': '~66M',
            'description': 'MobileNetV3-Small，极致轻量化',
            'features': ['参数量极少', '推理速度快', '适合移动端和边缘设备']
        }
    }

    if config['model_type'] in mobilenet_info:
        info = mobilenet_info[config['model_type']]
        print(f"  - 预训练参数量: {info['params']}")
        print(f"  - 预训练FLOPs: {info['flops']}")
        print(f"  - 描述: {info['description']}")
        print("  - 特点:")
        for feature in info['features']:
            print(f"    * {feature}")

    if config['pretrained']:
        print("✓ 使用ImageNet预训练权重进行迁移学习")
        print("  - 预训练权重有助于提高收敛速度和最终性能")
        print("  - MobileNetV3在ImageNet上表现优异，适合图像分类任务")
        print("  - 特别适合需要在移动设备上部署的植物分类应用")
    else:
        print("使用随机初始化权重从头训练")

    print(f"✓ 分类器最后全连接层已适配为 {num_classes} 类分类")
    print("✓ MobileNetV3特点:")
    print("  - 使用神经架构搜索(NAS)优化的网络结构")
    print("  - 引入Squeeze-and-Excitation(SE)注意力机制")
    print("  - 使用h-swish激活函数替代ReLU")
    print("  - 优化的反向残差块(Inverted Residual Block)")
    print("  - 专为移动设备优化，推理速度快")
    print("  - 模型轻量化，内存占用小")

    print("\n=== 学习率调整说明 ===")
    print(f"MobileNetV3模型学习率设置为 {config['learning_rate']:.2e}")
    print("  - MobileNetV3通常可以使用比VGG更高的学习率")
    print("  - 轻量化模型训练相对稳定，收敛较快")

    if config['lr_scheduler'] == 'cosine':
        print("✓ 使用余弦退火学习率调度器")
        print(f"  - 初始学习率: {config['learning_rate']:.2e}")
        print(f"  - 最小学习率: {config['learning_rate'] * config['cosine_eta_min_ratio']:.2e}")
        print(f"  - 余弦周期: {config['num_epochs']} epochs")
        print("  - 优势: 平滑的学习率衰减，有助于MobileNetV3收敛到更好的局部最优解")
    else:
        print("使用阶梯式学习率调度器")

    print("\n=== 数据集说明 ===")
    print("✓ 使用预分割的PlantDoc-Dataset数据集")
    print(f"  - 训练集: {len(train_paths)} 样本")
    print(f"  - 验证集: {len(val_paths)} 样本 (从训练集分出{config['val_ratio'] * 100:.0f}%)")
    print(f"  - 测试集: {len(test_paths)} 样本")
    print(f"  - 总类别数: {num_classes}")

    # MobileNetV3 vs VGG vs ResNet 比较提示
    print(f"\n=== MobileNetV3 vs 其他模型 比较 ===")
    print("MobileNetV3特点:")
    print("  ✓ 极致轻量化，参数量和计算量最小")
    print("  ✓ 推理速度极快，延迟低")
    print("  ✓ 内存占用小，适合移动设备")
    print("  ✓ 使用神经架构搜索优化，效率极高")
    print("  ✓ 支持量化和剪枝等进一步优化")
    print("  ✓ 特别适合边缘计算和实时应用")
    print("  - 在复杂任务上精度可能略低于大模型")

    print("VGG特点:")
    print("  ✓ 结构简单，易于理解")
    print("  ✓ 特征提取能力强")
    print("  - 参数量大(~138M)，计算开销高")
    print("  - 推理速度慢，内存占用大")

    print("ResNet特点:")
    print("  ✓ 残差连接，可训练更深网络")
    print("  ✓ 精度高，泛化能力强")
    print("  ✓ 训练稳定")
    print("  - 参数量中等(~25M)，计算量适中")

    print("\n=== 模型选择建议 ===")
    if config['model_type'] == 'mobilenet_v3_large':
        print("MobileNetV3-Large适用场景:")
        print("  ✓ 需要在移动设备/边缘设备上部署")
        print("  ✓ 对推理速度有较高要求")
        print("  ✓ 希望平衡精度和效率")
        print("  ✓ 资源受限的环境")
        print("  ✓ 需要实时植物识别的应用")
    else:
        print("MobileNetV3-Small适用场景:")
        print("  ✓ 极致轻量化需求")
        print("  ✓ 低功耗设备")
        print("  ✓ 对模型大小有严格限制")
        print("  ✓ 简单的植物分类任务")
        print("  ✓ 嵌入式设备应用")

    print("\n=== 部署优化建议 ===")
    print("MobileNetV3进一步优化选项:")
    print("  - 模型量化: 将float32转为int8，减少4倍模型大小")
    print("  - 模型剪枝: 移除不重要的连接，进一步减少参数")
    print("  - 知识蒸馏: 使用大模型指导小模型训练")
    print("  - TensorRT优化: 针对NVIDIA GPU的推理优化")
    print("  - ONNX转换: 跨平台部署")
    print("  - Core ML转换: iOS设备部署")
    print("  - TensorFlow Lite: Android设备部署")


if __name__ == '__main__':
    main()