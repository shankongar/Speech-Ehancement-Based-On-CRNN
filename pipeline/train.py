"""
训练脚本
功能：
1. 加载预处理后的数据
2. 训练 CRNN 语音增强模型
3. 保存最优模型和训练日志
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
import random

warnings.filterwarnings("ignore")

from core.model import CRNNSpeechEnhancement
from core.dataset import SpeechEnhancementDataset, collate_fn
from report.visualize import plot_training_loss
from report.logger import log_training_result


# ===================== 断点续训工具 =====================

def save_checkpoint(path, model, optimizer, scheduler, epoch,
                    train_losses, val_losses, lr_history,
                    best_val_loss, best_epoch, args, start_time):
    """保存完整训练状态到 checkpoint 文件"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'start_time': start_time,
        'args': args,
        # RNG 状态
        'rng_torch': torch.get_rng_state(),
        'rng_numpy': np.random.get_state(),
        'rng_random': random.getstate(),
    }
    if torch.cuda.is_available():
        checkpoint['rng_cuda'] = torch.cuda.get_rng_state_all()

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer, scheduler, device, current_args=None):
    """
    从 checkpoint 恢复训练状态

    返回 dict: {
        'epoch', 'train_losses', 'val_losses', 'lr_history',
        'best_val_loss', 'best_epoch', 'start_time'
    }
    """
    import random

    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint 文件不存在: {path}")

    print(f"\n{'='*60}")
    print(f"加载 Checkpoint: {path}")
    print(f"{'='*60}")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # 兼容性校验
    saved_args = checkpoint.get('args')
    if saved_args is not None and current_args is not None:
        mismatches = []
        for key in ['input_dim', 'hidden_dim', 'num_layers', 'batch_size']:
            saved_val = getattr(saved_args, key, None)
            curr_val = getattr(current_args, key, None)
            if saved_val != curr_val:
                mismatches.append(f"  ⚠ {key}: checkpoint={saved_val} vs current={curr_val}")
        if mismatches:
            print("  [超参数不匹配警告]")
            for m in mismatches:
                print(m)
        # 打印保存的超参数供参考
        print(f"  [Checkpoint 超参数]")
        print(f"    input_dim={getattr(saved_args, 'input_dim', '?')}, "
              f"hidden_dim={getattr(saved_args, 'hidden_dim', '?')}, "
              f"num_layers={getattr(saved_args, 'num_layers', '?')}, "
              f"batch_size={getattr(saved_args, 'batch_size', '?')}, "
              f"epochs={getattr(saved_args, 'epochs', '?')}, "
              f"lr={getattr(saved_args, 'lr', '?')}")

    # 恢复模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  ✓ 模型权重已恢复")

    # 恢复优化器状态
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("  ✓ 优化器状态已恢复")

    # 恢复调度器状态
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("  ✓ 学习率调度器状态已恢复")

    # 恢复 RNG 状态
    torch.set_rng_state(checkpoint['rng_torch'])
    np.random.set_state(checkpoint['rng_numpy'])
    random.setstate(checkpoint['rng_random'])
    if 'rng_cuda' in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])
    print("  ✓ 随机数生成器状态已恢复")

    # 提取元数据
    resume_epoch = checkpoint['epoch']
    print(f"\n  >>> 已恢复至 Epoch {resume_epoch} | "
          f"最优 Val Loss: {checkpoint.get('best_val_loss', '?'):.4f} "
          f"(Epoch {checkpoint.get('best_epoch', '?')})")
    print(f"  >>> 历史记录: {len(checkpoint.get('train_losses', []))} 个 epoch")
    print(f"{'='*60}\n")

    return {
        'epoch': resume_epoch,
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
        'lr_history': checkpoint.get('lr_history', []),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'best_epoch': checkpoint.get('best_epoch', 0),
        'start_time': checkpoint.get('start_time', time.time()),
    }


# ===================== 损失函数 =====================

class MaskLoss(nn.Module):
    """
    掩码损失函数
    计算预测掩码与理想掩码之间的 MSE
    理想掩码 = clean_mag / (noisy_mag + eps)
    """

    def __init__(self, eps=1e-8):
        super(MaskLoss, self).__init__()
        self.eps = eps

    def forward(self, pred_mask, noisy_mag, clean_mag):
        """
        :param pred_mask:  预测的掩码, shape = (B, 1, F, T)
        :param noisy_mag:  含噪语音幅度谱, shape = (B, 1, F, T)
        :param clean_mag:  干净语音幅度谱, shape = (B, 1, F, T)
        :return:           MSE 损失
        """
        # 计算理想掩码 (IRM)
        ideal_mask = clean_mag / (noisy_mag + self.eps)
        # 裁剪到 [0, 1]
        ideal_mask = torch.clamp(ideal_mask, 0, 1)

        # MSE 损失
        loss = nn.functional.mse_loss(pred_mask, ideal_mask)
        return loss


# ===================== 训练函数 =====================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0.0
    n_batches = 0

    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # 移动到设备
        noisy_mag = batch['noisy_mag'].to(device)      # (B, 1, F, T)
        clean_mag = batch['clean_mag'].to(device)      # (B, 1, F, T)

        # 前向传播
        pred_mask = model(noisy_mag)                    # (B, 1, F, T)

        # 计算损失
        loss = criterion(pred_mask, noisy_mag, clean_mag)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # 统计
        total_loss += loss.item()
        n_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / n_batches
    return avg_loss


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for batch in pbar:
            # 移动到设备
            noisy_mag = batch['noisy_mag'].to(device)
            clean_mag = batch['clean_mag'].to(device)

            # 前向传播
            pred_mask = model(noisy_mag)

            # 计算损失
            loss = criterion(pred_mask, noisy_mag, clean_mag)

            # 统计
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    return avg_loss


# ===================== 主训练流程 =====================

def train(args):
    """主训练函数"""
    import time
    _train_start_time = time.time()
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 加载数据集
    print("加载数据集...")

    if args.load_mode == 'wav':
        # wav模式: 先加载全部训练数据再划分
        full_dataset = SpeechEnhancementDataset(
            data_dir=args.data_dir,
            mode='train',
            sample_num=args.sample_num if args.sample_num > 0 else None,
            load_mode='wav'
        )

        # 划分训练/验证集
        indices = list(range(len(full_dataset)))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

        train_dataset = torch.utils.data.Subset(full_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(full_dataset, val_idx)

    else:  # npy模式
        npy_base = 'data/processed'
        train_dataset = SpeechEnhancementDataset(
            data_dir=os.path.join(npy_base, 'train'),
            mode='train',
            sample_num=int(args.sample_num * 0.8),  # 训练集占80%
            load_mode='npy'
        )
        val_dataset = SpeechEnhancementDataset(
            data_dir=os.path.join(npy_base, 'val'),
            mode='val',
            sample_num=int(args.sample_num * 0.2),  # 验证集占20%
            load_mode='npy'
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows 兼容性：避免多进程问题
        collate_fn=collate_fn,
        pin_memory=False,  # Windows 上 pin_memory 需要特殊配置
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Windows 兼容性：避免多进程问题
        collate_fn=collate_fn,
        pin_memory=False,  # Windows 上 pin_memory 需要特殊配置
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 打印数据集配置
    print("\n数据集配置:")
    print(f"  - 数据源: {args.data_dir}")
    print(f"  - 加载模式: {args.load_mode}")
    if args.sample_num > 0:
        print(f"  - 采样数量: {args.sample_num}")
    print(f"  - 训练集大小: {len(train_dataset)}")
    print(f"  - 验证集大小: {len(val_dataset)}")

    # 创建模型
    print("创建模型...")
    model = CRNNSpeechEnhancement(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = MaskLoss()

    # ===================== 断点续训：初始化或恢复 =====================
    is_resumed = False
    start_epoch = 1
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    lr_history = []

    # 确定 checkpoint 路径
    resume_path = getattr(args, 'resume', None)
    if not resume_path and getattr(args, 'resume_auto', False):
        _auto_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
        if os.path.exists(_auto_path):
            resume_path = _auto_path

    if resume_path:
        # ---- 恢复模式 ----
        try:
            state = load_checkpoint(resume_path, model, optimizer, scheduler, device, current_args=args)
            start_epoch = state['epoch'] + 1
            train_losses = state['train_losses']
            val_losses = state['val_losses']
            lr_history = state['lr_history']
            best_val_loss = state['best_val_loss']
            best_epoch = state['best_epoch']
            _train_start_time = state['start_time']
            is_resumed = True
        except Exception as e:
            print(f"⚠ Checkpoint 加载失败: {e}，将从头开始训练")
            _train_start_time = time.time()

    print(f"\n开始训练 {args.epochs} 个 epoch... (起始 Epoch: {start_epoch})")
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"训练损失: {train_loss:.4f}")

        # 验证
        val_loss = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        print(f"验证损失: {val_loss:.4f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        # 保存完整 checkpoint（每个 epoch 都保存，用于断点续训）
        _ckpt_path = os.path.join(args.output_dir, 'latest_checkpoint.pth')
        save_checkpoint(_ckpt_path, model, optimizer, scheduler, epoch,
                        train_losses, val_losses, lr_history,
                        best_val_loss, best_epoch, args, _train_start_time)

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            model_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args,
            }, model_path)
            print(f"✓ 保存最优模型 (验证损失: {val_loss:.4f})")

        # 每 2 个 epoch 保存一次
        if epoch % 2 == 0:
            model_path = os.path.join(args.output_dir, f'model_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': args,
            }, model_path)

    # 保存训练日志
    log_path = os.path.join(args.output_dir, 'training_log.npy')
    np.save(log_path, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
    })

    # ===================== 可视化与日志 =====================
    figures_dir = 'results/figures'
    os.makedirs(figures_dir, exist_ok=True)

    # 损失曲线图：同时保存 PNG 和 PDF
    loss_fig_path_png = os.path.join(figures_dir, 'training_loss_curve.png')
    loss_fig_path_pdf = os.path.join(figures_dir, 'training_loss_curve.pdf')
    plot_training_loss(train_losses, val_losses, save_path=loss_fig_path_png)
    plot_training_loss(train_losses, val_losses, save_path=loss_fig_path_pdf)
    print(f"✓ 训练损失曲线已保存: {loss_fig_path_png} / .pdf")

    # ===================== 新增：高级训练分析图 =====================
    training_fig_dir = os.path.join(figures_dir, 'training')
    os.makedirs(training_fig_dir, exist_ok=True)

    # V05: 平滑损失曲线
    try:
        _smooth_png = os.path.join(training_fig_dir, 'training_loss_smooth.png')
        _smooth_pdf = os.path.join(training_fig_dir, 'training_loss_smooth.pdf')
        from report.visualize import plot_training_loss_smooth
        plot_training_loss_smooth(train_losses, val_losses, save_path=_smooth_png)
        plot_training_loss_smooth(train_losses, val_losses, save_path=_smooth_pdf)
        print(f"✓ 平滑损失曲线已保存")
    except Exception as e:
        print(f"✗ 平滑损失曲线生成失败: {e}")

    # V07: 学习率全景图
    try:
        _lr_png = os.path.join(training_fig_dir, 'lr_schedule.png')
        _lr_pdf = os.path.join(training_fig_dir, 'lr_schedule.pdf')
        from report.visualize import plot_lr_warmup_schedule
        plot_lr_warmup_schedule(train_losses, lr_history, save_path=_lr_png)
        plot_lr_warmup_schedule(train_losses, lr_history, save_path=_lr_pdf)
        print(f"✓ 学习率全景图已保存")
    except Exception as e:
        print(f"✗ 学习率全景图生成失败: {e}")

    # V08: 早停指示器
    try:
        _es_png = os.path.join(training_fig_dir, 'early_stopping_indicator.png')
        _es_pdf = os.path.join(training_fig_dir, 'early_stopping_indicator.pdf')
        from report.visualize import plot_early_stopping_indicator
        plot_early_stopping_indicator(val_losses, patience=5, best_epoch=best_epoch,
                                       save_path=_es_png)
        plot_early_stopping_indicator(val_losses, patience=5, best_epoch=best_epoch,
                                       save_path=_es_pdf)
        print(f"✓ 早停指示器已保存 (best_epoch={best_epoch})")
    except Exception as e:
        print(f"✗ 早停指示器生成失败: {e}")

    # 计算训练耗时
    train_duration_sec = time.time() - _train_start_time

    # 构造 extra_info
    extra_info = {
        'device': str(device),
        'param_count': sum(p.numel() for p in model.parameters()),
        'duration_sec': train_duration_sec,
        'lr_history': lr_history,
        'best_epoch': best_epoch,
        'initial_lr': lr_history[0] if lr_history else args.lr,
        'final_lr': lr_history[-1] if lr_history else args.lr,
        'n_reductions': sum(1 for i in range(1, len(lr_history)) if lr_history[i] < lr_history[i-1]) if len(lr_history) > 1 else 0,
    }

    # 写入增强版实验日志
    log_path = 'results/logs/experiment_log.txt'
    # 续训模式：先写入分隔标记
    if is_resumed:
        with open(log_path, 'a', encoding='utf-8') as _f:
            _f.write('\n')
            _f.write('=' * 80 + '\n')
            _f.write(f'[=== RESUMED FROM EPOCH {start_epoch} ===] {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
            _f.write('=' * 80 + '\n\n')
    log_training_result(log_path, args, train_losses, val_losses, best_val_loss, extra_info=extra_info)
    print(f"✓ 训练日志已追加: {log_path}")

    print(f"\n训练完成！")
    print(f"最优验证损失: {best_val_loss:.4f}")
    print(f"模型和日志保存在: {args.output_dir}")


def init_seed(seed=42):
    """初始化随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    from config import get_config
    import argparse
    cfg = get_config()
    args = argparse.Namespace(
        data_dir=cfg.data_dir,
        load_mode=cfg.load_mode,
        sample_num=cfg.sample_num,
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        output_dir=cfg.output_dir,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        sr=cfg.sr,
        model_name=cfg.model_name,
        cpu=cfg.cpu,
        resume=cfg.resume_path or None,
        resume_auto=cfg.resume_auto,
    )
    init_seed()
    train(args)
