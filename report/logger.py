import os
import numpy as np
from datetime import datetime


def log_training_result(log_path='results/logs/experiment_log.txt', args=None,
                        train_losses=None, val_losses=None, best_val_loss=None,
                        extra_info=None):
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if args is None:
        args = type('Args', (), {})()

    extra = extra_info or {}

    # ---- 防御性读取 args ----
    model_name = getattr(args, 'model_name', None) or 'CRNN'
    input_dim = getattr(args, 'input_dim', 0)
    hidden_dim = getattr(args, 'hidden_dim', 0)
    num_layers = getattr(args, 'num_layers', 0)
    epochs = getattr(args, 'epochs', 0)
    batch_size = getattr(args, 'batch_size', 0)
    lr = getattr(args, 'lr', 0.0)
    weight_decay = getattr(args, 'weight_decay', 0.0)
    sample_num = getattr(args, 'sample_num', 0)
    load_mode = getattr(args, 'load_mode', 'N/A')
    data_dir = getattr(args, 'data_dir', 'N/A')
    output_dir = getattr(args, 'output_dir', 'N/A')

    # ---- 从 extra_info 获取额外信息，带默认值 ----
    device = extra.get('device', 'cpu')
    gpu_name = extra.get('gpu_name', 'N/A')
    python_version = extra.get('python_version', 'N/A')
    pytorch_version = extra.get('pytorch_version', 'N/A')
    param_count = extra.get('param_count', 0)
    best_epoch = extra.get('best_epoch', -1)
    duration_sec = extra.get('duration_sec', 0.0)
    lr_history = extra.get('lr_history', [])
    initial_lr = lr_history[0] if lr_history else float(lr)
    final_lr = lr_history[-1] if lr_history else float(lr)
    n_reductions = extra.get('n_reductions', 0)

    # ---- 计算派生值 ----
    total_epochs = len(train_losses) if train_losses is not None else 0
    last_train_loss = train_losses[-1] if (train_losses and len(train_losses)) else 0.0
    last_val_loss = val_losses[-1] if (val_losses and len(val_losses)) else 0.0
    best_vl = best_val_loss if best_val_loss is not None else last_val_loss
    duration_min = duration_sec / 60.0

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write('================================================================================\n')
        f.write(f'[训练记录] {timestamp}\n')
        f.write('--------------------------------------------------------------------------------\n')

        # [环境信息]
        f.write('[环境信息]\n')
        f.write(f'  设备: {device}\n')
        f.write(f'  GPU: {gpu_name}\n')
        f.write(f'  Python: {python_version}\n')
        f.write(f'  PyTorch: {pytorch_version}\n')
        f.write('\n')

        # [模型结构]
        f.write('[模型结构]\n')
        f.write(f'  架构: CRNN (Conv + BiLSTM + DeConv)\n')
        f.write(f'  input_dim: {input_dim}\n')
        f.write(f'  hidden_dim: {hidden_dim}\n')
        f.write(f'  num_layers: {num_layers}\n')
        f.write(f'  总参数量: {param_count:,}\n')
        f.write('\n')

        # [超参数配置]
        f.write('[超参数配置]\n')
        f.write(
            f'  epochs: {epochs} | batch_size: {batch_size} | lr: {lr}\n')
        f.write(
            f'  weight_decay: {weight_decay} | sample_num: {sample_num}\n')
        f.write(f'  load_mode: {load_mode}\n')
        f.write(f'  data_dir: {data_dir}\n')
        f.write(f'  output_dir: {output_dir}\n')
        f.write('\n')

        # [训练过程摘要]
        f.write('[训练过程摘要]\n')
        f.write(f'  总轮数: {total_epochs}\n')
        f.write(f'  最优 epoch: {best_epoch} (val_loss={best_vl:.4f})\n')
        f.write(f'  最终 train_loss: {last_train_loss:.4f}\n')
        f.write(f'  最终 val_loss:   {last_val_loss:.4f}\n')
        f.write('\n')

        # [Epoch 详情]
        f.write('[Epoch 详情]\n')
        f.write('  Epoch | Train Loss | Val Loss\n')
        f.write('  ------|------------|----------\n')
        if train_losses is not None and val_losses is not None:
            for i in range(len(train_losses)):
                tl = train_losses[i]
                vl = val_losses[i] if i < len(val_losses) else 0.0
                f.write(f'  {i+1:>5}  |   {tl:<8.4f} |  {vl:<8.4f}\n')
        f.write('\n')

        # [学习率调度]
        f.write('[学习率调度]\n')
        f.write(f'  初始 lr: {initial_lr:.6f}\n')
        f.write(f'  最终 lr: {final_lr:.6f}\n')
        f.write(f'  衰减次数: {n_reductions}\n')
        f.write('\n')

        # [耗时统计]
        f.write('[耗时统计]\n')
        f.write(f'  训练总时长: {duration_sec:.1f} 秒 ({duration_min:.1f} 分钟)\n')
        f.write('\n')

        # [输出文件]
        f.write('[输出文件]\n')
        f.write(f'  模型权重: {output_dir}/best_model.pth\n')
        f.write(f'  训练日志: {output_dir}/training_log.npy\n')
        f.write(f'  损失曲线图: results/figures/training_loss_curve.png (.pdf)\n')

        f.write('================================================================================\n\n')


def log_inference_result(log_path='results/logs/experiment_log.txt', args=None,
                         metrics_dict=None, dataset_source=None,
                         extra_info=None):
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if args is None:
        args = type('Args', (), {})()

    extra = extra_info or {}
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def format_value(val):
        if np.isnan(val) or np.isinf(val):
            return 'N/A'
        sign = '+' if val >= 0 else ''
        return f'{sign}{val:.2f}'

    snr_before_mean = np.mean(metrics_dict['snr_before'])
    snr_before_std = np.std(metrics_dict['snr_before'])
    snr_after_mean = np.mean(metrics_dict['snr_after'])
    snr_after_std = np.std(metrics_dict['snr_after'])
    si_snr_before_mean = np.mean(metrics_dict['si_snr_before'])
    si_snr_before_std = np.std(metrics_dict['si_snr_before'])
    si_snr_after_mean = np.mean(metrics_dict['si_snr_after'])
    si_snr_after_std = np.std(metrics_dict['si_snr_after'])
    seg_snr_before_mean = np.mean(metrics_dict['seg_snr_before'])
    seg_snr_before_std = np.std(metrics_dict['seg_snr_before'])
    seg_snr_after_mean = np.mean(metrics_dict['seg_snr_after'])
    seg_snr_after_std = np.std(metrics_dict['seg_snr_after'])

    snr_improve = snr_after_mean - snr_before_mean
    si_snr_improve = si_snr_after_mean - si_snr_before_mean
    seg_snr_improve = seg_snr_after_mean - seg_snr_before_mean

    model_path = getattr(args, 'model_path', 'N/A')
    sample_num = getattr(args, 'sample_num', 0)

    # ---- 从 extra_info 获取 STFT 等参数 ----
    n_fft = extra.get('n_fft', 512)
    hop_length = extra.get('hop_length', 128)
    win_length = extra.get('win_length', 512)
    sr = extra.get('sr', 16000)

    with open(log_path, mode='a', encoding='utf-8') as f:
        f.write('================================================================================\n')
        f.write(f'[评估记录] {timestamp}\n')
        f.write('--------------------------------------------------------------------------------\n')
        if dataset_source:
            f.write(f'数据集来源: {dataset_source}\n')
        f.write('模型信息:\n')
        f.write(f'  model_path: {model_path}\n')
        f.write(f'  sample_num: {sample_num}\n')

        # 如果有数据集信息，打印详细信息
        if 'dataset_info' in metrics_dict:
            info = metrics_dict['dataset_info']
            f.write(f'  数据模式: {info.get("mode", "N/A")}\n')
            f.write(f'  有效样本: {info.get("valid_samples", "N/A")}/{info.get("total_samples", "N/A")}\n')
            if info.get('skipped_files'):
                f.write(f'  跳过文件: {len(info["skipped_files"])} 个\n')
            if info.get('error_files'):
                f.write(f'  错误文件: {len(info["error_files"])} 个\n')

        f.write('评估指标 (均值 ± 标准差):\n')
        f.write(
            f"  SNR:      增强前 {format_value(snr_before_mean)} ± {format_value(snr_before_std)}  →  增强后 {format_value(snr_after_mean)} ± {format_value(snr_after_std)}  (提升 {format_value(snr_improve)} dB)\n")
        f.write(
            f"  SI-SNR:   增强前 {format_value(si_snr_before_mean)} ± {format_value(si_snr_before_std)}  →  增强后 {format_value(si_snr_after_mean)} ± {format_value(si_snr_after_std)}  (提升 {format_value(si_snr_improve)} dB)\n")
        f.write(
            f"  SegSNR:   增强前 {format_value(seg_snr_before_mean)} ± {format_value(seg_snr_before_std)} →  增强后 {format_value(seg_snr_after_mean)} ± {format_value(seg_snr_after_std)} (提升 {format_value(seg_snr_improve)} dB)\n")

        # ---- 新增区段: [STFT 参数] ----
        f.write('\n[STFT 参数]\n')
        f.write(f'  n_fft: {n_fft} | hop_length: {hop_length} | win_length: {win_length} | sr: {sr}\n')

        # ---- 新增区段: [逐样本统计] ----
        snr_diff = np.asarray(metrics_dict['snr_after']) - np.asarray(metrics_dict['snr_before'])
        sisnr_diff = np.asarray(metrics_dict['si_snr_after']) - np.asarray(metrics_dict['si_snr_before'])
        segsnr_diff = np.asarray(metrics_dict['seg_snr_after']) - np.asarray(metrics_dict['seg_snr_before'])

        f.write('\n[逐样本统计]\n')
        f.write(f"  SNR 提升:    最佳={format_value(np.max(snr_diff))} dB | 最差={format_value(np.min(snr_diff))} dB | 中位数={format_value(np.median(snr_diff))} dB\n")
        f.write(f"  SI-SNR 提升: 最佳={format_value(np.max(sisnr_diff))} dB | 最差={format_value(np.min(sisnr_diff))} dB | 中位数={format_value(np.median(sisnr_diff))} dB\n")
        f.write(f"  SegSNR 提升: 最佳={format_value(np.max(segsnr_diff))} dB | 最差={format_value(np.min(segsnr_diff))} dB | 中位数={format_value(np.median(segsnr_diff))} dB\n")

        # ---- 新增区段: [质量预警] ----
        snr_after_arr = np.asarray(metrics_dict['snr_after'])
        n_failed = int(np.sum(snr_after_arr < 0))
        total = len(snr_after_arr)
        pct = (n_failed / total * 100) if total > 0 else 0.0

        f.write('\n[质量预警]\n')
        f.write(f'  增强后 SNR < 0 dB 的样本数: {n_failed}/{total} ({pct:.1f}%)\n')

        f.write('================================================================================\n\n')
