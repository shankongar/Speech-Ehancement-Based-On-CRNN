import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import os
from scipy import signal as sig


def _setup_academic_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'SimHei'],
        'mathtext.fontset': 'stix',
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (8, 6),
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })


def _ensure_dir(path):
    if path:
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def _save_figure(fig, save_path):
    """根据扩展名自动选择格式保存图片"""
    if not save_path:
        plt.close(fig)
        return
    _ensure_dir(save_path)
    if save_path.lower().endswith('.pdf'):
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
    else:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_training_loss(train_losses, val_losses, save_path, title='Training Loss Curve'):
    _setup_academic_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-o', markersize=3, linewidth=1.5, label='Train Loss')
    ax.plot(epochs, val_losses, 'r-s', markersize=3, linewidth=1.5, label='Validation Loss')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    _save_figure(fig, save_path)


def plot_spectrogram_comparison(clean_mag, noisy_mag, enhanced_mag, sr=16000, hop_length=128, save_path=None):
    _setup_academic_style()
    fig, axes = plt.subplots(3, 1, figsize=(10, 9))
    
    specs = [clean_mag, noisy_mag, enhanced_mag]
    titles = ['Clean Speech', 'Noisy Speech', 'Enhanced Speech']
    
    for idx, (ax, spec, title) in enumerate(zip(axes, specs, titles)):
        im = ax.pcolormesh(spec.T, cmap='viridis', shading='auto')
        
        freq_bins, time_frames = spec.shape
        freqs = np.linspace(0, sr / 2, freq_bins) / 1000
        times = np.linspace(0, time_frames * hop_length / sr, time_frames)
        
        ax.set_ylabel('Frequency (kHz)')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, format='%.2f')
    
    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_metrics_comparison(metrics_dict, save_path):
    _setup_academic_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = ['SNR', 'SI-SNR', 'SegSNR']
    keys_before = ['snr_before', 'si_snr_before', 'seg_snr_before']
    keys_after = ['snr_after', 'si_snr_after', 'seg_snr_after']
    
    x = np.arange(len(groups))
    width = 0.35
    
    before_vals = [np.mean(metrics_dict.get(k, [0])) for k in keys_before]
    after_vals = [np.mean(metrics_dict.get(k, [0])) for k in keys_after]
    
    bars1 = ax.bar(x - width/2, before_vals, width, label='Before Enhancement', color='#A5C8E1')
    bars2 = ax.bar(x + width/2, after_vals, width, label='After Enhancement', color='#2171B5')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('dB')
    ax.set_title('Speech Enhancement Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_waveform_comparison(clean_wav, noisy_wav, enhanced_wav, sr=16000, save_path=None):
    _setup_academic_style()
    fig, axes = plt.subplots(3, 1, figsize=(12, 7))

    wavs = [clean_wav, noisy_wav, enhanced_wav]
    titles = ['Clean Speech', 'Noisy Speech', 'Enhanced Speech']

    for ax, wav, title in zip(axes, wavs, titles):
        times = np.linspace(0, len(wav) / sr, len(wav))
        ax.plot(times, wav, linewidth=0.5, color='#333333')
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Time (s)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_mask_heatmap(pred_mask, ideal_mask, noisy_mag, save_path=None):
    _setup_academic_style()

    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    if ideal_mask.ndim == 3:
        ideal_mask = ideal_mask.squeeze(0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    im0 = axes[0].imshow(pred_mask.T, aspect='auto', cmap='hot', origin='lower')
    axes[0].set_title('Predicted IRM')
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Frequency Bin')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(ideal_mask.T, aspect='auto', cmap='hot', origin='lower')
    axes[1].set_title('Ideal IRM')
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Frequency Bin')
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_sample_detail(clean_wav, noisy_wav, enhanced_wav, clean_mag, noisy_mag,
                       enhanced_mag, pred_mask, ideal_mask, sr=16000, hop_length=128,
                       save_path=None):
    _setup_academic_style()

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))

    times = np.linspace(0, len(clean_wav) / sr, len(clean_wav))
    axes[0, 0].plot(times, clean_wav, linewidth=0.3)
    axes[0, 0].set_title('Clean')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')

    axes[0, 1].plot(times, noisy_wav, linewidth=0.3)
    axes[0, 1].set_title('Noisy')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Amplitude')

    axes[1, 0].plot(times, enhanced_wav, linewidth=0.3)
    axes[1, 0].set_title('Enhanced')
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Amplitude')

    im00 = axes[1, 1].pcolormesh(clean_mag.T, cmap='viridis', shading='auto')
    axes[1, 1].set_title('Clean Spectrogram')
    plt.colorbar(im00, ax=axes[1, 1])

    im10 = axes[2, 0].pcolormesh(noisy_mag.T, cmap='viridis', shading='auto')
    axes[2, 0].set_title('Noisy Spectrogram')
    plt.colorbar(im10, ax=axes[2, 0])

    im11 = axes[2, 1].pcolormesh(enhanced_mag.T, cmap='viridis', shading='auto')
    axes[2, 1].set_title('Enhanced Spectrogram')
    plt.colorbar(im11, ax=axes[2, 1])

    if pred_mask.ndim == 3:
        pred_mask = pred_mask.squeeze(0)
    if ideal_mask.ndim == 3:
        ideal_mask = ideal_mask.squeeze(0)

    im20 = axes[3, 0].imshow(pred_mask.T, aspect='auto', cmap='hot', origin='lower')
    axes[3, 0].set_title('Predicted IRM')
    plt.colorbar(im20, ax=axes[3, 0])

    im21 = axes[3, 1].imshow(ideal_mask.T, aspect='auto', cmap='hot', origin='lower')
    axes[3, 1].set_title('Ideal IRM')
    plt.colorbar(im21, ax=axes[3, 1])

    fig.suptitle('Sample-level Speech Enhancement Analysis', fontsize=16, y=0.995)
    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_error_distribution(metrics_dict, save_path=None):
    _setup_academic_style()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    snr_before = np.array(metrics_dict['snr_before'])
    snr_after = np.array(metrics_dict['snr_after'])
    si_snr_before = np.array(metrics_dict['si_snr_before'])
    si_snr_after = np.array(metrics_dict['si_snr_after'])
    seg_snr_before = np.array(metrics_dict['seg_snr_before'])
    seg_snr_after = np.array(metrics_dict['seg_snr_after'])

    snr_imps = snr_after - snr_before
    si_snr_imps = si_snr_after - si_snr_before
    seg_snr_imps = seg_snr_after - seg_snr_before

    axes[0].hist(snr_imps, bins=20, alpha=0.7, color='#2171B5', edgecolor='white', density=True)
    mu, sigma = np.mean(snr_imps), np.std(snr_imps)
    x_range = np.linspace(snr_imps.min(), snr_imps.max(), 100)
    axes[0].plot(x_range, (1 / (sigma * np.sqrt(2 * np.pi))) *
                 np.exp(-0.5 * ((x_range - mu) / sigma) ** 2),
                 'r-', linewidth=2, label=f'Normal fit\nμ={mu:.2f}, σ={sigma:.2f}')
    axes[0].set_title('SNR Improvement Distribution')
    axes[0].set_xlabel('Improvement (dB)')
    axes[0].set_ylabel('Density')
    axes[0].legend()

    axes[1].boxplot([snr_imps, si_snr_imps, seg_snr_imps], labels=['SNR', 'SI-SNR', 'SegSNR'])
    axes[1].set_title('Improvement Box Plot')
    axes[1].set_ylabel('Improvement (dB)')
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    axes[2].scatter(snr_before, snr_after, alpha=0.6, c='#2171B5', s=20)
    min_val = min(snr_before.min(), snr_after.min())
    max_val = max(snr_before.max(), snr_after.max())
    axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x reference')
    axes[2].set_title('Before vs After SNR')
    axes[2].set_xlabel('Before Enhancement (dB)')
    axes[2].set_ylabel('After Enhancement (dB)')
    axes[2].legend()

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_model_architecture(save_path=None):
    _setup_academic_style()
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(12, 9))

    layers = [
        ('Input', '(1, 257, T)', (0.5, 8)),
        ('Conv1', '(32, 129, T)', (2.5, 8)),
        ('Conv2', '(64, 65, T)', (4.5, 8)),
        ('Flatten', '-', (6.5, 8)),
        ('BiLSTM', '(512)', (8.5, 8)),
        ('Linear', '-', (10.5, 8)),
        ('Reshape', '-', (10.5, 5)),
        ('DeConv1', '(32, 129, T)', (8.5, 5)),
        ('DeConv2', '(1, 257, T)', (6.5, 5)),
        ('Output Mask', '-', (4.5, 5)),
    ]

    for name, dims, (x, y) in layers:
        box = FancyBboxPatch((x - 0.7, y - 0.6), 1.4, 1.2,
                             boxstyle="round,pad=0.05",
                             facecolor='#D6EAF8', edgecolor='#2874A6',
                             linewidth=2)
        ax.add_patch(box)
        ax.text(x, y + 0.2, name, ha='center', va='center',
                fontsize=11, fontweight='bold', color='#1A5276')
        ax.text(x, y - 0.25, dims, ha='center', va='center',
                fontsize=9, color='#2E86AB')

    arrow_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
                   (6, 7), (7, 8), (8, 9)]
    for i, j in arrow_pairs:
        x1, y1 = layers[i][2]
        x2, y2 = layers[j][2]
        ax.annotate('', xy=(x2 - 0.75, y2), xytext=(x1 + 0.75, y1),
                    arrowprops=dict(arrowstyle='->', color='#34495E',
                                    lw=2, connectionstyle='arc3'))

    ax.set_xlim(-0.5, 12)
    ax.set_ylim(3, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('CRNN Speech Enhancement Model Architecture', fontsize=16,
                 fontweight='bold', pad=20)

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_training_deep_dive(train_losses, val_losses, lrs=None, save_path=None):
    _setup_academic_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs = range(1, len(train_losses) + 1)

    axes[0, 0].semilogy(epochs, train_losses, 'b-o', markersize=3, linewidth=1.5, label='Train Loss')
    axes[0, 0].semilogy(epochs, val_losses, 'r-s', markersize=3, linewidth=1.5, label='Val Loss')
    axes[0, 0].set_title('Loss Curve (Semi-log)')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (log scale)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    if lrs is not None:
        axes[0, 1].plot(epochs, lrs[:len(train_losses)], 'g-^', markersize=3, linewidth=1.5)
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    else:
        axes[0, 1].text(0.5, 0.5, 'N/A', transform=axes[0, 1].transAxes,
                        ha='center', va='center', fontsize=20, color='gray')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].axis('off')

    gaps = np.array(val_losses) - np.array(train_losses)
    colors_gaps = ['#E74C3C' if g > 0.01 else '#27AE60' for g in gaps]
    axes[1, 0].bar(epochs, gaps, color=colors_gaps, alpha=0.7, edgecolor='white')
    axes[1, 0].axhline(y=0.01, color='red', linestyle='--', alpha=0.7, label='Overfitting threshold')
    axes[1, 0].set_title('Train-Val Loss Gap')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Gap (Val - Train)')
    axes[1, 0].legend(fontsize=9)

    improvements = [val_losses[i - 1] - val_losses[i] for i in range(1, len(val_losses))]
    colors_imp = ['#2171B5' if imp > 0 else '#E74C3C' for imp in improvements]
    axes[1, 1].bar(range(2, len(val_losses) + 1), improvements, color=colors_imp,
                    alpha=0.7, edgecolor='white')
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Improvement Rate per Epoch')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Δ Val Loss')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_training_loss_smooth(train_losses, val_losses, window=5, save_path=None):
    """平滑损失曲线：散点+移动平均+置信区间"""
    _setup_academic_style()

    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    epochs = np.arange(1, len(train_losses) + 1)

    def moving_avg(data, w):
        kernel = np.ones(w) / w
        padded = np.pad(data, (w - 1, 0), mode='edge')
        return np.convolve(padded, kernel, mode='valid')

    def moving_std(data, w):
        padded = np.pad(data, (w - 1, 0), mode='edge')
        stds = []
        for i in range(len(data)):
            stds.append(np.std(padded[i:i + w]))
        return np.array(stds)

    train_smooth = moving_avg(train_losses, window)
    train_std = moving_std(train_losses, window)
    val_smooth = moving_avg(val_losses, window)
    val_std = moving_std(val_losses, window)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(epochs, train_losses, c='#2171B5', alpha=0.3, s=15, label='Train Loss (raw)', zorder=2)
    ax.scatter(epochs, val_losses, c='#CB433B', alpha=0.3, s=15, label='Val Loss (raw)', zorder=2)

    ax.plot(epochs, train_smooth, color='#2171B5', linewidth=2, label=f'Train Loss (MA, w={window})', zorder=3)
    ax.fill_between(epochs, train_smooth - train_std, train_smooth + train_std,
                    color='#2171B5', alpha=0.15, zorder=1)

    ax.plot(epochs, val_smooth, color='#CB433B', linewidth=2, label=f'Val Loss (MA, w={window})', zorder=3)
    ax.fill_between(epochs, val_smooth - val_std, val_smooth + val_std,
                    color='#CB433B', alpha=0.15, zorder=1)

    best_epoch = int(np.argmin(val_losses)) + 1
    ax.axvline(x=best_epoch, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.annotate(f'Best Epoch: {best_epoch}\nVal Loss: {val_losses[best_epoch - 1]:.4f}',
                xy=(best_epoch, val_losses[best_epoch - 1]),
                xytext=(best_epoch + len(epochs) * 0.1, val_losses[best_epoch - 1] + 0.05),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.2),
                fontsize=9, color='green', fontweight='bold')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Smoothed Training & Validation Loss Curves')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_gradient_flow(grad_norms_per_layer, layer_names=None, save_path=None):
    """梯度范数热力图：epoch × layer"""
    _setup_academic_style()

    grad_norms_per_layer = np.array(grad_norms_per_layer)
    n_epochs, n_layers = grad_norms_per_layer.shape

    if layer_names is None:
        layer_names = [f'Layer{i + 1}' for i in range(n_layers)]

    fig, ax = plt.subplots(figsize=(max(6, n_layers * 0.8), max(4, n_epochs * 0.12)))

    im = ax.imshow(grad_norms_per_layer, aspect='auto', cmap='YlOrRd', origin='lower')
    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label('Gradient L2 Norm', fontsize=11)

    ax.set_xticks(np.arange(n_layers))
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(np.arange(n_epochs))
    ax.set_yticklabels([str(i + 1) for i in range(n_epochs)], fontsize=9)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Epoch')
    ax.set_title('Gradient Norm Heatmap Across Layers and Epochs')

    max_idx = np.unravel_index(np.argmax(grad_norms_per_layer), grad_norms_per_layer.shape)
    min_idx = np.unravel_index(np.argmin(grad_norms_per_layer), grad_norms_per_layer.shape)

    ax.annotate(f'Max: {grad_norms_per_layer[max_idx]:.4f}',
                xy=(max_idx[1], max_idx[0]),
                xytext=(max_idx[1] + 0.5, max_idx[0] + n_epochs * 0.08),
                arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5),
                fontsize=8, color='darkred', fontweight='bold',
                ha='center')
    ax.annotate(f'Min: {grad_norms_per_layer[min_idx]:.4f}',
                xy=(min_idx[1], min_idx[0]),
                xytext=(min_idx[1] + 0.5, min_idx[0] - n_epochs * 0.08),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=1.5),
                fontsize=8, color='darkblue', fontweight='bold',
                ha='center')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_lr_warmup_schedule(train_losses, lr_history, save_path=None):
    """学习率全景：双Y轴 loss+lr 曲线，标注衰减事件"""
    _setup_academic_style()

    train_losses = np.array(train_losses)
    lr_history = np.array(lr_history)
    epochs = np.arange(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.semilogy(epochs, train_losses, color='#2171B5', alpha=0.5, linewidth=1.2,
                 label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)', color='#2171B5')
    ax1.tick_params(axis='y', labelcolor='#2171B5')
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = ax1.twinx()
    ax2.step(epochs, lr_history, where='mid', color='#E67E22', linewidth=2,
             label='Learning Rate')
    ax2.set_ylabel('Learning Rate', color='#E67E22')
    ax2.tick_params(axis='y', labelcolor='#E67E22')

    lr_change_epochs = []
    for i in range(1, len(lr_history)):
        if abs(lr_history[i] - lr_history[i - 1]) > 1e-10:
            lr_change_epochs.append(i + 1)

    colors_bg = ['#FFF9C4', '#E8F5E9']
    prev_ep = 1
    for idx, ep in enumerate(lr_change_epochs):
        ax1.axvspan(prev_ep - 0.5, ep - 0.5, alpha=0.25,
                    facecolor=colors_bg[idx % len(colors_bg)], zorder=0)
        ax1.axvline(x=ep, color='gray', linestyle=':', linewidth=1.2, alpha=0.7, zorder=1)
        ax1.text(ep, ax1.get_ylim()[1] * 0.95, 'lr↓',
                 fontsize=9, color='#E67E22', fontweight='bold',
                 ha='center', va='top',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        prev_ep = ep

    if lr_change_epochs:
        last_change = lr_change_epochs[-1]
        ax1.axvspan(last_change - 0.5, len(epochs) + 0.5, alpha=0.25,
                    facecolor=colors_bg[len(lr_change_epochs) % len(colors_bg)], zorder=0)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title('Learning Rate Schedule with Training Loss')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_early_stopping_indicator(val_losses, patience=5, best_epoch=None, save_path=None):
    """早停指示仪表盘：趋势+patience计数+状态信号灯"""
    _setup_academic_style()

    val_losses = np.array(val_losses)
    n_epochs = len(val_losses)

    if best_epoch is None:
        best_epoch = int(np.argmin(val_losses)) + 1

    current_no_improve = n_epochs - best_epoch

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1], hspace=0.35)

    # (a) 上部：val_loss 趋势迷你图
    ax_top = fig.add_subplot(gs[0])
    epochs_arr = np.arange(1, n_epochs + 1)
    colors_line = ['#95A5A6'] * n_epochs
    for i in range(min(best_epoch, n_epochs)):
        colors_line[i] = '#2980B9'
    for i in range(len(epochs_arr)):
        if i < n_epochs - 1:
            ax_top.plot(epochs_arr[i:i + 2], val_losses[i:i + 2],
                        color=colors_line[i], linewidth=2.5, solid_capstyle='round')

    ax_top.scatter([best_epoch], [val_losses[best_epoch - 1]],
                   marker='*', s=300, color='#27AE60', edgecolors='white',
                   linewidths=1.5, zorder=5, label=f'Best (Epoch {best_epoch})')
    ax_top.set_xlabel('Epoch')
    ax_top.set_ylabel('Validation Loss')
    ax_top.set_title('(a) Validation Loss Trend')
    ax_top.legend(loc='upper right', fontsize=9)
    ax_top.grid(True, alpha=0.3, linestyle='--')

    # (b) 中部：patience 计数进度条
    ax_mid = fig.add_subplot(gs[1])
    ratio = min(current_no_improve / patience, 1.0)
    bar_color = '#27AE60'
    hatch_pattern = ''
    if ratio >= 1.0:
        bar_color = '#E74C3C'
        hatch_pattern = '///'
    elif ratio >= 0.6:
        bar_color = '#F39C12'

    ax_mid.barh([0], [1.0], height=0.5, color='#ECF0F1', edgecolor='#BDC3C7',
                linewidth=1, zorder=0)
    ax_mid.barh([0], [ratio], height=0.5, color=bar_color, hatch=hatch_pattern,
                edgecolor='white', linewidth=2, zorder=1)

    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(-0.5, 0.5)
    ax_mid.set_xlabel(f'No-Improvement Count: {current_no_improve} / {patience} (patience)')
    ax_mid.set_title('(b) Patience Counter')
    ax_mid.set_yticks([])
    ax_mid.text(ratio + 0.02, 0, f'{ratio:.0%}', va='center', fontsize=12, fontweight='bold',
                color=bar_color)

    # (c) 下部：状态文字卡片
    ax_bot = fig.add_subplot(gs[2])
    ax_bot.axis('off')

    if ratio >= 1.0:
        status_text = 'SHOULD STOP'
        status_color = '#E74C3C'
        bg_color = '#FDEDEC'
        sub_text = f'Exceeded patience ({current_no_improve} > {patience}). Consider early stopping.'
    elif ratio >= 0.6:
        status_text = 'WARNING'
        status_color = '#F39C12'
        bg_color = '#FEF9E7'
        sub_text = f'Approaching patience limit ({current_no_improve}/{patience}). Monitor closely.'
    else:
        status_text = 'TRAINING'
        status_color = '#2980B9'
        bg_color = '#EBF5FB'
        sub_text = f'Training is progressing well ({current_no_improve}/{patience}).'

    ax_bot.text(0.5, 0.65, status_text, transform=ax_bot.transAxes,
                fontsize=28, fontweight='bold', color=status_color,
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor=bg_color,
                         edgecolor=status_color, linewidth=3))
    ax_bot.text(0.5, 0.18, sub_text, transform=ax_bot.transAxes,
                fontsize=11, color='#555555', ha='center', va='center',
                style='italic')
    ax_bot.set_title('(c) Early Stopping Status', fontsize=12, pad=10)

    fig.suptitle('Early Stopping Indicator Dashboard', fontsize=14, fontweight='bold', y=0.98)

    _save_figure(fig, save_path)


def plot_confusion_matrix_irm(pred_mask_flat, ideal_mask_flat, n_bins=5, save_path=None):
    """IRM 掩码分桶混淆矩阵"""
    _setup_academic_style()

    pred_mask_flat = np.asarray(pred_mask_flat).ravel()
    ideal_mask_flat = np.asarray(ideal_mask_flat).ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    pred_binned = np.digitize(pred_mask_flat, bins=bin_edges[1:-1])
    ideal_binned = np.digitize(ideal_mask_flat, bins=bin_edges[1:-1])

    cm = np.zeros((n_bins, n_bins), dtype=int)
    for p, r in zip(pred_binned, ideal_binned):
        cm[p, r] += 1

    bin_labels = [f'[{bin_edges[i]:.1f},{bin_edges[i + 1]:.1f})'
                  for i in range(n_bins)]
    bin_labels[-1] = f'[{bin_edges[-2]:.1f},{bin_edges[-1]:.1f}]'

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('Count', fontsize=11)

    total = cm.sum()
    for i in range(n_bins):
        for j in range(n_bins):
            pct = cm[i, j] / total * 100 if total > 0 else 0
            text_color = 'white' if cm[i, j] > cm.max() * 0.5 else 'black'
            ax.text(j, i, f'{cm[i, j]}\n({pct:.1f}%)',
                    ha='center', va='center', fontsize=9, color=text_color,
                    fontweight='bold' if i == j else 'normal')

    ax.set_xticks(np.arange(n_bins))
    ax.set_yticks(np.arange(n_bins))
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(bin_labels, fontsize=9)
    ax.set_xlabel('Ideal Mask Bin', fontsize=12)
    ax.set_ylabel('Predicted Mask Bin', fontsize=12)
    ax.set_title(f'IRM Mask Confusion Matrix (n_bins={n_bins})')

    accuracy = np.trace(cm) / total * 100 if total > 0 else 0
    ax.text(0.02, 0.98, f'Diagonal Accuracy: {accuracy:.1f}%\nTotal Samples: {total}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_dataset_overview(train_count, val_count, test_count, durations, snrs, save_path=None):
    """数据集总览图：样本数饼图 + 时长分布 + SNR分布"""
    _setup_academic_style()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) 样本数饼图
    counts = [train_count, val_count, test_count]
    labels = [f'Train\n({train_count})', f'Val\n({val_count})', f'Test\n({test_count})']
    colors_pie = ['#2171B5', '#6BAED6', '#BDD7E7']
    wedges, texts, autotexts = axes[0].pie(counts, labels=labels, colors=colors_pie,
                                            autopct='%1.1f%%', startangle=90)
    for t in autotexts:
        t.set_fontsize(10)
    axes[0].set_title('(a) Sample Distribution')

    # (b) 音频时长分布直方图
    durations = np.asarray(durations, dtype=float)
    axes[1].hist(durations, bins=30, color='#2E86AB', alpha=0.7, edgecolor='white')
    dur_mean = np.mean(durations)
    axes[1].axvline(dur_mean, color='#E74C3C', linestyle='--', linewidth=2,
                    label=f'Mean = {dur_mean:.2f}s')
    axes[1].set_xlabel('Duration (s)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('(b) Audio Duration Distribution')
    axes[1].legend(fontsize=9)

    # (c) SNR 分布直方图
    snrs = np.asarray(snrs, dtype=float)
    axes[2].hist(snrs, bins=30, color='#27AE60', alpha=0.7, edgecolor='white')
    snr_mean = np.mean(snrs)
    snr_median = np.median(snrs)
    axes[2].axvline(snr_mean, color='#E74C3C', linestyle='--', linewidth=2,
                    label=f'Mean = {snr_mean:.1f} dB')
    axes[2].axvline(snr_median, color='#8E44AD', linestyle='-.', linewidth=2,
                    label=f'Median = {snr_median:.1f} dB')
    axes[2].set_xlabel('SNR (dB)')
    axes[2].set_ylabel('Count')
    axes[2].set_title('(c) SNR Distribution')
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_audio_statistics(wav, sr, save_path=None):
    """音频多维统计：包络/能量/频谱质心/过零率/自相关/幅度谱"""
    _setup_academic_style()
    wav = np.asarray(wav, dtype=float).flatten()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    time_axis = np.arange(len(wav)) / sr

    # (0,0) 波形包络
    analytic = sig.hilbert(wav)
    envelope = np.abs(analytic)
    axes[0, 0].plot(time_axis, wav, color='#A5C8E1', linewidth=0.4, alpha=0.7, label='Waveform')
    axes[0, 0].plot(time_axis, envelope, color='#E74C3C', linewidth=1.2, label='Hilbert Envelope')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].set_title('(a) Waveform & Envelope')
    axes[0, 0].legend(fontsize=8)

    # (0,1) 短时能量曲线
    frame_length = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_frames = max(1, (len(wav) - frame_length) // hop + 1)
    energy = np.array([np.sum(wav[i * hop:i * hop + frame_length] ** 2) for i in range(n_frames)])
    energy_time = np.arange(n_frames) * hop / sr
    axes[0, 1].plot(energy_time, energy, color='#2171B5', linewidth=1)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Energy')
    axes[0, 1].set_title('(b) Short-Time Energy')

    # (0,2) 频谱质心
    freqs = np.fft.rfftfreq(frame_length, d=1 / sr)
    centroids = []
    for i in range(n_frames):
        frame = wav[i * hop:i * hop + frame_length]
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)))
        mag = np.abs(np.fft.rfft(frame))
        mag_sum = mag.sum()
        if mag_sum > 0:
            centroids.append(np.sum(freqs * mag) / mag_sum)
        else:
            centroids.append(0)
    axes[0, 2].plot(energy_time, centroids, color='#27AE60', linewidth=1)
    axes[0, 2].set_xlabel('Time (s)')
    axes[0, 2].set_ylabel('Frequency (Hz)')
    axes[0, 2].set_title('(c) Spectral Centroid')

    # (1,0) 过零率
    zcr_frame = frame_length
    zcr_hop = hop
    n_zcr = max(1, (len(wav) - zcr_frame) // zcr_hop + 1)
    zcr_values = []
    for i in range(n_zcr):
        chunk = wav[i * zcr_hop:i * zcr_hop + zcr_frame]
        signs = np.sign(chunk)
        crossings = np.sum(np.abs(np.diff(signs)) > 1) / 2
        zcr_values.append(crossings / len(chunk))
    zcr_time = np.arange(n_zcr) * zcr_hop / sr
    axes[1, 0].plot(zcr_time, zcr_values, color='#8E44AD', linewidth=1)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('ZCR')
    axes[1, 0].set_title('(d) Zero Crossing Rate')

    # (1,1) 自相关函数
    max_lag = min(len(wav), sr // 2)
    autocorr = np.correlate(wav[:max_lag] - np.mean(wav[:max_lag]),
                            wav[:max_lag] - np.mean(wav[:max_lag]), mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / autocorr[0]
    lag_axis = np.arange(len(autocorr)) / sr * 1000
    axes[1, 1].plot(lag_axis, autocorr, color='#D35400', linewidth=1)
    axes[1, 1].set_xlabel('Lag (ms)')
    axes[1, 1].set_ylabel('Normalized Correlation')
    axes[1, 1].set_title('(e) Autocorrelation Function')
    axes[1, 1].set_xlim([0, min(lag_axis[-1], 50)])

    # (1,2) 幅度谱（对数刻度）
    full_mag = np.abs(np.fft.rfft(wav))
    full_freqs = np.fft.rfftfreq(len(wav), d=1 / sr)
    axes[1, 2].semilogy(full_freqs, full_mag + 1e-10, color='#34495E', linewidth=0.8)
    axes[1, 2].set_xlabel('Frequency (Hz)')
    axes[1, 2].set_ylabel('Magnitude (log scale)')
    axes[1, 2].set_title('(f) Amplitude Spectrum')
    axes[1, 2].set_xlim([0, sr // 2])

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_noise_profile(save_path=None):
    """噪声特性对比：白噪/粉噪/Babble 的功率谱密度（双对数坐标）"""
    _setup_academic_style()

    sr = 16000
    duration = 1.0
    N = int(sr * duration)

    # 白噪声
    white = np.random.randn(N)

    # 粉红噪声 (Voss-McCartney 频域方法)
    X_white = np.fft.rfft(np.random.randn(N))
    pink_filter = np.ones_like(X_white)
    pink_filter[1:] = 1.0 / (np.sqrt(np.arange(1, len(pink_filter))))
    X_pink = X_white * pink_filter
    pink = np.real(np.fft.irfft(X_pink, n=N))

    # Babble 噪声：叠加多个正弦波模拟多人说话
    babble = np.zeros(N)
    base_freqs = np.linspace(80, 400, 12)
    for f in base_freqs:
        phase = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.3, 1.0)
        fm = f + 10 * np.sin(2 * np.pi * 2 * np.arange(N) / sr)
        instantaneous_phase = 2 * np.pi * np.cumsum(fm) / sr + phase
        babble += amplitude * np.sin(instantaneous_phase)
    babble = babble / np.max(np.abs(babble)) * np.std(white)

    fig, ax = plt.subplots(figsize=(8, 6))

    for noise_signal, label, color in [(white, 'White Noise', '#2171B5'),
                                        (pink, 'Pink Noise', '#E74C3C'),
                                        (babble, 'Babble Noise', '#27AE60')]:
        freqs, psd = sig.welch(noise_signal, fs=sr, nperseg=1024)
        ax.loglog(freqs, psd, linewidth=1.5, label=label, color=color)

    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Spectral Density (dB/Hz)')
    ax.set_title('Noise Profile Comparison (Welch PSD, log-log)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both', linestyle='--')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_snr_distribution(sample_snrs, save_path=None):
    """训练集 SNR 分布：直方图 + KDE + 统计标注"""
    _setup_academic_style()

    sample_snrs = np.asarray(sample_snrs, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))

    # 直方图（密度归一化）
    ax.hist(sample_snrs, bins=20, density=True, alpha=0.65,
            color='#2171B5', edgecolor='white', label='Histogram')

    # KDE 曲线
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(sample_snrs)
    x_range = np.linspace(sample_snrs.min() - 1, sample_snrs.max() + 1, 200)
    ax.plot(x_range, kde(x_range), color='#E74C3C', linewidth=2, label='KDE')

    # 均匀分布参考线（理论范围 0~15 dB）
    if sample_snrs.min() >= 0 and sample_snrs.max() <= 18:
        ax.axhline(y=1.0 / 16, color='gray', linestyle=':', linewidth=1.5,
                   alpha=0.7, label='Uniform reference (0-16 dB)')

    # 统计标注
    mu = np.mean(sample_snrs)
    sigma = np.std(sample_snrs)
    median_val = np.median(sample_snrs)
    stats_text = f'μ = {mu:.2f} dB\nσ = {sigma:.2f} dB\nMedian = {median_val:.2f} dB'
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Density')
    ax.set_title('Training Set SNR Distribution')
    ax.legend()

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_frequency_band_analysis(metrics_dict, n_bands=4, sr=16000, n_fft=512, save_path=None):
    """分频带增强效果：按原始SNR分组的提升量柱状图"""
    _setup_academic_style()

    snr_before = np.array(metrics_dict['snr_before'])
    snr_after = np.array(metrics_dict['snr_after'])
    snr_improvement = snr_after - snr_before

    quantiles = np.linspace(0, 100, n_bands + 1)
    band_edges = np.percentile(snr_before, quantiles)
    band_labels = []
    band_means = []
    band_sems = []

    for i in range(n_bands):
        if i == n_bands - 1:
            mask = (snr_before >= band_edges[i]) & (snr_before <= band_edges[i + 1])
        else:
            mask = (snr_before >= band_edges[i]) & (snr_before < band_edges[i + 1])
        group_imp = snr_improvement[mask]
        band_means.append(np.mean(group_imp))
        band_sems.append(np.std(group_imp) / np.sqrt(len(group_imp)))
        low_edge = f"{band_edges[i]:.1f}"
        high_edge = f"{band_edges[i + 1]:.1f}"
        band_labels.append(f"{low_edge}~{high_edge}\ndB")

    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(n_bands)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_bands))

    bars = ax.bar(x_pos, band_means, yerr=band_sems, capsize=5,
                  color=colors, edgecolor='#1A5276', linewidth=1.2,
                  error_kw={'elinewidth': 1.5, 'capthick': 1.5})

    ax.set_xticks(x_pos)
    ax.set_xticklabels(band_labels, fontsize=10)
    ax.set_xlabel('Original SNR Group (dB)', fontsize=12)
    ax.set_ylabel('Average SNR Improvement (dB)', fontsize=12)
    ax.set_title('Frequency Band Enhancement Effect Analysis', fontsize=14)

    for bar, mean_val in zip(bars, band_means):
        ax.annotate(f'{mean_val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.axhline(y=np.mean(snr_improvement), color='red', linestyle='--',
               linewidth=1.5, alpha=0.7, label=f'Overall Mean: {np.mean(snr_improvement):.2f} dB')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_phase_importance(noisy_wav, enhanced_noisy_phase_wav, enhanced_est_phase_wav, sr, save_path=None):
    """相位保留分析：含噪相位 vs 估计相位的重建质量对比"""
    _setup_academic_style()

    def _safe_sisnr(target, estimate):
        target = target.astype(np.float64)
        estimate = estimate.astype(np.float64)
        if len(target) != len(estimate):
            min_len = min(len(target), len(estimate))
            target = target[:min_len]
            estimate = estimate[:min_len]
        s_target = np.dot(target, target) / (np.dot(target, target) + 1e-8) * target
        e_noise = estimate - s_target
        return 10 * np.log10(np.dot(s_target, s_target) + 1e-8) - 10 * np.log10(np.dot(e_noise, e_noise) + 1e-8)

    sisnr_noisy_phase = _safe_sisnr(noisy_wav, enhanced_noisy_phase_wav)
    sisnr_est_phase = _safe_sisnr(noisy_wav, enhanced_est_phase_wav)

    better_strategy = "Estimated Phase" if sisnr_est_phase > sisnr_noisy_phase else "Noisy Phase"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    t_noisy = np.linspace(0, len(noisy_wav) / sr, len(noisy_wav))
    t_enhanced_np = np.linspace(0, len(enhanced_noisy_phase_wav) / sr, len(enhanced_noisy_phase_wav))
    t_enhanced_ep = np.linspace(0, len(enhanced_est_phase_wav) / sr, len(enhanced_est_phase_wav))

    min_len_np = min(len(t_noisy), len(t_enhanced_np))
    min_len_ep = min(len(t_noisy), len(t_enhanced_ep))

    axes[0].plot(t_noisy[:min_len_np], noisy_wav[:min_len_np], linewidth=0.6,
                 color='#95A5A6', alpha=0.6, label='Noisy Reference')
    axes[0].plot(t_enhanced_np[:min_len_np], enhanced_noisy_phase_wav[:min_len_np],
                 linewidth=0.6, color='#E74C3C', alpha=0.85, label='Enhanced (Noisy Phase)')
    axes[0].set_title(f'Enhanced with Noisy Phase   |   SI-SNR: {sisnr_noisy_phase:.2f} dB', fontsize=13)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')

    axes[1].plot(t_noisy[:min_len_ep], noisy_wav[:min_len_ep], linewidth=0.6,
                 color='#95A5A6', alpha=0.6, label='Noisy Reference')
    axes[1].plot(t_enhanced_ep[:min_len_ep], enhanced_est_phase_wav[:min_len_ep],
                 linewidth=0.6, color='#27AE60', alpha=0.85, label='Enhanced (Estimated Phase)')
    axes[1].set_title(f'Enhanced with Estimated Phase   |   SI-SNR: {sisnr_est_phase:.2f} dB', fontsize=13)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')

    fig.suptitle('Phase Importance Analysis', fontsize=15, fontweight='bold', y=1.01)
    conclusion_text = f'Conclusion: {better_strategy} yields higher reconstruction quality'
    fig.text(0.5, -0.01, conclusion_text, ha='center', fontsize=11,
             style='italic', color='#2C3E50',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#EBF5FB', edgecolor='#AED6F1'))

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_mask_confidence(pred_masks_list, save_path=None):
    """掩码置信度分析：确定性指标随帧变化"""
    _setup_academic_style()

    if isinstance(pred_masks_list, list):
        pred_masks_list = np.array(pred_masks_list)

    if pred_masks_list.ndim == 1:
        confidence_scores = np.abs(pred_masks_list - 0.5) * 2
    elif pred_masks_list.ndim == 2:
        confidence_scores = np.mean(np.abs(pred_masks_list - 0.5) * 2, axis=1)
    else:
        raise ValueError(f"pred_masks_list should be 1D or 2D array, got shape {pred_masks_list.shape}")

    n_frames = len(confidence_scores)
    frame_indices = np.arange(n_frames)

    avg_confidence = np.mean(confidence_scores)
    low_conf_ratio = np.mean(confidence_scores < 0.5) * 100
    mid_conf_ratio = np.mean((confidence_scores >= 0.5) & (confidence_scores < 0.8)) * 100
    high_conf_ratio = np.mean(confidence_scores >= 0.8) * 100

    fig, (ax_main, ax_zone) = plt.subplots(2, 1, figsize=(12, 7),
                                            gridspec_kw={'height_ratios': [3, 1]})

    ax_main.plot(frame_indices, confidence_scores, color='#2171B5',
                 linewidth=1.0, alpha=0.85)
    ax_main.axhline(y=avg_confidence, color='#E74C3C', linestyle='--',
                    linewidth=1.5, label=f'Mean Confidence: {avg_confidence:.3f}')
    ax_main.fill_between(frame_indices, avg_confidence, confidence_scores,
                         where=(confidence_scores >= avg_confidence),
                         color='#27AE60', alpha=0.15, interpolate=True)
    ax_main.fill_between(frame_indices, avg_confidence, confidence_scores,
                         where=(confidence_scores < avg_confidence),
                         color='#E74C3C', alpha=0.15, interpolate=True)
    ax_main.set_xlabel('Frame Index', fontsize=12)
    ax_main.set_ylabel('Confidence Score', fontsize=12)
    ax_main.set_title('Mask Confidence Analysis Over Frames', fontsize=14)
    ax_main.legend(loc='upper right', fontsize=10)
    ax_main.set_ylim(-0.05, 1.05)
    ax_main.grid(True, alpha=0.3, linestyle='--')

    zone_colors = ['#E74C3C', '#F39C12', '#27AE60']
    zone_labels = ['Low (<0.5)', 'Medium (0.5~0.8)', 'High (\u22650.8)']
    zone_values = [low_conf_ratio, mid_conf_ratio, high_conf_ratio]

    left = 0
    for color, label, val in zip(zone_colors, zone_labels, zone_values):
        ax_zone.barh(0, val, left=left, height=0.5, color=color, alpha=0.75,
                     edgecolor='white', linewidth=0.8, label=f'{label}: {val:.1f}%')
        left += val

    ax_zone.set_xlim(0, 100)
    ax_zone.set_yticks([])
    ax_zone.set_xlabel('Percentage (%)', fontsize=11)
    ax_zone.set_title('Confidence Zone Distribution', fontsize=12)
    ax_zone.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                   ncol=3, fontsize=9, frameon=False)
    ax_zone.grid(True, alpha=0.3, linestyle='--', axis='x')

    stats_text = (f'Avg Confidence: {avg_confidence:.3f}  |  '
                  f'Low Conf Frames: {low_conf_ratio:.1f}%  |  '
                  f'Mid Conf Frames: {mid_conf_ratio:.1f}%  |  '
                  f'High Conf Frames: {high_conf_ratio:.1f}%')
    fig.suptitle(stats_text, fontsize=11, y=1.005, style='italic', color='#555555')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_enhancement_trajectory(noisy_mag, enhanced_mag, sr=16000, hop_length=128, n_steps=6, save_path=None):
    """增强轨迹：noisy→enhanced 幅度谱渐变过程"""
    _setup_academic_style()

    vmin = min(noisy_mag.min(), enhanced_mag.min())
    vmax = max(noisy_mag.max(), enhanced_mag.max())

    ncols = min(n_steps, 3)
    nrows = (n_steps + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
    if n_steps == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)

    for step in range(n_steps):
        r, c = divmod(step, ncols)
        ax = axes[r, c]

        if step == 0:
            mag = noisy_mag
            alpha_val = 0.0
        elif step == n_steps - 1:
            mag = enhanced_mag
            alpha_val = 1.0
        else:
            alpha_val = step / (n_steps - 1)
            mag = noisy_mag + alpha_val * (enhanced_mag - noisy_mag)

        im = ax.imshow(mag.T, aspect='auto', origin='lower',
                       cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Step {step}: $\\alpha$={alpha_val:.2f}', fontsize=11)
        ax.set_xlabel('Frame')
        ax.set_ylabel('Freq Bin')

        if step == n_steps - 1:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for extra in range(n_steps, nrows * ncols):
        r, c = divmod(extra, ncols)
        axes[r, c].axis('off')

    fig.suptitle('Enhancement Trajectory: Noisy \u2192 Enhanced Magnitude Spectrum',
                 fontsize=14, fontweight='bold', y=1.01)

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_radar_chart(metrics_dict, sample_ids=None, top_n=5, save_path=None):
    """三轴雷达图：SNR/SI-SNR/SegSNR 增强前后对比"""
    _setup_academic_style()

    snr_b = np.array(metrics_dict['snr_before'])
    snr_a = np.array(metrics_dict['snr_after'])
    si_snr_b = np.array(metrics_dict['si_snr_before'])
    si_snr_a = np.array(metrics_dict['si_snr_after'])
    seg_snr_b = np.array(metrics_dict['seg_snr_before'])
    seg_snr_a = np.array(metrics_dict['seg_snr_after'])

    labels = ['SNR (dB)', 'SI-SNR (dB)', 'SegSNR (dB)']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    max_vals = [snr_a.max(), si_snr_a.max(), seg_snr_a.max()]
    min_vals = [snr_b.min(), si_snr_b.min(), seg_snr_b.min()]

    def normalize(val, idx):
        return (val - min_vals[idx]) / (max_vals[idx] - min_vals[idx])

    before_mean = [
        normalize(np.mean(snr_b), 0),
        normalize(np.mean(si_snr_b), 1),
        normalize(np.mean(seg_snr_b), 2),
    ]
    before_mean += before_mean[:1]

    after_mean = [
        normalize(np.mean(snr_a), 0),
        normalize(np.mean(si_snr_a), 1),
        normalize(np.mean(seg_snr_a), 2),
    ]
    after_mean += after_mean[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    ax.plot(angles, before_mean, 'o-', color='#6BAED6', linewidth=2,
            label='Before Enhancement')
    ax.fill(angles, before_mean, color='#6BAED6', alpha=0.25)

    ax.plot(angles, after_mean, 'o-', color='#E6550D', linewidth=2,
            label='After Enhancement')
    ax.fill(angles, after_mean, color='#E6550D', alpha=0.25)

    if sample_ids is not None and top_n > 0:
        diffs = (snr_a - snr_b) + (si_snr_a - si_snr_b) + (seg_snr_a - seg_snr_b)
        top_indices = np.argsort(diffs)[-top_n:]
        for idx in top_indices:
            sample_vals = [
                normalize(snr_a[idx], 0),
                normalize(si_snr_a[idx], 1),
                normalize(seg_snr_a[idx], 2),
            ]
            sample_vals += sample_vals[:1]
            ax.plot(angles, sample_vals, '-', color='gray', alpha=0.2, linewidth=0.8)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.00'], fontsize=8, color='gray')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    ax.set_title('Multi-Metric Radar Chart: Before vs After Enhancement',
                 fontsize=14, pad=20)

    _save_figure(fig, save_path)


def plot_correlation_heatmap(metrics_dict, save_path=None):
    """6 指标 Pearson 相关性热力图"""
    _setup_academic_style()

    keys = ['snr_before', 'snr_after', 'si_snr_before', 'si_snr_after',
            'seg_snr_before', 'seg_snr_after']
    data = np.column_stack([np.array(metrics_dict[k]) for k in keys])

    corr_matrix = np.corrcoef(data.T)

    fig, ax = plt.subplots(figsize=(8, 7))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')

    short_labels = [r'SNR$^-$', r'SNR$^+$', r'SSR$^-$', r'SSR$^+$',
                    r'SSG$^-$', r'SSG$^+$']
    ax.set_xticks(range(len(short_labels)))
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=11)

    for i in range(len(keys)):
        for j in range(len(keys)):
            val = corr_matrix[i, j]
            text_color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=10, color=text_color, fontweight='bold')

    for i in range(0, len(keys), 2):
        rect = plt.Rectangle((i - 0.45, i - 0.45), 1.9, 1.9,
                              linewidth=3, edgecolor='#333333',
                              facecolor='none')
        ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Pearson Correlation', fontsize=11)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    ax.set_title('Cross-Metric Pearson Correlation Heatmap', fontsize=14, pad=12)

    _save_figure(fig, save_path)


def plot_improvement_ranking(metrics_dict, top_k=15, metric='snr', save_path=None):
    """样本提升排名：水平条形图，标注 Top-K 最佳/最差"""
    _setup_academic_style()

    key_map = {
        'snr': ('snr_before', 'snr_after'),
        'si_snr': ('si_snr_before', 'si_snr_after'),
        'seg_snr': ('seg_snr_before', 'seg_snr_after'),
    }
    if metric not in key_map:
        raise ValueError(f"metric must be one of {list(key_map.keys())}, got '{metric}'")

    key_before, key_after = key_map[metric]
    before = np.array(metrics_dict[key_before])
    after = np.array(metrics_dict[key_after])
    diff = after - before

    sorted_indices = np.argsort(diff)[::-1]
    top_best = sorted_indices[:top_k]

    values = diff[top_best]
    labels = [str(i) for i in top_best]
    colors = ['#27AE60' if v >= 0 else '#E74C3C' for v in values]

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.4)))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white', height=0.6)

    mean_val = np.mean(diff)
    ax.axvline(x=mean_val, color='#F39C12', linestyle='--', linewidth=2,
               label=f'Mean = {mean_val:.2f} dB')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(f'{metric.upper()} Improvement (dB)', fontsize=12)
    ax.set_ylabel('Sample Index', fontsize=12)
    ax.set_title(f'Top-{top_k} Sample Improvements ({metric.upper()})',
                 fontsize=14, pad=10)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='x')

    for bar, val in zip(bars, values):
        x_pos = val + 0.02 if val >= 0 else val - 0.02
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', ha=ha, fontsize=9)

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_pareto_frontier(metrics_dict, x_metric='si_snr', y_metric='seg_snr', save_path=None):
    """Pareto 前沿分析：两指标提升量的前沿边界"""
    _setup_academic_style()

    x_key_before = f'{x_metric}_before'
    x_key_after = f'{x_metric}_after'
    y_key_before = f'{y_metric}_before'
    y_key_after = f'{y_metric}_after'

    x_diff = np.array(metrics_dict[x_key_after]) - np.array(metrics_dict[x_key_before])
    y_diff = np.array(metrics_dict[y_key_after]) - np.array(metrics_dict[y_key_before])

    pareto_mask = []
    n = len(x_diff)
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (x_diff[j] >= x_diff[i] and y_diff[j] > y_diff[i]) or \
               (x_diff[j] > x_diff[i] and y_diff[j] >= y_diff[i]):
                dominated = True
                break
        pareto_mask.append(not dominated)
    pareto_mask = np.array(pareto_mask)

    pareto_x = x_diff[pareto_mask]
    pareto_y = y_diff[pareto_mask]
    pareto_idx = np.where(pareto_mask)[0]

    sort_order = np.argsort(pareto_x)
    pareto_x_sorted = pareto_x[sort_order]
    pareto_y_sorted = pareto_y[sort_order]
    pareto_idx_sorted = pareto_idx[sort_order]

    fig, ax = plt.subplots(figsize=(9, 7))

    ax.scatter(x_diff, y_diff, c='gray', alpha=0.3, s=30, label='All Samples',
               edgecolors='none')
    ax.scatter(pareto_x, pareto_y, c='#E6550D', s=80, zorder=5,
               label='Pareto Frontier', edgecolors='black', linewidths=0.8)
    ax.plot(pareto_x_sorted, pareto_y_sorted, '-', color='#E6550D',
            linewidth=2, alpha=0.7, zorder=4)

    for px, py, pid in zip(pareto_x_sorted, pareto_y_sorted, pareto_idx_sorted):
        ax.annotate(str(pid), (px, py), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color='#E6550D', fontweight='bold')

    ax.set_xlabel(f'{x_metric.upper()} Improvement (dB)', fontsize=12)
    ax.set_ylabel(f'{y_metric.upper()} Improvement (dB)', fontsize=12)
    ax.set_title(f'Pareto Frontier: {x_metric.upper()} vs {y_metric.upper()}',
                 fontsize=14, pad=10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.4)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_ablation_comparison(results_dict, save_path=None):
    """消融实验对比：分组柱状图 + 显著性星号"""
    _setup_academic_style()

    exp_names = list(results_dict.keys())
    metrics = ['snr', 'si_snr', 'seg_snr']
    group_labels = ['SNR', 'SI-SNR', 'SegSNR']

    baseline_name = exp_names[0]
    baseline_vals = np.array([results_dict[baseline_name][m] for m in metrics])

    n_exps = len(exp_names)
    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    width = 0.8 / n_exps

    cmap = plt.cm.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(n_exps)]

    fig, ax = plt.subplots(figsize=(max(8, n_exps * 1.5), 6))

    for i, exp_name in enumerate(exp_names):
        vals = [results_dict[exp_name][m] for m in metrics]
        offset = (i - (n_exps - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, label=exp_name, color=colors[i],
                      edgecolor='white', linewidth=0.5)

        if i > 0:
            for j, (bar, val) in enumerate(zip(bars, vals)):
                if val > baseline_vals[j]:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.15,
                            '***', ha='center', va='bottom', fontsize=11,
                            color='#C0392B', fontweight='bold')

    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Value (dB)', fontsize=12)
    ax.set_title('Ablation Study Comparison', fontsize=14, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()
    _save_figure(fig, save_path)


def plot_experiment_summary_dashboard(args, metrics_dict, train_losses, val_losses, save_path=None):
    """实验总览仪表盘：KPI卡片 + 迷你损失曲线 + 迷你雷达图 + 超参数表格"""
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch

    _setup_academic_style()

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[0.8, 2.2, 1.5],
                           hspace=0.35, wspace=0.25,
                           left=0.05, right=0.95, top=0.93, bottom=0.06)

    # ---- Row 0: KPI 卡片区 (3 个子图) ----
    kpi_metrics = [
        ('SNR Improvement', 'snr_before', 'snr_after'),
        ('SI-SNR Improvement', 'si_snr_before', 'si_snr_after'),
        ('SegSNR Improvement', 'seg_snr_before', 'seg_snr_after'),
    ]

    for idx, (label, key_before, key_after) in enumerate(kpi_metrics):
        ax_kpi = fig.add_subplot(gs[0, idx])
        ax_kpi.axis('off')

        before_vals = np.array(metrics_dict.get(key_before, [0]))
        after_vals = np.array(metrics_dict.get(key_after, [0]))
        mean_before = np.mean(before_vals) if len(before_vals) > 0 else 0
        mean_after = np.mean(after_vals) if len(after_vals) > 0 else 0
        improvement = mean_after - mean_before

        color = '#27AE60' if improvement >= 0 else '#E74C3C'
        arrow = '\u2191' if improvement >= 0 else '\u2193'

        bg = FancyBboxPatch((0.05, 0.1), 0.9, 0.8,
                            transform=ax_kpi.transAxes,
                            boxstyle="round,pad=0.03",
                            facecolor='#F8F9FA',
                            edgecolor=color, linewidth=2.5)
        ax_kpi.add_patch(bg)

        ax_kpi.text(0.5, 0.72, label, transform=ax_kpi.transAxes,
                    ha='center', va='center', fontsize=11,
                    fontweight='bold', color='#2C3E50')
        ax_kpi.text(0.5, 0.45, f'{improvement:+.2f} dB',
                    transform=ax_kpi.transAxes,
                    ha='center', va='center', fontsize=22,
                    fontweight='bold', color=color)
        ax_kpi.text(0.5, 0.20, f'{arrow} from {mean_before:.2f} to {mean_after:.2f}',
                    transform=ax_kpi.transAxes,
                    ha='center', va='center', fontsize=10,
                    color='#7F8C8D')

    # ---- Row 1: 左 - 迷你损失曲线 ----
    ax_loss = fig.add_subplot(gs[1, 0])
    epochs = range(1, len(train_losses) + 1)
    ax_loss.plot(epochs, train_losses, color='#2171B5', linewidth=1.5,
                 label='Train Loss', alpha=0.85)
    if val_losses is not None and len(val_losses) == len(train_losses):
        ax_loss.plot(epochs, val_losses, color='#E74C3C', linewidth=1.5,
                     label='Val Loss', alpha=0.85)
    ax_loss.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Epoch', fontsize=10)
    ax_loss.set_ylabel('Loss', fontsize=10)
    ax_loss.legend(fontsize=9, loc='upper right')
    ax_loss.grid(True, alpha=0.25, linestyle='--')

    # ---- Row 1: 右 - 迷你雷达图 ----
    ax_radar = fig.add_subplot(gs[1, 1], projection='polar')
    radar_labels = ['SNR', 'SI-SNR', 'SegSNR']
    keys_after = ['snr_after', 'si_snr_after', 'seg_snr_after']
    values = [np.mean(np.array(metrics_dict.get(k, [0]))) for k in keys_after]
    angles = np.linspace(0, 2 * np.pi, len(radar_labels), endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]

    ax_radar.plot(angles, values_plot, 'o-', linewidth=2, color='#2171B5', markersize=6)
    ax_radar.fill(angles, values_plot, alpha=0.2, color='#2171B5')
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(radar_labels, fontsize=10)
    ax_radar.set_title('Metrics Radar Overview', fontsize=12, fontweight='bold',
                       pad=15)
    ax_radar.set_yticklabels([])

    # ---- Row 2: 超参数表格 (跨两列) ----
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    param_rows = [
        ['Model', getattr(args, 'model_type', 'CRNN')],
        ['Optimizer', getattr(args, 'optimizer', 'Adam')],
        ['Epochs', str(getattr(args, 'epochs', '-'))],
        ['Batch Size', str(getattr(args, 'batch_size', '-'))],
        ['Learning Rate', f'{getattr(args, "lr", 0.0):.0e}' if hasattr(args, 'lr') else '-'],
        ['Hidden Dim', str(getattr(args, 'hidden_dim', '-'))],
        ['Layers', str(getattr(args, 'num_layers', '-'))],
        ['Data Mode', getattr(args, 'data_mode', '-')],
    ]
    table = ax_table.table(cellText=param_rows,
                           colLabels=['Hyperparameter', 'Value'],
                           loc='center',
                           cellLoc='center',
                           colWidths=[0.28, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#2171B5')
            cell.set_text_props(color='white', fontweight='bold')
        else:
            cell.set_facecolor('#F4F6F7' if row % 2 == 0 else 'white')
            cell.set_edgecolor('#D5DBDB')

    fig.suptitle('Experiment Summary Dashboard', fontsize=16, fontweight='bold', y=0.98)
    _save_figure(fig, save_path)


def generate_all_visualizations(args, model, device, data_dir, output_dir,
                                 metrics_dict=None, train_losses=None,
                                 val_losses=None, lr_history=None,
                                 pred_masks=None, sample_wavs=None):
    """一键生成全部可视化图表，分类存放到不同子目录"""
    base = output_dir
    dirs = {
        'training': os.path.join(base, 'training'),
        'analysis': os.path.join(base, 'analysis'),
        'samples': os.path.join(base, 'samples'),
        'data': os.path.join(base, 'data'),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    print("=" * 60)
    print("开始生成全部可视化图表...")
    print("=" * 60)

    # === 类别 1: 数据分析 ===
    if metrics_dict is not None:
        print("[1/5] 数据样本分析...")
        try:
            plot_radar_chart(metrics_dict, save_path=os.path.join(dirs['analysis'], 'radar_chart'))
            print("  ✓ 雷达图")
        except Exception as e:
            print(f"  ✗ 雷达图失败: {e}")
        try:
            plot_correlation_heatmap(metrics_dict, save_path=os.path.join(dirs['analysis'], 'correlation_heatmap'))
            print("  ✓ 相关性热力图")
        except Exception as e:
            print(f"  ✗ 相关性热力图失败: {e}")
        try:
            plot_improvement_ranking(metrics_dict, save_path=os.path.join(dirs['analysis'], 'improvement_ranking'))
            print("  ✓ 排名图")
        except Exception as e:
            print(f"  ✗ 排名图失败: {e}")
        try:
            plot_pareto_frontier(metrics_dict, save_path=os.path.join(dirs['analysis'], 'pareto_frontier'))
            print("  ✓ Pareto 前沿")
        except Exception as e:
            print(f"  ✗ Pareto 前沿失败: {e}")

    # === 类别 2: 训练验证 ===
    if train_losses is not None and len(train_losses) > 0:
        print("[2/5] 模型训练验证...")
        try:
            plot_training_loss_smooth(train_losses, val_losses,
                                       save_path=os.path.join(dirs['training'], 'training_loss_smooth'))
            print("  ✓ 平滑损失曲线")
        except Exception as e:
            print(f"  ✗ 平滑损失曲线失败: {e}")

        if lr_history is not None:
            try:
                plot_lr_warmup_schedule(train_losses, lr_history,
                                        save_path=os.path.join(dirs['training'], 'lr_schedule'))
                print("  ✓ 学习率调度")
            except Exception as e:
                print(f"  ✗ 学习率调度失败: {e}")

        if val_losses is not None:
            best_epoch = int(np.argmin(val_losses)) + 1
            try:
                plot_early_stopping_indicator(val_losses, best_epoch=best_epoch,
                                              save_path=os.path.join(dirs['training'],
                                                                     'early_stopping_indicator'))
                print("  ✓ 早停指示器")
            except Exception as e:
                print(f"  ✗ 早停指示器失败: {e}")

    # === 类别 3: 模型推理 ===
    if pred_masks is not None:
        print("[3/5] 模型推理分析...")
        try:
            plot_mask_confidence(pred_masks, save_path=os.path.join(dirs['samples'], 'mask_confidence'))
            print("  ✓ 掩码置信度")
        except Exception as e:
            print(f"  ✗ 掩码置信度失败: {e}")

    if metrics_dict is not None:
        try:
            plot_frequency_band_analysis(metrics_dict,
                                          save_path=os.path.join(dirs['samples'], 'frequency_band_analysis'))
            print("  ✓ 频带分析")
        except Exception as e:
            print(f"  ✗ 频带分析失败: {e}")

    # === 类别 4: 结果分析 (已在类别1处理了部分) ===
    print("[4/5] 结果分析完成")

    # === 类别 5: 综合报告 ===
    if metrics_dict is not None and train_losses is not None and len(train_losses) > 0:
        print("[5/5] 生成综合报告...")
        try:
            plot_experiment_summary_dashboard(args, metrics_dict, train_losses, val_losses,
                                              save_path=os.path.join(base, 'experiment_summary_dashboard'))
            print("  ✓ 实验总览仪表盘")
        except Exception as e:
            print(f"  ✗ 实验总览仪表盘失败: {e}")

    # 数据集分析（需要外部提供数据时才生成）
    print("[额外] 数据集分析（需外部数据）...")
    try:
        plot_noise_profile(save_path=os.path.join(dirs['data'], 'noise_profile'))
        print("  ✓ 噪声概况")
    except Exception as e:
        print(f"  ✗ 噪声概况失败: {e}")

    print("\n✓ 全部可视化图表生成完毕！")
    print(f"  训练图表: {dirs['training']}/")
    print(f"  分析图表: {dirs['analysis']}/")
    print(f"  样本图表: {dirs['samples']}/")
    print(f"  数据图表: {dirs['data']}/")
    print(f"  总览仪表盘: {base}/experiment_summary_dashboard.*")
