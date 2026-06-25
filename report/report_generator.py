import os
import numpy as np
import torch
from datetime import datetime


def generate_config_report(args, metrics_dict, save_path='results/report/config_report.md'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

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

    data_dir = getattr(args, 'train_data_dir', None) or getattr(args, 'data_dir', None)

    try:
        import sys
        python_version = sys.version.split()[0]
    except Exception:
        python_version = 'N/A'

    try:
        pytorch_version = torch.__version__
    except Exception:
        pytorch_version = 'N/A'

    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = 'N/A'

    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = 'N/A'

    report = f"""# CRNN Speech Enhancement - Experiment Report

**Generated:** {timestamp}

---

## 1. Hyperparameters

| Parameter | Value |
|-----------|-------|
| Model | CRNN (Conv + BiLSTM + DeConv) |
| Input Dimension | {args.input_dim} |
| Hidden Dimension | {args.hidden_dim} |
| LSTM Layers | {args.num_layers} |
| Epochs | {args.epochs} |
| Batch Size | {args.batch_size} |
| Learning Rate | {args.lr} |
| Training Samples | {args.sample_num} |

## 2. Dataset

| Item | Value |
|------|-------|
| Data Directory | {data_dir} |
| Noise Types | White, Pink, Babble |
| SNR Range | [-5, 15] dB |

## 3. Results Summary

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| SNR (dB) | {snr_before_mean:.2f}±{snr_before_std:.2f} | {snr_after_mean:.2f}±{snr_after_std:.2f} | {snr_improve:+.2f} |
| SI-SNR (dB) | {si_snr_before_mean:.2f}±{si_snr_before_std:.2f} | {si_snr_after_mean:.2f}±{si_snr_after_std:.2f} | {si_snr_improve:+.2f} |
| SegSNR (dB) | {seg_snr_before_mean:.2f}±{seg_snr_before_std:.2f} | {seg_snr_after_mean:.2f}±{seg_snr_after_std:.2f} | {seg_snr_improve:+.2f} |

## 4. Output Files

### Figures
- results/report/figures/training_loss_curve.pdf
- results/report/figures/metrics_comparison.pdf
- results/report/figures/training_deep_dive.pdf
- results/report/figures/model_architecture.pdf
- results/report/figures/error_distribution.pdf

### Audio
- results/enhanced/*_enhanced.wav

## 5. Environment
- Python: {python_version}
- PyTorch: {pytorch_version}
- CUDA Available: {cuda_available}
- GPU: {gpu_name}
"""

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Config report saved to: {save_path}")


def generate_latex_table(metrics_dict, args, save_path='results/report/latex_tables.tex'):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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

    latex_content = f"""%% Table: Speech Enhancement Results
\\begin{{table}}[htbp]
\\centering
\\caption{{Speech Enhancement Results on Validation Set}}
\\label{{tab:results}}
\\begin{{tabular}}{{lccc}}
\\hline
Metric & Before (dB) & After (dB) & Improvement (dB) \\\\
\\hline
SNR      & {snr_before_mean:.2f}$\\pm${snr_before_std:.2f} & {snr_after_mean:.2f}$\\pm${snr_after_std:.2f} & {snr_improve:+.2f} \\\\
SI-SNR   & {si_snr_before_mean:.2f}$\\pm${si_snr_before_std:.2f} & {si_snr_after_mean:.2f}$\\pm${si_snr_after_std:.2f} & {si_snr_improve:+.2f} \\\\
SegSNR   & {seg_snr_before_mean:.2f}$\\pm${seg_snr_before_std:.2f} & {seg_snr_after_mean:.2f}$\\pm${seg_snr_after_std:.2f} & {seg_snr_improve:+.2f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}

%% Table: Model Hyperparameters
\\begin{{table}}[htbp]
\\centering
\\caption{{Model Hyperparameters}}
\\label{{tab:hyp}}
\\begin{{tabular}}{{ll}}
\\hline
Parameter & Value \\\\
\\hline
Model Architecture & CRNN (Enc-BLSTM-Dec) \\\\
Input Dim & {args.input_dim} \\\\
Hidden Dim & {args.hidden_dim} \\\\
LSTM Layers & {args.num_layers} \\\\
Epochs & {args.epochs} \\\\
Batch Size & {args.batch_size} \\\\
Learning Rate & {args.lr} \\\\
Training Samples & {args.sample_num} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)

    print(f"✓ LaTeX tables saved to: {save_path}")


def generate_all_figures(args, model, data_dir, device, train_losses=None, val_losses=None, lrs=None):
    import glob as glob_module

    from report.visualize import (
        plot_training_loss, plot_spectrogram_comparison, plot_metrics_comparison,
        plot_waveform_comparison, plot_mask_heatmap, plot_sample_detail,
        plot_error_distribution, plot_model_architecture, plot_training_deep_dive
    )
    from core.utils import wav_to_mag_phase, mag_phase_to_wav, snr, si_snr, segmental_snr

    figures_dir = 'results/report/figures'
    os.makedirs(figures_dir, exist_ok=True)

    if train_losses is not None:
        plot_training_loss(train_losses, val_losses,
                           os.path.join(figures_dir, 'training_loss_curve.pdf'))
        plot_training_deep_dive(train_losses, val_losses, lrs,
                                os.path.join(figures_dir, 'training_deep_dive.pdf'))
        plot_model_architecture(os.path.join(figures_dir, 'model_architecture.pdf'))

    noisy_files = sorted(glob_module.glob(os.path.join(data_dir, '*_noisy_mag.npy')))

    if hasattr(args, 'sample_num') and args.sample_num > 0 and len(noisy_files) > args.sample_num:
        indices = np.random.choice(len(noisy_files), args.sample_num, replace=False)
        noisy_files = [noisy_files[i] for i in indices]

    n_samples = len(noisy_files)
    n_vis = min(3, n_samples)

    metrics_dict = {
        'snr_before': [],
        'snr_after': [],
        'si_snr_before': [],
        'si_snr_after': [],
        'seg_snr_before': [],
        'seg_snr_after': [],
    }

    n_fft = getattr(args, 'n_fft', 512)
    hop_length = getattr(args, 'hop_length', 128)
    win_length = getattr(args, 'win_length', 512)

    for i in range(n_vis):
        base = noisy_files[i].replace('_noisy_mag.npy', '')

        clean_mag = np.load(base + '_clean_mag.npy')
        clean_phase = np.load(base + '_clean_phase.npy')
        noisy_mag = np.load(base + '_noisy_mag.npy')
        noisy_phase = np.load(base + '_noisy_phase.npy')

        clean_wav = mag_phase_to_wav(clean_mag, clean_phase, n_fft, hop_length, win_length)
        noisy_wav = mag_phase_to_wav(noisy_mag, noisy_phase, n_fft, hop_length, win_length)

        noisy_mag_tensor = torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0).to(device)
        noisy_phase_tensor = torch.from_numpy(noisy_phase).float().unsqueeze(0).to(device)

        enhanced_mag = model.enhance(noisy_mag_tensor, noisy_phase_tensor)
        enhanced_mag_np = enhanced_mag.squeeze(0).cpu().numpy()

        enhanced_wav = mag_phase_to_wav(enhanced_mag_np, noisy_phase, n_fft, hop_length,
                                        win_length, length=len(noisy_wav))

        ideal_mask = clean_mag / (noisy_mag + 1e-8)
        ideal_mask = np.clip(ideal_mask, 0, 1)

        pred_mask = enhanced_mag_np / (noisy_mag + 1e-8)
        pred_mask = np.clip(pred_mask, 0, 1)

        plot_sample_detail(
            clean_wav, noisy_wav, enhanced_wav,
            clean_mag, noisy_mag, enhanced_mag_np,
            pred_mask, ideal_mask,
            save_path=os.path.join(figures_dir, f'sample_detail_{i}.pdf')
        )

        plot_mask_heatmap(
            pred_mask, ideal_mask, noisy_mag,
            save_path=os.path.join(figures_dir, f'mask_heatmap_{i}.pdf')
        )

        plot_spectrogram_comparison(
            clean_mag, noisy_mag, enhanced_mag_np,
            sr=getattr(args, 'sr', 16000), hop_length=hop_length,
            save_path=os.path.join(figures_dir, f'spectrogram_{i}.pdf')
        )

        plot_waveform_comparison(
            clean_wav, noisy_wav, enhanced_wav,
            sr=getattr(args, 'sr', 16000),
            save_path=os.path.join(figures_dir, f'waveform_{i}.pdf')
        )

        snr_before_val = snr(clean_wav, noisy_wav)
        snr_after_val = snr(clean_wav, enhanced_wav)
        si_snr_before_val = si_snr(clean_wav, noisy_wav)
        si_snr_after_val = si_snr(clean_wav, enhanced_wav)
        seg_snr_before_val = segmental_snr(clean_wav, noisy_wav, getattr(args, 'sr', 16000))
        seg_snr_after_val = segmental_snr(clean_wav, enhanced_wav, getattr(args, 'sr', 16000))

        metrics_dict['snr_before'].append(snr_before_val)
        metrics_dict['snr_after'].append(snr_after_val)
        metrics_dict['si_snr_before'].append(si_snr_before_val)
        metrics_dict['si_snr_after'].append(si_snr_after_val)
        metrics_dict['seg_snr_before'].append(seg_snr_before_val)
        metrics_dict['seg_snr_after'].append(seg_snr_after_val)

    for j in range(n_vis, n_samples):
        base = noisy_files[j].replace('_noisy_mag.npy', '')

        clean_mag = np.load(base + '_clean_mag.npy')
        clean_phase = np.load(base + '_clean_phase.npy')
        noisy_mag = np.load(base + '_noisy_mag.npy')
        noisy_phase = np.load(base + '_noisy_phase.npy')

        clean_wav = mag_phase_to_wav(clean_mag, clean_phase, n_fft, hop_length, win_length)
        noisy_wav = mag_phase_to_wav(noisy_mag, noisy_phase, n_fft, hop_length, win_length)

        noisy_mag_tensor = torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0).to(device)
        noisy_phase_tensor = torch.from_numpy(noisy_phase).float().unsqueeze(0).to(device)

        enhanced_mag = model.enhance(noisy_mag_tensor, noisy_phase_tensor)
        enhanced_mag_np = enhanced_mag.squeeze(0).cpu().numpy()

        enhanced_wav = mag_phase_to_wav(enhanced_mag_np, noisy_phase, n_fft, hop_length,
                                        win_length, length=len(noisy_wav))

        metrics_dict['snr_before'].append(snr(clean_wav, noisy_wav))
        metrics_dict['snr_after'].append(snr(clean_wav, enhanced_wav))
        metrics_dict['si_snr_before'].append(si_snr(clean_wav, noisy_wav))
        metrics_dict['si_snr_after'].append(si_snr(clean_wav, enhanced_wav))
        metrics_dict['seg_snr_before'].append(segmental_snr(clean_wav, noisy_wav, getattr(args, 'sr', 16000)))
        metrics_dict['seg_snr_after'].append(segmental_snr(clean_wav, enhanced_wav, getattr(args, 'sr', 16000)))

    if len(metrics_dict['snr_before']) > 0:
        plot_error_distribution(metrics_dict,
                                save_path=os.path.join(figures_dir, 'error_distribution.pdf'))
        plot_metrics_comparison(metrics_dict,
                               save_path=os.path.join(figures_dir, 'metrics_comparison.pdf'))

    print(f"All figures saved to {figures_dir}/")

    return metrics_dict


def generate_full_report(args, model, data_dir, device, train_losses=None, val_losses=None, lrs=None):
    from report.logger import log_inference_result

    metrics = generate_all_figures(args, model, data_dir, device, train_losses, val_losses, lrs)

    generate_config_report(args, metrics)

    generate_latex_table(metrics, args)

    extra_info = {
        'n_fft': getattr(args, 'n_fft', 512),
        'hop_length': getattr(args, 'hop_length', 128),
        'win_length': getattr(args, 'win_length', 512),
        'sr': getattr(args, 'sr', 16000),
    }
    log_inference_result('results/logs/experiment_log.txt', args, metrics,
                         dataset_source=getattr(args, 'data_dir', None),
                         extra_info=extra_info)

    print("\n" + "=" * 60)
    print("完整报告索引清单")
    print("=" * 60)
    print("📄 报告文件:")
    print("   - results/report/config_report.md")
    print("   - results/report/latex_tables.tex")
    print("📊 图表文件:")
    figures = [
        "training_loss_curve.pdf",
        "training_deep_dive.pdf",
        "model_architecture.pdf",
        "error_distribution.pdf",
        "metrics_comparison.pdf"
    ]
    for fig in figures:
        print(f"   - results/report/figures/{fig}")
    for i in range(min(3, len(os.listdir('results/report/figures')) - 5)):
        detail_figs = [
            f"sample_detail_{i}.pdf",
            f"mask_heatmap_{i}.pdf",
            f"spectrogram_{i}.pdf",
            f"waveform_{i}.pdf"
        ]
        for fig in detail_figs:
            print(f"   - results/report/figures/{fig}")
    print("📝 日志文件:")
    print("   - results/logs/experiment_log.txt")
    print("=" * 60)

    return metrics
