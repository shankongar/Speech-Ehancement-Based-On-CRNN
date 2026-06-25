"""
推理/评估脚本
功能：
1. 加载训练好的模型
2. 对含噪语音进行增强
3. 计算评估指标（SNR、SI-SNR、SegSNR）
4. 保存增强后的语音
"""

import os
import glob
import argparse
import numpy as np
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

from core.model import CRNNSpeechEnhancement
from core.utils import (
    load_audio, save_audio, wav_to_mag_phase, mag_phase_to_wav,
    snr, si_snr, segmental_snr
)
from report.visualize import plot_spectrogram_comparison, plot_metrics_comparison, plot_waveform_comparison
from report.logger import log_inference_result


def enhance_audio(model, noisy_wav, device, n_fft=512, hop_length=128, win_length=512):
    """
    对单条语音进行增强
    :param model:       训练好的模型
    :param noisy_wav:   含噪语音 numpy 数组
    :param device:      设备
    :param n_fft:       FFT 窗口大小
    :param hop_length:  帧移
    :param win_length:  窗长
    :return:            增强后的语音 numpy 数组
    """
    model.eval()

    # 提取频谱特征
    noisy_mag, noisy_phase = wav_to_mag_phase(noisy_wav, n_fft, hop_length, win_length)

    # 转为 tensor
    noisy_mag_tensor = torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0).to(device)
    noisy_phase_tensor = torch.from_numpy(noisy_phase).float().unsqueeze(0).to(device)

    # 增强
    with torch.no_grad():
        clean_mag = model.enhance(noisy_mag_tensor, noisy_phase_tensor)

    # 转回 numpy
    clean_mag = clean_mag.squeeze(0).cpu().numpy()

    # 重建波形
    enhanced_wav = mag_phase_to_wav(
        clean_mag, noisy_phase, n_fft, hop_length, win_length, length=len(noisy_wav)
    )

    return enhanced_wav.astype(np.float32)


def evaluate_dataset(model, data_dir, output_dir, device, args):
    """
    评估整个数据集（根据 load_mode 选择模式）
    :param model:      训练好的模型
    :param data_dir:   数据目录
    :param output_dir: 输出目录
    :param device:     设备
    :param args:       参数
    """
    if args.load_mode == 'wav':
        print(f"\n{'='*60}")
        print(f"使用 WAV 模式 - VoiceBank-DEMAND Testset")
        print(f"数据目录: {data_dir}")
        print(f"{'='*60}\n")
        evaluate_from_wav(model, data_dir, output_dir, device, args)
    else:
        npy_test_dir = os.path.join('data', 'processed', 'test')
        print(f"\n{'='*60}")
        print(f"使用 NPY 模式 - 预处理数据")
        print(f"数据目录: {npy_test_dir}")
        print(f"{'='*60}\n")
        evaluate_from_npy(model, npy_test_dir, output_dir, device, args)


def evaluate_from_wav(model, voicebank_dir, output_dir, device, args):
    """
    从 VoiceBank-DEMAND testset 的 wav 文件进行评估
    :param model:         训练好的模型
    :param voicebank_dir: VoiceBank-DEMAND 数据集根目录
    :param output_dir:    输出目录
    :param device:        设备
    :param args:          参数
    """
    import glob

    os.makedirs(output_dir, exist_ok=True)

    noisy_dir = os.path.join(voicebank_dir, 'noisy_testset_wav')
    clean_dir = os.path.join(voicebank_dir, 'clean_testset_wav')

    if not os.path.exists(noisy_dir):
        raise FileNotFoundError(f"找不到 noisy 测试集目录: {noisy_dir}")
    if not os.path.exists(clean_dir):
        raise FileNotFoundError(f"找不到 clean 测试集目录: {clean_dir}")

    # 获取所有 noisy 测试文件
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))

    if len(noisy_files) == 0:
        raise ValueError(f"在 {noisy_dir} 中未找到任何 .wav 文件")

    # 采样
    if args.sample_num > 0 and len(noisy_files) > args.sample_num:
        noisy_files = noisy_files[:args.sample_num]

    print(f"找到 {len(noisy_files)} 个测试样本 (共 {len(sorted(glob.glob(os.path.join(noisy_dir, '*.wav'))))} 个)")
    print(f"Noisy 目录: {noisy_dir}")
    print(f"Clean 目录: {clean_dir}")
    print()

    # 评估指标统计
    metrics = {
        'snr_before': [],
        'snr_after': [],
        'si_snr_before': [],
        'si_snr_after': [],
        'seg_snr_before': [],
        'seg_snr_after': [],
    }

    # 用于可视化的样本信息
    vis_samples = []

    pbar = tqdm(noisy_files, desc='Enhancing', total=len(noisy_files))
    skipped_files = []
    error_files = []

    for idx, noisy_path in enumerate(pbar):
        file_id = os.path.splitext(os.path.basename(noisy_path))[0]
        clean_path = os.path.join(clean_dir, os.path.basename(noisy_path))

        try:
            # 检查 clean 文件是否存在
            if not os.path.exists(clean_path):
                warning_msg = f"⚠ 跳过 {file_id}: 找不到对应的 clean 文件"
                print(warning_msg)
                skipped_files.append(file_id)
                continue

            # 加载 wav 文件
            try:
                noisy_wav = load_audio(noisy_path, sr=args.sr, mono=True)
                clean_wav = load_audio(clean_path, sr=args.sr, mono=True)
            except Exception as e:
                error_msg = f"⚠ 跳过 {file_id}: wav 文件加载失败 - {str(e)}"
                print(error_msg)
                error_files.append((file_id, str(e)))
                continue

            # 增强
            try:
                enhanced_wav = enhance_audio(
                    model, noisy_wav, device,
                    args.n_fft, args.hop_length, args.win_length
                )
            except Exception as e:
                warning_msg = f"⚠ {file_id}: 增强失败，使用原始含噪语音作为 fallback - {str(e)}"
                print(warning_msg)
                enhanced_wav = noisy_wav

            # 计算评估指标
            # 增强前
            snr_before = snr(clean_wav, noisy_wav)
            si_snr_before = si_snr(clean_wav, noisy_wav)
            seg_snr_before = segmental_snr(clean_wav, noisy_wav, args.sr)

            # 增强后
            snr_after = snr(clean_wav, enhanced_wav)
            si_snr_after = si_snr(clean_wav, enhanced_wav)
            seg_snr_after = segmental_snr(clean_wav, enhanced_wav, args.sr)

            # 保存指标
            metrics['snr_before'].append(snr_before)
            metrics['snr_after'].append(snr_after)
            metrics['si_snr_before'].append(si_snr_before)
            metrics['si_snr_after'].append(si_snr_after)
            metrics['seg_snr_before'].append(seg_snr_before)
            metrics['seg_snr_after'].append(seg_snr_after)

            # 保存增强后的音频
            if args.save_audio:
                save_audio(
                    os.path.join(output_dir, f'{file_id}_enhanced.wav'),
                    enhanced_wav, args.sr
                )

            # 收集前3个样本用于可视化
            if idx < 3:
                vis_samples.append({
                    'file_id': file_id,
                    'clean_wav': clean_wav,
                    'noisy_wav': noisy_wav,
                    'enhanced_wav': enhanced_wav,
                    'noisy_path': noisy_path,
                })

            # 更新进度条
            pbar.set_postfix({
                'SNR': f'{snr_before:.2f}->{snr_after:.2f}',
                'SI-SNR': f'{si_snr_before:.2f}->{si_snr_after:.2f}',
            })

        except Exception as e:
            error_msg = f"⚠ 处理 {file_id} 时发生未知错误: {str(e)}"
            print(error_msg)
            error_files.append((file_id, str(e)))
            continue

    # 打印汇总统计
    n_valid = len(metrics['snr_before'])
    print("\n" + "=" * 70)
    print("VoiceBank-DEMAND Testset 评估结果汇总")
    print("=" * 70)
    print(f"\n有效样本数: {n_valid}/{len(noisy_files)}")
    if skipped_files:
        print(f"跳过文件 (缺少 clean): {len(skipped_files)} 个")
    if error_files:
        print(f"错误文件: {len(error_files)} 个")

    if n_valid == 0:
        print("\n❌ 没有有效的评估结果！")
        return

    print(f"\nSNR (dB):")
    print(f"  增强前: {np.mean(metrics['snr_before']):.2f} ± {np.std(metrics['snr_before']):.2f}")
    print(f"  增强后: {np.mean(metrics['snr_after']):.2f} ± {np.std(metrics['snr_after']):.2f}")
    print(f"  提升:   {np.mean(metrics['snr_after']) - np.mean(metrics['snr_before']):.2f}")

    print(f"\nSI-SNR (dB):")
    print(f"  增强前: {np.mean(metrics['si_snr_before']):.2f} ± {np.std(metrics['si_snr_before']):.2f}")
    print(f"  增强后: {np.mean(metrics['si_snr_after']):.2f} ± {np.std(metrics['si_snr_after']):.2f}")
    print(f"  提升:   {np.mean(metrics['si_snr_after']) - np.mean(metrics['si_snr_before']):.2f}")

    print(f"\nSegSNR (dB):")
    print(f"  增强前: {np.mean(metrics['seg_snr_before']):.2f} ± {np.std(metrics['seg_snr_before']):.2f}")
    print(f"  增强后: {np.mean(metrics['seg_snr_after']):.2f} ± {np.std(metrics['seg_snr_after']):.2f}")
    print(f"  提升:   {np.mean(metrics['seg_snr_after']) - np.mean(metrics['seg_snr_before']):.2f}")
    print("=" * 70)

    # 保存评估结果
    metrics['dataset_info'] = {
        'mode': 'VoiceBank-DEMAND testset (wav)',
        'voicebank_dir': voicebank_dir,
        'total_samples': len(noisy_files),
        'valid_samples': n_valid,
        'skipped_files': skipped_files,
        'error_files': error_files,
    }
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)

    # ===================== 可视化 =====================
    visualize_results(model, vis_samples, metrics, output_dir, device, args, mode='wav')

    # 追加评估日志
    log_path = 'results/logs/experiment_log.txt'
    inf_extra_info = {
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'win_length': args.win_length,
        'sr': args.sr,
    }
    log_inference_result(log_path, args, metrics,
                         dataset_source='VoiceBank-DEMAND testset (wav)',
                         extra_info=inf_extra_info)
    print(f"✓ 评估日志已追加: {log_path}")

    # ===================== 新增：全局结果分析图 =====================
    analysis_dir = os.path.join('results', 'figures', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    try:
        from report.visualize import (
            plot_radar_chart, plot_correlation_heatmap,
            plot_improvement_ranking, plot_pareto_frontier,
            plot_frequency_band_analysis,
            plot_experiment_summary_dashboard,
        )

        # V14: 雷达图
        for fmt in ['png', 'pdf']:
            plot_radar_chart(metrics, save_path=os.path.join(analysis_dir, f'radar_chart.{fmt}'))
        print("✓ 雷达图已保存")

        # V15: 相关性热力图
        for fmt in ['png', 'pdf']:
            plot_correlation_heatmap(metrics, save_path=os.path.join(analysis_dir, f'correlation_heatmap.{fmt}'))
        print("✓ 相关性热力图已保存")

        # V16: 排名图
        for fmt in ['png', 'pdf']:
            plot_improvement_ranking(metrics, top_k=15, save_path=os.path.join(analysis_dir, f'improvement_ranking.{fmt}'))
        print("✓ 排名图已保存")

        # V17: Pareto 前沿
        for fmt in ['png', 'pdf']:
            plot_pareto_frontier(metrics, save_path=os.path.join(analysis_dir, f'pareto_frontier.{fmt}'))
        print("✓ Pareto 前沿图已保存")

        # V10: 分频带分析
        for fmt in ['png', 'pdf']:
            plot_frequency_band_analysis(metrics, sr=args.sr, n_fft=args.n_fft,
                                          save_path=os.path.join(analysis_dir, f'frequency_band.{fmt}'))
        print("✓ 分频带分析图已保存")

        # V18: 实验总览仪表盘
        try:
            import numpy as _np
            _train_losses = _np.load('checkpoints/training_log.npy', allow_pickle=True).item() \
                if os.path.exists('checkpoints/training_log.npy') else None
            _tl = _train_losses.get('train_losses') if _train_losses else None
            _vl = _train_losses.get('val_losses') if _train_losses else None
            for fmt in ['png', 'pdf']:
                plot_experiment_summary_dashboard(args, metrics, _tl or [], _vl or [],
                    save_path=os.path.join(analysis_dir, f'experiment_dashboard.{fmt}'))
            print("✓ 实验总览仪表盘已保存")
        except Exception:
            pass  # 无训练日志时跳过

    except Exception as e:
        print(f"✗ 全局分析图生成失败: {e}")

    print(f"\n✓ 评估结果已保存到: {output_dir}")


def evaluate_from_npy(model, data_dir, output_dir, device, args):
    """
    从预处理后的 npy 文件进行评估（原有逻辑）
    :param model:      训练好的模型
    :param data_dir:   预处理数据目录
    :param output_dir: 输出目录
    :param device:     设备
    :param args:       参数
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有含噪语音文件
    noisy_files = sorted(glob.glob(os.path.join(data_dir, '*_noisy_mag.npy')))

    if len(noisy_files) == 0:
        raise ValueError(f"在 {data_dir} 中未找到任何 *_noisy_mag.npy 文件")

    # 随机采样
    if args.sample_num > 0 and len(noisy_files) > args.sample_num:
        noisy_files = np.random.choice(noisy_files, args.sample_num, replace=False).tolist()

    print(f"找到 {len(noisy_files)} 个含噪语音文件 (共 {len(sorted(glob.glob(os.path.join(data_dir, '*_noisy_mag.npy'))))} 个)")

    # 评估指标统计
    metrics = {
        'snr_before': [],
        'snr_after': [],
        'si_snr_before': [],
        'si_snr_after': [],
        'seg_snr_before': [],
        'seg_snr_after': [],
    }

    # 用于可视化的样本信息
    vis_samples = []

    pbar = tqdm(noisy_files, desc='Enhancing')

    for idx, noisy_path in enumerate(pbar):
        try:
            # 获取基础文件名
            base = noisy_path.replace('_noisy_mag.npy', '')
            file_id = os.path.basename(base)

            # 加载特征
            clean_mag = np.load(base + '_clean_mag.npy')
            clean_phase = np.load(base + '_clean_phase.npy')
            noisy_mag = np.load(base + '_noisy_mag.npy')
            noisy_phase = np.load(base + '_noisy_phase.npy')

            # 重建波形
            clean_wav = mag_phase_to_wav(clean_mag, clean_phase, args.n_fft, args.hop_length, args.win_length)
            noisy_wav = mag_phase_to_wav(noisy_mag, noisy_phase, args.n_fft, args.hop_length, args.win_length)

            # 增强
            enhanced_wav = enhance_audio(model, noisy_wav, device, args.n_fft, args.hop_length, args.win_length)

            # 计算评估指标
            # 增强前
            snr_before = snr(clean_wav, noisy_wav)
            si_snr_before = si_snr(clean_wav, noisy_wav)
            seg_snr_before = segmental_snr(clean_wav, noisy_wav, args.sr)

            # 增强后
            snr_after = snr(clean_wav, enhanced_wav)
            si_snr_after = si_snr(clean_wav, enhanced_wav)
            seg_snr_after = segmental_snr(clean_wav, enhanced_wav, args.sr)

            # 保存指标
            metrics['snr_before'].append(snr_before)
            metrics['snr_after'].append(snr_after)
            metrics['si_snr_before'].append(si_snr_before)
            metrics['si_snr_after'].append(si_snr_after)
            metrics['seg_snr_before'].append(seg_snr_before)
            metrics['seg_snr_after'].append(seg_snr_after)

            # 保存增强后的语音
            if args.save_audio:
                save_audio(os.path.join(output_dir, f'{file_id}_enhanced.wav'), enhanced_wav, args.sr)

            # 收集前3个样本用于可视化
            if idx < 3:
                vis_samples.append({
                    'file_id': file_id,
                    'clean_wav': clean_wav,
                    'noisy_wav': noisy_wav,
                    'enhanced_wav': enhanced_wav,
                    'clean_mag': clean_mag,
                    'noisy_mag': noisy_mag,
                    'noisy_phase': noisy_phase,
                    'base_path': base,
                })

            # 更新进度条
            pbar.set_postfix({
                'SNR': f'{snr_before:.2f} -> {snr_after:.2f}',
            })

        except Exception as e:
            print(f"\n⚠ 处理文件 {noisy_path} 时出错: {str(e)}")
            continue

    # 计算平均指标
    n_valid = len(metrics['snr_before'])
    print("\n" + "=" * 60)
    print("NPY 预处理数据评估结果汇总")
    print("=" * 60)
    print(f"\n有效样本数: {n_valid}")

    if n_valid == 0:
        print("\n❌ 没有有效的评估结果！")
        return

    print(f"\nSNR (dB):")
    print(f"  增强前: {np.mean(metrics['snr_before']):.2f} ± {np.std(metrics['snr_before']):.2f}")
    print(f"  增强后: {np.mean(metrics['snr_after']):.2f} ± {np.std(metrics['snr_after']):.2f}")
    print(f"  提升:   {np.mean(metrics['snr_after']) - np.mean(metrics['snr_before']):.2f}")

    print(f"\nSI-SNR (dB):")
    print(f"  增强前: {np.mean(metrics['si_snr_before']):.2f} ± {np.std(metrics['si_snr_before']):.2f}")
    print(f"  增强后: {np.mean(metrics['si_snr_after']):.2f} ± {np.std(metrics['si_snr_after']):.2f}")
    print(f"  提升:   {np.mean(metrics['si_snr_after']) - np.mean(metrics['si_snr_before']):.2f}")

    print(f"\nSegSNR (dB):")
    print(f"  增强前: {np.mean(metrics['seg_snr_before']):.2f} ± {np.std(metrics['seg_snr_before']):.2f}")
    print(f"  增强后: {np.mean(metrics['seg_snr_after']):.2f} ± {np.std(metrics['seg_snr_after']):.2f}")
    print(f"  提升:   {np.mean(metrics['seg_snr_after']) - np.mean(metrics['seg_snr_before']):.2f}")
    print("=" * 60)

    # 保存评估结果
    metrics['dataset_info'] = {
        'mode': 'Preprocessed NPY',
        'data_dir': data_dir,
        'total_samples': len(noisy_files),
        'valid_samples': n_valid,
    }
    np.save(os.path.join(output_dir, 'metrics.npy'), metrics)

    # ===================== 可视化 =====================
    visualize_results(model, vis_samples, metrics, output_dir, device, args, mode='npy')

    # 追加评估日志
    log_path = 'results/logs/experiment_log.txt'
    npy_extra_info = {
        'n_fft': args.n_fft,
        'hop_length': args.hop_length,
        'win_length': args.win_length,
        'sr': args.sr,
    }
    log_inference_result(log_path, args, metrics,
                         dataset_source='Preprocessed NPY data',
                         extra_info=npy_extra_info)
    print(f"✓ 评估日志已追加: {log_path}")

    # ===================== 新增：全局结果分析图 =====================
    analysis_dir = os.path.join('results', 'figures', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)

    try:
        from report.visualize import (
            plot_radar_chart, plot_correlation_heatmap,
            plot_improvement_ranking, plot_pareto_frontier,
            plot_frequency_band_analysis,
            plot_experiment_summary_dashboard,
        )

        # V14: 雷达图
        for fmt in ['png', 'pdf']:
            plot_radar_chart(metrics, save_path=os.path.join(analysis_dir, f'radar_chart.{fmt}'))
        print("✓ 雷达图已保存")

        # V15: 相关性热力图
        for fmt in ['png', 'pdf']:
            plot_correlation_heatmap(metrics, save_path=os.path.join(analysis_dir, f'correlation_heatmap.{fmt}'))
        print("✓ 相关性热力图已保存")

        # V16: 排名图
        for fmt in ['png', 'pdf']:
            plot_improvement_ranking(metrics, top_k=15, save_path=os.path.join(analysis_dir, f'improvement_ranking.{fmt}'))
        print("✓ 排名图已保存")

        # V17: Pareto 前沿
        for fmt in ['png', 'pdf']:
            plot_pareto_frontier(metrics, save_path=os.path.join(analysis_dir, f'pareto_frontier.{fmt}'))
        print("✓ Pareto 前沿图已保存")

        # V10: 分频带分析
        for fmt in ['png', 'pdf']:
            plot_frequency_band_analysis(metrics, sr=args.sr, n_fft=args.n_fft,
                                          save_path=os.path.join(analysis_dir, f'frequency_band.{fmt}'))
        print("✓ 分频带分析图已保存")

        # V18: 实验总览仪表盘
        try:
            import numpy as _np
            _train_losses = _np.load('checkpoints/training_log.npy', allow_pickle=True).item() \
                if os.path.exists('checkpoints/training_log.npy') else None
            _tl = _train_losses.get('train_losses') if _train_losses else None
            _vl = _train_losses.get('val_losses') if _train_losses else None
            for fmt in ['png', 'pdf']:
                plot_experiment_summary_dashboard(args, metrics, _tl or [], _vl or [],
                    save_path=os.path.join(analysis_dir, f'experiment_dashboard.{fmt}'))
            print("✓ 实验总览仪表盘已保存")
        except Exception:
            pass  # 无训练日志时跳过

    except Exception as e:
        print(f"✗ 全局分析图生成失败: {e}")

    print(f"\n✓ 评估结果已保存到: {output_dir}")


def visualize_results(model, vis_samples, metrics, output_dir, device, args, mode='wav'):
    """
    统一的可视化函数
    :param model:        模型
    :param vis_samples:  可视化样本列表
    :param metrics:      评估指标
    :param output_dir:   输出目录
    :param device:       设备
    :param args:         参数
    :param mode:         数据模式 ('wav' 或 'npy')
    """
    figures_dir = 'results/figures'
    os.makedirs(figures_dir, exist_ok=True)

    n_vis = min(3, len(vis_samples))

    for i in range(n_vis):
        sample = vis_samples[i]
        file_id = sample['file_id']
        clean_wav = sample['clean_wav']
        noisy_wav = sample['noisy_wav']
        enhanced_wav = sample['enhanced_wav']

        if mode == 'wav':
            # wav 模式：从波形提取频谱
            clean_mag, _ = wav_to_mag_phase(clean_wav, args.n_fft, args.hop_length, args.win_length)
            noisy_mag, noisy_phase = wav_to_mag_phase(noisy_wav, args.n_fft, args.hop_length, args.win_length)

            # 使用模型增强频谱
            enhanced_mag = model.enhance(
                torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0).to(device),
                torch.from_numpy(noisy_phase).float().unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()
        else:
            # npy 模式：直接使用预加载的频谱
            clean_mag = sample.get('clean_mag')
            noisy_mag = sample.get('noisy_mag')
            noisy_phase = sample.get('noisy_phase')

            enhanced_mag = model.enhance(
                torch.from_numpy(noisy_mag).float().unsqueeze(0).unsqueeze(0).to(device),
                torch.from_numpy(noisy_phase).float().unsqueeze(0).to(device)
            ).squeeze(0).cpu().numpy()

        # 频谱对比图
        png_base = os.path.join(figures_dir, f'spectrogram_{file_id}')
        plot_spectrogram_comparison(clean_mag, noisy_mag, enhanced_mag,
            sr=args.sr, hop_length=args.hop_length, save_path=png_base + '.png')
        plot_spectrogram_comparison(clean_mag, noisy_mag, enhanced_mag,
            sr=args.sr, hop_length=args.hop_length, save_path=png_base + '.pdf')

        # 波形对比图
        png_base = os.path.join(figures_dir, f'waveform_{file_id}')
        plot_waveform_comparison(clean_wav, noisy_wav, enhanced_wav,
            sr=args.sr, save_path=png_base + '.png')
        plot_waveform_comparison(clean_wav, noisy_wav, enhanced_wav,
            sr=args.sr, save_path=png_base + '.pdf')

    # 指标对比柱状图
    if len(metrics.get('snr_before', [])) > 0:
        png_base = os.path.join(figures_dir, 'metrics_comparison')
        plot_metrics_comparison(metrics, save_path=png_base + '.png')
        plot_metrics_comparison(metrics, save_path=png_base + '.pdf')
        print(f"✓ 可视化图表已保存到: {figures_dir}")

    # ===================== 新增：逐样本深度分析 =====================
    sample_dir = os.path.join(figures_dir, 'samples')
    os.makedirs(sample_dir, exist_ok=True)

    try:
        from report.visualize import plot_audio_statistics

        # V02: 音频多维统计（对前 3 个样本生成）
        n_sample_vis = min(3, len(vis_samples))
        for idx in range(n_sample_vis):
            sample = vis_samples[idx]
            file_id = sample['file_id']
            noisy_wav = sample.get('noisy_wav')
            if noisy_wav is not None:
                for fmt in ['png', 'pdf']:
                    plot_audio_statistics(noisy_wav, args.sr,
                        save_path=os.path.join(sample_dir, f'audio_stats_{file_id}.{fmt}'))
                print(f"  ✓ 音频统计图: audio_stats_{file_id}")

        # V12: 掩码置信度分析（对有预测掩码的样本）
        try:
            from report.visualize import plot_mask_confidence
            for idx in range(n_sample_vis):
                sample = vis_samples[idx]
                file_id = sample['file_id']
                pred_mask = sample.get('pred_mask')
                if pred_mask is not None:
                    for fmt in ['png', 'pdf']:
                        plot_mask_confidence(pred_mask,
                            save_path=os.path.join(sample_dir, f'mask_confidence_{file_id}.{fmt}'))
                    print(f"  ✓ 掩码置信度图: mask_confidence_{file_id}")
        except Exception as e:
            print(f"  ✗ 掩码置信度图生成失败: {e}")
    except Exception as e:
        print(f"  ✗ 音频统计图生成失败: {e}")


def main(args):
    """主函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"使用设备: {device}")

    # 加载模型
    print(f"加载模型: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # 提取参数
    model_args = checkpoint['args']

    # 创建模型
    model = CRNNSpeechEnhancement(
        input_dim=model_args.input_dim,
        hidden_dim=model_args.hidden_dim,
        num_layers=model_args.num_layers,
    ).to(device)

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"模型加载成功 (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss']:.4f})")

    # 评估
    evaluate_dataset(model, args.data_dir, args.enhance_output_dir, device, args)


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
        model_path=cfg.model_path or cfg.best_model_path,
        data_dir=cfg.data_dir,
        load_mode=cfg.load_mode,
        enhance_output_dir=cfg.enhance_output_dir,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        sr=cfg.sr,
        sample_num=cfg.eval_sample_num,
        save_audio=cfg.save_audio,
        cpu=cfg.cpu,
    )
    init_seed()
    main(args)
