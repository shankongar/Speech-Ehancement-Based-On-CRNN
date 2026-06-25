"""
工具函数模块
包含：音频读写、STFT/iSTFT、特征提取、评估指标（SNR / SI-SNR）
"""

import os
import numpy as np

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    sf = None
    HAS_SOUNDFILE = False
    print("⚠️ 警告: 未安装 soundfile 库，将使用 scipy 作为备用方案")
    print("   安装方法: pip install soundfile")

import librosa
import torch
import torch.nn.functional as F


# ===================== 音频 I/O =====================

def load_audio(path, sr=16000, mono=True):
    """
    读取音频文件，返回 numpy 数组。
    :param path:  音频文件路径
    :param sr:    目标采样率（默认 16 kHz）
    :param mono:  是否转为单声道
    :return:      np.ndarray, shape = (samples,) 或 (channels, samples)
    """
    if HAS_SOUNDFILE and sf is not None:
        wav, sample_rate = sf.read(path, always_2d=False)
    else:
        from scipy.io import wavfile
        sample_rate, wav = wavfile.read(path)
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype == np.int32:
            wav = wav.astype(np.float32) / 2147483648.0
        else:
            wav = wav.astype(np.float32)
    
    if mono and wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sample_rate != sr:
        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=sr)
    return wav.astype(np.float32)


def save_audio(path, wav, sr=16000):
    """将 numpy 数组保存为 WAV 文件。"""
    sf.write(path, wav, sr)


# ===================== STFT / iSTFT =====================

def stft(wav, n_fft=512, hop_length=128, win_length=512, window='hann'):
    """
    对时域信号做短时傅里叶变换。
    :return: complex ndarray, shape = (n_fft//2+1, T)
    """
    return librosa.stft(
        wav,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
    )


def istft(spec, hop_length=128, win_length=512, window='hann', length=None):
    """
    从复数频谱恢复时域信号。
    :param spec:   complex ndarray, shape = (n_fft//2+1, T)
    :param length: 期望的输出采样点数（用于对齐）
    :return:       np.ndarray, shape = (samples,)
    """
    return librosa.istft(
        spec,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=length,
    )


def wav_to_mag_phase(wav, n_fft=512, hop_length=128, win_length=512):
    """
    将时域波形转换为幅度谱和相位谱（用于模型输入）。
    :return: (magnitude, phase)  均为 np.ndarray, shape = (F, T)
             F = n_fft//2+1
    """
    S = stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    mag = np.abs(S)
    phase = np.angle(S)
    return mag, phase


def mag_phase_to_wav(mag, phase, n_fft=512, hop_length=128, win_length=512, length=None):
    """从幅度谱和相位谱重建时域波形。"""
    S = mag * np.exp(1j * phase)
    return istft(S, hop_length=hop_length, win_length=win_length, length=length)


# ===================== 特征提取 =====================

def extract_log_mel(wav, sr=16000, n_fft=512, hop_length=128, win_length=512,
                    n_mels=80, fmin=0, fmax=None):
    """
    提取 Log-Mel 频谱特征。
    :return: np.ndarray, shape = (n_mels, T)
    """
    if fmax is None:
        fmax = sr / 2
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


# ===================== 评估指标 =====================

def snr(clean, enhanced):
    """
    信噪比 (SNR)，单位 dB。
    SNR = 10 * log10( ||clean||^2 / ||clean - enhanced||^2 )
    """
    noise = clean - enhanced
    signal_power = np.sum(clean ** 2)
    noise_power = np.sum(noise ** 2)
    if noise_power < 1e-10:
        return 100.0  # 完美重建
    return 10.0 * np.log10(signal_power / noise_power)


def si_snr(target, estimate):
    """
    尺度不变信噪比 (SI-SNR)，单位 dB。
    常用于语音分离 / 增强评估。
    """
    target = target - np.mean(target)
    estimate = estimate - np.mean(estimate)
    # s_target = <estimate, target> / ||target||^2 * target
    dot = np.sum(estimate * target)
    s_target = (dot / (np.sum(target ** 2) + 1e-8)) * target
    e_noise = estimate - s_target
    si_snr_val = 10.0 * np.log10(
        np.sum(s_target ** 2) / (np.sum(e_noise ** 2) + 1e-8)
    )
    return si_snr_val


def segmental_snr(clean, enhanced, sr=16000, frame_len=0.03):
    """
    分段信噪比 (SegSNR)，单位 dB。
    对每一帧分别计算 SNR 再取平均，更接近人耳感知。
    """
    frame_samples = int(sr * frame_len)
    n_frames = len(clean) // frame_samples
    if n_frames == 0:
        return snr(clean, enhanced)

    seg_snrs = []
    for i in range(n_frames):
        start = i * frame_samples
        end = start + frame_samples
        c = clean[start:end]
        e = enhanced[start:end]
        noise = c - e
        sig_power = np.sum(c ** 2)
        noise_power = np.sum(noise ** 2)
        # 跳过静音帧（信号能量极低）
        if sig_power < 1e-10:
            continue
        if noise_power < 1e-10:
            seg_snrs.append(50.0)
        else:
            seg_snrs.append(10.0 * np.log10(sig_power / noise_power))
    
    # 如果没有有效帧，返回全局 SNR
    if len(seg_snrs) == 0:
        return snr(clean, enhanced)
    return float(np.mean(seg_snrs))


# ===================== PyTorch 辅助 =====================

def numpy_to_tensor(arr, device='cpu'):
    """将 numpy 数组转为 PyTorch tensor (float32)。"""
    return torch.from_numpy(arr).float().to(device)


def tensor_to_numpy(tensor):
    """将 PyTorch tensor 转为 numpy 数组。"""
    return tensor.detach().cpu().numpy()
