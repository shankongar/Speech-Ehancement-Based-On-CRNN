"""
PyTorch Dataset 类
支持两种数据加载模式:
1. npy模式: 从 data/processed/{train,val,test} 加载预处理的 .npy 特征文件
2. wav模式: 从 VoiceBank-DEMAND 数据集直接加载 wav 文件并在线提取 STFT 特征
"""

import os
import glob
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

from core.utils import load_audio, wav_to_mag_phase


def load_wav_features(clean_path, noisy_path, n_fft=512, hop_length=128, win_length=512):
    """
    从 clean/noisy wav 文件对在线提取 STFT 特征。

    :param clean_path:   干净语音 wav 路径
    :param noisy_path:   含噪语音 wav 路径
    :param n_fft:        FFT 窗口大小
    :param hop_length:   帧移
    :param win_length:   窗长
    :return: (clean_mag, clean_phase, noisy_mag, noisy_phase) 均为 np.ndarray shape=(F,T)
    :raises: FileNotFoundError / ValueError 当文件不存在或提取失败时
    """
    if not os.path.exists(clean_path):
        raise FileNotFoundError(f"干净语音文件不存在: {clean_path}")
    if not os.path.exists(noisy_path):
        raise FileNotFoundError(f"含噪语音文件不存在: {noisy_path}")

    try:
        clean_wav = load_audio(clean_path)
        noisy_wav = load_audio(noisy_path)

        clean_mag, clean_phase = wav_to_mag_phase(clean_wav, n_fft=n_fft,
                                                   hop_length=hop_length, win_length=win_length)
        noisy_mag, noisy_phase = wav_to_mag_phase(noisy_wav, n_fft=n_fft,
                                                   hop_length=hop_length, win_length=win_length)

        return (
            clean_mag.astype(np.float32),
            clean_phase.astype(np.float32),
            noisy_mag.astype(np.float32),
            noisy_phase.astype(np.float32),
        )
    except Exception as e:
        raise ValueError(f"STFT 特征提取失败 ({os.path.basename(clean_path)}): {e}")


class SpeechEnhancementDataset(Dataset):
    """
    语音增强数据集，支持 npy / wav 双模式加载。
    
    返回格式（两种模式一致）:
      {
          'noisy_mag': tensor (1, F, T),
          'clean_mag': tensor (1, F, T),
          'noisy_phase': tensor (F, T),
          'file_id': str,
      }
    """

    def __init__(self, data_dir, mode='train', sample_num=500, load_mode='npy'):
        """
        初始化数据集。

        :param data_dir: 数据根目录
                         - npy 模式: data/processed/train (或 val/test)
                         - wav 模式: data/VoiceBank-DEMAND/
        :param mode:     'train' / 'val' / 'test'
        :param sample_num: 采样数量，默认 500
        :param load_mode: 'npy' 从预处理文件加载 / 'wav' 直接从 wav 加载
        """
        self.data_dir = data_dir
        self.mode = mode
        self.sample_num = sample_num
        self.load_mode = load_mode

        if load_mode == 'wav':
            self._init_wav_mode()
        elif load_mode == 'npy':
            self._init_npy_mode()
        else:
            raise ValueError(f"不支持的 load_mode: {load_mode}，请选择 'npy' 或 'wav'")

    def _init_npy_mode(self):
        """初始化 npy 加载模式"""
        root_dir = self.data_dir
        pattern = "*_clean_mag.npy"

        print(f"[{self.mode}|npy] 开始加载 {root_dir} 中的 {pattern} 文件")

        self.file_list = sorted(
            glob.glob(os.path.join(root_dir, pattern))
        )

        if len(self.file_list) == 0:
            raise FileNotFoundError(
                f"未找到 {pattern} 文件，请确认目录: {root_dir}\n"
                f"提示: 请先运行 python main.py preprocess 生成预处理数据，或使用 --load_mode wav 模式"
            )

        actual_num = min(self.sample_num, len(self.file_list))
        self.file_list = list(np.random.choice(
            self.file_list, actual_num, replace=False
        ))

        print(f"[{self.mode}|npy] 加载 {len(self.file_list)} 个样本 from {root_dir}")

    def _init_wav_mode(self):
        """初始化 wav 加载模式（VoiceBank-DEMAND 格式）"""
        base_dir = self.data_dir

        # 根据模式确定子目录名称
        noisy_subdir_map = {
            'train': 'noisy_trainset_28spk_wav',
            'val':   'noisy_testset_wav',
            'test':  'noisy_testset_wav',
        }
        clean_subdir_map = {
            'train': 'clean_trainset_28spk_wav',
            'val':   'clean_testset_wav',
            'test':  'clean_testset_wav',
        }

        noisy_subdir = noisy_subdir_map.get(self.mode)
        clean_subdir = clean_subdir_map.get(self.mode)

        if noisy_subdir is None or clean_subdir is None:
            raise ValueError(f"不支持的模式: {self.mode}，请选择 train/val/test")

        noisy_dir = os.path.join(base_dir, noisy_subdir)
        clean_dir = os.path.join(base_dir, clean_subdir)

        if not os.path.isdir(noisy_dir):
            raise FileNotFoundError(f"含噪语音目录不存在: {noisy_dir}")

        # 获取所有 wav 文件
        noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))

        if len(noisy_files) == 0:
            raise FileNotFoundError(f"在 {noisy_dir} 中未找到 .wav 文件")

        # 匹配 clean/noisy 文件对
        paired_samples = []
        missing_clean = []

        for noisy_path in noisy_files:
            filename = os.path.basename(noisy_path)
            clean_path = os.path.join(clean_dir, filename)

            if os.path.exists(clean_path):
                paired_samples.append((clean_path, noisy_path))
            else:
                missing_clean.append(filename)

        if missing_clean:
            warnings.warn(
                f"[{len(missing_clean)}] 个含噪文件缺少对应的 clean 文件 (示例: {missing_clean[:3]})"
            )

        if len(paired_samples) == 0:
            raise FileNotFoundError("没有匹配的 clean/noisy 文件对")

        # 采样
        actual_num = min(self.sample_num, len(paired_samples))
        indices = np.random.choice(len(paired_samples), actual_num, replace=False)
        self.file_pairs = [paired_samples[i] for i in indices]

        print(
            f"[{self.mode}|wav] 加载 {len(self.file_pairs)} 个样本 "
            f"(共 {len(paired_samples)} 对) from {base_dir}"
        )

    def __len__(self):
        if self.load_mode == 'wav':
            return len(self.file_pairs)
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.load_mode == 'wav':
            return _get_item_wav(self, idx)
        return _get_item_npy(self, idx)


def _get_item_npy(dataset, idx):
    """npy 模式的 __getitem__ 实现"""
    clean_mag_path = dataset.file_list[idx]
    base = clean_mag_path.replace('_clean_mag.npy', '')

    clean_mag = np.load(base + '_clean_mag.npy').astype(np.float32)
    clean_phase = np.load(base + '_clean_phase.npy').astype(np.float32)
    noisy_mag = np.load(base + '_noisy_mag.npy').astype(np.float32)
    noisy_phase = np.load(base + '_noisy_phase.npy').astype(np.float32)

    min_len = min(clean_mag.shape[1], noisy_mag.shape[1])
    clean_mag = clean_mag[:, :min_len]
    clean_phase = clean_phase[:, :min_len]
    noisy_mag = noisy_mag[:, :min_len]
    noisy_phase = noisy_phase[:, :min_len]

    return _build_output(clean_mag, clean_phase, noisy_mag, noisy_phase, base)


def _get_item_wav(dataset, idx):
    """wav 模式的 __getitem__ 实现"""
    clean_path, noisy_path = dataset.file_pairs[idx]

    try:
        clean_mag, clean_phase, noisy_mag, noisy_phase = load_wav_features(
            clean_path, noisy_path
        )
    except (FileNotFoundError, ValueError) as e:
        warnings.warn(f"跳过样本 [{idx}]: {e}")
        # 返回一个 dummy 样本避免训练中断（或可考虑重新采样）
        F, T = 257, 100
        clean_mag = np.zeros((F, T), dtype=np.float32)
        clean_phase = np.zeros((F, T), dtype=np.float32)
        noisy_mag = np.zeros((F, T), dtype=np.float32)
        noisy_phase = np.zeros((F, T), dtype=np.float32)

    min_len = min(clean_mag.shape[1], noisy_mag.shape[1])
    clean_mag = clean_mag[:, :min_len]
    clean_phase = clean_phase[:, :min_len]
    noisy_mag = noisy_mag[:, :min_len]
    noisy_phase = noisy_phase[:, :min_len]

    base = os.path.splitext(os.path.basename(clean_path))[0]
    return _build_output(clean_mag, clean_phase, noisy_mag, noisy_phase, base)


def _build_output(clean_mag, clean_phase, noisy_mag, noisy_phase, base):
    """统一构建输出字典"""
    noisy_mag_tensor = torch.from_numpy(noisy_mag).unsqueeze(0)
    clean_mag_tensor = torch.from_numpy(clean_mag).unsqueeze(0)
    noisy_phase_tensor = torch.from_numpy(noisy_phase)

    return {
        'noisy_mag': noisy_mag_tensor,
        'clean_mag': clean_mag_tensor,
        'noisy_phase': noisy_phase_tensor,
        'file_id': os.path.basename(base),
    }


def collate_fn(batch):
    """
    自定义 collate 函数
    由于不同语音的帧数 T 可能不同，需要 pad 到相同长度
    """
    max_T = max(item['noisy_mag'].shape[2] for item in batch)

    noisy_mags = []
    clean_mags = []
    noisy_phases = []
    file_ids = []

    for item in batch:
        T = item['noisy_mag'].shape[2]
        pad_T = max_T - T

        if pad_T > 0:
            noisy_mag = torch.nn.functional.pad(item['noisy_mag'], (0, pad_T))
            clean_mag = torch.nn.functional.pad(item['clean_mag'], (0, pad_T))
            noisy_phase = torch.nn.functional.pad(item['noisy_phase'], (0, pad_T))
        else:
            noisy_mag = item['noisy_mag']
            clean_mag = item['clean_mag']
            noisy_phase = item['noisy_phase']

        noisy_mags.append(noisy_mag)
        clean_mags.append(clean_mag)
        noisy_phases.append(noisy_phase)
        file_ids.append(item['file_id'])

    return {
        'noisy_mag': torch.stack(noisy_mags),      # (B, 1, F, T)
        'clean_mag': torch.stack(clean_mags),      # (B, 1, F, T)
        'noisy_phase': torch.stack(noisy_phases),  # (B, F, T)
        'file_id': file_ids,
    }
