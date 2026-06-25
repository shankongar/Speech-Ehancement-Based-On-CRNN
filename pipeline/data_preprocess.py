"""
数据预处理模块 - 支持 VoiceBank-DEMAND 数据集
功能：
1. 加载 VoiceBank-DEMAND 数据集（含噪语音 + 干净语音）
2. 训练集/验证集划分与采样
3. 测试集处理（clean_testset_wav + noisy_testset_wav）
4. 提取 STFT 特征（幅度谱和相位谱）并保存为 .npy 文件
5. 保留噪声生成函数用于数据增强
"""

import os
import glob
import argparse
import numpy as np
try:
    import soundfile as sf
except ImportError:
    sf = None  # 本模块不直接使用，仅为兼容性
from tqdm import tqdm
from core.utils import load_audio, stft, wav_to_mag_phase

# ===================== 配置参数 =====================
VOICEBANK_DIR = 'data/VoiceBank-DEMAND'  # VoiceBank-DEMAND 数据集目录
CLEAN_TRAIN_DIR = 'data/BZNSYP/Wave'     # 备用干净语音源（当 clean_trainset 不存在时使用）
OUTPUT_DIR = 'data/processed'            # 处理后数据保存目录
SR = 16000                               # 采样率
N_FFT = 512                              # FFT 窗口大小
HOP_LENGTH = 128                         # 帧移
WIN_LENGTH = 512                         # 窗长

# 训练/验证集划分比例
TRAIN_RATIO = 0.8

# 默认采样数量
SAMPLE_NUM = 500

# VoiceBank-DEMAND 子目录名称
NOISY_TRAIN_SUBDIR = 'noisy_trainset_28spk_wav'
CLEAN_TRAIN_SUBDIR = 'clean_trainset_28spk_wav'
NOISY_TEST_SUBDIR = 'noisy_testset_wav'
CLEAN_TEST_SUBDIR = 'clean_testset_wav'

# 噪声类型及对应的 SNR 范围（dB）- 用于数据增强
NOISE_TYPES = ['white', 'pink', 'babble']
SNR_RANGE = (-5, 15)


# ===================== 噪声生成函数（保留用于数据增强）=====================

def generate_white_noise(length, sr=16000):
    """生成白噪声"""
    return np.random.randn(length).astype(np.float32)


def generate_pink_noise(length, sr=16000):
    """生成粉噪声（1/f 噪声）"""
    white = np.random.randn(length).astype(np.float32)
    b = np.array([1.0])
    a = np.array([1.0, -0.99])
    from scipy.signal import lfilter
    pink = lfilter(b, a, white)
    pink = pink / (np.max(np.abs(pink)) + 1e-8)
    return pink.astype(np.float32)


def generate_babble_noise(length, sr=16000, n_speakers=5):
    """
    生成 babble 噪声（多人说话叠加）
    这里简化为随机叠加多个语音片段
    """
    wav_files = glob.glob(os.path.join(CLEAN_TRAIN_DIR, '*.wav'))
    if len(wav_files) < n_speakers:
        n_speakers = len(wav_files)

    babble = np.zeros(length, dtype=np.float32)
    for i in range(n_speakers):
        wav_path = np.random.choice(wav_files)
        wav = load_audio(wav_path, sr=sr, mono=True)
        if len(wav) > length:
            start = np.random.randint(0, len(wav) - length)
            segment = wav[start:start+length]
        else:
            repeats = int(np.ceil(length / len(wav)))
            segment = np.tile(wav, repeats)[:length]
        scale = np.random.uniform(0.3, 1.0)
        babble += segment * scale

    babble = babble / (np.max(np.abs(babble)) + 1e-8)
    return babble.astype(np.float32)


def add_noise(clean_wav, noise_type='white', snr_db=0, sr=16000):
    """
    给干净语音添加噪声
    :param clean_wav:  干净语音 numpy 数组
    :param noise_type: 噪声类型 ('white', 'pink', 'babble')
    :param snr_db:     信噪比 (dB)
    :return:           含噪语音 numpy 数组
    """
    length = len(clean_wav)

    if noise_type == 'white':
        noise = generate_white_noise(length, sr)
    elif noise_type == 'pink':
        noise = generate_pink_noise(length, sr)
    elif noise_type == 'babble':
        noise = generate_babble_noise(length, sr)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    signal_power = np.sum(clean_wav ** 2)
    noise_power = np.sum(noise ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    noise_scale = np.sqrt(target_noise_power / (noise_power + 1e-8))
    noise = noise * noise_scale

    noisy_wav = clean_wav + noise
    return noisy_wav.astype(np.float32)


# ===================== 辅助函数 =====================

def extract_and_save_features(clean_wav, noisy_wav, file_id, output_dir,
                               n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
    """
    提取 STFT 特征并保存
    :param clean_wav:   干净语音波形
    :param noisy_wav:   含噪语音波形
    :param file_id:     文件标识符
    :param output_dir:  输出目录
    """
    clean_mag, clean_phase = wav_to_mag_phase(clean_wav, n_fft, hop_length, win_length)
    noisy_mag, noisy_phase = wav_to_mag_phase(noisy_wav, n_fft, hop_length, win_length)

    np.save(os.path.join(output_dir, f'{file_id}_clean_mag.npy'), clean_mag)
    np.save(os.path.join(output_dir, f'{file_id}_clean_phase.npy'), clean_phase)
    np.save(os.path.join(output_dir, f'{file_id}_noisy_mag.npy'), noisy_mag)
    np.save(os.path.join(output_dir, f'{file_id}_noisy_phase.npy'), noisy_phase)

    return clean_mag.shape


def load_voicebank_train_files():
    """
    加载 VoiceBank-DEMAND 训练集文件
    :return: (noisy_files, clean_files) 文件路径列表
    """
    noisy_dir = os.path.join(VOICEBANK_DIR, NOISY_TRAIN_SUBDIR)
    clean_dir = os.path.join(VOICEBANK_DIR, CLEAN_TRAIN_SUBDIR)

    # 加载含噪训练文件
    noisy_files = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
    print(f"找到 {len(noisy_files)} 个含噪训练语音文件")

    # 检查是否存在干净训练集
    if os.path.exists(clean_dir):
        clean_files = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))
        print(f"找到 {len(clean_files)} 个干净训练语音文件")
    else:
        print(f"警告: 未找到干净训练集目录 {clean_dir}")
        print(f"将使用备用干净语音源: {CLEAN_TRAIN_DIR}")
        clean_files = sorted(glob.glob(os.path.join(CLEAN_TRAIN_DIR, '*.wav')))
        print(f"从备用源找到 {len(clean_files)} 个干净语音文件")

    return noisy_files, clean_files


def match_clean_noisy_pairs(noisy_files, clean_files):
    """
    匹配含噪和干净的文件对
    如果有对应的 clean_trainset，则根据文件名匹配
    否则随机配对或按顺序配对
    """
    pairs = []
    clean_dir = os.path.join(VOICEBANK_DIR, CLEAN_TRAIN_SUBDIR)

    if os.path.exists(clean_dir):
        # 根据文件名匹配
        clean_dict = {}
        for cf in clean_files:
            basename = os.path.basename(cf)
            clean_dict[basename] = cf

        for nf in noisy_files:
            basename = os.path.basename(nf)
            if basename in clean_dict:
                pairs.append((nf, clean_dict[basename]))
            else:
                print(f"警告: 未找到匹配的干净文件: {basename}")
    else:
        # 使用备用源：随机或顺序配对
        min_len = min(len(noisy_files), len(clean_files))
        for i in range(min_len):
            pairs.append((noisy_files[i], clean_files[i]))

        if len(noisy_files) > min_len:
            print(f"警告: 含噪文件 ({len(noisy_files)}) 多于干净文件 ({len(clean_files)})，"
                  f"只处理前 {min_len} 对")

    return pairs


# ===================== 训练集预处理 =====================

def preprocess_dataset(sample_num=SAMPLE_NUM, train_ratio=TRAIN_RATIO):
    """
    预处理 VoiceBank-DEMAND 训练数据集：
    1. 加载含噪和干净语音
    2. 划分训练/验证集
    3. 可选采样
    4. 提取 STFT 特征并保存
    """
    print("=" * 60)
    print("VoiceBank-DEMAND 训练集预处理")
    print("=" * 60)

    # 创建输出目录
    train_dir = os.path.join(OUTPUT_DIR, 'train')
    val_dir = os.path.join(OUTPUT_DIR, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 加载文件
    noisy_files, clean_files = load_voicebank_train_files()

    if len(noisy_files) == 0:
        raise FileNotFoundError("未找到任何含噪训练文件！")

    # 匹配文件对
    pairs = match_clean_noisy_pairs(noisy_files, clean_files)
    total_pairs = len(pairs)
    print(f"\n成功匹配 {total_pairs} 个文件对")

    # 随机打乱并划分
    np.random.seed(42)
    indices = np.random.permutation(total_pairs)
    train_size = int(total_pairs * train_ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    print(f"\n数据划分:")
    print(f"  总样本数: {total_pairs}")
    print(f"  训练集: {len(train_indices)} ({train_ratio*100:.0f}%)")
    print(f"  验证集: {len(val_indices)} ({(1-train_ratio)*100:.0f}%)")

    # 应用采样
    actual_sample_num = min(sample_num, total_pairs) if sample_num > 0 else total_pairs
    if sample_num > 0 and sample_num < total_pairs:
        print(f"\n采样设置: 从全部数据中选择 {actual_sample_num} 个样本")

        # 分别对训练集和验证集按比例采样
        train_sample_size = int(actual_sample_num * train_ratio)
        val_sample_size = actual_sample_num - train_sample_size

        if train_sample_size > 0:
            train_indices = np.random.choice(train_indices, size=train_sample_size, replace=False)
        if val_sample_size > 0 and len(val_indices) > 0:
            val_indices = np.random.choice(val_indices, size=min(val_sample_size, len(val_indices)), replace=False)

        print(f"  实际训练集: {len(train_indices)}")
        print(f"  实际验证集: {len(val_indices)}")

    # 处理每个 split
    all_meta = {'train': [], 'val': []}

    for split, split_indices in [('train', train_indices), ('val', val_indices)]:
        if len(split_indices) == 0:
            print(f"\n跳过 {split} 集（无样本）")
            continue

        print(f"\n处理 {split} 集 ({len(split_indices)} 个样本)...")
        split_dir = train_dir if split == 'train' else val_dir

        split_meta = []

        for idx in tqdm(split_indices, desc=split):
            idx = int(idx)
            noisy_path, clean_path = pairs[idx]

            file_id = os.path.splitext(os.path.basename(noisy_path))[0]

            try:
                # 加载音频
                noisy_wav = load_audio(noisy_path, sr=SR, mono=True)
                clean_wav = load_audio(clean_path, sr=SR, mono=True)

                # 确保长度一致（截取较短者）
                min_len = min(len(noisy_wav), len(clean_wav))
                noisy_wav = noisy_wav[:min_len]
                clean_wav = clean_wav[:min_len]

                # 提取特征
                feature_shape = extract_and_save_features(
                    clean_wav, noisy_wav, file_id, split_dir
                )

                # 保存元信息
                meta = {
                    'file_id': file_id,
                    'noisy_path': noisy_path,
                    'clean_path': clean_path,
                    'length': min_len,
                    'feature_shape': feature_shape,
                    'sr': SR,
                }
                np.save(os.path.join(split_dir, f'{file_id}_meta.npy'), meta)
                split_meta.append(meta)

            except Exception as e:
                print(f"\n错误: 处理文件 {file_id} 时出错: {e}")
                continue

        all_meta[split] = split_meta
        print(f"{split} 集处理完成: {len(split_meta)} 个样本")

    # 保存汇总元信息
    summary_file = os.path.join(OUTPUT_DIR, 'train_meta_summary.npy')
    np.save(summary_file, all_meta)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("预处理完成！统计信息:")
    print("=" * 60)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练集样本数: {len(all_meta['train'])}")
    print(f"验证集样本数: {len(all_meta['val'])}")
    print(f"总样本数: {len(all_meta['train']) + len(all_meta['val'])}")
    print(f"采样率: {SR} Hz")
    print(f"STFT 参数: N_FFT={N_FFT}, HOP={HOP_LENGTH}, WIN={WIN_LENGTH}")


# ===================== 测试集预处理 =====================

def preprocess_testset():
    """
    预处理 VoiceBank-DEMAND 测试数据集：
    1. 加载 clean_testset_wav 和 noisy_testset_wav
    2. 根据文件名匹配文件对
    3. 提取 STFT 特征并保存到 data/processed/test/
    """
    print("=" * 60)
    print("VoiceBank-DEMAND 测试集预处理")
    print("=" * 60)

    # 创建测试集输出目录
    test_dir = os.path.join(OUTPUT_DIR, 'test')
    os.makedirs(test_dir, exist_ok=True)

    # 加载测试集文件
    noisy_test_dir = os.path.join(VOICEBANK_DIR, NOISY_TEST_SUBDIR)
    clean_test_dir = os.path.join(VOICEBANK_DIR, CLEAN_TEST_SUBDIR)

    if not os.path.exists(noisy_test_dir):
        raise FileNotFoundError(f"未找到测试集含噪目录: {noisy_test_dir}")
    if not os.path.exists(clean_test_dir):
        raise FileNotFoundError(f"未找到测试集干净目录: {clean_test_dir}")

    noisy_files = sorted(glob.glob(os.path.join(noisy_test_dir, '*.wav')))
    clean_files = sorted(glob.glob(os.path.join(clean_test_dir, '*.wav')))

    print(f"\n找到 {len(noisy_files)} 个含噪测试文件")
    print(f"找到 {len(clean_files)} 个干净测试文件")

    # 根据文件名建立字典进行匹配
    clean_dict = {}
    for cf in clean_files:
        basename = os.path.basename(cf)
        clean_dict[basename] = cf

    # 匹配文件对
    matched_pairs = []
    unmatched = []

    for nf in noisy_files:
        basename = os.path.basename(nf)
        if basename in clean_dict:
            matched_pairs.append((nf, clean_dict[basename]))
        else:
            unmatched.append(basename)

    print(f"\n成功匹配 {len(matched_pairs)} 个文件对")
    if unmatched:
        print(f"警告: {len(unmatched)} 个含噪文件未找到匹配的干净文件")

    # 处理每个文件对
    test_meta = []
    print(f"\n处理测试集...")

    for noisy_path, clean_path in tqdm(matched_pairs, desc='test'):
        file_id = os.path.splitext(os.path.basename(noisy_path))[0]

        try:
            # 加载音频
            noisy_wav = load_audio(noisy_path, sr=SR, mono=True)
            clean_wav = load_audio(clean_path, sr=SR, mono=True)

            # 确保长度一致
            min_len = min(len(noisy_wav), len(clean_wav))
            noisy_wav = noisy_wav[:min_len]
            clean_wav = clean_wav[:min_len]

            # 提取特征
            feature_shape = extract_and_save_features(
                clean_wav, noisy_wav, file_id, test_dir
            )

            # 解析 VoiceBank-DEMAND 文件名中的信息
            # 格式示例: p232_001.wav (说话人ID_句子编号.wav)
            meta = {
                'file_id': file_id,
                'noisy_path': noisy_path,
                'clean_path': clean_path,
                'length': min_len,
                'feature_shape': feature_shape,
                'sr': SR,
            }

            # 尝试解析额外的元信息（如果文件名包含噪声类型等信息）
            # VoiceBank-DEMAND 的噪声类型通常在目录结构中体现
            np.save(os.path.join(test_dir, f'{file_id}_meta.npy'), meta)
            test_meta.append(meta)

        except Exception as e:
            print(f"\n错误: 处理测试文件 {file_id} 时出错: {e}")
            continue

    # 保存测试集汇总元信息
    summary_file = os.path.join(OUTPUT_DIR, 'test_meta_summary.npy')
    np.save(summary_file, test_meta)

    # 打印统计信息
    print("\n" + "=" * 60)
    print("测试集预处理完成！统计信息:")
    print("=" * 60)
    print(f"输出目录: {test_dir}")
    print(f"测试集样本数: {len(test_meta)}")
    print(f"匹配成功率: {len(matched_pairs)}/{len(noisy_files)} "
          f"({100*len(matched_pairs)/max(len(noisy_files),1):.1f}%)")
    print(f"采样率: {SR} Hz")
    print(f"STFT 参数: N_FFT={N_FFT}, HOP={HOP_LENGTH}, WIN={WIN_LENGTH}")


# ===================== 主程序入口 =====================

def main(args=None):
    """
    VoiceBank-DEMAND 数据预处理主函数
    :param args: 可选的参数对象（argparse.Namespace），如果为None则从命令行解析
    """
    if args is None:
        raise ValueError("args must be provided")

    print("\n" + "=" * 60)
    print("VoiceBank-DEMAND 数据预处理工具")
    print("=" * 60)
    print(f"\n配置参数:")
    print(f"  数据集目录: {VOICEBANK_DIR}")
    print(f"  备用干净语音: {CLEAN_TRAIN_DIR}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  采样数量: {args.sample_num}")
    print(f"  数据集类型: {args.dataset_type}")
    print(f"  训练比例: {args.train_ratio}")

    try:
        if args.dataset_type in ['train', 'all']:
            preprocess_dataset(
                sample_num=args.sample_num,
                train_ratio=args.train_ratio
            )

        if args.dataset_type in ['test', 'all']:
            preprocess_testset()

        print("\n✓ 所有预处理任务完成！")

    except Exception as e:
        print(f"\n✗ 预处理过程中出错: {e}")
        raise


if __name__ == '__main__':
    from config import get_config
    import argparse
    cfg = get_config()
    args = argparse.Namespace(
        dataset_type=cfg.preprocess_dataset_type,
        sample_num=cfg.sample_num,
        train_ratio=cfg.train_ratio,
        output_dir=cfg.processed_data_dir,
    )
    main(args)
