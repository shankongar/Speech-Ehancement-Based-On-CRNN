"""
快速测试脚本 - 验证代码核心功能
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

print("=" * 60)
print("测试 1: 导入依赖库")
print("=" * 60)

try:
    import torch
    import numpy as np
    import librosa
    import soundfile as sf
    print("✓ PyTorch 版本:", torch.__version__)
    print("✓ NumPy 版本:", np.__version__)
    print("✓ Librosa 版本:", librosa.__version__)
    print("✓ SoundFile 版本:", sf.__version__)
    print("\n所有依赖库导入成功！\n")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

print("=" * 60)
print("测试 2: 测试工具函数")
print("=" * 60)

try:
    from core.utils import stft, istft, wav_to_mag_phase, snr, si_snr
    
    # 生成测试信号
    test_signal = np.random.randn(16000).astype(np.float32)
    
    # 测试 STFT
    S = stft(test_signal, n_fft=512, hop_length=128)
    print(f"✓ STFT 输出形状: {S.shape}")
    
    # 测试 iSTFT
    reconstructed = istft(S, hop_length=128, length=len(test_signal))
    print(f"✓ iSTFT 输出形状: {reconstructed.shape}")
    
    # 测试评估指标
    test_snr = snr(test_signal, test_signal + np.random.randn(16000) * 0.1)
    print(f"✓ SNR 计算成功: {test_snr:.2f} dB")
    
    test_si_snr = si_snr(test_signal, test_signal + np.random.randn(16000) * 0.1)
    print(f"✓ SI-SNR 计算成功: {test_si_snr:.2f} dB")
    
    print("\n工具函数测试通过！\n")
except Exception as e:
    print(f"✗ 工具函数测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("测试 3: 测试模型定义")
print("=" * 60)

try:
    from core.model import CRNNSpeechEnhancement
    
    # 创建模型
    model = CRNNSpeechEnhancement(input_dim=257, hidden_dim=256, num_layers=2)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✓ 模型创建成功")
    print(f"✓ 模型参数量: {param_count:,}")
    
    # 测试前向传播
    batch_size = 2
    F = 257  # n_fft//2 + 1
    T = 100  # 帧数
    
    x = torch.randn(batch_size, 1, F, T)
    print(f"✓ 输入张量形状: {x.shape}")
    
    mask = model(x)
    print(f"✓ 输出掩码形状: {mask.shape}")
    print(f"✓ 掩码值范围: [{mask.min().item():.3f}, {mask.max().item():.3f}]")
    
    # 测试增强功能
    noisy_mag = torch.randn(batch_size, 1, F, T)
    noisy_phase = torch.randn(batch_size, F, T)
    clean_mag = model.enhance(noisy_mag, noisy_phase)
    print(f"✓ 增强输出形状: {clean_mag.shape}")
    
    print("\n模型测试通过！\n")
except Exception as e:
    print(f"✗ 模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("测试 4: 测试数据集类")
print("=" * 60)

try:
    from core.dataset import SpeechEnhancementDataset, collate_fn
    
    # 注意：这里只是测试类定义，不实际加载数据
    print("✓ SpeechEnhancementDataset 类导入成功")
    print("✓ collate_fn 函数导入成功")
    
    print("\n数据集类测试通过！\n")
except Exception as e:
    print(f"✗ 数据集类测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("=" * 60)
print("所有测试通过！")
print("=" * 60)
print("\n代码核心功能验证成功，可以正常运行。")
print("\n下一步操作：")
print("1. 运行数据预处理: python data_preprocess.py")
print("2. 训练模型: python train.py")
print("3. 推理评估: python inference.py --save_audio")