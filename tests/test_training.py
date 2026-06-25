#!/usr/bin/env python
"""
快速训练测试脚本
用于验证训练流程是否能正常启动和运行
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_training():
    """测试训练流程"""
    print("=" * 60)
    print("CRNN 语音增强系统 - 训练流程快速测试")
    print("=" * 60)

    # 1. 测试导入
    print("\n[1/5] 测试模块导入...")
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        
        from core.model import CRNNSpeechEnhancement
        print("  ✓ 模型模块")
        
        from core.dataset import SpeechEnhancementDataset, collate_fn
        print("  ✓ 数据集模块")
        
        from pipeline.train import train, MaskLoss
        print("  ✓ 训练模块")
    except Exception as e:
        print(f"  ✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 2. 测试数据集加载
    print("\n[2/5] 测试数据集加载 (sample_num=10)...")
    try:
        dataset = SpeechEnhancementDataset(
            data_dir='data/VoiceBank-DEMAND',
            mode='train',
            sample_num=10,
            load_mode='wav'
        )
        print(f"  ✓ 数据集创建成功: {len(dataset)} 个样本")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  ✓ 样本格式正确:")
            for k, v in sample.items():
                if hasattr(v, 'shape'):
                    print(f"      {k}: {v.shape}")
                else:
                    print(f"      {k}: {type(v)}")
    except Exception as e:
        print(f"  ✗ 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 测试 DataLoader
    print("\n[3/5] 测试 DataLoader (num_workers=0)...")
    try:
        from torch.utils.data import DataLoader
        
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,  # Windows 兼容性
            collate_fn=collate_fn,
        )
        
        batch = next(iter(loader))
        print(f"  ✓ Batch 加载成功:")
        for k, v in batch.items():
            if hasattr(v, 'shape'):
                print(f"      {k}: {v.shape}")
    except Exception as e:
        print(f"  ✗ DataLoader 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. 测试模型前向传播
    print("\n[4/5] 测试模型前向传播...")
    try:
        device = torch.device('cpu')
        model = CRNNSpeechEnhancement(
            input_dim=257,
            hidden_dim=256,
            num_layers=2,
        ).to(device)
        
        noisy_mag = batch['noisy_mag'].to(device)
        pred_mask = model(noisy_mag)
        
        print(f"  ✓ 模型前向传播成功:")
        print(f"      输入: {noisy_mag.shape}")
        print(f"      输出: {pred_mask.shape}")
        print(f"      参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        print(f"  ✗ 模型前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 测试损失计算
    print("\n[5/5] 测试损失计算...")
    try:
        criterion = MaskLoss()
        loss = criterion(pred_mask, 
                        batch['noisy_mag'].to(device),
                        batch['clean_mag'].to(device))
        print(f"  ✓ 损失计算成功: {loss.item():.4f}")
    except Exception as e:
        print(f"  ✗ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！训练流程可以正常运行。")
    print("=" * 60)
    print("\n建议运行命令:")
    print("  python main.py demo          # 快速演示（50样本，2轮）")
    print("  python main.py train         # 完整训练")
    print("=" * 60)

    return True

if __name__ == '__main__':
    success = test_training()
    sys.exit(0 if success else 1)