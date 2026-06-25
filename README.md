# CRNN 语音增强系统

基于**卷积循环神经网络（CRNN）**的语音增强系统，采用**理想比率掩码（IRM）估计**方法，从含噪语音中恢复干净语音。

## 运行环境

### 系统要求
- **Python**: 3.8+
- **GPU**: 推荐使用 NVIDIA GPU + CUDA（CPU 也可运行）

### 依赖安装

```bash
pip install -r requirements.txt
```

核心依赖：`torch>=1.9.0`, `librosa>=0.9.0`, `numpy>=1.21.0`, `scipy>=1.7.0`, `matplotlib>=3.5.0`

## 数据来源

使用 **VoiceBank-DEMAND** 数据集。

### 下载地址
[百度 AI Studio](https://aistudio.baidu.com/datasetdetail/62188)

### 数据结构
```
data/VoiceBank-DEMAND/
├── clean_trainset_28spk_wav/    # 训练集干净语音
├── noisy_trainset_28spk_wav/    # 训练集含噪语音
├── clean_testset_wav/           # 测试集干净语音
└── noisy_testset_wav/           # 测试集含噪语音
```

## 如何运行

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置参数

编辑 `config.py` 调整参数（默认配置可直接运行）：

```python
# 关键参数示例
epochs = 48              # 训练轮数
batch_size = 16          # 批次大小
lr = 1e-3                # 学习率
hidden_dim = 256         # 隐藏层维度
data_dir = "data/VoiceBank-DEMAND"  # 数据集路径
```

### 3. 运行命令

```bash
# 完整流程（推荐首次使用）
python main.py all

# 分步执行
python main.py preprocess    # 数据预处理
python main.py train         # 训练模型
python main.py inference     # 推理评估
```

### 4. 断点续训

在 `config.py` 中启用：

```python
resume_auto = True    # 自动检测 checkpoint 并续训
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 48 | 训练轮数 |
| `batch_size` | 16 | 批次大小 |
| `lr` | 1e-3 | 初始学习率 |
| `hidden_dim` | 256 | BLSTM 隐藏层维度 |
| `num_layers` | 2 | BLSTM 层数 |
| `n_fft` | 512 | FFT 窗口大小 |
| `hop_length` | 128 | 帧移（采样点） |
| `sr` | 16000 | 采样率 (Hz) |
| `sample_num` | 3000 | 训练采样数（0=全部） |
| `train_ratio` | 0.8 | 训练/验证集划分比例 |

完整参数配置详见 [`config.py`](config.py)。

## 预期输出示例

### 训练输出

```
Epoch 1/48
  Train Loss: 0.0842 | Val Loss: 0.0756
  LR: 0.001000 | Best Val Loss: 0.0756
  ✓ Checkpoint saved

Epoch 2/48
  Train Loss: 0.0623 | Val Loss: 0.0589
  LR: 0.001000 | Best Val Loss: 0.0589
  ✓ New best model saved
```

### 评估指标

```
Inference Results:
  SNR Improvement:    8.42 ± 2.31 dB
  SI-SNR Improvement: 9.15 ± 2.48 dB
  SegSNR Improvement: 7.89 ± 1.92 dB

Output Files:
  ├── checkpoints/best_model.pth           # 最优模型
  ├── results/enhanced/metrics.npy         # 评估指标
  └── results/figures/                      # 可视化图表
```

### 输出目录结构

```
checkpoints/
  ├── best_model.pth              # 最优模型权重
  └── latest_checkpoint.pth       # 最新检查点

results/
  ├── enhanced/                   # 增强后音频（需开启 save_audio）
  ├── figures/                    # 训练曲线、频谱对比等图表
  └── logs/                       # 实验日志
```

---

## 📄 许可证

本项目采用 **MIT License** 开源协议。详见 [LICENSE](LICENSE) 文件。

```
MIT License

Copyright (c) 2026 CRNN Speech Enhancement

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---
