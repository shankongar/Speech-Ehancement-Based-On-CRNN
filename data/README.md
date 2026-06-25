# 数据目录说明

## VoiceBank-DEMAND 数据集

本项目使用 **VoiceBank-DEMAND** 数据集进行语音增强实验。

### 数据集来源
- **语音库**：VoiceBank（英国爱丁堡大学发布）
- **噪声库**：DEMAND（Database of Multi-channel Acoustic Noise）
- **获取方式**：百度 AI Studio 公开下载
  - 链接：https://aistudio.baidu.com/datasetdetail/62188

### 数据集规格
- 采样率：16 kHz
- 训练集：28 名说话人，约 11,572 条样本
- 测试集：2 名说话人（p232、p257），824 条样本
- 混合 SNR：0, 5, 10, 15 dB

### 目录结构

```
data/
├── VoiceBank-DEMAND/       # 原始数据集（wav 文件）
├── processed/              # 预处理后特征（npy 文件）
│   ├── train/
│   ├── val/
│   └── test/
```

### 数据获取说明
1. 从上述链接下载数据集
2. 解压到 `data/VoiceBank-DEMAND/` 目录
3. 运行预处理：`python main.py preprocess`