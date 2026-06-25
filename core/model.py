"""
CRNN 语音增强模型
结构：卷积 + 循环神经网络 + 卷积
输入：含噪语音的幅度谱 (1, F, T)
输出：理想比率掩码 (IRM) 或干净语音幅度谱
"""

import torch
import torch.nn as nn


class CRNNSpeechEnhancement(nn.Module):
    """
    CRNN 语音增强模型
    
    架构：
    1. 编码器：2 层卷积下采样
    2. 瓶颈层：BLSTM 捕获时序依赖
    3. 解码器：2 层卷积上采样 + 跳跃连接
    
    输出：理想比率掩码 (IRM)，范围 [0, 1]
    增强：clean_mag = noisy_mag * IRM
    """

    def __init__(self, input_dim=257, hidden_dim=256, num_layers=2):
        """
        :param input_dim:  输入特征维度 (n_fft//2 + 1 = 257 for n_fft=512)
        :param hidden_dim: LSTM 隐藏层维度
        :param num_layers: LSTM 层数
        """
        super(CRNNSpeechEnhancement, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # ===================== 编码器 =====================
        # 第一层卷积：(1, F, T) -> (32, F1, T)，F1 = (F+2*1-3)//2+1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 第二层卷积：(32, F1, T) -> (64, F2, T)
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 动态计算卷积后的频率维度（避免整除误差）
        # Conv2d: out = (in + 2*padding - kernel) // stride + 1
        self.freq_conv1 = (input_dim + 2 * 1 - 3) // 2 + 1   # 257 -> 129
        self.freq_conv2 = (self.freq_conv1 + 2 * 1 - 3) // 2 + 1  # 129 -> 65
        lstm_input_dim = 64 * self.freq_conv2

        # ===================== 瓶颈层：BLSTM =====================
        self.blstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0,
        )

        # ===================== 解码器 =====================
        lstm_output_dim = hidden_dim * 2  # 双向

        # 线性投影：将 LSTM 输出投影到解码器所需的特征维度
        self.lstm_proj = nn.Linear(lstm_output_dim, 64 * self.freq_conv2)

        # 第一层反卷积：(64, F2, T) -> (32, F1, T)
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32,
                kernel_size=(3, 3), stride=(2, 1), padding=(1, 1),
                output_padding=(1, 0)
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 第二层反卷积：(32, F1, T) -> (1, >=F, T)
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(
                32, 1,
                kernel_size=(3, 3), stride=(2, 1), padding=(1, 1),
                output_padding=(1, 0)
            ),
            nn.Sigmoid(),  # 输出掩码，范围 [0, 1]
        )

    def forward(self, x):
        """
        前向传播
        :param x: 含噪语音幅度谱, shape = (B, 1, F, T)
        :return:  理想比率掩码 (IRM), shape = (B, 1, F, T)
        """
        B, C, F, T = x.shape

        # 编码器
        enc1 = self.conv1(x)      # (B, 32, freq_conv1, T)
        enc2 = self.conv2(enc1)   # (B, 64, freq_conv2, T)

        # 展平为 LSTM 输入: (B, T, lstm_input_dim)
        enc2_flat = enc2.permute(0, 3, 1, 2).contiguous()  # (B, T, 64, freq_conv2)
        enc2_flat = enc2_flat.view(B, T, -1)               # (B, T, lstm_input_dim)

        # BLSTM
        lstm_out, _ = self.blstm(enc2_flat)  # (B, T, lstm_output_dim)

        # 线性投影
        lstm_out = self.lstm_proj(lstm_out)  # (B, T, 64 * freq_conv2)

        # 重塑为卷积输入: (B, 64, freq_conv2, T)
        lstm_out = lstm_out.view(B, T, 64, self.freq_conv2)
        lstm_out = lstm_out.permute(0, 2, 3, 1).contiguous()  # (B, 64, freq_conv2, T)

        # 解码器
        dec1 = self.deconv1(lstm_out)  # (B, 32, freq_conv1, T)
        mask = self.deconv2(dec1)      # (B, 1, >=F, T)

        # 确保输出维度与输入一致
        if mask.shape[2] != F or mask.shape[3] != T:
            mask = mask[:, :, :F, :T]

        return mask

    def enhance(self, noisy_mag, noisy_phase):
        """
        语音增强推理
        :param noisy_mag:   含噪语音幅度谱, shape = (B, 1, F, T)
        :param noisy_phase: 含噪语音相位谱, shape = (B, F, T)
        :return:            增强后的干净语音幅度谱, shape = (B, F, T)
        """
        self.eval()
        with torch.no_grad():
            # 预测掩码
            mask = self.forward(noisy_mag)  # (B, 1, F, T)
            mask = mask.squeeze(1)          # (B, F, T)

            # 应用掩码
            clean_mag = noisy_mag.squeeze(1) * mask  # (B, F, T)

        return clean_mag


# ===================== 测试代码 =====================
if __name__ == '__main__':
    # 测试模型
    model = CRNNSpeechEnhancement(input_dim=257, hidden_dim=256, num_layers=2)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 模拟输入
    batch_size = 2
    F = 257  # n_fft//2 + 1
    T = 100  # 帧数

    x = torch.randn(batch_size, 1, F, T)
    print(f"输入 shape: {x.shape}")

    mask = model(x)
    print(f"输出掩码 shape: {mask.shape}")
    print(f"掩码范围: [{mask.min().item():.3f}, {mask.max().item():.3f}]")
