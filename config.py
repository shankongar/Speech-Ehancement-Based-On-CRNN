"""
CRNN 语音增强系统 - 集中配置文件

使用方法：
  1. 直接修改下方配置值即可调整所有参数
  2. 各模块通过 from config import get_config 获取配置
"""

import os


class Config:
    """全局配置类 - 修改此处即可调整全部参数"""

    # ==================== 数据配置 ====================

    # 数据集根目录（VoiceBank-DEMAND）
    data_dir: str = "data/VoiceBank-DEMAND"

    # 数据加载模式：'wav' = 从 wav 文件在线提取 STFT | 'npy' = 加载预处理好的 .npy
    load_mode: str = "wav"  # 'wav' | 'npy'

    # 训练采样数（0 = 使用全部数据）
    sample_num: int = 3000

    # 训练/验证集划分比例
    train_ratio: float = 0.8

    # 预处理数据输出根目录
    processed_data_dir: str = "data/processed"

    # ==================== 模型结构 ====================

    # 输入特征维度 (= n_fft // 2 + 1)
    input_dim: int = 257

    # BLSTM 隐藏层维度
    hidden_dim: int = 256

    # BLSTM 层数
    num_layers: int = 2

    # 模型名称（用于日志和报告）
    model_name: str = "CRNN"

    # ==================== 训练超参数 ====================

    # 训练轮数
    epochs: int = 48

    # 批次大小
    batch_size: int = 16

    # 初始学习率
    lr: float = 1e-3

    # 权重衰减 (L2 正则化)
    weight_decay: float = 1e-5

    # 梯度裁剪最大范数
    grad_clip_max_norm: float = 5.0

    # 学习率调度 patience（ReduceLROnPlateau）
    scheduler_patience: int = 5

    # 学习率衰减因子
    scheduler_factor: float = 0.5

    # 随机种子（确保可复现）
    seed: int = 42

    # 每 N 个 epoch 保存一次检查点（除 best_model 外）
    save_every_n_epochs: int = 1

    # ==================== STFT 参数 ====================

    # FFT 窗口大小
    n_fft: int = 512

    # 帧移（采样点）
    hop_length: int = 128

    # 窗长（采样点）
    win_length: int = 512

    # 采样率 (Hz)
    sr: int = 16000

    # ==================== 路径配置 ====================

    # 模型保存目录
    output_dir: str = "checkpoints"

    # 最优模型文件名
    best_model_name: str = "best_model.pth"

    # 断点续训 checkpoint 文件名
    checkpoint_name: str = "latest_checkpoint.pth"

    # 推理时使用的模型路径（默认用训练输出的最优模型）
    model_path: str = ""  # 空字符串表示自动取 output_dir/best_model_name

    # 增强音频输出目录
    enhance_output_dir: str = "results/enhanced"

    # 可视化图表输出目录
    figures_dir: str = "results/figures"

    # 日志输出目录
    log_dir: str = "results/logs"

    # ==================== 功能开关 ====================

    # 强制使用 CPU（禁用 GPU）
    cpu: bool = False

    # 推理时是否保存增强后的音频文件
    save_audio: bool = False

    # 推理评估时的测试采样数（0 = 全部）
    eval_sample_num: int = 0

    # ==================== 断点续训 ====================

    # 是否自动检测并恢复训练（True=存在 checkpoint 时自动续训）
    resume_auto: bool = True

    # 手动指定 checkpoint 路径（优先于 resume_auto）
    resume_path: str = ""

    # ==================== 可视化设置 ====================

    # 图表 DPI
    figure_dpi: int = 300

    # 图表输出格式: 'png', 'pdf', or 'both'
    figure_format: str = "both"

    # ==================== 预处理专用 ====================

    # 预处理数据集类型: 'all' | 'train' | 'test'
    preprocess_dataset_type: str = "all"

    # ==================== 衍生属性（自动计算，不要手动修改）====================

    @property
    def best_model_path(self) -> str:
        """最优模型完整路径"""
        return os.path.join(self.output_dir, self.best_model_name)

    @property
    def checkpoint_path(self) -> str:
        """断点续训 checkpoint 完整路径"""
        return os.path.join(self.output_dir, self.checkpoint_name)

    @property
    def train_processed_dir(self) -> str:
        """训练集预处理数据目录"""
        return os.path.join(self.processed_data_dir, "train")

    @property
    def val_processed_dir(self) -> str:
        """验证集预处理数据目录"""
        return os.path.join(self.processed_data_dir, "val")

    @property
    def test_processed_dir(self) -> str:
        """测试集预处理数据目录"""
        return os.path.join(self.processed_data_dir, "test")

    @property
    def frequency_dim(self) -> int:
        """频率维度（= n_fft // 2 + 1）"""
        return self.n_fft // 2 + 1


# 全局单例
_config_instance = None


def get_config() -> Config:
    """获取全局配置实例（单例模式）"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance


def reset_config():
    """重置配置（用于测试）"""
    global _config_instance
    _config_instance = None
