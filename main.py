"""
CRNN 语音增强系统 - 统一入口

用法:
  python main.py preprocess              # 数据预处理
  python main.py train                   # 训练模型
  python main.py inference               # 推理评估
  python main.py all                     # 完整流程（预处理 + 训练 + 推理）
  python main.py demo                    # 快速演示（少量数据，快速验证）
  python main.py report                  # 生成学术报告素材

配置: 编辑 config.py 调整所有参数
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")

from config import get_config


# ===================== 从 config 构建 args =====================

def _build_args_from_config(overrides=None):
    """
    从 config.py 构建 argparse.Namespace 对象

    overrides: dict, 可选的参数覆盖（用于 demo 等需要临时修改配置的场景）
    """
    cfg = get_config()

    import argparse
    args = argparse.Namespace(
        # --- 数据 ---
        data_dir=cfg.data_dir,
        load_mode=cfg.load_mode,
        sample_num=cfg.sample_num,
        train_ratio=cfg.train_ratio,
        dataset_type=cfg.preprocess_dataset_type,

        # --- 模型 ---
        input_dim=cfg.input_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        model_name=cfg.model_name,

        # --- 训练 ---
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        output_dir=cfg.output_dir,
        resume=cfg.resume_path if cfg.resume_path else None,
        resume_auto=cfg.resume_auto,

        # --- STFT ---
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        win_length=cfg.win_length,
        sr=cfg.sr,

        # --- 路径 ---
        model_path=cfg.model_path or cfg.best_model_path,
        enhance_dir=cfg.enhance_output_dir,
        enhance_output_dir=cfg.enhance_output_dir,

        # --- 开关 ---
        cpu=cfg.cpu,
        save_audio=cfg.save_audio,
        eval_sample_num=cfg.eval_sample_num,
    )

    # 应用覆盖（如 demo 模式的小规模配置）
    if overrides:
        for k, v in overrides.items():
            setattr(args, k, v)

    return args


# ===================== 子命令实现 =====================

def cmd_preprocess(_args=None):
    """数据预处理"""
    from pipeline.data_preprocess import main as preprocess_main

    import argparse
    cfg = get_config()
    pp_args = argparse.Namespace(
        dataset_type=cfg.preprocess_dataset_type,
        sample_num=cfg.sample_num,
        train_ratio=cfg.train_ratio,
    )
    preprocess_main(pp_args)


def cmd_train(_args=None):
    """训练模型"""
    from pipeline.train import train, init_seed

    init_seed()
    print("=" * 60)
    print("[2/3] 模型训练")
    print("=" * 60)
    args = _build_args_from_config()
    train(args)


def cmd_inference(_args=None):
    """推理评估"""
    from pipeline.inference import main as inference_main, init_seed

    init_seed()
    print("=" * 60)
    print("[3/3] 推理评估")
    print("=" * 60)
    args = _build_args_from_config()
    inference_main(args)


def cmd_all(_args=None):
    """完整流程：预处理 → 训练 → 推理"""
    cfg = get_config()

    print("=" * 60)
    print("CRNN 语音增强系统 - 完整流程")
    print("=" * 60)

    cmd_preprocess()
    cmd_train()

    # 推理阶段覆盖部分参数
    args = _build_args_from_config()
    args.model_path = cfg.best_model_path
    args.save_audio = True
    if args.sample_num > 0:
        args.sample_num = max(args.sample_num, 20)

    # 统一字段名映射：main.py 的 enhance_dir → inference.py 的 enhance_output_dir
    args.enhance_output_dir = args.enhance_dir
    cmd_inference()

    print("\n" + "=" * 60)
    print("全部完成！")
    print(f"  模型: {cfg.best_model_path}")
    print(f"  图表: {cfg.figures_dir}/")
    print(f"  日志: {cfg.log_dir}/experiment_log.txt")
    print(f"  音频: {cfg.enhance_output_dir}/")
    print("=" * 60)


def cmd_demo(_args=None):
    """快速演示模式：小规模配置"""
    cfg = get_config()
    overrides = {
        'sample_num': 50,
        'hidden_dim': 128,
        'num_layers': 1,
        'epochs': 2,
        'batch_size': 8,
        'output_dir': 'checkpoints_demo',
        'enhance_dir': 'results/demo_enhanced',
        'enhance_output_dir': 'results/demo_enhanced',
        'save_audio': True,
        'dataset_type': 'all',
    }

    # 临时修改全局配置（不持久化到文件）
    original_attrs = {}
    for k, v in overrides.items():
        original_attrs[k] = getattr(cfg, k, None)
        setattr(cfg, k, v)

    try:
        print("=" * 60)
        print("CRNN 语音增强系统 - 快速演示模式")
        print(f"  样本数: {cfg.sample_num} | 轮数: {cfg.epochs} | 模式: {cfg.load_mode}")
        print("=" * 60)

        cmd_all()
    finally:
        # 恢复原始配置
        for k, v in original_attrs.items():
            setattr(cfg, k, v)


def cmd_report(_args=None):
    """生成学术报告素材"""
    import torch
    import numpy as np
    from report.report_generator import generate_full_report

    print("=" * 60)
    print("CRNN 语音增强系统 - 学术报告生成")
    print("=" * 60)

    args = _build_args_from_config()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"使用设备: {device}")

    # 加载模型
    model_path = args.model_path
    print(f"加载模型: {model_path}")

    from core.model import CRNNSpeechEnhancement
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_args = checkpoint.get("args")
    if model_args is None:
        raise RuntimeError(f"Checkpoint '{model_path}' 缺少 'args' 字段，无法恢复模型配置")

    model = CRNNSpeechEnhancement(
        input_dim=model_args.input_dim,
        hidden_dim=model_args.hidden_dim,
        num_layers=model_args.num_layers,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"模型加载成功 (epoch {checkpoint['epoch']})")

    # 数据目录
    data_dir = args.data_dir

    # 加载训练日志
    train_losses = val_losses = lrs = None
    log_file = os.path.join(args.output_dir, "training_log.npy")
    if os.path.exists(log_file):
        try:
            log_data = np.load(log_file, allow_pickle=True).item()
            train_losses = log_data.get("train_losses")
            val_losses = log_data.get("val_losses")
            print(f"已加载训练日志: {log_file}")
        except Exception:
            pass

    # 从 checkpoint 补全超参数信息
    args.input_dim = model_args.input_dim
    args.hidden_dim = model_args.hidden_dim
    args.num_layers = model_args.num_layers
    args.epochs = checkpoint.get("epoch", getattr(args, "epochs", 0))
    args.model_name = getattr(model_args, "model_name", "CRNN")

    generate_full_report(args, model, data_dir, device, train_losses, val_losses, lrs)

    # ===================== 一键生成全部可视化图表 =====================
    try:
        from report.visualize import generate_all_visualizations

        output_vis_dir = 'results/figures'

        # 构造 metrics_dict（从 report_generator 的输出或重新加载）
        metrics_npy_path = os.path.join(getattr(args, 'enhance_dir', 'results/enhanced'), 'metrics.npy')
        metrics_for_vis = None
        if os.path.exists(metrics_npy_path):
            metrics_for_vis = np.load(metrics_npy_path, allow_pickle=True).item()

        generate_all_visualizations(
            args=args,
            model=model,
            device=device,
            data_dir=data_dir,
            output_dir=output_vis_dir,
            metrics_dict=metrics_for_vis,
            train_losses=train_losses,
            val_losses=val_losses,
            lr_history=lrs if lrs else None,
        )
    except Exception as e:
        print(f"⚠ 一键可视化生成部分失败（不影响已完成的部分）: {e}")

    print("\n" + "=" * 60)
    print("学术报告素材全部生成！")
    print(f"  PDF 图表:  results/report/figures/")
    print(f"  可视化图表: results/figures/")
    print(f"  配置报告:  results/report/config_report.md")
    print(f"  LaTeX 表格: results/report/latex_tables.tex")
    print("=" * 60)


# ===================== 入口 =====================

if __name__ == "__main__":
    COMMANDS = {
        'preprocess': cmd_preprocess,
        'train': cmd_train,
        'inference': cmd_inference,
        'all': cmd_all,
        'demo': cmd_demo,
        'report': cmd_report,
    }

    if len(sys.argv) < 2:
        print(__doc__)
        print("\n可用命令:", ", ".join(COMMANDS.keys()))
        sys.exit(0)

    command = sys.argv[1].lower()
    if command not in COMMANDS:
        print(f"未知命令: {command}")
        print(f"可用命令:", ", ".join(COMMANDS.keys()))
        sys.exit(1)

    COMMANDS[command]()
