"""
PNG → PDF 批量转换工具
用法:
  python tools/convert_figures_to_pdf.py --input_dir results/figures
  python tools/convert_figures_to_pdf.py --input_dir results/figures --replace
"""

import os
import glob
import argparse
from PIL import Image


def convert_png_to_pdf(png_path, pdf_path):
    """将单个 PNG 文件转换为 PDF"""
    img = Image.open(png_path)
    # RGB 模式才能保存为 PDF（如果是 RGBA 则转 RGB，白色背景）
    if img.mode == 'RGBA':
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(pdf_path, 'PDF')


def main():
    parser = argparse.ArgumentParser(description='批量将 PNG 图片转换为 PDF')
    parser.add_argument('--input_dir', required=True, help='输入目录（包含 .png 文件）')
    parser.add_argument('--output_dir', default=None, help='输出目录（默认同 input_dir）')
    parser.add_argument('--replace', action='store_true', help='转换后删除原 PNG 文件')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir if args.output_dir else input_dir

    if not os.path.isdir(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        return

    # 自动创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 扫描所有 png 文件
    png_files = sorted(glob.glob(os.path.join(input_dir, '*.png')))

    if not png_files:
        print(f"在 {input_dir} 中未找到任何 .png 文件")
        return

    print(f"找到 {len(png_files)} 个 PNG 文件，开始转换...")
    print(f"输出目录: {output_dir}")
    print("-" * 50)

    success_count = 0
    failed_count = 0
    failed_files = []
    converted_paths = []  # 记录成功转换的源文件路径，用于 --replace

    for i, png_path in enumerate(png_files, 1):
        basename = os.path.splitext(os.path.basename(png_path))[0]
        pdf_path = os.path.join(output_dir, basename + '.pdf')

        try:
            convert_png_to_pdf(png_path, pdf_path)
            success_count += 1
            converted_paths.append(png_path)
            print(f"[{i}/{len(png_files)}] ✓ {os.path.basename(png_path)} -> {os.path.basename(pdf_path)}")
        except Exception as e:
            failed_count += 1
            failed_files.append((os.path.basename(png_path), str(e)))
            print(f"[{i}/{len(png_files)}] ✗ {os.path.basename(png_path)} 失败: {e}")

    print("-" * 50)

    # 如果指定了 --replace，删除成功转换的原 png 文件
    if args.replace and converted_paths:
        deleted_count = 0
        for png_path in converted_paths:
            try:
                os.remove(png_path)
                deleted_count += 1
            except OSError as e:
                print(f"警告: 无法删除 {png_path}: {e}")
        print(f"已删除 {deleted_count} 个原 PNG 文件")

    # 汇总信息
    print("\n========== 转换汇总 ==========")
    print(f"总共扫描: {len(png_files)} 个文件")
    print(f"成功转换: {success_count} 个文件")
    if failed_count > 0:
        print(f"转换失败: {failed_count} 个文件:")
        for name, err in failed_files:
            print(f"  - {name}: {err}")
    print(f"输出目录: {os.path.abspath(output_dir)}")


if __name__ == '__main__':
    main()
