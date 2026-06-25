#!/usr/bin/env python
"""
依赖安装和检查脚本
运行此脚本自动检查并安装缺失的依赖
"""

import subprocess
import sys

# 必需的依赖包列表
REQUIRED_PACKAGES = [
    'torch',
    'numpy',
    'scipy',
    'soundfile',
    'librosa',
    'tqdm',
    'scikit-learn',
    'matplotlib',
]

def check_package(package_name):
    """检查包是否已安装"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """使用 pip 安装包"""
    print(f"  正在安装 {package_name}...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    print("=" * 60)
    print("CRNN 语音增强系统 - 依赖检查")
    print("=" * 60)

    missing_packages = []
    installed_packages = []

    print("\n检查依赖包...")
    for package in REQUIRED_PACKAGES:
        if check_package(package):
            print(f"  ✓ {package}")
            installed_packages.append(package)
        else:
            print(f"  ✗ {package} (缺失)")
            missing_packages.append(package)

    print("\n" + "-" * 60)
    print(f"已安装: {len(installed_packages)}/{len(REQUIRED_PACKAGES)}")
    print(f"缺失: {len(missing_packages)}/{len(REQUIRED_PACKAGES)}")

    if missing_packages:
        print("\n" + "=" * 60)
        print("正在安装缺失的依赖...")
        print("=" * 60)

        failed = []
        for package in missing_packages:
            if install_package(package):
                print(f"  ✓ {package} 安装成功")
            else:
                print(f"  ✗ {package} 安装失败")
                failed.append(package)

        if failed:
            print("\n" + "!" * 60)
            print("以下包安装失败，请手动安装:")
            for pkg in failed:
                print(f"  pip install {pkg}")
            print("!" * 60)
            return False
        else:
            print("\n✅ 所有依赖安装成功！")
            return True
    else:
        print("\n✅ 所有依赖已满足！")
        return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
