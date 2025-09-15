#!/usr/bin/env python3
"""
检查当前Python环境并提供正确的安装步骤
"""

import sys
import os
import subprocess
import site

def check_current_environment():
    print("=" * 60)
    print("当前Python环境信息")
    print("=" * 60)
    
    # 1. Python版本和路径
    print(f"\nPython版本: {sys.version}")
    print(f"Python可执行文件: {sys.executable}")
    print(f"Python前缀: {sys.prefix}")
    
    # 2. 检查是否在虚拟环境中
    print("\n" + "-" * 40)
    print("虚拟环境检查:")
    print("-" * 40)
    
    # 检查虚拟环境的几种方式
    in_virtualenv = False
    env_type = "系统Python"
    
    # 检查VIRTUAL_ENV环境变量（venv/virtualenv）
    if 'VIRTUAL_ENV' in os.environ:
        in_virtualenv = True
        env_type = "venv/virtualenv"
        print(f"✓ 在虚拟环境中: {os.environ['VIRTUAL_ENV']}")
    
    # 检查CONDA环境
    elif 'CONDA_DEFAULT_ENV' in os.environ:
        in_virtualenv = True
        env_type = "conda"
        print(f"✓ 在Conda环境中: {os.environ['CONDA_DEFAULT_ENV']}")
        if 'CONDA_PREFIX' in os.environ:
            print(f"  Conda前缀: {os.environ['CONDA_PREFIX']}")
    
    # 检查是否是venv（通过路径）
    elif hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        in_virtualenv = True
        env_type = "venv"
        print(f"✓ 在虚拟环境中")
    
    else:
        print("✗ 不在虚拟环境中（使用系统Python）")
    
    print(f"环境类型: {env_type}")
    
    # 3. 显示pip信息
    print("\n" + "-" * 40)
    print("Pip信息:")
    print("-" * 40)
    
    try:
        pip_version = subprocess.check_output([sys.executable, '-m', 'pip', '--version'], text=True)
        print(f"Pip版本: {pip_version.strip()}")
    except:
        print("✗ Pip未安装或无法访问")
    
    # 4. 站点包路径
    print("\n" + "-" * 40)
    print("包安装路径:")
    print("-" * 40)
    site_packages = site.getsitepackages()
    for i, path in enumerate(site_packages, 1):
        print(f"{i}. {path}")
    
    return in_virtualenv, env_type

def list_environments():
    """列出可能的Python环境"""
    print("\n" + "=" * 60)
    print("查找可能的Python环境")
    print("=" * 60)
    
    possible_locations = []
    
    # 1. 检查当前目录的虚拟环境
    current_dir = os.getcwd()
    common_env_names = ['venv', 'env', '.venv', '.env', 'virtualenv', 'deepfake_env', 'llava_env']
    
    print("\n当前目录中的虚拟环境:")
    for env_name in common_env_names:
        env_path = os.path.join(current_dir, env_name)
        if os.path.exists(env_path):
            activate_script = os.path.join(env_path, 'Scripts', 'activate.bat')
            if os.path.exists(activate_script):
                print(f"✓ 找到: {env_path}")
                possible_locations.append(env_path)
    
    if not possible_locations:
        print("✗ 当前目录没有找到虚拟环境")
    
    # 2. 检查Conda环境
    print("\nConda环境:")
    try:
        conda_envs = subprocess.check_output("conda env list", shell=True, text=True)
        print(conda_envs)
    except:
        print("✗ Conda未安装或不在PATH中")
    
    return possible_locations

def provide_instructions(in_virtualenv, env_type):
    """提供基于当前环境的指令"""
    print("\n" + "=" * 60)
    print("建议的操作步骤")
    print("=" * 60)
    
    if not in_virtualenv:
        print("\n⚠️ 你当前不在虚拟环境中！")
        print("\n请先激活你的Python环境：")
        print("\n如果你有venv虚拟环境:")
        print("  Windows: your_env\\Scripts\\activate")
        print("  Linux/Mac: source your_env/bin/activate")
        print("\n如果你有Conda环境:")
        print("  conda activate your_env_name")
        print("\n然后重新运行这个脚本。")
    else:
        print(f"\n✓ 你在{env_type}环境中")
        print("\n现在可以安装CUDA版本的PyTorch：")
        print("\n1. 首先卸载现有的CPU版本:")
        print(f"   {sys.executable} -m pip uninstall torch torchvision torchaudio -y")
        print("\n2. 清理pip缓存:")
        print(f"   {sys.executable} -m pip cache purge")
        print("\n3. 安装CUDA版本:")
        
        # 检查Python版本
        if sys.version_info[:2] == (3, 13):
            print("\n⚠️ 警告: Python 3.13可能还不支持CUDA版本的PyTorch")
            print("   尝试安装最新版本:")
            print(f"   {sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\n   如果失败，你可能需要:")
            print("   - 创建一个Python 3.11的新环境")
            print("   - 或使用Conda环境")
        else:
            print(f"   {sys.executable} -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        
        print("\n4. 验证安装:")
        print(f'   {sys.executable} -c "import torch; print(torch.cuda.is_available())"')

def main():
    print("\n" + "=" * 60)
    print(" Python环境检查工具 ".center(60))
    print("=" * 60)
    
    # 检查当前环境
    in_virtualenv, env_type = check_current_environment()
    
    # 列出可能的环境
    environments = list_environments()
    
    # 提供指令
    provide_instructions(in_virtualenv, env_type)
    
    # 如果找到了环境但未激活
    if not in_virtualenv and environments:
        print("\n" + "=" * 60)
        print("快速激活命令")
        print("=" * 60)
        print("\n检测到以下环境，使用相应命令激活：")
        for env_path in environments:
            env_name = os.path.basename(env_path)
            print(f"\n环境: {env_name}")
            print(f"激活命令: {env_path}\\Scripts\\activate")

if __name__ == "__main__":
    main()