#!/usr/bin/env python3
"""
Step 1: Check GPU and System Information
"""

import subprocess
import sys
import platform

def run_command(command):
    """Run a command and return output"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.stdout.strip()
    except:
        return "Command failed"

def check_system_info():
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    
    # Basic system info
    print(f"\nOperating System: {platform.system()} {platform.release()}")
    print(f"Platform: {platform.platform()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    # Check if running on Windows
    if platform.system() == "Windows":
        print("\n✓ Running on Windows")
        
        # Check for NVIDIA GPU using Windows commands
        print("\n" + "=" * 60)
        print("GPU DETECTION (Windows)")
        print("=" * 60)
        
        # Method 1: Check using wmic
        print("\nMethod 1 - WMIC Check:")
        gpu_info = run_command("wmic path win32_VideoController get name")
        print(gpu_info)
        
        # Method 2: Check using nvidia-smi
        print("\nMethod 2 - NVIDIA-SMI Check:")
        nvidia_smi = run_command("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
        if "failed" not in nvidia_smi.lower() and "not found" not in nvidia_smi.lower():
            print("✓ NVIDIA GPU detected!")
            print(nvidia_smi)
            
            # Get more detailed info
            print("\nDetailed GPU Info:")
            detailed = run_command("nvidia-smi")
            print(detailed)
        else:
            print("✗ nvidia-smi not found or failed")
            print("  This means either:")
            print("  1. You don't have an NVIDIA GPU")
            print("  2. NVIDIA drivers are not installed")
            print("  3. nvidia-smi is not in PATH")
        
        # Check CUDA installation
        print("\n" + "=" * 60)
        print("CUDA CHECK")
        print("=" * 60)
        
        # Check for CUDA in common locations
        import os
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
            r"C:\Program Files\NVIDIA Corporation\CUDA",
            os.environ.get("CUDA_PATH", ""),
            os.environ.get("CUDA_HOME", "")
        ]
        
        cuda_found = False
        for cuda_path in cuda_paths:
            if cuda_path and os.path.exists(cuda_path):
                print(f"✓ CUDA installation found at: {cuda_path}")
                cuda_found = True
                
                # Try to find version
                version_file = os.path.join(cuda_path, "version.txt")
                if os.path.exists(version_file):
                    with open(version_file, 'r') as f:
                        print(f"  Version: {f.read().strip()}")
                
                # Check for nvcc
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc.exe")
                if os.path.exists(nvcc_path):
                    nvcc_version = run_command(f'"{nvcc_path}" --version')
                    if nvcc_version:
                        for line in nvcc_version.split('\n'):
                            if 'release' in line.lower():
                                print(f"  NVCC: {line.strip()}")
                break
        
        if not cuda_found:
            print("✗ CUDA installation not found in common locations")
            
        # Check environment variables
        print("\n" + "=" * 60)
        print("ENVIRONMENT VARIABLES")
        print("=" * 60)
        
        env_vars = ["CUDA_PATH", "CUDA_HOME", "PATH"]
        for var in env_vars:
            value = os.environ.get(var, "Not set")
            if var == "PATH" and "cuda" in value.lower():
                # Only show CUDA-related paths
                paths = value.split(';')
                cuda_paths = [p for p in paths if 'cuda' in p.lower() or 'nvidia' in p.lower()]
                if cuda_paths:
                    print(f"{var}: (CUDA-related paths only)")
                    for p in cuda_paths[:5]:  # Show first 5 CUDA paths
                        print(f"  - {p}")
            elif var != "PATH":
                print(f"{var}: {value}")

def check_pytorch_cuda():
    print("\n" + "=" * 60)
    print("PYTORCH CUDA CHECK")
    print("=" * 60)
    
    try:
        import torch
        print(f"\n✓ PyTorch Version: {torch.__version__}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        
        if not torch.cuda.is_available():
            print("\n⚠ PyTorch cannot access CUDA")
            print("\nPossible reasons:")
            print("1. PyTorch CPU-only version is installed")
            print("2. CUDA is not properly installed")
            print("3. CUDA version incompatible with PyTorch")
            
            # Check if PyTorch was built with CUDA
            print(f"\nPyTorch built with CUDA: {torch.version.cuda is not None}")
            if torch.version.cuda:
                print(f"PyTorch CUDA version: {torch.version.cuda}")
            else:
                print("✗ This PyTorch installation is CPU-only!")
                print("\nTo fix: Reinstall PyTorch with CUDA support")
        else:
            print(f"✓ CUDA Version (PyTorch): {torch.version.cuda}")
            print(f"✓ cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\nGPU {i}: {props.name}")
                print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
        
    except ImportError:
        print("✗ PyTorch is not installed")
        print("Install with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")

def main():
    print("\n" + "=" * 60)
    print(" GPU AND CUDA ENVIRONMENT CHECK ".center(60))
    print("=" * 60 + "\n")
    
    # Run all checks
    check_system_info()
    check_pytorch_cuda()
    
    # Recommendations
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("\nBased on the checks above, follow the appropriate steps:")
    print("\n1. If you have an NVIDIA GPU but CUDA is not detected:")
    print("   - Install NVIDIA drivers: https://www.nvidia.com/Download/index.aspx")
    print("   - Install CUDA Toolkit 11.8 or 12.1: https://developer.nvidia.com/cuda-downloads")
    print("\n2. If PyTorch is CPU-only:")
    print("   - Uninstall current PyTorch: pip uninstall torch torchvision torchaudio")
    print("   - Reinstall with CUDA support:")
    print("     # For CUDA 11.8:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
    print("     # For CUDA 12.1:")
    print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\n3. If you don't have an NVIDIA GPU:")
    print("   - The code will run on CPU (much slower)")
    print("   - Consider using Google Colab or cloud GPU services")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()