#!/usr/bin/env python3
"""
验证PyTorch CUDA安装是否成功
"""

import sys

def verify_pytorch_cuda():
    print("=" * 60)
    print("PyTorch CUDA Verification")
    print("=" * 60)
    
    try:
        import torch
        print(f"\n✓ PyTorch imported successfully")
        print(f"PyTorch version: {torch.__version__}")
        
        # 检查CUDA是否可用
        cuda_available = torch.cuda.is_available()
        print(f"\nCUDA available: {cuda_available}")
        
        if cuda_available:
            print("✅ SUCCESS! CUDA is now available!")
            print(f"\nCUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"Number of GPUs: {torch.cuda.device_count()}")
            
            # 显示GPU信息
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print(f"  Total Memory: {props.total_memory / 1024**3:.1f} GB")
                
                # 测试GPU
                print(f"\n  Testing GPU {i}...")
                try:
                    # 创建测试张量
                    x = torch.randn(1000, 1000).cuda(i)
                    y = torch.randn(1000, 1000).cuda(i)
                    z = torch.matmul(x, y)
                    print(f"  ✓ GPU {i} computation test passed!")
                    
                    # 显示当前GPU内存使用
                    print(f"  Memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.1f} MB")
                    print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.1f} MB")
                except Exception as e:
                    print(f"  ✗ GPU {i} test failed: {e}")
            
            print("\n" + "=" * 60)
            print("✅ PyTorch CUDA setup is complete!")
            print("You can now use GPU acceleration for your models.")
            print("=" * 60)
            
        else:
            print("\n❌ CUDA is still not available!")
            print("\nTroubleshooting steps:")
            print("1. Make sure you uninstalled the CPU-only version completely:")
            print("   pip uninstall torch torchvision torchaudio -y")
            print("\n2. Clear pip cache:")
            print("   pip cache purge")
            print("\n3. Install the correct version:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
            print("\n4. If still not working, try CUDA 11.8 version:")
            print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            
            # 检查PyTorch是否包含CUDA
            if hasattr(torch.version, 'cuda') and torch.version.cuda:
                print(f"\nNote: PyTorch was built with CUDA {torch.version.cuda}, but cannot access GPU.")
                print("This might be a driver compatibility issue.")
            else:
                print("\n⚠ This PyTorch installation is CPU-only!")
                
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        print("\nInstall PyTorch with:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

def check_other_packages():
    """检查其他相关包"""
    print("\n" + "=" * 60)
    print("Other Package Checks")
    print("=" * 60)
    
    packages = {
        'transformers': None,
        'accelerate': None,
        'xformers': None,  # 可选，但能加速transformer模型
        'bitsandbytes': None,  # 可选，用于8bit量化
    }
    
    for package_name in packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'Unknown')
            packages[package_name] = version
            print(f"✓ {package_name:15s}: {version}")
        except ImportError:
            print(f"✗ {package_name:15s}: Not installed")
    
    # 建议安装缺失的包
    missing = [k for k, v in packages.items() if v is None]
    if missing:
        print(f"\nTo install missing packages:")
        print(f"pip install {' '.join(missing)}")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" PYTORCH CUDA VERIFICATION ".center(60))
    print("=" * 60 + "\n")
    
    verify_pytorch_cuda()
    check_other_packages()
    
    print("\n" + "=" * 60)
    print("Next steps:")
    print("1. If CUDA is working, you can proceed with your deepfake detection code")
    print("2. The RTX 4090 should provide excellent performance for LLaVA model")
    print("=" * 60)